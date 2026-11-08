# Copyright 2026 The muscle-supermimic Authors. Modified for MyoSuite by the MyoSuite Authors.
#
# Licensed under the Apache License, Version 2.0. See NOTICE-muscle-supermimic.txt in
# this package for the notices of the code it includes.
"""Reference motion library: clips resampled to the control rate, with FK of the tracked bodies.

Port of supermimic_vibed's ``MotionLib`` (github.com/zita-ch/supermimic_vibed, MIT). A clip is one
``.npz`` in one of these formats:

* **canonical** -- ``fps``, ``root_pos`` (T,3), ``root_quat`` (T,4, wxyz, canonical root frame: x
  forward, y left, z up), ``joint_pos`` (T,J) and ``joint_names`` (J,). Joints the model lacks are dropped;
* **MuscleMimic trajectory** -- ``qpos`` (T,nq) in the layout given by ``joint_names`` (+ ``jnt_type``),
  the root free joint first, and ``frequency``, as MuscleMimic's retargeted MyoFullBody caches
  (``MyoFullBody/gmr/KIT/<subject>/<clip>.npz``).

Coupled joints in a clip are ignored (they follow from the independent ones). A directory is searched
recursively, and a ``.txt`` file lists clips (see :func:`motion_files`); clips are named by their path
relative to the directory they share. Changed for MyoSuite: a list's paths may be relative to a separate root
(the dataset's download directory), and the canonical format needs ``joint_names``.

Ported from muscle-supermimic (github.com/kaibohex/muscle-supermimic, Apache-2.0) at commit
8a6244b, ``src/muscle_supermimic/tasks/tracking/mdp/motion.py``.
"""

from __future__ import annotations

import glob
import os

import mujoco
import numpy as np
import torch

from myosuite.core.model_names import resolve_model_names

from .model_info import ModelInfo
from .rotations import (
    mat_to_quat,
    quat_slerp,
    quat_to_mat,
    so3_log,
)

from .kinematics import SkeletonKinematics

QPOS_WIDTH = {int(mujoco.mjtJoint.mjJNT_FREE): 7, int(mujoco.mjtJoint.mjJNT_BALL): 4,
              int(mujoco.mjtJoint.mjJNT_SLIDE): 1, int(mujoco.mjtJoint.mjJNT_HINGE): 1}  # fmt: skip


def motion_files(path: str, root: str = "") -> list[str]:
    """The clips a motion path names.

    Args:
        path: A clip, a directory (every ``.npz`` under it, sorted), or a ``.txt`` list of clips, one path per line
            (``#`` starts a comment).
        root: The directory a list's relative paths start from (default: the list's own directory).

    Returns:
        The clip files.

    Raises:
        FileNotFoundError: If a listed clip is missing.
    """
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "**", "*.npz"), recursive=True))
    if not path.endswith(".txt"):
        return [path]
    with open(path) as f:
        lines = [line.split("#", 1)[0].strip() for line in f]
    files = [
        os.path.join(root or os.path.dirname(path), line) for line in lines if line
    ]
    missing = [f for f in files if not os.path.isfile(f)]
    if missing:
        raise FileNotFoundError(
            f"{path}: {len(missing)} of {len(files)} listed clips missing, e.g. {missing[:3]}"
        )
    return files


def _model_names(names, info: ModelInfo) -> list[str]:
    """A clip's joint names in the model's naming (MuscleMimic's ``elbow_flex_*`` are myo_sim's ``elbow_flexion_*``)."""
    model = [info.root_joint_name, *info.joint_names]
    return resolve_model_names([str(n) for n in names], model, field_name="qpos")


def load_clip(
    path: str, info: ModelInfo, R_off: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """-> (fps, root_pos (T,3), root_quat (T,4) canonical wxyz, joint_pos (T,J) in ``info.indep_names`` order)."""
    data = np.load(path, allow_pickle=False)
    if "qpos" in data.files:
        return (float(data["frequency"]),) + trajectory_to_canonical(data, info, R_off)
    q = np.asarray(data["joint_pos"], dtype=np.float64)
    names = _model_names(data["joint_names"], info)
    assert q.shape[1] == len(
        names
    ), f"{path}: joint_pos has {q.shape[1]} columns for {len(names)} joint names"
    missing = [n for n in info.indep_names if n not in names]
    assert not missing, f"{path}: no data for joints {missing}"
    q = q[:, [names.index(n) for n in info.indep_names]]
    return (
        float(data["fps"]),
        np.asarray(data["root_pos"], np.float64),
        np.asarray(data["root_quat"], np.float64),
        q,
    )


def trajectory_to_canonical(data, info: ModelInfo, R_off: np.ndarray):
    """MuscleMimic trajectory (root free joint + named joints in ``qpos``) -> canonical root pose + independent
    joint angles. The trajectory's free joint must be the model's root."""
    names = _model_names(data["joint_names"], info)
    free, hinge = mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_HINGE
    types = (
        data["jnt_type"]
        if "jnt_type" in data.files
        else [free] + [hinge] * (len(names) - 1)
    )
    adr = np.cumsum([0] + [QPOS_WIDTH[int(t)] for t in types])
    qpos = np.asarray(data["qpos"], np.float64)
    assert (
        qpos.shape[1] == adr[-1]
    ), f"qpos has {qpos.shape[1]} columns, the joints need {adr[-1]}"
    assert (
        names[0] == info.root_joint_name and int(types[0]) == free
    ), f"the first joint must be the root free joint {info.root_joint_name!r}, got {names[0]!r}"
    assert (
        "split_points" not in data.files or len(data["split_points"]) == 2
    ), "several trajectories in one file"
    missing = [n for n in info.indep_names if n not in names]
    assert not missing, f"no data for joints {missing}"
    R = quat_to_mat(torch.from_numpy(qpos[:, 3:7])) @ torch.from_numpy(R_off)
    return (
        qpos[:, :3].copy(),
        mat_to_quat(R).numpy(),
        qpos[:, [adr[names.index(n)] for n in info.indep_names]],
    )


class MotionLib:
    def __init__(
        self,
        files: list[str],
        kin: SkeletonKinematics,
        control_dt: float,
        device: str,
        bin_frames: int = 50,
    ):
        assert files, "no motion clips"
        self.device, self.dt, self.bin_frames = device, control_dt, bin_frames
        info, R_off = kin.info, kin.R_off.double().cpu().numpy()
        base = os.path.commonpath([os.path.dirname(os.path.abspath(f)) for f in files])
        self.names = [
            os.path.splitext(os.path.relpath(os.path.abspath(f), base))[0]
            for f in files
        ]
        ps, qs, js, lens = [], [], [], []
        for f in files:
            fps, p, quat, j = load_clip(f, info, R_off)
            p, quat, j = (torch.tensor(x, dtype=torch.float32) for x in (p, quat, j))
            # resample to the control rate
            T_src = p.shape[0]
            dur = (T_src - 1) / fps
            n = int(np.floor(dur / control_dt + 1e-6)) + 1
            t = torch.arange(n, dtype=torch.float32) * control_dt * fps
            i0 = torch.clamp(t.floor().long(), max=T_src - 2)
            a = (t - i0.float()).clamp(0, 1)
            p = p[i0] * (1 - a[:, None]) + p[i0 + 1] * a[:, None]
            j = j[i0] * (1 - a[:, None]) + j[i0 + 1] * a[:, None]
            quat = quat_slerp(quat[i0], quat[i0 + 1], a)
            ps.append(p), qs.append(quat), js.append(j)
            lens.append(n)
        self.clip_len = torch.tensor(lens, device=device)
        self.clip_start = torch.cumsum(
            torch.tensor([0] + lens[:-1], dtype=torch.long), 0
        ).to(device)
        self.clip_end = (
            self.clip_start + self.clip_len - 1
        )  # last valid frame (inclusive)
        self.num_frames = int(sum(lens))
        self.clip_id = torch.cat(
            [torch.full((n,), i, dtype=torch.long) for i, n in enumerate(lens)]
        ).to(device)

        root_p, root_R, root_v, root_w, jq, jqd = [], [], [], [], [], []
        for p, quat, j in zip(ps, qs, js):
            R = quat_to_mat(quat)
            v = _gradient(p, control_dt)
            qd = _gradient(j, control_dt)
            # angular velocity (world) by central differences of rotations
            w = torch.zeros_like(p)
            w[1:-1] = so3_log(R[2:] @ R[:-2].transpose(-1, -2)) / (2 * control_dt)
            w[0] = so3_log(R[1] @ R[0].T) / control_dt
            w[-1] = so3_log(R[-1] @ R[-2].T) / control_dt
            (
                root_p.append(p),
                root_R.append(R),
                root_v.append(v),
                root_w.append(w),
                jq.append(j),
                jqd.append(qd),
            )
        cat = lambda xs: torch.cat(xs).to(device)  # noqa: E731
        self.root_p, self.root_R, self.root_v, self.root_w = (
            cat(root_p),
            cat(root_R),
            cat(root_v),
            cat(root_w),
        )
        self.q, self.qd = cat(jq), cat(jqd)
        self.root_quat = mat_to_quat(self.root_R)
        with torch.no_grad():
            bp, bR, bv, bw = [], [], [], []
            for s in range(0, self.num_frames, 65536):
                e = min(s + 65536, self.num_frames)
                out = kin.forward(
                    self.root_p[s:e],
                    self.root_R[s:e],
                    self.root_v[s:e],
                    self.root_w[s:e],
                    self.q[s:e],
                    self.qd[s:e],
                )
                (
                    bp.append(out[0]),
                    bR.append(out[1]),
                    bv.append(out[2]),
                    bw.append(out[3]),
                )
            self.body_p, self.body_R, self.body_v, self.body_w = (
                torch.cat(bp),
                torch.cat(bR),
                torch.cat(bv),
                torch.cat(bw),
            )
        # 1 s bins for return-based reset sampling
        bin_clip, bin_start = [], []
        for c in range(len(lens)):
            for b in range(int(np.ceil(lens[c] / bin_frames))):
                bin_clip.append(c)
                bin_start.append(int(self.clip_start[c]) + b * bin_frames)
        self.bin_clip = torch.tensor(bin_clip, device=device)
        self.bin_start = torch.tensor(bin_start, device=device)
        self.num_bins = len(bin_start)
        self.frame_bin = torch.zeros(self.num_frames, dtype=torch.long, device=device)
        for b in range(self.num_bins):
            s = bin_start[b]
            e = min(s + bin_frames, int(self.clip_end[bin_clip[b]]) + 1)
            self.frame_bin[s:e] = b

    @staticmethod
    def from_path(path: str, kin: SkeletonKinematics, control_dt: float, device: str, bin_frames: int = 50,
                  root: str = "") -> MotionLib:  # fmt: skip
        """The library of the clips ``path`` names (see :func:`motion_files`)."""
        return MotionLib(motion_files(path, root), kin, control_dt, device, bin_frames)

    def ref_state(self, f: torch.Tensor):
        return (
            self.root_p[f],
            self.root_R[f],
            self.root_v[f],
            self.root_w[f],
            self.q[f],
            self.qd[f],
        )


def _gradient(x: torch.Tensor, dt: float) -> torch.Tensor:
    g = torch.zeros_like(x)
    g[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    g[0] = (x[1] - x[0]) / dt
    g[-1] = (x[-1] - x[-2]) / dt
    return g


def write_synthetic_clip(path: str | os.PathLike, m: mujoco.MjModel, seconds: float, speed: float, amp: float,
                         seed: int, climb: float = 0.0) -> None:  # fmt: skip
    """A MuscleMimic-format clip for tests and smoke runs: the default pose moving forward at ``speed`` m/s (and up at
    ``climb``) with every independent joint oscillating by ``amp`` of its half range at 1 Hz, sampled at 100 Hz.

    Args:
        path: The ``.npz`` to write.
        m: The body model (e.g. :func:`.body.compiled_body`).
        seconds: Duration.
        speed: Forward speed along the world x axis.
        amp: Oscillation amplitude as a fraction of each joint's half range.
        seed: Seed of the oscillation phases.
        climb: Vertical speed.
    """
    fps = 100.0
    t = np.arange(int(seconds * fps) + 1) / fps
    qpos = np.tile(m.qpos0, (len(t), 1))
    qpos[:, 0] = speed * t
    qpos[:, 2] += climb * t
    rng = np.random.default_rng(seed)
    for j in ModelInfo(m).indep_ids:
        adr, (lo, hi) = m.jnt_qposadr[j], m.jnt_range[j]
        wave = amp * 0.5 * (hi - lo) * np.sin(2 * np.pi * t + rng.uniform(0, 2 * np.pi))
        qpos[:, adr] = np.clip(m.qpos0[adr] + wave, lo, hi)
    names = np.array([m.joint(j).name for j in range(m.njnt)])
    np.savez(path, qpos=qpos, joint_names=names, jnt_type=m.jnt_type, frequency=fps)
