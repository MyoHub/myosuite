# Copyright 2026 The muscle-supermimic Authors. Modified for MyoSuite by the MyoSuite Authors.
#
# Licensed under the Apache License, Version 2.0. See NOTICE-muscle-supermimic.txt in
# this package for the notices of the code it includes.
"""Reference motion library: clips resampled to the control step, with the reference
quantities of every frame.

A clip is one ``.npz`` in one of these formats:

* **MuscleMimic trajectory**: ``qpos`` (T, nq) in the layout of ``joint_names`` (+
  ``jnt_type``), the root free joint first, and ``frequency``, as MuscleMimic's
  retargeted MyoFullBody clips (``MyoFullBody/gmr/KIT/<subject>/<clip>.npz``);
* **canonical**: ``fps``, ``root_pos`` (T, 3), ``root_quat`` (T, 4, wxyz, canonical
  root frame: x forward, z up), ``joint_pos`` (T, J) and ``joint_names``.

Coupled joints in a clip are ignored (they follow from the independent ones). Clips are
resampled to the control step (linear, slerp for the root), velocities are central
differences, and the tracked bodies' poses and velocities are MuJoCo's forward
kinematics of each frame on the task body. Arrays are NumPy (float32);
:meth:`MotionLib.map` converts them (e.g. to torch tensors on a device).

Ported from muscle-supermimic (github.com/kaibohex/muscle-supermimic, Apache-2.0) at
commit 8a6244b, ``src/muscle_supermimic/tasks/tracking/mdp/motion.py`` (itself a port
of supermimic_vibed's ``MotionLib``, MIT).
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import functools
import json
import glob
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from myosuite.core.model_names import resolve_model_names
from myosuite.physics.joint_coupling import project_joint_couplings
from myosuite.terms import mimic_track as mt

from .models import ModelInfo, root_offset

QPOS_WIDTH = {
    int(mujoco.mjtJoint.mjJNT_FREE): 7,
    int(mujoco.mjtJoint.mjJNT_BALL): 4,
    int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
    int(mujoco.mjtJoint.mjJNT_HINGE): 1,
}


def motion_files(path: str | os.PathLike, root: str | os.PathLike = "") -> list[str]:
    """The clips a motion path names.

    Args:
        path: A clip, a directory (every ``.npz`` under it, sorted), or a ``.txt``
            list of clips, one path per line (``#`` starts a comment).
        root: Directory a list's relative paths start from (default: the list's).

    Returns:
        The clip files.

    Raises:
        FileNotFoundError: If a listed clip is missing.
    """
    path = str(path)
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "**", "*.npz"), recursive=True))
    if not path.endswith(".txt"):
        return [path]
    with open(path) as f:
        lines = [line.split("#", 1)[0].strip() for line in f]
    base = str(root) or os.path.dirname(path)
    files = [os.path.join(base, line) for line in lines if line]
    missing = [f for f in files if not os.path.isfile(f)]
    if missing:
        raise FileNotFoundError(
            f"{path}: {len(missing)} of {len(files)} listed clips missing, e.g. {missing[:3]}"
        )
    return files


def _model_names(names: Sequence[str], info: ModelInfo) -> list[str]:
    """A clip's joint names in the model's naming (``elbow_flex_*`` -> ``elbow_flexion_*``)."""
    model = [info.root_joint_name, *info.joint_names]
    return resolve_model_names([str(n) for n in names], model, field_name="qpos")


def load_clip(
    path: str, info: ModelInfo, R_off: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """``(fps, root_pos (T,3), root_quat (T,4) of the free-joint body, indep. joints (T,J))``."""
    data = np.load(path, allow_pickle=False)
    names = _model_names(data["joint_names"], info)
    missing = [n for n in info.indep_names if n not in names]
    if missing:
        raise ValueError(f"{path}: no data for joints {missing}")
    if "qpos" not in data.files:
        q = np.asarray(data["joint_pos"], np.float64)
        if q.shape[1] != len(names):
            raise ValueError(
                f"{path}: {q.shape[1]} joint columns for {len(names)} names"
            )
        R = mt.quat_to_mat(np, np.asarray(data["root_quat"], np.float64)) @ R_off.T
        return (
            float(data["fps"]),
            np.asarray(data["root_pos"], np.float64),
            mt.mat_to_quat(np, R),
            q[:, [names.index(n) for n in info.indep_names]],
        )
    free = int(mujoco.mjtJoint.mjJNT_FREE)
    types = (
        data["jnt_type"]
        if "jnt_type" in data.files
        else [free] + [1] * (len(names) - 1)
    )
    adr = np.cumsum([0] + [QPOS_WIDTH[int(t)] for t in types])
    qpos = np.asarray(data["qpos"], np.float64)
    if qpos.shape[1] != adr[-1]:
        raise ValueError(
            f"{path}: qpos has {qpos.shape[1]} columns, the joints need {adr[-1]}"
        )
    if names[0] != info.root_joint_name or int(types[0]) != free:
        raise ValueError(f"{path}: the first joint must be the root free joint")
    if "split_points" in data.files and len(data["split_points"]) != 2:
        raise ValueError(f"{path}: several trajectories in one file")
    quat = qpos[:, 3:7] / np.linalg.norm(qpos[:, 3:7], axis=-1, keepdims=True)
    return (
        float(data["frequency"]),
        qpos[:, :3].copy(),
        np.where(quat[:, :1] < 0, -quat, quat),
        qpos[:, [adr[names.index(n)] for n in info.indep_names]],
    )


def _slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    dot = (q0 * q1).sum(-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    theta = np.arccos(np.clip(np.abs(dot), None, 1.0))
    s = np.sin(theta)
    t = t[..., None]
    w0 = np.where(s < 1e-6, 1 - t, np.sin((1 - t) * theta) / (s + 1e-12))
    w1 = np.where(s < 1e-6, t, np.sin(t * theta) / (s + 1e-12))
    q = w0 * q0 + w1 * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _gradient(x: np.ndarray, dt: float) -> np.ndarray:
    g = np.empty_like(x)
    g[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    g[0] = (x[1] - x[0]) / dt
    g[-1] = (x[-1] - x[-2]) / dt
    return g


def _angular_velocity(R: np.ndarray, dt: float) -> np.ndarray:
    """World angular velocity by central differences of rotations."""
    w = np.empty((len(R), 3))
    w[1:-1] = mt.so3_log(np, R[2:] @ R[:-2].swapaxes(-1, -2)) / (2 * dt)
    w[0] = mt.so3_log(np, R[1] @ R[0].T) / dt
    w[-1] = mt.so3_log(np, R[-1] @ R[-2].T) / dt
    return w


def _process_clip(
    path: str, m: mujoco.MjModel, info: ModelInfo, R_off: np.ndarray,
    tracked: np.ndarray, dt: float,
) -> dict[str, np.ndarray]:  # fmt: skip
    fps, p, quat, j = load_clip(path, info, R_off)
    T_src = p.shape[0]
    if T_src < 2:
        raise ValueError(f"{path}: a clip needs at least two frames")
    n = int(np.floor((T_src - 1) / fps / dt + 1e-6)) + 1
    t = np.arange(n) * dt * fps
    i0 = np.minimum(np.floor(t).astype(np.int64), T_src - 2)
    a = np.clip(t - i0, 0.0, 1.0)[:, None]
    p = p[i0] * (1 - a) + p[i0 + 1] * a
    j = j[i0] * (1 - a) + j[i0 + 1] * a
    quat = _slerp(quat[i0], quat[i0 + 1], a[:, 0])
    R = mt.quat_to_mat(np, quat)
    v, w, jd = _gradient(p, dt), _angular_velocity(R, dt), _gradient(j, dt)
    w_body = (R.swapaxes(-1, -2) @ w[..., None])[..., 0]

    def full_state(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        qpos = np.tile(m.qpos0, (n, 1))
        qvel = np.zeros((n, m.nv))
        qpos[:, info.root_qadr : info.root_qadr + 3] = p
        qpos[:, info.root_qadr + 3 : info.root_qadr + 7] = quat
        qvel[:, info.root_dadr : info.root_dadr + 3] = v
        qvel[:, info.root_dadr + 3 : info.root_dadr + 6] = w_body
        qpos[:, info.indep_qadr] = q
        qvel[:, info.indep_dadr] = jd
        project_joint_couplings(info.couplings, qpos, qvel)
        return qpos, qvel

    qpos, qvel = full_state(j)
    reset_qpos, reset_qvel = full_state(np.clip(j, info.q_lo, info.q_hi))
    # forward kinematics of the tracked bodies
    d = mujoco.MjData(m)
    B = len(tracked)
    P, Rb, cvel, com = (np.empty((n, B, 3)), np.empty((n, B, 9)), np.empty((n, B, 6)),
                        np.empty((n, B, 3)))  # fmt: skip
    rootid = m.body_rootid[tracked]
    for k in range(n):
        d.qpos[:], d.qvel[:] = qpos[k], qvel[k]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_comVel(m, d)
        P[k], Rb[k], cvel[k], com[k] = (d.xpos[tracked], d.xmat[tracked], d.cvel[tracked],
                                        d.subtree_com[rootid])  # fmt: skip
    Rb = Rb.reshape(n, B, 3, 3)
    W = cvel[..., :3]
    V = cvel[..., 3:] + np.cross(W, P - com)
    rpos, rangles, rvel = mt.relative_sites(np, P, Rb, V, W)
    root_qvel = np.concatenate((v, w_body), -1)
    out = {
        "root_pos": p,
        "root_rot": R,
        "root_qvel": root_qvel,
        "root_vel_local": mt.root_vel_local(np, R, root_qvel),
        "joint_pos": qpos[:, info.joint_qadr],
        "joint_vel": qvel[:, info.joint_dadr],
        "site_rpos": rpos,
        "site_rangles": rangles,
        "site_rvel": rvel,
        "body_pos": P,
        "body_lin_vel": V,
        "reset_qpos": reset_qpos,
        "reset_qvel": reset_qvel,
    }
    return {k: x.astype(np.float32) for k, x in out.items()}


REF_FIELDS = tuple(f.name for f in dataclasses.fields(mt.TrackRef))


@dataclass
class MotionLib:
    """Clips resampled to one control step, concatenated frame-wise.

    Attributes:
        names: Clip names (paths relative to the clips' common directory).
        dt: Control step.
        clip_start, clip_len, clip_end: Frame index range per clip (``clip_end``
            inclusive).
        clip_id: Clip of every frame.
        bin_frames: Frames per reset bin.
        bin_clip, bin_start: Clip and first frame of each bin.
        frame_bin: Bin of every frame.
        frames: Per-frame arrays: the :class:`~myosuite.terms.mimic_track.TrackRef`
            fields plus ``body_pos`` / ``body_lin_vel`` (tracked bodies, world) and
            ``reset_qpos`` / ``reset_qvel`` (the full state written at a reset, joints
            clamped to their ranges).
    """

    names: list[str]
    dt: float
    clip_start: Any
    clip_len: Any
    clip_end: Any
    clip_id: Any
    bin_frames: int
    bin_clip: Any
    bin_start: Any
    frame_bin: Any
    frames: dict[str, Any]

    @property
    def num_frames(self) -> int:
        """Total frames."""
        return int(self.clip_id.shape[0])

    @property
    def num_bins(self) -> int:
        """Total reset bins."""
        return int(self.bin_start.shape[0])

    def ref(self, f: Any) -> mt.TrackRef:
        """The reference at frames *f* (any index shape)."""
        return mt.TrackRef(**{k: self.frames[k][f] for k in REF_FIELDS})

    def phase(self, f: Any) -> Any:
        """Progress ``(f - start) / len`` of frames *f* through their clip."""
        c = self.clip_id[f]
        return (f - self.clip_start[c]) / self.clip_len[c]

    def map(self, fn: Callable[[np.ndarray], Any]) -> MotionLib:
        """A copy with every array converted by *fn* (e.g. ``torch.as_tensor``)."""
        arrays = {
            k: fn(getattr(self, k))
            for k in ("clip_start", "clip_len", "clip_end", "clip_id", "bin_clip",
                      "bin_start", "frame_bin")
        }  # fmt: skip
        return dataclasses.replace(
            self, frames={k: fn(v) for k, v in self.frames.items()}, **arrays
        )


def build_motion_lib(
    files: Sequence[str], m: mujoco.MjModel, tracked: Sequence[str], dt: float,
    bin_s: float = 1.0, workers: int = 8,
) -> MotionLib:  # fmt: skip
    """Load, resample and process clips on the body model *m*.

    Args:
        files: Clip files.
        m: The body (or scene) model.
        tracked: Tracked body names, the one welded to the root first.
        dt: Control step.
        bin_s: Reset bin length in seconds.
        workers: Threads processing clips (MuJoCo releases the GIL).

    Returns:
        The library.
    """
    if not files:
        raise ValueError("no motion clips")
    info = ModelInfo(m)
    R_off = root_offset(m)
    ids = np.array([m.body(n).id for n in tracked])
    if m.body_weldid[ids[0]] != info.root_id:
        raise ValueError(
            f"the first tracked body {tracked[0]!r} must be welded to the root"
        )
    work = functools.partial(
        _process_clip, m=m, info=info, R_off=R_off, tracked=ids, dt=dt
    )
    with concurrent.futures.ThreadPoolExecutor(max(1, workers)) as pool:
        clips = list(pool.map(work, files))
    base = os.path.commonpath([os.path.dirname(os.path.abspath(f)) for f in files])
    names = [
        os.path.splitext(os.path.relpath(os.path.abspath(f), base))[0] for f in files
    ]
    lens = np.array([len(c["root_pos"]) for c in clips])
    start = np.concatenate(([0], np.cumsum(lens)[:-1]))
    bin_frames = max(1, round(bin_s / dt))
    bin_clip = np.concatenate(
        [np.full(-(-n // bin_frames), i) for i, n in enumerate(lens)]
    )
    bin_start = np.concatenate(
        [s + bin_frames * np.arange(-(-n // bin_frames)) for s, n in zip(start, lens)]
    )
    frames = {k: np.concatenate([c[k] for c in clips]) for k in clips[0]}
    clip_id = np.repeat(np.arange(len(lens)), lens)
    frame_bin = np.empty(int(lens.sum()), np.int64)
    for b, (c, s) in enumerate(zip(bin_clip, bin_start)):
        frame_bin[s : min(s + bin_frames, start[c] + lens[c])] = b
    return MotionLib(
        names=names, dt=dt, clip_start=start, clip_len=lens, clip_end=start + lens - 1,
        clip_id=clip_id, bin_frames=bin_frames, bin_clip=bin_clip, bin_start=bin_start,
        frame_bin=frame_bin, frames=frames,
    )  # fmt: skip


def write_synthetic_clip(
    path: str | os.PathLike, m: mujoco.MjModel, seconds: float, speed: float,
    amp: float, seed: int, climb: float = 0.0, model_name: str = "myofullbody",
) -> None:  # fmt: skip
    """A clip for tests: the default pose moving along x at *speed* (and up at *climb*),
    each independent joint oscillating by *amp* of its half range at 1 Hz, sampled at
    100 Hz, with the metadata of a clip fitted on *model_name* (its current version)."""
    from myosuite.methods.biomech.retarget import REFIT_META_KEY  # noqa: PLC0415

    from .models import model_version  # noqa: PLC0415

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
    meta = {"source": "synthetic", "model": model_name,
            "model_version": model_version(model_name)}  # fmt: skip
    np.savez(
        Path(path), qpos=qpos, joint_names=names, jnt_type=m.jnt_type, frequency=fps,
        **{REFIT_META_KEY: np.array(json.dumps(meta))},
    )  # fmt: skip


@functools.lru_cache(maxsize=2)
def load_motion_lib(
    clips: str, root: str, model: str, self_collisions: bool, dt: float, bin_s: float,
) -> MotionLib:  # fmt: skip
    """The (cached) library of a :class:`~.config.MotionCfg` on a task model.

    Args:
        clips: ``MotionCfg.clips``.
        root: ``MotionCfg.root``.
        model: Model name (:data:`~.models.MODELS`).
        self_collisions: The scene variant (does not change the kinematics).
        dt: Control step.
        bin_s: Reset bin length.

    Returns:
        The library (NumPy; do not modify).

    Raises:
        ValueError: For a clip fitted on another model or model version, or a clip
            without that metadata.
    """
    from myosuite.methods.biomech.retarget import check_refit_meta  # noqa: PLC0415

    from . import resolve_clips  # noqa: PLC0415
    from .models import compiled_scene, model_cfg, model_version  # noqa: PLC0415

    files = motion_files(*resolve_clips(clips, root, model))
    version = model_version(model)
    for f in files:
        check_refit_meta(f, model, version)
    return build_motion_lib(
        files, compiled_scene(model, self_collisions),
        model_cfg(model).tracked_bodies, dt, bin_s,
    )  # fmt: skip
