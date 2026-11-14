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

A body with a fixed root needs no root data (a trajectory without a free joint, or a
canonical clip without ``root_pos`` / ``root_quat``); its root arrays have width 0.

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
import hashlib
import inspect
import json
import glob
import os
import tempfile
import warnings
import zipfile
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
    model = ([] if info.fixed_root else [info.root_joint_name]) + info.joint_names
    return resolve_model_names([str(n) for n in names], model, field_name="qpos")


def load_clip(
    path: str, info: ModelInfo, R_off: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """``(fps, root_pos (T,3), root_quat (T,4) of the free-joint body, indep. joints (T,J))``;
    the root entries are ``None`` for a fixed root."""
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
        cols = q[:, [names.index(n) for n in info.indep_names]]
        if info.fixed_root:
            return float(data["fps"]), None, None, cols
        R = mt.quat_to_mat(np, np.asarray(data["root_quat"], np.float64)) @ R_off.T
        return (
            float(data["fps"]),
            np.asarray(data["root_pos"], np.float64),
            mt.mat_to_quat(np, R),
            cols,
        )
    free = int(mujoco.mjtJoint.mjJNT_FREE)
    types = (
        data["jnt_type"]
        if "jnt_type" in data.files
        else ([] if info.fixed_root else [free])
        + [1] * (len(names) - (not info.fixed_root))
    )
    adr = np.cumsum([0] + [QPOS_WIDTH[int(t)] for t in types])
    qpos = np.asarray(data["qpos"], np.float64)
    if qpos.shape[1] != adr[-1]:
        raise ValueError(
            f"{path}: qpos has {qpos.shape[1]} columns, the joints need {adr[-1]}"
        )
    if "split_points" in data.files and len(data["split_points"]) != 2:
        raise ValueError(f"{path}: several trajectories in one file")
    joints = qpos[:, [adr[names.index(n)] for n in info.indep_names]]
    if info.fixed_root:
        return float(data["frequency"]), None, None, joints
    if names[0] != info.root_joint_name or int(types[0]) != free:
        raise ValueError(f"{path}: the first joint must be the root free joint")
    quat = qpos[:, 3:7] / np.linalg.norm(qpos[:, 3:7], axis=-1, keepdims=True)
    return (
        float(data["frequency"]),
        qpos[:, :3].copy(),
        np.where(quat[:, :1] < 0, -quat, quat),
        joints,
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
    fixed = info.fixed_root
    T_src = j.shape[0]
    if T_src < 2:
        raise ValueError(f"{path}: a clip needs at least two frames")
    n = int(np.floor((T_src - 1) / fps / dt + 1e-6)) + 1
    if n < 2:
        raise ValueError(f"{path}: the resampled clip needs at least two frames")
    t = np.arange(n) * dt * fps
    i0 = np.minimum(np.floor(t).astype(np.int64), T_src - 2)
    a = np.clip(t - i0, 0.0, 1.0)[:, None]
    j = j[i0] * (1 - a) + j[i0 + 1] * a
    jd = _gradient(j, dt)
    if fixed:  # zero-width root, identity rotation
        p, v, w_body = np.zeros((n, 0)), np.zeros((n, 0)), np.zeros((n, 0))
        R = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    else:
        p = p[i0] * (1 - a) + p[i0 + 1] * a
        quat = _slerp(quat[i0], quat[i0 + 1], a[:, 0])
        R = mt.quat_to_mat(np, quat)
        v, w = _gradient(p, dt), _angular_velocity(R, dt)
        w_body = (R.swapaxes(-1, -2) @ w[..., None])[..., 0]

    def full_state(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        qpos = np.tile(m.qpos0, (n, 1))
        qvel = np.zeros((n, m.nv))
        if not fixed:
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
        "root_vel_local": root_qvel if fixed else mt.root_vel_local(np, R, root_qvel),
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
_CACHE_SCHEMA = 1


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
        quality_flags: Per clip, the ``quality_flags`` of its retarget metadata (e.g.
            ``"source_knee_hyperextension"``); empty for an unflagged clip (the default).
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
    quality_flags: list[list[str]] = dataclasses.field(default_factory=list)

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
    bin_s: float = 1.0, workers: int = 1, cache_dir: str | os.PathLike = "",
) -> MotionLib:  # fmt: skip
    """Load, resample and process clips on the body model *m*.

    Args:
        files: Clip files.
        m: The body (or scene) model.
        tracked: Tracked body names, the anchor first (with a free root, welded to it).
        dt: Control step.
        bin_s: Reset bin length in seconds.
        workers: Threads processing clips (MuJoCo releases the GIL).
        cache_dir: Optional persistent data cache; empty disables disk caching.

    Returns:
        The library.
    """
    if not files:
        raise ValueError("no motion clips")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(bin_s) or bin_s <= 0:
        raise ValueError("dt and bin_s must be finite and positive")
    if len(tracked) == 0:
        raise ValueError("at least one tracked body is required")
    files = tuple(str(Path(f).resolve()) for f in files)
    if not cache_dir:
        return _build_motion_lib(files, m, tracked, dt, bin_s, workers)
    key = _motion_key(files, m, tracked, dt, bin_s)
    return _load_or_build(key, files, m, tracked, dt, bin_s, workers, cache_dir)


def _build_motion_lib(
    files: Sequence[str], m: mujoco.MjModel, tracked: Sequence[str], dt: float,
    bin_s: float, workers: int,
) -> MotionLib:  # fmt: skip
    from myosuite.biomechanics.retarget import read_refit_meta  # noqa: PLC0415

    info = ModelInfo(m)
    R_off = root_offset(m)
    ids = np.array([m.body(n).id for n in tracked])
    if not info.fixed_root and m.body_weldid[ids[0]] != info.root_id:
        raise ValueError(
            f"the first tracked body {tracked[0]!r} must be welded to the root"
        )
    work = functools.partial(
        _process_clip, m=m, info=info, R_off=R_off, tracked=ids, dt=dt
    )
    if workers == 1:
        clips = list(map(work, files))
    else:
        with concurrent.futures.ThreadPoolExecutor(min(workers, len(files))) as pool:
            clips = list(pool.map(work, files))
    names = _clip_names(files)
    lens = np.array([len(c["joint_pos"]) for c in clips])
    frames = {k: np.concatenate([c[k] for c in clips]) for k in clips[0]}
    flags = [list((read_refit_meta(f) or {}).get("quality_flags", [])) for f in files]
    return _assemble_lib(names, dt, bin_s, lens, frames, flags)


def _assemble_lib(
    names: list[str], dt: float, bin_s: float, lens: np.ndarray,
    frames: dict[str, np.ndarray], flags: list[list[str]],
) -> MotionLib:  # fmt: skip
    start = np.concatenate(([0], np.cumsum(lens)[:-1]))
    bin_frames = max(1, round(bin_s / dt))
    bin_clip = np.concatenate(
        [np.full(-(-n // bin_frames), i) for i, n in enumerate(lens)]
    )
    bin_start = np.concatenate(
        [s + bin_frames * np.arange(-(-n // bin_frames)) for s, n in zip(start, lens)]
    )
    clip_id = np.repeat(np.arange(len(lens)), lens)
    frame_bin = np.empty(int(lens.sum()), np.int64)
    for b, (c, s) in enumerate(zip(bin_clip, bin_start)):
        frame_bin[s : min(s + bin_frames, start[c] + lens[c])] = b
    return MotionLib(
        names=names, dt=dt, clip_start=start, clip_len=lens, clip_end=start + lens - 1,
        clip_id=clip_id, bin_frames=bin_frames, bin_clip=bin_clip, bin_start=bin_start,
        frame_bin=frame_bin, frames=frames, quality_flags=flags,
    )  # fmt: skip


def _motion_key(
    files: Sequence[str], m: mujoco.MjModel, tracked: Sequence[str], dt: float,
    bin_s: float,
) -> str:  # fmt: skip
    from myosuite.biomechanics.retarget import file_sha256  # noqa: PLC0415

    binary = np.empty(mujoco.mj_sizeModel(m), dtype=np.uint8)
    mujoco.mj_saveModel(m, buffer=binary)
    implementations = {__file__} | {
        inspect.getfile(obj)
        for obj in (
            ModelInfo,
            resolve_model_names,
            project_joint_couplings,
            mt.relative_sites,
        )
    }
    identity = {
        "schema": _CACHE_SCHEMA,
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "model": hashlib.sha256(binary).hexdigest(),
        "sources": [(str(Path(f).resolve()), file_sha256(f)) for f in files],
        "implementation": sorted(file_sha256(f) for f in implementations),
        "tracked": list(tracked),
        "dt": dt,
        "bin_s": bin_s,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _clip_names(files: Sequence[str]) -> list[str]:
    base = Path(os.path.commonpath([str(Path(f).parent) for f in files]))
    return [str(Path(f).relative_to(base).with_suffix("")) for f in files]


def _frame_shapes(m: mujoco.MjModel, n_tracked: int) -> dict[str, tuple[int, ...]]:
    info = ModelInfo(m)
    joints, root = len(info.joint_ids), 0 if info.fixed_root else 1  # fixed: width 0
    return {
        "root_pos": (3 * root,),
        "root_rot": (3, 3),
        "root_qvel": (6 * root,),
        "root_vel_local": (6 * root,),
        "joint_pos": (joints,),
        "joint_vel": (joints,),
        "site_rpos": (n_tracked - 1, 3),
        "site_rangles": (n_tracked - 1, 3),
        "site_rvel": (n_tracked - 1, 6),
        "body_pos": (n_tracked, 3),
        "body_lin_vel": (n_tracked, 3),
        "reset_qpos": (m.nq,),
        "reset_qvel": (m.nv,),
    }


def _array_digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def _read_cache(
    path: Path, key: str, m: mujoco.MjModel, tracked: Sequence[str],
    files: Sequence[str], dt: float, bin_s: float,
) -> MotionLib | None:  # fmt: skip
    try:
        n_clips = len(files)
        with np.load(path, allow_pickle=False) as data:
            shapes = _frame_shapes(m, len(tracked))
            if set(data.files) != {"metadata", "clip_len", *shapes}:
                raise ValueError("unexpected array fields")
            text = data["metadata"]
            if text.shape != () or text.dtype.kind != "U":
                raise ValueError("metadata must be a Unicode scalar")
            meta = json.loads(str(text))
            if meta["schema"] != _CACHE_SCHEMA or meta["key"] != key:
                raise ValueError("cache identity differs")
            names, flags = meta["names"], meta["quality_flags"]
            if (
                names != _clip_names(files)
                or not isinstance(flags, list)
                or len(flags) != n_clips
                or not all(
                    isinstance(f, list) and all(isinstance(s, str) for s in f)
                    for f in flags
                )
            ):
                raise ValueError("invalid clip metadata")
            lens = data["clip_len"]
            if lens.dtype != np.int64 or lens.shape != (n_clips,) or (lens < 2).any():
                raise ValueError("invalid clip lengths")
            total = sum(int(n) for n in lens)
            frames = {name: data[name] for name in shapes}
            for name, array in frames.items():
                if (
                    array.dtype != np.float32
                    or array.shape != (total, *shapes[name])
                    or not np.isfinite(array).all()
                ):
                    raise ValueError(f"invalid {name} shape, dtype or values")
            for name, array in {"clip_len": lens, **frames}.items():
                if _array_digest(array) != meta["sha256"][name]:
                    raise ValueError(f"invalid {name} checksum")
        return _assemble_lib(names, dt, bin_s, lens, frames, flags)
    except FileNotFoundError:
        return None
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        EOFError,
        zipfile.BadZipFile,
    ) as exc:
        warnings.warn(
            f"Ignoring invalid motion cache {path}: {exc}", RuntimeWarning, stacklevel=2
        )
        return None


def _write_cache(path: Path, key: str, lib: MotionLib) -> None:
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"clip_len": lib.clip_len, **lib.frames}
        meta = {
            "schema": _CACHE_SCHEMA,
            "key": key,
            "names": lib.names,
            "quality_flags": lib.quality_flags,
            "sha256": {name: _array_digest(a) for name, a in arrays.items()},
        }
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".npz", delete=False
        ) as f:
            temporary = Path(f.name)
            np.savez(f, metadata=np.array(json.dumps(meta)), **arrays)
            f.flush()
            os.fsync(f.fileno())
        temporary.replace(path)
    except OSError as exc:
        warnings.warn(
            f"Cannot write motion cache {path}: {exc}", RuntimeWarning, stacklevel=2
        )
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _load_or_build(
    key: str, files: Sequence[str], m: mujoco.MjModel, tracked: Sequence[str],
    dt: float, bin_s: float, workers: int, cache_dir: str | os.PathLike,
) -> MotionLib:  # fmt: skip
    path = (
        Path(cache_dir).expanduser() / f"motion-v{_CACHE_SCHEMA}-{key}.npz"
        if cache_dir
        else None
    )
    if path is not None:
        cached = _read_cache(path, key, m, tracked, files, dt, bin_s)
        if cached is not None:
            return cached
    lib = _build_motion_lib(files, m, tracked, dt, bin_s, workers)
    if _motion_key(files, m, tracked, dt, bin_s) != key:
        raise RuntimeError("motion inputs changed during preprocessing; retry loading")
    if path is not None:
        _write_cache(path, key, lib)
    return lib


def write_synthetic_clip(
    path: str | os.PathLike, m: mujoco.MjModel, seconds: float, speed: float,
    amp: float, seed: int, climb: float = 0.0, model_name: str = "myofullbody",
) -> None:  # fmt: skip
    """A clip for tests: the default pose moving along x at *speed* (and up at *climb*;
    both ignored for a fixed root), each independent joint oscillating by *amp* of its
    half range at 1 Hz, sampled at 100 Hz, with the metadata of a clip fitted on
    *model_name* (its current version and kinematic fingerprint)."""
    from myosuite.biomechanics.retarget import REFIT_META_KEY  # noqa: PLC0415

    from .models import kinematic_fingerprint, model_version  # noqa: PLC0415

    fps = 100.0
    t = np.arange(int(seconds * fps) + 1) / fps
    qpos = np.tile(m.qpos0, (len(t), 1))
    info = ModelInfo(m)
    if not info.fixed_root:
        qpos[:, 0] = speed * t
        qpos[:, 2] += climb * t
    rng = np.random.default_rng(seed)
    for j in info.indep_ids:
        adr, (lo, hi) = m.jnt_qposadr[j], m.jnt_range[j]
        wave = amp * 0.5 * (hi - lo) * np.sin(2 * np.pi * t + rng.uniform(0, 2 * np.pi))
        qpos[:, adr] = np.clip(m.qpos0[adr] + wave, lo, hi)
    names = np.array([m.joint(j).name for j in range(m.njnt)])
    meta = {"source": "synthetic", "model": model_name,
            "model_version": model_version(model_name),
            "model_fingerprint": kinematic_fingerprint(model_name)}  # fmt: skip
    np.savez(
        Path(path), qpos=qpos, joint_names=names, jnt_type=m.jnt_type, frequency=fps,
        **{REFIT_META_KEY: np.array(json.dumps(meta))},
    )  # fmt: skip


def load_motion_lib(
    clips: str, root: str, model: str, self_collisions: bool, dt: float, bin_s: float,
    *, workers: int = 1, cache_dir: str | os.PathLike = "",
) -> MotionLib:  # fmt: skip
    """The (cached) library of a :class:`~.config.MotionCfg` on a task model.

    Args:
        clips: ``MotionCfg.clips``.
        root: ``MotionCfg.root``.
        model: Model name (:data:`~.models.MODELS`).
        self_collisions: The scene variant (does not change the kinematics).
        dt: Control step.
        bin_s: Reset bin length.
        workers: Threads preprocessing clips.
        cache_dir: Optional directory for persistent, data-only NumPy archives.

    Returns:
        The library (NumPy; do not modify).

    Raises:
        ValueError: For a clip fitted on another model, on a body with other
            kinematics (another fingerprint; without one: another myo_sim version), or
            a clip without that metadata.
    """
    from myosuite.biomechanics.retarget import check_refit_meta  # noqa: PLC0415

    from . import resolve_clips  # noqa: PLC0415
    from .models import (  # noqa: PLC0415
        compiled_scene,
        kinematic_fingerprint,
        model_cfg,
        model_version,
    )

    files = motion_files(*resolve_clips(clips, root, model))
    version, fingerprint = model_version(model), kinematic_fingerprint(model)
    for f in files:
        check_refit_meta(f, model, version, fingerprint)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if not files:
        raise ValueError("no motion clips")
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(bin_s) or bin_s <= 0:
        raise ValueError("dt and bin_s must be finite and positive")
    files = tuple(str(Path(f).resolve()) for f in files)
    scene = compiled_scene(model, self_collisions)
    tracked = model_cfg(model).tracked_bodies
    key = _motion_key(files, scene, tracked, dt, bin_s)
    directory = str(Path(cache_dir).expanduser().resolve()) if cache_dir else ""
    return _cached_motion_lib(key, files, scene, tracked, dt, bin_s, workers, directory)


@functools.lru_cache(maxsize=2)
def _cached_motion_lib(
    key: str, files: tuple[str, ...], m: mujoco.MjModel, tracked: tuple[str, ...],
    dt: float, bin_s: float, workers: int, cache_dir: str,
) -> MotionLib:  # fmt: skip
    return _load_or_build(key, files, m, tracked, dt, bin_s, workers, cache_dir)
