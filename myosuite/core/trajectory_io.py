# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Generic trajectory IO helpers for MuJoCo-based playback."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from myosuite.core.hf_io import default_musclemimic_cache_root

_QPOS_NAME_KEYS: tuple[str, ...] = ("qpos_names", "qpos_joint_names", "joint_names")
_QVEL_NAME_KEYS: tuple[str, ...] = ("qvel_names", "qvel_joint_names", "joint_names")


@dataclass(frozen=True)
class MotionClip:
    """Trajectory clip loaded from NPZ.

    ``weights`` is an optional per-frame weight ``(T,)`` (e.g. to down-weight noisy
    frames of a hand-authored trajectory); nothing in MyoSuite reads it, it only
    travels with the clip.
    """

    qpos: np.ndarray
    qvel: np.ndarray | None
    site_xpos: np.ndarray | None
    site_names: list[str] | None
    qpos_joint_names: list[str] | None = None
    qvel_joint_names: list[str] | None = None
    qpos_model_indices: np.ndarray | None = None
    qvel_model_indices: np.ndarray | None = None
    frequency_hz: float | None = None
    source_path: Path | None = None
    weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.weights is not None:
            frames = next(
                (
                    int(a.shape[0])
                    for a in (self.qpos, self.qvel, self.site_xpos)
                    if a is not None
                ),
                None,
            )
            if self.weights.ndim != 1 or (
                frames is not None and int(self.weights.shape[0]) != frames
            ):
                raise ValueError(
                    f"MotionClip.weights must have shape ({frames},), "
                    f"got {self.weights.shape}."
                )
        if (
            self.qpos is not None
            and self.qpos_model_indices is None
            and self.qpos_joint_names is None
        ):
            object.__setattr__(
                self,
                "qpos_model_indices",
                np.arange(self.qpos.shape[1], dtype=np.int32),
            )
        if (
            self.qvel is not None
            and self.qvel_model_indices is None
            and self.qvel_joint_names is None
        ):
            object.__setattr__(
                self,
                "qvel_model_indices",
                np.arange(self.qvel.shape[1], dtype=np.int32),
            )


def _decode_name_list(
    npz: np.lib.npyio.NpzFile,
    candidate_keys: tuple[str, ...],
) -> list[str] | None:
    for key in candidate_keys:
        if key not in npz.files:
            continue
        values = [str(name) for name in np.asarray(npz[key]).reshape(-1).tolist()]
        if not values:
            raise ValueError(f"Motion file metadata key {key!r} is empty.")
        return values
    return None


def _joint_qpos_size(jnt_type: int) -> int:
    import mujoco

    if jnt_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7
    if jnt_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return 4
    return 1


def _joint_qvel_size(jnt_type: int) -> int:
    import mujoco

    if jnt_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return 6
    if jnt_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return 3
    return 1


def _build_model_joint_index_map(
    mj_model: Any,
    *,
    use_qvel: bool,
) -> dict[str, np.ndarray]:
    import mujoco

    index_map: dict[str, np.ndarray] = {}
    for joint_id in range(int(mj_model.njnt)):
        joint_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if not joint_name:
            continue
        joint_type = int(mj_model.jnt_type[joint_id])
        if use_qvel:
            start = int(mj_model.jnt_dofadr[joint_id])
            size = _joint_qvel_size(joint_type)
        else:
            start = int(mj_model.jnt_qposadr[joint_id])
            size = _joint_qpos_size(joint_type)
        index_map[joint_name] = np.arange(start, start + size, dtype=np.int32)
    return index_map


def _resolve_model_indices(
    *,
    clip_width: int,
    expected_width: int,
    joint_names: list[str] | None,
    mj_model: Any,
    field_name: str,
    use_qvel: bool,
) -> np.ndarray:
    if joint_names is None:
        if clip_width != expected_width:
            raise ValueError(
                f"{field_name} width mismatch: expected {expected_width}, got {clip_width}. "
                f"Provide {field_name}_names/{field_name}_joint_names or joint_names metadata."
            )
        return np.arange(expected_width, dtype=np.int32)

    index_map = _build_model_joint_index_map(mj_model, use_qvel=use_qvel)
    missing = sorted(set(joint_names) - set(index_map))
    if missing:
        raise ValueError(
            f"Motion clip {field_name} metadata references unknown model joints: {missing}."
        )

    if len(joint_names) == clip_width:
        counters = {name: 0 for name in joint_names}
        resolved: list[int] = []
        for joint_name in joint_names:
            joint_indices = index_map[joint_name]
            offset = counters[joint_name]
            if offset >= len(joint_indices):
                raise ValueError(
                    f"Motion clip {field_name} metadata over-specifies joint {joint_name!r}."
                )
            resolved.append(int(joint_indices[offset]))
            counters[joint_name] = offset + 1
        incomplete = sorted(
            name for name, count in counters.items() if count != len(index_map[name])
        )
        if incomplete:
            raise ValueError(
                f"Coordinate-level {field_name} metadata must cover complete joints; "
                f"got incomplete coverage for {incomplete}."
            )
        return np.asarray(resolved, dtype=np.int32)

    if len(set(joint_names)) != len(joint_names):
        raise ValueError(
            f"Joint-level {field_name} metadata must not repeat joint names: {joint_names}."
        )
    resolved_arr = np.concatenate([index_map[name] for name in joint_names])
    if int(resolved_arr.size) != clip_width:
        raise ValueError(
            f"{field_name} width {clip_width} does not match the selected model coordinates "
            f"{int(resolved_arr.size)} for joints {joint_names}."
        )
    return resolved_arr.astype(np.int32)


def _default_state(mj_model: Any, *, use_qvel: bool) -> np.ndarray:
    if use_qvel:
        if getattr(mj_model, "key_qvel", None) is not None and int(mj_model.nkey) > 0:
            return np.asarray(mj_model.key_qvel[0], dtype=np.float64)
        return np.zeros(int(mj_model.nv), dtype=np.float64)
    if getattr(mj_model, "key_qpos", None) is not None and int(mj_model.nkey) > 0:
        return np.asarray(mj_model.key_qpos[0], dtype=np.float64)
    return np.zeros(int(mj_model.nq), dtype=np.float64)


def _expand_state_field(
    partial_values: np.ndarray,
    *,
    mj_model: Any,
    model_indices: np.ndarray,
    use_qvel: bool,
) -> np.ndarray:
    # In the clip's dtype, so a float32 clip stays float32.
    default = _default_state(mj_model, use_qvel=use_qvel).astype(
        partial_values.dtype, copy=False
    )
    expanded = np.repeat(default[None, :], partial_values.shape[0], axis=0)
    expanded[:, model_indices] = partial_values
    return expanded


def _resolve_clip_field(
    values: np.ndarray,
    model_indices: np.ndarray | None,
    joint_names: list[str] | None,
    *,
    mj_model: Any,
    field_name: str,
    use_qvel: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(model-ordered values, model indices of the clip columns)``.

    A full-width field that already carries model indices (positional clips get
    ``arange`` in :class:`MotionClip`, expanded clips keep theirs) is in model
    order.  Otherwise the columns are mapped by joint name, at any width, and
    scattered into the model's default state.
    """
    width = int(values.shape[1])
    expected_width = int(mj_model.nv if use_qvel else mj_model.nq)
    if width == expected_width and model_indices is not None:
        return values, np.asarray(model_indices, dtype=np.int32)
    resolved = _resolve_model_indices(
        clip_width=width,
        expected_width=expected_width,
        joint_names=joint_names,
        mj_model=mj_model,
        field_name=field_name,
        use_qvel=use_qvel,
    )
    expanded = _expand_state_field(
        values, mj_model=mj_model, model_indices=resolved, use_qvel=use_qvel
    )
    return expanded, resolved


def expand_motion_clip_to_model(
    clip: MotionClip,
    mj_model: Any,
) -> MotionClip:
    """Map a clip's qpos/qvel columns onto the model by joint name.

    Clips with ``qpos_names`` / ``qvel_names`` or shared ``joint_names``
    metadata are resolved by name at any width: a full-width clip whose joints
    are stored in another order is reordered, and a partial clip is expanded
    to the full model width (missing coordinates take the model's default
    state).  Clips without names must already be in model order.  The result
    is idempotent: ``expanded.qpos[:, expanded.qpos_model_indices]`` equals the
    clip's own columns.
    """

    expanded_qpos, qpos_indices = clip.qpos, clip.qpos_model_indices
    if clip.qpos is not None:
        expanded_qpos, qpos_indices = _resolve_clip_field(
            clip.qpos,
            clip.qpos_model_indices,
            clip.qpos_joint_names,
            mj_model=mj_model,
            field_name="qpos",
            use_qvel=False,
        )
    expanded_qvel, qvel_indices = clip.qvel, clip.qvel_model_indices
    if clip.qvel is not None:
        expanded_qvel, qvel_indices = _resolve_clip_field(
            clip.qvel,
            clip.qvel_model_indices,
            clip.qvel_joint_names,
            mj_model=mj_model,
            field_name="qvel",
            use_qvel=True,
        )

    return MotionClip(
        qpos=expanded_qpos,
        qvel=expanded_qvel,
        site_xpos=clip.site_xpos,
        site_names=clip.site_names,
        qpos_joint_names=clip.qpos_joint_names,
        qvel_joint_names=clip.qvel_joint_names,
        qpos_model_indices=qpos_indices,
        qvel_model_indices=qvel_indices,
        frequency_hz=clip.frequency_hz,
        source_path=clip.source_path,
        weights=clip.weights,
    )


def resolve_motion_path(
    motion_path: str,
    env_name: str = "MyoFullBody",
    cache_root: Path | None = None,
) -> Path:
    """Resolve motion path from absolute, relative, or cache layout."""
    raw = motion_path.strip()
    if not raw:
        raise ValueError("motion_path is empty")
    p = Path(raw).expanduser()
    if p.is_file():
        return p.resolve()
    cwd = Path.cwd() / raw
    if cwd.is_file():
        return cwd.resolve()
    base = cache_root or default_musclemimic_cache_root()
    cached = base / env_name / "gmr" / raw
    if cached.is_file():
        return cached.resolve()
    if not raw.endswith(".npz"):
        cached_npz = cached.with_suffix(".npz")
        if cached_npz.is_file():
            return cached_npz.resolve()
    raise FileNotFoundError(
        "Could not resolve motion_path. Expected absolute/cwd-relative file, "
        f"or cache file under {cached.parent}. "
        "For bundled MyoFullBody demos install huggingface_hub and run "
        "`myosuite-musclemimic-setup-demo-cache --env_name MyoFullBody`, or call "
        "`setup_demo_for_myo_fullbody()` from "
        "`myosuite.integrations.musclemimic.hf_demo_cache`."
    )


def load_motion_clip(
    path: Path,
    expected_nq: int,
    expected_nv: int,
    dtype: Any = np.float64,
) -> MotionClip:
    """Load motion NPZ and validate core tensor shapes.

    Full-width qpos/qvel arrays are accepted as before. Reduced-width arrays are
    also accepted when the NPZ contains ``qpos_names`` / ``qvel_names`` or
    shared ``joint_names`` metadata.  A clip with such names (at any width)
    must be resolved against a concrete MuJoCo model via
    :func:`expand_motion_clip_to_model` before its columns are used as model
    qpos/qvel.

    Args:
        path: The NPZ file.
        expected_nq: Model qpos width (checked unless the clip names its joints).
        expected_nv: Model qvel width (likewise).
        dtype: dtype of qpos, qvel and site_xpos. ``np.float32`` halves the host
            memory of a large mjlab clip bank and leaves its float32 device bank
            unchanged; the CPU envs compute in float64.
    """
    npz = np.load(path, allow_pickle=True)
    if "qpos" not in npz.files:
        raise KeyError(f"Motion file missing required key 'qpos': {path}")
    qpos = np.asarray(npz["qpos"], dtype=dtype)
    if qpos.ndim != 2:
        raise ValueError(f"qpos must be rank-2, got shape {qpos.shape}")
    qpos_joint_names = _decode_name_list(npz, _QPOS_NAME_KEYS)
    if qpos.shape[1] != expected_nq and qpos_joint_names is None:
        raise ValueError(
            f"qpos width mismatch: expected nq={expected_nq}, got {qpos.shape[1]}"
        )
    qvel: np.ndarray | None = None
    qvel_joint_names: list[str] | None = None
    if "qvel" in npz.files:
        qvel_arr = np.asarray(npz["qvel"], dtype=dtype)
        qvel_joint_names = _decode_name_list(npz, _QVEL_NAME_KEYS)
        if qvel_arr.ndim != 2:
            raise ValueError(f"qvel must be rank-2, got shape {qvel_arr.shape}")
        if qvel_arr.shape[1] != expected_nv and qvel_joint_names is None:
            raise ValueError(
                f"qvel width mismatch: expected nv={expected_nv}, got {qvel_arr.shape}"
            )
        qvel = qvel_arr
    site_xpos: np.ndarray | None = None
    if "site_xpos" in npz.files:
        site_xpos = np.asarray(npz["site_xpos"], dtype=dtype)
    site_names: list[str] | None = None
    if "site_names" in npz.files:
        try:
            site_names = [str(n) for n in npz["site_names"]]
        except Exception as exc:
            import warnings

            warnings.warn(
                f"Could not decode site_names from motion file: {exc}", stacklevel=2
            )
            site_names = None
    frequency_hz: float | None = None
    if "frequency" in npz.files:
        try:
            frequency_hz = float(np.asarray(npz["frequency"]).reshape(()))
        except Exception as exc:
            import warnings

            warnings.warn(
                f"Could not decode frequency from motion file: {exc}", stacklevel=2
            )
            frequency_hz = None
    weights: np.ndarray | None = None
    if "weights" in npz.files:
        weights = np.asarray(npz["weights"], dtype=np.float64).reshape(-1)
    return MotionClip(
        qpos=qpos,
        qvel=qvel,
        site_xpos=site_xpos,
        site_names=site_names,
        qpos_joint_names=qpos_joint_names,
        qvel_joint_names=qvel_joint_names,
        frequency_hz=frequency_hz,
        source_path=path.resolve(),
        weights=weights,
    )


def check_clip_rate(
    frequency_hz: float | None, ctrl_dt: float, name: str = "clip"
) -> None:
    """Warn when a clip's frame rate differs from the control rate.

    Clips play back one frame per control step, so a clip recorded at another
    rate would run too fast or too slow.

    Args:
        frequency_hz: Clip frame rate, or ``None`` if the clip does not record it.
        ctrl_dt: Control timestep in seconds.
        name: Label used in the warning.
    """
    if frequency_hz is None:
        return
    if abs(frequency_hz * ctrl_dt - 1.0) > 1e-3:
        warnings.warn(
            f"{name} is recorded at {frequency_hz:g} Hz but plays back at the control "
            f"rate {1.0 / ctrl_dt:g} Hz (one frame per step); playback will be "
            "time-scaled.",
            stacklevel=2,
        )


__all__ = [
    "MotionClip",
    "check_clip_rate",
    "expand_motion_clip_to_model",
    "load_motion_clip",
    "resolve_motion_path",
]
