# Copyright (c) 2026 Embodied AI Nexus / Genesis Authors (Eden, Apache-2.0).
# Modified for MyoSuite by the MyoSuite Authors.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""GVHMR Y-up world-frame output to Z-up AMASS clip converter.

GVHMR (Shen et al., *"World-Grounded Human Motion Recovery via Gravity-View
Coordinates"*, SIGGRAPH Asia 2024) predicts gravity-aligned SMPL-X trajectories in a
``+Y``-up world frame, whereas AMASS and MyoSuite's
:mod:`myosuite.integrations.retarget_smplh` expect ``+Z`` up. Because SMPL rotates the
body around the shaped rest pelvis ``J0(betas)``, rotating the world frame about the
origin requires shifting the root translation by::

    transl_z = R_y2z @ (transl + J0) - J0

Attribution and licensing:
    - Adapted from ``eden/extensions/pose_estimation/gvhmr/convert.py``
      (github.com/embodied-ai-nexus/Eden, Apache-2.0) in pure NumPy.
    - No SMPL-X or SMPL+H body model files are bundled in MyoSuite; any body model
      passed via ``body_model`` must be obtained from MPI-IS under its upstream
      non-commercial research license (https://smpl-x.is.tue.mpg.de/).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from myosuite.integrations.smplh import (
    SMPLH,
    axis_angle_to_matrix,
    matrix_to_axis_angle,
)

SMPLH_POSE_DIM = 156
SMPLX_POSE_DIM = 165
ROOT_ORIENT_SLICE = slice(0, 3)
BODY_POSE_SLICE = slice(3, 66)

Y_UP_TO_Z_UP_AXIS_ANGLE: tuple[float, float, float] = (0.5 * float(np.pi), 0.0, 0.0)
Y_UP_TO_Z_UP_MATRIX: np.ndarray = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)


def y_up_to_z_up_root(
    global_orient: np.ndarray,
    transl: np.ndarray,
    pelvis_j0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate root ``(global_orient, transl)`` from GVHMR ``+Y``-up to ``+Z``-up."""
    orient = np.asarray(global_orient, dtype=np.float64)
    trans = np.asarray(transl, dtype=np.float64)
    if orient.ndim != 2 or orient.shape[-1] != 3 or trans.shape != orient.shape:
        raise ValueError(
            f"Expected (T, 3) arrays; got global_orient {orient.shape}, transl {trans.shape}."
        )
    j0 = np.broadcast_to(np.asarray(pelvis_j0, dtype=np.float64), trans.shape)
    r_root = axis_angle_to_matrix(orient)
    r_z = np.einsum("ab,tbc->tac", Y_UP_TO_Z_UP_MATRIX, r_root)
    orient_z = matrix_to_axis_angle(r_z).astype(np.float32)
    trans_z = ((trans + j0) @ Y_UP_TO_Z_UP_MATRIX.T - j0).astype(np.float32)
    return orient_z, trans_z


def compute_pelvis_j0(body_model: str | Path | SMPLH, betas: np.ndarray) -> np.ndarray:
    """Compute per-frame or static rest-pose pelvis ``J0(betas)`` from a body model."""
    model = body_model if isinstance(body_model, SMPLH) else SMPLH(body_model)
    b = np.asarray(betas, dtype=np.float64)
    if b.ndim == 1:
        return model.shaped(b)[1][0].astype(np.float32)
    if np.allclose(b, b[0:1], atol=1e-7):
        j0 = model.shaped(b[0])[1][0].astype(np.float32)
        return np.broadcast_to(j0[None, :], (len(b), 3)).copy()
    return np.stack([model.shaped(row)[1][0] for row in b], axis=0).astype(np.float32)


def _require_keys(mapping: dict[str, Any], keys: tuple[str, ...], where: str) -> None:
    missing = [k for k in keys if k not in mapping]
    if missing:
        raise ValueError(f"{where} is missing {missing}; got {sorted(mapping)}.")


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return np.asarray(x.detach().cpu().numpy())
    return np.asarray(x)


def to_amass_clip(
    result: dict[str, Any],
    *,
    motion_fps: float,
    pelvis_j0: np.ndarray | None = None,
    body_model: str | Path | SMPLH | None = None,
    pose_layout: str = "smplx",
    source_video: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Pack a GVHMR estimation result into AMASS ``.npz`` fields."""
    if pose_layout not in ("smplx", "smplh"):
        raise ValueError(
            f"pose_layout must be 'smplx' or 'smplh', got {pose_layout!r}."
        )
    _require_keys(result, ("global",), "GVHMR result")
    g = result["global"]
    _require_keys(
        g, ("global_orient", "body_pose", "betas", "transl"), "GVHMR result['global']"
    )

    orient = _to_numpy(g["global_orient"]).astype(np.float32)
    body_pose = _to_numpy(g["body_pose"]).astype(np.float32)
    betas = _to_numpy(g["betas"]).astype(np.float32)
    transl = _to_numpy(g["transl"]).astype(np.float32)

    if betas.ndim == 1:
        betas = np.broadcast_to(betas[None, :], (len(orient), len(betas))).copy()

    if pelvis_j0 is None:
        if "pelvis_j0" in result:
            pelvis_j0 = _to_numpy(result["pelvis_j0"]).astype(np.float32)
        elif body_model is not None:
            pelvis_j0 = compute_pelvis_j0(body_model, betas)
        else:
            raise ValueError(
                "pelvis_j0 is required when missing from result; pass pelvis_j0 or body_model."
            )
    else:
        pelvis_j0 = np.asarray(pelvis_j0, dtype=np.float32)
    if pelvis_j0.ndim == 1:
        pelvis_j0 = np.broadcast_to(pelvis_j0[None, :], (len(orient), 3)).copy()

    num_frames = len(orient)
    for name, (arr, expected) in {
        "global_orient": (orient, (num_frames, 3)),
        "body_pose": (body_pose, (num_frames, 63)),
        "transl": (transl, (num_frames, 3)),
        "pelvis_j0": (pelvis_j0, (num_frames, 3)),
    }.items():
        if arr.shape != expected:
            raise ValueError(
                f"GVHMR field {name!r} has shape {arr.shape}, expected {expected}."
            )

    orient_z, trans_z = y_up_to_z_up_root(orient, transl, pelvis_j0)
    pose_dim = SMPLX_POSE_DIM if pose_layout == "smplx" else SMPLH_POSE_DIM
    poses = np.zeros((num_frames, pose_dim), dtype=np.float32)
    poses[:, ROOT_ORIENT_SLICE] = orient_z
    poses[:, BODY_POSE_SLICE] = body_pose

    payload: dict[str, np.ndarray] = {
        "poses": poses,
        "trans": trans_z,
        "betas": betas.mean(axis=0).astype(np.float32),
        "gender": np.array("neutral"),
        "mocap_framerate": np.array(motion_fps, dtype=np.float32),
        "mocap_frame_rate": np.array(motion_fps, dtype=np.float32),
        "surface_model_type": np.array(pose_layout),
        "gvhmr_global_orient": orient,
        "gvhmr_transl": transl,
        "gvhmr_betas": betas,
        "gvhmr_pelvis_j0": pelvis_j0,
    }
    if "K_fullimg" in result:
        payload["gvhmr_K_fullimg"] = _to_numpy(result["K_fullimg"]).astype(np.float32)
    if result.get("video_fps") is not None:
        payload["gvhmr_video_fps"] = np.array(result["video_fps"], dtype=np.float32)
    if source_video is not None:
        payload["source_video"] = np.array(str(source_video))
    return payload


def load_gvhmr_result(path: str | Path) -> dict[str, Any]:
    """Load a GVHMR ``.npz`` or ``.pkl`` archive into a Python dict."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".npz":
        with np.load(file_path, allow_pickle=True) as z:
            if "global" in z.files and z["global"].ndim == 0:
                return {k: z[k].item() if z[k].ndim == 0 else z[k] for k in z.files}
            return {
                "global": {
                    "global_orient": z["global_orient"],
                    "body_pose": z["body_pose"],
                    "betas": z["betas"],
                    "transl": z["transl"],
                },
                **({"pelvis_j0": z["pelvis_j0"]} if "pelvis_j0" in z.files else {}),
                **({"K_fullimg": z["K_fullimg"]} if "K_fullimg" in z.files else {}),
            }
    if suffix == ".pkl":
        with file_path.open("rb") as f:
            raw = pickle.load(f)
        if not isinstance(raw, dict):
            raise TypeError(f"Expected dict in {file_path}, got {type(raw).__name__}.")
        return raw
    raise ValueError(
        f"Unsupported archive suffix {suffix!r} for pure-NumPy loader ({file_path}); "
        "use .npz/.pkl, or load .pt via scripts/mimic/convert_gvhmr.py or torch.load()."
    )


def save_amass_npz(payload: dict[str, Any], path: str | Path) -> Path:
    """Write a :func:`to_amass_clip` payload to ``path`` (with ``.npz`` suffix)."""
    out = Path(path).with_suffix(".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **payload)
    return out


__all__ = [
    "BODY_POSE_SLICE",
    "ROOT_ORIENT_SLICE",
    "SMPLH_POSE_DIM",
    "SMPLX_POSE_DIM",
    "Y_UP_TO_Z_UP_AXIS_ANGLE",
    "Y_UP_TO_Z_UP_MATRIX",
    "compute_pelvis_j0",
    "load_gvhmr_result",
    "save_amass_npz",
    "to_amass_clip",
    "y_up_to_z_up_root",
]
