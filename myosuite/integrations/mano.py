# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""MANO forward kinematics in NumPy (hand and hand-object motions).

Linear blend skinning of the MANO hand model (Romero, Tzionas & Black, *"Embodied
Hands: Modeling and Capturing Hands and Bodies Together"*, SIGGRAPH Asia 2017): shape
blend shapes, joint regression, rigid kinematic-tree transforms, pose blend shapes,
and skinning of the five fingertip vertices (or full mesh when requested) to emit the
canonical 21-keypoint hand layout used by DexYCB, ARCTIC, DexPilot, and Eden.

Reads both ``.npz`` archives and official ``MANO_RIGHT.pkl`` / ``MANO_LEFT.pkl``
files directly via :mod:`myosuite.integrations.chumpy_pickle` without ``chumpy``.

Licensing note:
    MyoSuite does **not** bundle, download, or redistribute ``MANO_RIGHT.pkl``,
    ``MANO_LEFT.pkl``, or any derived MANO parameters. Official MANO model files must
    be obtained directly from the Max Planck Institute for Intelligent Systems
    (https://mano.is.tue.mpg.de/) under the MANO Software License (non-commercial
    scientific research only).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from myosuite.integrations.smplh import axis_angle_to_matrix

MANO_KEYPOINTS: tuple[str, ...] = (
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)  # fmt: skip
"""Canonical 21-keypoint ordering (wrist + 4 keypoints per finger, thumb to pinky)."""

MANO_TIP_VERTICES: dict[str, int] = {
    "thumb_tip": 744,
    "index_tip": 320,
    "middle_tip": 443,
    "ring_tip": 554,
    "pinky_tip": 671,
}
"""Surface vertex indices of the five MANO fingertips on the 778-vertex mesh."""

_KEYPOINT_JOINT_SLOTS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (1, 13), (2, 14), (3, 15),
    (5, 1), (6, 2), (7, 3),
    (9, 4), (10, 5), (11, 6),
    (13, 10), (14, 11), (15, 12),
    (17, 7), (18, 8), (19, 9),
)  # fmt: skip
_KEYPOINT_TIP_SLOTS: tuple[tuple[int, str], ...] = (
    (4, "thumb_tip"),
    (8, "index_tip"),
    (12, "middle_tip"),
    (16, "ring_tip"),
    (20, "pinky_tip"),
)


@dataclass
class MANOPose:
    """A posed MANO hand sequence.

    Attributes:
        keypoints: ``(T, 21, 3)`` canonical 21 keypoints in world coordinates.
        joints: ``(T, 16, 3)`` internal MANO joints in world coordinates.
        rotations: ``(T, 16, 3, 3)`` global joint rotation matrices.
        vertices: ``(T, V, 3)`` skinned mesh vertices if ``return_vertices=True``,
            otherwise ``(T, 0, 3)``.
    """

    keypoints: np.ndarray
    joints: np.ndarray
    rotations: np.ndarray
    vertices: np.ndarray


class MANO:
    """Pure-NumPy MANO hand model (``.pkl`` or ``.npz``).

    Args:
        path: Path to ``MANO_RIGHT.pkl``, ``MANO_LEFT.pkl``, or an equivalent ``.npz``.
        is_rhand: Handedness when not encoded in the file name (default inferred from
            ``LEFT`` in ``path``, otherwise ``True``).
        tip_vertices: Optional override of fingertip vertex indices (useful for
            reduced-vertex synthetic test fixtures).
    """

    def __init__(
        self,
        path: str | Path,
        is_rhand: bool | None = None,
        tip_vertices: dict[str, int] | None = None,
    ) -> None:
        model_path = Path(path)
        if model_path.suffix.lower() == ".pkl":
            from myosuite.integrations.chumpy_pickle import load_body_model_pkl

            d = load_body_model_pkl(model_path)
            raw_keys = set(d)
            get = d.__getitem__
        else:
            z = np.load(model_path, allow_pickle=False)
            raw_keys = set(z.files)
            get = z.__getitem__

        try:
            self.v_template = np.asarray(get("v_template"), np.float64)
            self.shapedirs = np.asarray(get("shapedirs"), np.float64)
            self.posedirs = np.asarray(get("posedirs"), np.float64)
            self.J_regressor = np.asarray(get("J_regressor"), np.float64)
            self.weights = np.asarray(get("weights"), np.float64)
            parents = np.asarray(get("kintree_table")[0], np.int64).copy()
            self.hands_mean = (
                np.asarray(get("hands_mean"), np.float64).ravel()
                if "hands_mean" in raw_keys
                else np.zeros(3 * (len(parents) - 1), np.float64)
            )
            self.hands_components = (
                np.asarray(get("hands_components"), np.float64)
                if "hands_components" in raw_keys
                else np.eye(3 * (len(parents) - 1), dtype=np.float64)
            )
        finally:
            if model_path.suffix.lower() != ".pkl":
                z.close()

        if self.posedirs.ndim == 2:
            self.posedirs = self.posedirs.T.reshape(len(self.v_template), 3, -1)
        parents[0] = -1
        self.parents = parents
        self.num_joints = len(parents)
        self.num_betas = self.shapedirs.shape[-1]
        self.is_rhand = (
            ("left" not in model_path.stem.lower())
            if is_rhand is None
            else bool(is_rhand)
        )
        self.tip_vertices = dict(tip_vertices or MANO_TIP_VERTICES)

    def shaped(self, betas: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return shaped rest vertices ``(V, 3)`` and rest joints ``(16, 3)``."""
        b = np.zeros(self.num_betas, dtype=np.float64)
        if betas is not None:
            arr = np.asarray(betas, np.float64).ravel()[: self.num_betas]
            b[: len(arr)] = arr
        v = self.v_template + self.shapedirs @ b
        return v, self.J_regressor @ v

    def decode_hand_pose(
        self,
        hand_pose: np.ndarray,
        *,
        use_pca: bool | None = None,
        flat_hand_mean: bool = False,
    ) -> np.ndarray:
        """Convert PCA or axis-angle finger pose to ``(T, 45)`` axis-angles."""
        pose = np.asarray(hand_pose, np.float64)
        if pose.ndim == 1:
            pose = pose[None, :]
        full_dim = 3 * (self.num_joints - 1)
        is_pca = (pose.shape[1] != full_dim) if use_pca is None else use_pca
        if is_pca:
            ncomps = pose.shape[1]
            if ncomps > len(self.hands_components):
                raise ValueError(
                    f"PCA pose width {ncomps} exceeds hands_components ({len(self.hands_components)})."
                )
            aa = pose @ self.hands_components[:ncomps]
        else:
            if pose.shape[1] != full_dim:
                raise ValueError(
                    f"Expected {full_dim}-d axis-angle hand_pose when use_pca=False, got {pose.shape}."
                )
            aa = pose.copy()
        if not flat_hand_mean:
            aa = aa + self.hands_mean[None, :]
        return aa

    def forward(
        self,
        global_orient: np.ndarray,
        hand_pose: np.ndarray,
        betas: np.ndarray | None = None,
        trans: np.ndarray | None = None,
        *,
        use_pca: bool | None = None,
        flat_hand_mean: bool = False,
        return_vertices: bool = False,
    ) -> MANOPose:
        """Pose a MANO sequence and return 21 canonical keypoints (and optional mesh)."""
        orient = np.asarray(global_orient, np.float64)
        if orient.ndim == 1:
            orient = orient[None, :]
        fingers = self.decode_hand_pose(
            hand_pose, use_pca=use_pca, flat_hand_mean=flat_hand_mean
        )
        n_frames = len(orient)
        if len(fingers) != n_frames:
            raise ValueError(
                f"Frame count mismatch: global_orient has {n_frames} frames, hand_pose has {len(fingers)}."
            )
        aa = np.concatenate(
            [orient[:, None, :], fingers.reshape(n_frames, self.num_joints - 1, 3)],
            axis=1,
        )
        r_local = axis_angle_to_matrix(aa)
        v_rest, j_rest = self.shaped(betas)

        r_glob = np.empty_like(r_local)
        p_glob = np.empty((n_frames, self.num_joints, 3), dtype=np.float64)
        r_glob[:, 0], p_glob[:, 0] = r_local[:, 0], j_rest[0]
        for j in range(1, self.num_joints):
            p = self.parents[j]
            r_glob[:, j] = r_glob[:, p] @ r_local[:, j]
            p_glob[:, j] = p_glob[:, p] + r_glob[:, p] @ (j_rest[j] - j_rest[p])

        t = (
            np.zeros((n_frames, 3), dtype=np.float64)
            if trans is None
            else np.asarray(trans, np.float64)
        )
        if t.ndim == 1:
            t = np.broadcast_to(t[None, :], (n_frames, 3))

        feat = (r_local[:, 1:] - np.eye(3)).reshape(n_frames, -1)
        tip_names = [name for _, name in _KEYPOINT_TIP_SLOTS]
        tip_ids = np.array(
            [self.tip_vertices[name] for name in tip_names], dtype=np.int64
        )

        if return_vertices:
            v_posed = v_rest[None] + np.einsum("vcp,tp->tvc", self.posedirs, feat)
            loc = v_posed[:, :, None, :] - j_rest[None, None]
            moved = np.einsum("tjab,tvjb->tvja", r_glob, loc) + p_glob[:, None]
            verts = np.einsum("vj,tvja->tva", self.weights, moved) + t[:, None]
            tips = verts[:, tip_ids]
        else:
            v_posed = v_rest[tip_ids][None] + np.einsum(
                "vcp,tp->tvc", self.posedirs[tip_ids], feat
            )
            loc = v_posed[:, :, None, :] - j_rest[None, None]
            moved = np.einsum("tjab,tvjb->tvja", r_glob, loc) + p_glob[:, None]
            tips = np.einsum("vj,tvja->tva", self.weights[tip_ids], moved) + t[:, None]
            verts = np.empty((n_frames, 0, 3), dtype=np.float64)

        joints_w = p_glob + t[:, None]
        keypoints = np.empty((n_frames, 21, 3), dtype=np.float64)
        for kp_idx, j_idx in _KEYPOINT_JOINT_SLOTS:
            keypoints[:, kp_idx] = joints_w[:, j_idx]
        for i, (kp_idx, _) in enumerate(_KEYPOINT_TIP_SLOTS):
            keypoints[:, kp_idx] = tips[:, i]

        return MANOPose(
            keypoints=keypoints, joints=joints_w, rotations=r_glob, vertices=verts
        )


__all__ = [
    "MANO",
    "MANO_KEYPOINTS",
    "MANO_TIP_VERTICES",
    "MANOPose",
]
