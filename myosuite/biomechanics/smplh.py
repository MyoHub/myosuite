# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""SMPL+H forward kinematics in NumPy (AMASS motions).

Linear blend skinning of the SMPL+H body model (Romero et al. 2017, MANO/SMPL+H; the
16-beta gender-specific models AMASS ships): shape blend shapes, joint regression,
rigid joint transforms along the kinematic tree, and, for a chosen subset of vertices,
pose blend shapes and skinning. Only the rigid transforms are needed for joint
positions and orientations, so posing a long clip costs a few matrix products per
joint; vertices are skinned only where asked for (e.g. heels and toe tips).

The body model files (``model.npz``) are not part of MyoSuite: they come with AMASS
under the SMPL+H licence. Nothing here downloads or redistributes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SMPLH_JOINTS: tuple[str, ...] = (
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2", "L_Ankle",
    "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar", "R_Collar", "Head",
    "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
)  # fmt: skip
"""The 22 body joints (the hand joints 22-51 follow in the model)."""

SMPLH_VERTICES: dict[str, int] = {
    "L_BigToe": 3216, "L_SmallToe": 3226, "L_Heel": 3387,
    "R_BigToe": 6617, "R_SmallToe": 6624, "R_Heel": 6787,
}  # fmt: skip
"""Named surface vertices of the SMPL(+H) mesh (smplx's OpenPose foot keypoints)."""


def axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    """Rodrigues: ``(..., 3)`` axis-angle vectors to ``(..., 3, 3)`` rotation matrices."""
    aa = np.asarray(aa, np.float64)
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    small = theta < 1e-8
    k = aa / np.where(small, 1.0, theta)
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zero = np.zeros_like(kx)
    K = np.stack([zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], -1).reshape(
        *aa.shape[:-1], 3, 3
    )
    s, c = np.sin(theta)[..., None], np.cos(theta)[..., None]
    R = np.eye(3) + s * K + (1 - c) * (K @ K)
    # second-order expansion near zero keeps the derivative right
    Ks = np.stack([zero, -aa[..., 2], aa[..., 1], aa[..., 2], zero, -aa[..., 0],
                   -aa[..., 1], aa[..., 0], zero], -1).reshape(*aa.shape[:-1], 3, 3)  # fmt: skip
    return np.where(small[..., None], np.eye(3) + Ks + 0.5 * Ks @ Ks, R)


@dataclass
class SMPLHPose:
    """A posed SMPL+H sequence.

    Attributes:
        joints: ``(T, J, 3)`` joint positions (world, with ``trans``).
        rotations: ``(T, J, 3, 3)`` global joint orientations.
        vertices: ``(T, V, 3)`` the requested vertices (world).
        vertex_names: Names of ``vertices`` (same order).
    """

    joints: np.ndarray
    rotations: np.ndarray
    vertices: np.ndarray
    vertex_names: tuple[str, ...]


class SMPLH:
    """An SMPL+H body model (``model.npz`` of the AMASS SMPL+H release).

    Args:
        path: The ``model.npz`` (e.g. ``body_models/smplh/female/model.npz``).
    """

    def __init__(self, path: str | Path) -> None:
        with np.load(path, allow_pickle=False) as d:
            self.v_template = np.asarray(d["v_template"], np.float64)
            self.shapedirs = np.asarray(d["shapedirs"], np.float64)
            self.posedirs = np.asarray(d["posedirs"], np.float64)
            self.J_regressor = np.asarray(d["J_regressor"], np.float64)
            self.weights = np.asarray(d["weights"], np.float64)
            parents = np.asarray(d["kintree_table"][0], np.int64)
        parents[0] = -1
        self.parents = parents
        self.num_joints = len(parents)
        self.num_betas = self.shapedirs.shape[-1]

    def shaped(self, betas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Rest vertices and joints of a shape: ``(V, 3)``, ``(J, 3)``."""
        b = np.zeros(self.num_betas)
        betas = np.asarray(betas, np.float64).ravel()[: self.num_betas]
        b[: len(betas)] = betas
        v = self.v_template + self.shapedirs @ b
        return v, self.J_regressor @ v

    def forward(
        self,
        poses: np.ndarray,
        betas: np.ndarray,
        trans: np.ndarray | None = None,
        vertex_ids: dict[str, int] | None = None,
    ) -> SMPLHPose:
        """Pose a sequence.

        Args:
            poses: ``(T, 3 * J')`` axis-angle per joint (AMASS ``poses``: root, 21 body,
                30 hand joints); missing trailing joints are zero (default hands).
            betas: Shape coefficients (up to ``num_betas``).
            trans: ``(T, 3)`` root translation (zero if ``None``).
            vertex_ids: Vertices to skin, by name (default: none).

        Returns:
            The posed joints, joint orientations and requested vertices.
        """
        poses = np.asarray(poses, np.float64)
        T = len(poses)
        aa = np.zeros((T, self.num_joints, 3))
        n = min(poses.shape[1] // 3, self.num_joints)
        aa[:, :n] = poses[:, : 3 * n].reshape(T, n, 3)
        R_local = axis_angle_to_matrix(aa)
        v_rest, J = self.shaped(betas)
        # rigid transforms along the tree (parents precede children)
        R = np.empty_like(R_local)
        P = np.empty((T, self.num_joints, 3))
        R[:, 0], P[:, 0] = R_local[:, 0], J[0]
        for j in range(1, self.num_joints):
            p = self.parents[j]
            R[:, j] = R[:, p] @ R_local[:, j]
            P[:, j] = P[:, p] + R[:, p] @ (J[j] - J[p])
        t = np.zeros((T, 3)) if trans is None else np.asarray(trans, np.float64)
        names = tuple(vertex_ids or ())
        verts = np.empty((T, len(names), 3))
        if names:
            ids = np.array([vertex_ids[k] for k in names])
            feat = (R_local[:, 1:] - np.eye(3)).reshape(T, -1)
            v = v_rest[ids][None] + np.einsum("vcp,tp->tvc", self.posedirs[ids], feat)
            # skinning: sum_j w_vj (R_j (v - J_j) + P_j)
            w = self.weights[ids]
            loc = v[:, :, None, :] - J[None, None]
            moved = np.einsum("tjab,tvjb->tvja", R, loc) + P[:, None]
            verts = np.einsum("vj,tvja->tva", w, moved) + t[:, None]
        return SMPLHPose(P + t[:, None], R, verts, names)


def load_amass(path: str | Path) -> dict[str, np.ndarray | str | float]:
    """An AMASS SMPL+H ``*_poses.npz``: ``poses``, ``betas``, ``trans``, ``gender``, ``fps``."""
    with np.load(path, allow_pickle=False) as d:
        g = d["gender"].item()
        gender = g.decode() if isinstance(g, bytes) else str(g)
        return {
            "poses": np.asarray(d["poses"], np.float64),
            "betas": np.asarray(d["betas"], np.float64),
            "trans": np.asarray(d["trans"], np.float64),
            "gender": gender.strip().lower(),
            "fps": float(d["mocap_framerate"]),
        }


__all__ = [
    "SMPLH",
    "SMPLH_JOINTS",
    "SMPLH_VERTICES",
    "SMPLHPose",
    "axis_angle_to_matrix",
    "load_amass",
]
