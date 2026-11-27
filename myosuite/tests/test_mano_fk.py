# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for pure-NumPy MANO forward kinematics and PCA pose decoding."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse

from myosuite.biomechanics.chumpy_pickle import install_chumpy_pickle_modules
from myosuite.biomechanics.mano import MANO, MANO_KEYPOINTS

pytestmark = pytest.mark.tier1


def _write_synthetic_mano(path: Path, payload: dict, as_pkl: bool = True) -> None:
    if not as_pkl:
        np.savez(path, **payload)
        return
    install_chumpy_pickle_modules()
    ch_cls = sys.modules["chumpy.ch"].Ch
    sel_cls = sys.modules["chumpy.reordering"].Select
    ch_obj = ch_cls.__new__(ch_cls)
    raw_s = np.asarray(payload["shapedirs"], dtype=np.float64)
    ch_obj.x = raw_s
    sel_obj = sel_cls.__new__(sel_cls)
    sel_obj.a = ch_obj
    sel_obj.idxs = np.arange(raw_s.size, dtype=np.int64)
    sel_obj.preferred_shape = raw_s.shape
    out = dict(payload)
    out["shapedirs"] = sel_obj
    out["J_regressor"] = scipy.sparse.csr_matrix(payload["J_regressor"])
    with path.open("wb") as f:
        pickle.dump(out, f, protocol=2)


def _synthetic_mano_arrays() -> dict[str, np.ndarray]:
    num_joints, num_verts = 16, 30
    rng = np.random.default_rng(5)
    parents = np.arange(num_joints, dtype=np.int64) - 1
    parents[0] = -1
    parents[1::3] = 0
    kintree = np.vstack([parents, np.arange(num_joints, dtype=np.int64)])
    j_reg = np.zeros((num_joints, num_verts), dtype=np.float64)
    for j in range(num_joints):
        j_reg[j, j % num_verts] = 1.0
    weights = np.zeros((num_verts, num_joints), dtype=np.float64)
    for v in range(num_verts):
        weights[v, v % num_joints] = 1.0
    q, _ = np.linalg.qr(rng.normal(size=(45, 45)))
    return {
        "v_template": rng.normal(scale=0.1, size=(num_verts, 3)),
        "shapedirs": rng.normal(scale=0.01, size=(num_verts, 3, 10)),
        "posedirs": rng.normal(scale=0.002, size=(num_verts, 3, 9 * (num_joints - 1))),
        "J_regressor": j_reg,
        "weights": weights,
        "kintree_table": kintree,
        "hands_mean": rng.normal(scale=0.05, size=(45,)),
        "hands_components": q.T,
    }


def test_mano_forward_kinematics_pkl_npz_and_pca(tmp_path: Path) -> None:
    arrays = _synthetic_mano_arrays()
    tip_map = {
        "thumb_tip": 5,
        "index_tip": 10,
        "middle_tip": 15,
        "ring_tip": 20,
        "pinky_tip": 25,
    }

    right_pkl = tmp_path / "MANO_RIGHT.pkl"
    left_npz = tmp_path / "MANO_LEFT.npz"
    _write_synthetic_mano(right_pkl, arrays, as_pkl=True)
    _write_synthetic_mano(left_npz, arrays, as_pkl=False)

    mano_r = MANO(right_pkl, tip_vertices=tip_map)
    mano_l = MANO(left_npz, tip_vertices=tip_map)
    assert mano_r.is_rhand is True
    assert mano_l.is_rhand is False
    assert len(MANO_KEYPOINTS) == 21

    rng = np.random.default_rng(7)
    pca6 = rng.normal(scale=0.3, size=(6, 6))
    orient = rng.normal(scale=0.2, size=(6, 3))
    trans = rng.normal(scale=0.1, size=(6, 3))
    pose_pca = mano_r.forward(orient, pca6, trans=trans, return_vertices=True)
    assert pose_pca.keypoints.shape == (6, 21, 3)
    assert pose_pca.vertices.shape == (6, 30, 3)

    full_aa = mano_r.decode_hand_pose(pca6, use_pca=True, flat_hand_mean=False)
    pose_aa = mano_r.forward(
        orient, full_aa, trans=trans, use_pca=False, flat_hand_mean=True
    )
    np.testing.assert_allclose(pose_pca.keypoints, pose_aa.keypoints, atol=1e-12)

    pose_npz = mano_l.forward(
        orient, full_aa, trans=trans, use_pca=False, flat_hand_mean=True
    )
    np.testing.assert_allclose(pose_npz.keypoints, pose_aa.keypoints, atol=1e-12)

    with pytest.raises(ValueError, match="PCA pose width"):
        mano_r.decode_hand_pose(np.zeros((2, 60)), use_pca=True)
    with pytest.raises(ValueError, match="Frame count mismatch"):
        mano_r.forward(orient[:2], pca6)
