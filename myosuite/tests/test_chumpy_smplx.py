# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for chumpy-free pickle loading and SMPL+H / SMPL-X adaptation."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse

from myosuite.biomechanics.chumpy_pickle import (
    install_chumpy_pickle_modules,
    load_body_model_pkl,
)
from myosuite.biomechanics.retarget_smplh import _resolve_body_model_path
from myosuite.biomechanics.smplh import (
    SMPLH,
    adapt_smplx_poses_to_smplh,
    axis_angle_to_matrix,
    load_amass,
    matrix_to_axis_angle,
)

pytestmark = pytest.mark.tier1


def _write_synthetic_chumpy_pkl(path: Path, payload: dict) -> None:
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


def _synthetic_smpl_arrays(
    num_joints: int = 52, num_verts: int = 24
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    parents = np.arange(num_joints, dtype=np.int64) - 1
    parents[0] = -1
    parents[1:4] = 0
    kintree = np.vstack([parents, np.arange(num_joints, dtype=np.int64)])
    j_reg = np.zeros((num_joints, num_verts), dtype=np.float64)
    for j in range(num_joints):
        j_reg[j, j % num_verts] = 1.0
    weights = np.zeros((num_verts, num_joints), dtype=np.float64)
    for v in range(num_verts):
        weights[v, v % num_joints] = 1.0
    return {
        "v_template": rng.normal(scale=0.1, size=(num_verts, 3)),
        "shapedirs": rng.normal(scale=0.01, size=(num_verts, 3, 10)),
        "posedirs": rng.normal(scale=0.002, size=(num_verts, 3, 9 * (num_joints - 1))),
        "J_regressor": j_reg,
        "weights": weights,
        "kintree_table": kintree,
    }


def test_axis_angle_roundtrip_and_smplx_pose_adaptation() -> None:
    rng = np.random.default_rng(11)
    aa = rng.uniform(-0.8, 0.8, size=(12, 3))
    aa[0] = 0.0
    aa[1] = np.array([np.pi - 1e-4, 0.0, 0.0])
    rot = axis_angle_to_matrix(aa)
    aa_rec = matrix_to_axis_angle(rot)
    rot_rec = axis_angle_to_matrix(aa_rec)
    np.testing.assert_allclose(rot_rec, rot, atol=1e-6)

    smplx_poses = np.arange(5 * 165, dtype=np.float64).reshape(5, 165)
    adapted = adapt_smplx_poses_to_smplh(smplx_poses)
    assert adapted.shape == (5, 156)
    np.testing.assert_array_equal(adapted[:, :66], smplx_poses[:, :66])
    np.testing.assert_array_equal(adapted[:, 66:], smplx_poses[:, 75:165])

    with pytest.raises(ValueError, match="165"):
        adapt_smplx_poses_to_smplh(np.zeros((3, 156)))


def test_chumpy_pkl_and_smplh_npz_parity_and_smplx_55_joint_load(
    tmp_path: Path,
) -> None:
    arrays52 = _synthetic_smpl_arrays(num_joints=52)
    npz_path = tmp_path / "SMPLH_NEUTRAL.npz"
    pkl_path = tmp_path / "SMPLH_NEUTRAL.pkl"
    np.savez(npz_path, **arrays52)
    _write_synthetic_chumpy_pkl(pkl_path, arrays52)

    loaded_dict = load_body_model_pkl(pkl_path)
    assert isinstance(loaded_dict["shapedirs"], np.ndarray)
    assert isinstance(loaded_dict["J_regressor"], np.ndarray)

    m_npz = SMPLH(npz_path)
    m_pkl = SMPLH(pkl_path)
    poses165 = np.random.default_rng(2).normal(scale=0.1, size=(4, 165))
    trans = np.random.default_rng(3).normal(scale=0.2, size=(4, 3))
    betas = np.ones(10) * 0.25

    out_npz = m_npz.forward(poses165, betas, trans)
    out_pkl = m_pkl.forward(poses165, betas, trans)
    np.testing.assert_allclose(out_pkl.joints, out_npz.joints, atol=1e-12)
    np.testing.assert_allclose(out_pkl.rotations, out_npz.rotations, atol=1e-12)

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    smplx_pkl = model_dir / "SMPLX_FEMALE.pkl"
    _write_synthetic_chumpy_pkl(smplx_pkl, _synthetic_smpl_arrays(num_joints=55))
    resolved = _resolve_body_model_path(model_dir, "female")
    assert Path(resolved) == smplx_pkl
    m_x = SMPLH(resolved)
    out_x = m_x.forward(poses165, betas, trans)
    assert out_x.joints.shape == (4, 55, 3)

    clip_path = tmp_path / "smplx_clip.npz"
    np.savez(
        clip_path,
        poses=poses165,
        betas=betas,
        trans=trans,
        mocap_frame_rate=np.array(60.0),
    )
    loaded_clip = load_amass(clip_path)
    assert loaded_clip["gender"] == "neutral"
    assert loaded_clip["fps"] == pytest.approx(60.0)


def test_chumpy_unpickler_rejects_non_dict_payload(tmp_path: Path) -> None:
    bad_pkl = tmp_path / "bad.pkl"
    with bad_pkl.open("wb") as f:
        pickle.dump([1, 2, 3], f)
    with pytest.raises(TypeError, match="Expected dict"):
        load_body_model_pkl(bad_pkl)
