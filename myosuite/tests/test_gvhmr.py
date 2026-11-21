# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for GVHMR Y-up to Z-up AMASS clip conversion and CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from myosuite.biomechanics.gvhmr import (
    Y_UP_TO_Z_UP_MATRIX,
    compute_pelvis_j0,
    save_amass_npz,
    to_amass_clip,
    y_up_to_z_up_root,
)
from myosuite.biomechanics.smplh import SMPLH, load_amass

pytestmark = pytest.mark.tier1


def _synthetic_smplh_npz(path: Path) -> SMPLH:
    num_joints, num_verts = 52, 24
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
    np.savez(
        path,
        v_template=rng.normal(scale=0.1, size=(num_verts, 3)),
        shapedirs=rng.normal(scale=0.01, size=(num_verts, 3, 10)),
        posedirs=rng.normal(scale=0.002, size=(num_verts, 3, 9 * (num_joints - 1))),
        J_regressor=j_reg,
        weights=weights,
        kintree_table=kintree,
    )
    return SMPLH(path)


def test_gvhmr_pelvis_pivot_invariance_and_cli(tmp_path: Path) -> None:
    bm = _synthetic_smplh_npz(tmp_path / "SMPLH_NEUTRAL.npz")
    rng = np.random.default_rng(9)
    n_frames = 8
    betas = rng.normal(scale=0.5, size=(n_frames, 10)).astype(np.float32)
    betas[:] = betas[0]
    orient_y = rng.normal(scale=0.4, size=(n_frames, 3)).astype(np.float32)
    body_pose = rng.normal(scale=0.2, size=(n_frames, 63)).astype(np.float32)
    transl_y = rng.normal(scale=1.0, size=(n_frames, 3)).astype(np.float32)

    j0 = compute_pelvis_j0(bm, betas[0])
    orient_z, transl_z = y_up_to_z_up_root(orient_y, transl_y, j0)
    assert orient_z.shape == (n_frames, 3) and transl_z.shape == (n_frames, 3)

    poses_y = np.zeros((n_frames, 156), dtype=np.float64)
    poses_y[:, :3] = orient_y
    poses_y[:, 3:66] = body_pose
    fk_y = bm.forward(poses_y, betas[0], transl_y)

    clip = to_amass_clip(
        {
            "global": {
                "global_orient": orient_y,
                "body_pose": body_pose,
                "betas": betas,
                "transl": transl_y,
            }
        },
        motion_fps=30.0,
        body_model=bm,
        pose_layout="smplx",
    )
    out_npz = save_amass_npz(clip, tmp_path / "converted_poses.npz")
    amass = load_amass(out_npz)
    assert amass["fps"] == pytest.approx(30.0)
    fk_z = bm.forward(amass["poses"], amass["betas"], amass["trans"])

    expected_joints_z = fk_y.joints @ Y_UP_TO_Z_UP_MATRIX.T
    np.testing.assert_allclose(fk_z.joints, expected_joints_z, atol=1e-5)

    raw_in = tmp_path / "gvhmr_raw.npz"
    np.savez(
        raw_in,
        global_orient=orient_y,
        body_pose=body_pose,
        betas=betas,
        transl=transl_y,
        pelvis_j0=j0,
    )
    cli_out = tmp_path / "cli_poses.npz"
    script = (
        Path(__file__).resolve().parents[2] / "scripts" / "mimic" / "convert_gvhmr.py"
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            str(raw_in),
            "--out",
            str(cli_out),
            "--fps",
            "60",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert cli_out.is_file()
    loaded_cli = load_amass(cli_out)
    assert loaded_cli["fps"] == pytest.approx(60.0)

    with pytest.raises(ValueError, match="pelvis_j0 is required"):
        to_amass_clip(
            {
                "global": {
                    "global_orient": orient_y,
                    "body_pose": body_pose,
                    "betas": betas,
                    "transl": transl_y,
                }
            },
            motion_fps=30.0,
        )
