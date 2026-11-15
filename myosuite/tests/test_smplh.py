# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""SMPL+H forward kinematics (``myosuite.biomechanics.smplh``) on a synthetic model.

The SMPL+H files are licensed and not in the repo; a random model with the SMPL+H tree
checks the kinematics. (Checked once against ``smplx.lbs`` on AMASS KIT clips: 7e-9 m.)
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from myosuite.biomechanics.smplh import SMPLH, axis_angle_to_matrix

pytestmark = pytest.mark.tier2

PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
PARENTS += [20] * 15 + [21] * 15  # hand joints (flat chains are enough here)


@pytest.fixture(scope="module")
def body(tmp_path_factory: pytest.TempPathFactory) -> SMPLH:
    rng = np.random.default_rng(0)
    V, J = 40, len(PARENTS)
    w = rng.random((V, J))
    path = Path(tmp_path_factory.mktemp("smplh")) / "model.npz"
    np.savez(
        path, v_template=rng.normal(size=(V, 3)), shapedirs=0.01 * rng.normal(size=(V, 3, 16)),
        posedirs=0.01 * rng.normal(size=(V, 3, 9 * (J - 1))),
        J_regressor=rng.dirichlet(np.ones(V), J), weights=w / w.sum(1, keepdims=True),
        kintree_table=np.stack([np.array(PARENTS) % 2**32, np.arange(J)]),
    )  # fmt: skip
    return SMPLH(path)


def test_rodrigues_matches_mujoco() -> None:
    rng = np.random.default_rng(1)
    aa = np.concatenate([rng.normal(size=(20, 3)), 1e-10 * rng.normal(size=(2, 3))])
    for v, R in zip(aa, axis_angle_to_matrix(aa)):
        q, M = np.empty(4), np.empty(9)
        angle = np.linalg.norm(v)
        mujoco.mju_axisAngle2Quat(q, v / max(angle, 1e-300), angle)
        mujoco.mju_quat2Mat(M, q)
        np.testing.assert_allclose(R, M.reshape(3, 3), atol=1e-12)


def test_rest_pose_and_rigid_bones(body: SMPLH) -> None:
    betas = np.linspace(-1, 1, 16)
    v_rest, j_rest = body.shaped(betas)
    ids = {"a": 3, "b": 17}
    rest = body.forward(np.zeros((1, 156)), betas, None, ids)
    np.testing.assert_allclose(rest.joints[0], j_rest, atol=1e-12)
    np.testing.assert_allclose(rest.vertices[0], v_rest[[3, 17]], atol=1e-12)
    rng = np.random.default_rng(2)
    trans = rng.normal(size=(5, 3))
    out = body.forward(0.5 * rng.normal(size=(5, 156)), betas, trans, ids)
    for j in range(1, body.num_joints):
        p = body.parents[j]
        bone = np.einsum("tab,b->ta", out.rotations[:, p], j_rest[j] - j_rest[p])
        np.testing.assert_allclose(
            out.joints[:, j] - out.joints[:, p], bone, atol=1e-12
        )
    np.testing.assert_allclose(out.joints[:, 0], j_rest[0] + trans, atol=1e-12)
