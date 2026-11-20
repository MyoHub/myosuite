# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for dexterous hand frame calibration, presets, and SLSQP retargeting."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from myosuite.biomechanics.hand_retarget import (
    TIP_MANO_INDICES,
    DexHandRetargeter,
    hand_frame_from_sites,
    mano_to_wrist_rot,
    preset_for_model,
)
from myosuite.core.model_builder import ModelBuilder
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings

pytestmark = pytest.mark.tier1


def _synthetic_mano_keypoints_from_hand(
    model: mujoco.MjModel, qpos: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    preset = preset_for_model(model)
    d = mujoco.MjData(model)
    d.qpos[:] = qpos
    project_joint_couplings(JointCouplings.from_model(model), d.qpos)
    mujoco.mj_kinematics(model, d)

    wid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, preset.wrist_body)
    p_w = d.xpos[wid].copy()
    r_w = d.xmat[wid].reshape(3, 3).copy()
    r_m2w = mano_to_wrist_rot(preset.wrist_up, preset.wrist_front, side=preset.side)

    kp = np.zeros((21, 3), dtype=np.float64)
    tips_world = np.stack(
        [
            d.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s)]
            for s in preset.tip_sites
        ],
        axis=0,
    )
    rel_robot_local = (tips_world - p_w[None, :]) @ r_w
    rel_mano_local = rel_robot_local @ r_m2w
    for k_idx, vec in zip(TIP_MANO_INDICES, rel_mano_local, strict=True):
        kp[k_idx] = vec
    return kp, tips_world


def test_frame_calibration_and_presets_on_hand_and_arm() -> None:
    hand_model, _ = ModelBuilder().attach_fragment("hand").build()
    arm_model, _ = ModelBuilder().attach_fragment("arm").build()

    p_hand = preset_for_model(hand_model)
    p_arm = preset_for_model(arm_model)
    assert p_hand.wrist_body == "lunate_r"
    assert p_arm.wrist_body == "lunate"

    up_h, front_h = hand_frame_from_sites(
        hand_model, "lunate_r", "MFtip_r", "THtip_r", "LFtip_r"
    )
    assert up_h[1] < -0.85
    assert front_h[2] < -0.85

    r_mat = mano_to_wrist_rot((0.0, -1.0, 0.0), (0.0, 0.0, -1.0), side="right")
    np.testing.assert_allclose(r_mat @ r_mat.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(r_mat) == pytest.approx(1.0)
    np.testing.assert_allclose(
        r_mat @ np.array([-1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]), atol=1e-12
    )
    np.testing.assert_allclose(
        r_mat @ np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, -1.0]), atol=1e-12
    )


@pytest.mark.parametrize("mode", ["position", "vector", "dexpilot"])
def test_dex_hand_retargeter_recovers_flexed_grasp(mode: str) -> None:
    model, _ = ModelBuilder().attach_fragment("hand").build()
    q_goal = model.qpos0.copy()
    for jname, val in (
        ("cmc_flexion_r", 0.35),
        ("mp_flexion_r", 0.30),
        ("mcp2_flexion_r", 0.75),
        ("pm2_flexion_r", 0.70),
        ("mcp3_flexion_r", 0.65),
        ("pm3_flexion_r", 0.60),
    ):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        q_goal[model.jnt_qposadr[jid]] = val

    kp, tips_goal = _synthetic_mano_keypoints_from_hand(model, q_goal)
    _, tips_init = _synthetic_mano_keypoints_from_hand(model, model.qpos0)
    err_before = float(np.mean(np.linalg.norm(tips_init - tips_goal, axis=-1)))

    retargeter = DexHandRetargeter(model, mode=mode, norm_delta=1e-5, maxiter=50)
    q_sol = retargeter.retarget_keypoints(kp, wrist_rot=np.eye(3))
    _, tips_after = _synthetic_mano_keypoints_from_hand(model, q_sol)
    err_after = float(np.mean(np.linalg.norm(tips_after - tips_goal, axis=-1)))

    assert err_after < err_before * 0.25
    for qadr, (lo, hi) in zip(retargeter.qpos_adrs, retargeter.bounds, strict=True):
        assert lo - 1e-6 <= q_sol[qadr] <= hi + 1e-6


def test_dexpilot_pinch_hysteresis_and_reset() -> None:
    model, _ = ModelBuilder().attach_fragment("hand").build()
    ret = DexHandRetargeter(
        model, mode="dexpilot", project_dist=0.04, escape_dist=0.06, maxiter=15
    )
    kp_pinch = np.zeros((21, 3), dtype=np.float64)
    kp_pinch[4] = [-0.08, 0.0, 0.02]
    kp_pinch[8] = [-0.08, 0.0, 0.03]  # 1 cm from thumb (< project_dist)
    kp_pinch[12] = [-0.10, 0.0, 0.0]
    kp_pinch[16] = [-0.095, 0.0, -0.02]
    kp_pinch[20] = [-0.085, 0.0, -0.04]
    ret.retarget_keypoints(kp_pinch)
    assert bool(ret._projected.any())
    ret.reset()
    assert not bool(ret._projected.any())
