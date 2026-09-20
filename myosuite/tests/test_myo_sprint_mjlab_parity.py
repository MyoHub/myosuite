# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Phase 5 gate (myo_bolt_vv porting plan): myoSprint-v0, driven through mjlab's own
ManagerBasedRlEnv.step() decimation loop, must reproduce a raw myo_sim.warp rollout (mirroring
myo_bolt_vv's own WarpBackend compute/apply/advance convention, already gated bit-exact against
myo_bolt_vv's originals in myo_sim's own test suite) on an identical action tape.

Run (needs the myosuite-ms3 uv env, mjlab+myo_sim editable sources -- see pyproject.toml
[tool.uv.sources]):
    KMP_DUPLICATE_LIB_OK=TRUE uv run python -m pytest myosuite/tests/test_myo_sprint_mjlab_parity.py -q
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("VV_WARP_DEVICE", "cpu")

NUM_ENVS = 1
STEPS = 3  # env.step() calls; at DECIMATION substeps each this is still hundreds of physics steps -- kept
# small since the point is confirming exactness holds at the real control-rate decimation, not a long rollout
TOL_QPOS = 1e-4
TOL_QVEL = 1e-4


def _raw_reference_rollout(ctrl_sequence: np.ndarray):
    """Same call pattern as myo_sim's own test_bolt_warp_combined_parity.py / test_myosprinter_fragment.py
    (already gated bit-exact against myo_bolt_vv's WarpBackend), used here as the independent oracle."""
    import mujoco
    import mujoco_warp as mjw
    import warp as wp

    from myo_sim.sprinter_assets import load_contact, load_muscles
    from myo_sim.warp.contact_system import ContactSystem
    from myo_sim.warp.muscle_system import MuscleSystem

    wp.init()
    muscles = load_muscles()
    cz = load_contact()
    import myo_sim

    mjm = myo_sim.load_spec("myosprinter").compile()
    mjm.opt.timestep = 1e-4
    mjm.opt.gravity[:] = (0.0, -9.80665, 0.0)
    wm = mjw.put_model(mjm)
    wd = mjw.put_data(mjm, mujoco.MjData(mjm), nworld=NUM_ENVS)
    qpos0 = np.tile(mjm.qpos0.astype(np.float32), (NUM_ENVS, 1))
    qpos0[:, 1] = 1.05
    wd.qpos.assign(qpos0)
    mjw.kinematics(wm, wd)
    mjw.tendon(wm, wd)
    ms = MuscleSystem(muscles, NUM_ENVS)
    cs = ContactSystem(cz["geom_flat"], cz["pairs"], NUM_ENVS, mjm.nbody, naconmax=cz["naconmax"])
    ms.equilibrate(wd)
    dt = float(mjm.opt.timestep)
    # Each of mjlab's env.step() calls holds one action constant across DECIMATION physics substeps.
    # Replicate that here (imported from env_cfgs, not hardcoded) so both rollouts see the same
    # action-vs-substep structure -- otherwise this "reference" would be answering a different question.
    from myosuite.envs.myo.backends.mjlab.tasks.myosprint.env_cfgs import DECIMATION

    for t in range(STEPS):
        ctrl_t = np.tile(ctrl_sequence[t], (NUM_ENVS, 1)).astype(np.float32)
        for _ in range(DECIMATION):
            wd.ctrl.assign(ctrl_t)
            mjw.fwd_position(wm, wd)
            mjw.fwd_velocity(wm, wd)
            ms.compute(wd)
            ms.apply(wd, m=wm, zero=True)
            wd.xfrc_applied.zero_()
            cs.compute(wm, wd)
            cs.apply(wd)
            mjw.step(wm, wd)
            ms.advance(dt)
    return wd.qpos.numpy().copy(), wd.qvel.numpy().copy()


def _mjlab_env_rollout(ctrl_sequence: np.ndarray):
    import torch
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

    from myosuite.envs.myo.backends.mjlab.tasks.myosprint.env_cfgs import make_sprint_env_cfg

    cfg = make_sprint_env_cfg(num_envs=NUM_ENVS)
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset()
    for t in range(STEPS):
        act = torch.as_tensor(ctrl_sequence[t], dtype=torch.float32).unsqueeze(0).repeat(NUM_ENVS, 1)
        env.step(act)
    qpos = env.sim.wp_data.qpos.numpy().copy()
    qvel = env.sim.wp_data.qvel.numpy().copy()
    return qpos, qvel


@pytest.mark.tier2
def test_myosprint_mjlab_matches_raw_myo_sim_warp():
    import myo_sim

    nu = myo_sim.load_spec("myosprinter").compile().nu
    rng = np.random.default_rng(23)
    ctrl_sequence = rng.uniform(0.2, 0.8, (STEPS, nu)).astype(np.float32)

    qpos_ref, qvel_ref = _raw_reference_rollout(ctrl_sequence)
    qpos_mjlab, qvel_mjlab = _mjlab_env_rollout(ctrl_sequence)

    d_qpos = np.abs(qpos_ref - qpos_mjlab).max()
    d_qvel = np.abs(qvel_ref - qvel_mjlab).max()
    print(f"myoSprint-v0 vs raw myo_sim.warp, {STEPS} steps: max|d qpos|={d_qpos:.3e} max|d qvel|={d_qvel:.3e}")
    assert d_qpos < TOL_QPOS, f"qpos parity failed: {d_qpos} >= {TOL_QPOS}"
    assert d_qvel < TOL_QVEL, f"qvel parity failed: {d_qvel} >= {TOL_QVEL}"


if __name__ == "__main__":
    test_myosprint_mjlab_matches_raw_myo_sim_warp()
    print("PASS")
