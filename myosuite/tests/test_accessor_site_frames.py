# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Site orientation and angular velocity of the accessors (``site_xmat``, ``site_ang_vel``), on the elbow's moving
wrist site: the CPU accessor against MuJoCo, and both mjlab accessors against the CPU on a matched state."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from myosuite import make_env
from myosuite.envs.gymnasium_env import CpuEnvAccessor
from myosuite.tests.support.optional_deps import require_mjlab, require_mujoco_warp

pytestmark = pytest.mark.tier2

ENV_ID = "myoElbowPose1D6MRandom-v0"


def _moving_elbow() -> object:
    env = make_env(ENV_ID)
    env.reset(seed=2)
    env.action_space.seed(2)
    for _ in range(5):
        env.step(env.action_space.sample())
    return env.unwrapped


def test_the_cpu_accessor_reads_the_site_frame_and_its_angular_velocity() -> None:
    base = _moving_elbow()
    accessor = CpuEnvAccessor(base.model, base.data, base._ctrl_dt)
    site = accessor.site_id("wrist")
    np.testing.assert_array_equal(
        accessor.site_xmat(site), base.data.site_xmat[site].reshape(3, 3)
    )
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(
        base.model, base.data, mujoco.mjtObj.mjOBJ_SITE, site, velocity, 0
    )
    angular = accessor.site_ang_vel(site)
    assert np.linalg.norm(angular) > 0.1  # the forearm turns
    np.testing.assert_allclose(angular, velocity[:3], atol=1e-12)
    assert accessor.site_xmat([site, 0]).shape == (2, 3, 3)
    assert accessor.site_ang_vel([site, 0]).shape == (2, 3)


def test_both_mjlab_accessors_read_the_cpu_values_on_a_matched_state() -> None:
    require_mjlab()
    require_mujoco_warp()
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    from myosuite.envs.myo.backends.mjlab.mjlab_env_base import (
        MjlabEntityAccessor,
        MjlabEnvAccessor,
    )
    from myosuite.envs.myo.backends.mjlab.tasks.mdp import write_cpu_state

    base = _moving_elbow()
    cpu = CpuEnvAccessor(base.model, base.data, base._ctrl_dt)
    cfg = load_env_cfg(ENV_ID)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        env.reset()
        row = lambda x: torch.as_tensor(x[None], dtype=torch.float32)  # noqa: E731
        write_cpu_state(
            env,
            "robot",
            torch.zeros(1, dtype=torch.long),
            row(base.data.qpos),
            row(base.data.qvel),
        )
        env.sim.data.act[:] = row(base.data.act)
        env.sim.forward()
        entity = MjlabEntityAccessor(env, "robot")
        raw = MjlabEnvAccessor(env.sim.model, env.sim.data, env.step_dt)
        site = cpu.site_id("wrist")
        global_site = mujoco.mj_name2id(
            env.sim.mj_model, mujoco.mjtObj.mjOBJ_SITE, "wrist"
        )
        for accessor, sid in ((entity, entity.site_id("wrist")), (raw, global_site)):
            np.testing.assert_allclose(
                accessor.site_xmat(sid)[0].numpy(), cpu.site_xmat(site), atol=1e-5
            )
            np.testing.assert_allclose(
                accessor.site_ang_vel(sid)[0].numpy(), cpu.site_ang_vel(site), atol=1e-4
            )
    finally:
        env.close()
