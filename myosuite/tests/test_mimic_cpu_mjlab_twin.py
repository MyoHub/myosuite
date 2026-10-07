# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""CPU <-> mjlab one-step parity of the random-target ``myoMimic*-v0`` twins.

As in ``test_mjlab_cpu_twins``: the mjlab env is synced to the CPU state and
target, the observations of that state are compared, both take the same
action, and observation and reward are compared again. Also: the mjlab step
makes no avoidable host syncs.
"""

from __future__ import annotations

import copy
import functools
from typing import Any

import numpy as np
import pytest

from myosuite.tests.support.optional_deps import (
    require_mjlab,
    require_mujoco_warp,
    require_musclemimic_models,
)
from myosuite import make_env

pytestmark = pytest.mark.tier2

torch = pytest.importorskip("torch")

_ENTITY = {
    "myoMimicBimanual-v0": "mimic_bimanual_robot",
    "myoMimicFullbody-v0": "mimic_fullbody_robot",
}
# float32 Warp vs float64 MuJoCo one step after a synced state, max |diff| over
# 20 random-action steps: obs (qvel * dt) 8e-3 bimanual (the arm wrapping-tendon
# difference of test_mjlab_cpu_twins) and 4e-3 full body (foot contacts); reward
# 9e-4 / 1.3e-4. Without the twin's sync_forward term the reward read one-substep
# stale sites: 9e-3 / 2.8e-2.
_STEP_OBS_ATOL, _REW_ATOL = 1e-2, 5e-3


@functools.cache
def _registered_play_cfgs() -> dict[str, Any]:
    """Play env cfgs of the random-target registration (``num_envs=1``)."""
    from myosuite.envs.myo.backends.mjlab import mimic_mjlab_env as mimic

    cfgs: dict[str, Any] = {}

    def _capture(**task: Any) -> None:
        cfgs[task["task_id"]] = task["play_env_cfg"]

    mimic.register_mimic_mjlab_tasks(
        register_mjlab_task=_capture,
        rl_cfg_fn=mimic.default_mimic_clip_on_policy_runner_cfg,
    )
    return cfgs


@pytest.mark.parametrize("env_id", sorted(_ENTITY))
def test_one_step_parity_with_the_mjlab_twin(env_id: str) -> None:
    """Same state, target and action -> same obs and reward on both halves."""
    require_mjlab()
    require_mujoco_warp()
    require_musclemimic_models()
    from mjlab.envs import ManagerBasedRlEnv

    from myosuite.envs.gymnasium_env import CpuEnvAccessor
    from myosuite.envs.myo.backends.mjlab import mimic_mjlab_env as mimic
    from myosuite.envs.myo.backends.mjlab.tasks.mdp import write_cpu_state

    cpu = make_env(env_id).unwrapped
    cpu.reset(seed=0)
    mj = ManagerBasedRlEnv(cfg=_registered_play_cfgs()[env_id], device="cpu")
    mj.reset()
    entity = _ENTITY[env_id]
    assert mj.action_manager.total_action_dim == cpu.action_space.shape[0]
    assert (
        mj.observation_manager.group_obs_dim["actor"][0]
        == cpu.observation_space.shape[0]
    )
    assert cpu._ctrl_dt == pytest.approx(float(mj.step_dt))

    def _t(x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x)[None], dtype=torch.float32)

    variant = "bimanual" if "Bimanual" in env_id else "fullbody"
    cache = mimic._resolve_mimic_mjlab_ids(mj, entity, variant)
    env0 = torch.zeros(1, dtype=torch.long)
    rng = np.random.default_rng(0)
    for _ in range(5):
        # Targets ~1 cm from the sites, so the tracking reward is far from 0.
        sites = cpu.data.site_xpos[cpu._site_ids]
        cpu._target_site_pos = (sites + rng.normal(0.0, 0.01, sites.shape)).astype(
            np.float32
        )
        write_cpu_state(mj, entity, env0, _t(cpu.data.qpos), _t(cpu.data.qvel))
        mj.sim.data.act[:] = _t(cpu.data.act)
        cache["target_torch"] = _t(cpu._target_site_pos)
        mj.sim.forward()
        accessor = CpuEnvAccessor(cpu.model, cpu.data, cpu._ctrl_dt)
        np.testing.assert_allclose(
            mj.observation_manager.compute_group("actor")[0].numpy(),
            cpu._obs_dict_to_vec(cpu._get_obs_dict(accessor)),
            atol=1e-5,
        )
        action = rng.uniform(-2.5, 2.5, cpu.action_space.shape).astype(np.float32)
        cpu_obs, cpu_rew, _, _, _ = cpu.step(action)
        mj_obs, mj_rew, _, _, _ = mj.step(torch.as_tensor(action[None]))
        np.testing.assert_allclose(
            mj_obs["actor"][0].numpy(), cpu_obs, atol=_STEP_OBS_ATOL
        )
        np.testing.assert_allclose(float(mj_rew[0]), cpu_rew, atol=_REW_ATOL)
        assert cpu_rew > 0.1
    mj.close()
    cpu.close()


def test_random_target_steps_sync_only_to_detect_resets() -> None:
    """A step reads no host data, except one reset check on a step that reset an env.

    Every observation term (in three groups) and the reward re-checked the
    targets for an episode restart (a ``reset.any()`` host read) and indexed
    sites with a NumPy array (a host-to-device copy): 17 host syncs per step.
    """
    require_mjlab()
    require_mujoco_warp()
    require_musclemimic_models()
    from mjlab.envs import ManagerBasedRlEnv

    from myosuite.tests.support.host_sync import HostSyncCounter

    # A copy: the mimic term cache is per env cfg.
    cfg = copy.deepcopy(_registered_play_cfgs()["myoMimicBimanual-v0"])
    mj = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        mj.reset()
        mj.episode_length_buf[0] = mj.max_episode_length - 2  # time-out on step 2
        action = torch.zeros(1, mj.action_manager.total_action_dim)
        mj.step(action)  # sees the counter write: one reset check
        resets = []
        for _ in range(3):
            with HostSyncCounter(package_only=True) as syncs:
                mj.step(action)
            reset = bool(mj.reset_buf.any())
            resets.append(reset)
            assert syncs.total <= int(reset), syncs.report()
        assert resets == [True, False, False]
    finally:
        mj.close()
