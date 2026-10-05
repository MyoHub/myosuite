"""The reach far check starts at the control step the twins derive with ``first_step_after``.

CPU ``ReachEnvV0`` enables the far penalty/termination once ``data.time > 2 * ctrl_dt``.
The mjlab twin (``penalty_start_step``) and the MJX env (``_far_check_time``, tested in
``mjx/test_mjx_parity.py``) both replay that float64 time sum with ``first_step_after``.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from myosuite.utils.step_timing import first_step_after

pytestmark = pytest.mark.tier1

# frame_skip 10 (myo) and 5 (motor)
_IDS = ("myoFingerReachRandom-v0", "motorFingerReachRandom-v0")


def _first_far_termination_step(env_id: str) -> int:
    """Control step of the first termination with the target 1 m from the fingertip."""
    env = gym.make(env_id)
    u = env.unwrapped
    env.reset(seed=0)
    u.model.site_pos[u.target_sids[0]] += np.array([1.0, 0.0, 0.0])
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    for step in range(1, 6):
        if env.step(action)[2]:
            env.close()
            return step
    raise AssertionError("no far termination within 5 steps")


@pytest.mark.parametrize("env_id", _IDS)
def test_cpu_far_check_starts_at_first_step_after(env_id: str) -> None:
    u = gym.make(env_id).unwrapped
    expected = first_step_after(2 * u.dt, u.model.opt.timestep, u.frame_skip)
    assert expected == 2
    assert _first_far_termination_step(env_id) == expected


@pytest.mark.parametrize("env_id", _IDS)
def test_mjlab_twin_penalty_start_step_matches_cpu(env_id: str) -> None:
    pytest.importorskip("mjlab")
    from myosuite.envs.myo.backends.mjlab.tasks.reach.reach_env_cfg import (
        make_reach_env_cfg,
    )

    cfg = make_reach_env_cfg(env_id)
    params = cfg.terminations["reach_failed"].params
    assert params["penalty_start_step"] == _first_far_termination_step(env_id)
