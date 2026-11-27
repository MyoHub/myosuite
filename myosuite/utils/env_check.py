# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Shared single-environment reset, step, observation/reward dict, and pickle check."""

from __future__ import annotations

import copy
import os
import pickle
from typing import Any

import numpy as np

from myosuite import make_env
from myosuite.utils.dict_utils import assert_nested_close


def is_multi_agent_env_spec(spec: Any) -> bool:
    """Return ``True`` if *spec* registers a multi-agent environment."""
    return "MultiAgent" in str(getattr(spec, "entry_point", ""))


def check_reset_tuple(reset_data: Any) -> None:
    """Assert Gymnasium ``(obs, info_dict)`` reset return shape."""
    assert (
        isinstance(reset_data, tuple) and len(reset_data) == 2
    ), "Reset should return a tuple of length 2"
    assert isinstance(reset_data[1], dict), "second element returned should be a dict"


def check_single_env(
    environment_id: str,
    input_seed: int = 1234,
    *,
    extra_steps: int = 0,
    respect_lite_env: bool = True,
) -> None:
    """Verify initialization, seeding, step dicts, and ``pickle`` parity for *environment_id*.

    Shared by ``myosuite.tests.test_envs.TestEnvs.check_env`` and ``.github/smoke_wheel.py``
    so source-checkout tests and wheel release gates run the exact same checks.

    Args:
        environment_id: Registered Gymnasium environment ID.
        input_seed: Deterministic seed passed to ``make_env`` and ``seed()``.
        extra_steps: Additional seeded ``env.step`` round-trip checks on the wrapper
            after the unwrapped step/dict/proprioception/exteroception checks.
        respect_lite_env: When ``True`` and ``ROBOHIVE_TEST=LITE``, skip visual-encoder
            environments (``r3m``, ``rrl``, ``vc1``).
    """
    if respect_lite_env and os.getenv("ROBOHIVE_TEST") == "LITE":
        if (
            "r3m" in environment_id
            or "rrl" in environment_id
            or "vc1" in environment_id
        ):
            return

    env1w = make_env(environment_id, seed=input_seed)
    env2w: Any = None
    try:
        env1 = env1w.unwrapped
        env1.seed(input_seed)
        assert env1.get_input_seed() == input_seed
        reset_obs1, *_ = env1.reset()

        mj_model = env1.model if hasattr(env1, "model") else env1.mj_model
        mj_data = env1.data if hasattr(env1, "data") else env1.mj_data
        nu = env1.action_space.shape[0]

        rng = np.random.default_rng(input_seed)
        u = 0.01 * rng.uniform(low=0.0, high=1.0, size=nu)
        step_out = env1.step(u.copy())
        obs1, rwd1, done1, *rest = step_out
        infos1 = copy.deepcopy(rest[-1] if rest else {})
        _, proprio1_vec, _ = env1.get_proprioception()
        extero1 = env1.get_exteroception()
        assert len(obs1) > 0
        assert mj_model is not None and mj_data is not None
        assert len(infos1) > 0
        obs_dict1 = env1.get_obs_dict(mj_model, mj_data)
        assert len(obs_dict1) > 0
        rwd_dict1 = env1.get_reward_dict(obs_dict1)
        assert len(rwd_dict1) > 0

        reset_data = env1.reset()
        check_reset_tuple(reset_data)

        env2w = pickle.loads(pickle.dumps(env1w))
        env2 = env2w.unwrapped
        env2.seed(input_seed)
        assert env2.get_input_seed() == input_seed
        assert env1.get_input_seed() == env2.get_input_seed(), {
            env1.get_input_seed(),
            env2.get_input_seed(),
        }
        assert env1.action_space == env2.action_space, (
            env1.action_space,
            env2.action_space,
        )
        assert env1.observation_space == env2.observation_space, (
            env1.observation_space,
            env2.observation_space,
        )

        reset_obs2, *_ = env2.reset()
        assert_nested_close(reset_obs1, reset_obs2)

        obs2, rwd2, done2, *_, infos2 = env2.step(u)
        infos2 = copy.deepcopy(infos2)
        _, proprio2_vec, _ = env2.get_proprioception()
        extero2 = env2.get_exteroception()

        assert_nested_close(obs1, obs2)
        assert_nested_close(proprio1_vec, proprio2_vec)
        assert_nested_close(extero1, extero2, atol=2, rtol=0.04)
        assert_nested_close(rwd1, rwd2)
        assert done1 == done2, (done1, done2)
        assert len(infos1) == len(infos2), (infos1, infos2)
        assert_nested_close(infos1, infos2)
        env2.reset()

        if extra_steps > 0:
            w_obs1, w_info1 = env1w.reset(seed=input_seed)
            w_obs2, w_info2 = env2w.reset(seed=input_seed)
            assert_nested_close(w_obs1, w_obs2, atol=1e-6, rtol=1e-6)
            assert_nested_close(w_info1, w_info2, atol=1e-6, rtol=1e-6)
            low = np.asarray(env1w.action_space.low, dtype=np.float32)
            high = np.asarray(env1w.action_space.high, dtype=np.float32)
            for step_idx in range(extra_steps):
                act = rng.uniform(low, high).astype(np.float32)
                s_obs1, s_rwd1, s_term1, s_trunc1, s_info1 = env1w.step(act)
                s_obs2, s_rwd2, s_term2, s_trunc2, s_info2 = env2w.step(act)
                assert np.all(np.isfinite(s_obs1)) and np.isfinite(float(s_rwd1)), (
                    environment_id,
                    step_idx,
                )
                assert_nested_close(s_obs1, s_obs2, atol=1e-6, rtol=1e-6)
                assert_nested_close(float(s_rwd1), float(s_rwd2), atol=1e-6, rtol=1e-6)
                assert bool(s_term1) == bool(s_term2) and bool(s_trunc1) == bool(
                    s_trunc2
                ), (environment_id, step_idx)
                assert_nested_close(s_info1, s_info2, atol=1e-6, rtol=1e-6)
                if s_term1 or s_trunc1:
                    r_obs1, r_info1 = env1w.reset(seed=input_seed + step_idx + 1)
                    r_obs2, r_info2 = env2w.reset(seed=input_seed + step_idx + 1)
                    assert_nested_close(r_obs1, r_obs2, atol=1e-6, rtol=1e-6)
                    assert_nested_close(r_info1, r_info2, atol=1e-6, rtol=1e-6)
    finally:
        if env2w is not None:
            env2w.close()
        env1w.close()
