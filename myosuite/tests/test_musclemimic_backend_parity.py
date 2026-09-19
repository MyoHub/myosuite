# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Parity checks for Mimic model wiring across backend surfaces."""

from __future__ import annotations

import numpy as np
import pytest

from myosuite.tests.support.optional_deps import require_musclemimic_models


@pytest.mark.parametrize(
    ("env_id", "compile_fn", "default_cfg_fn"),
    (
        (
            "myoMimicBimanual-v0",
            "compile_mimic_bimanual_mjmodel",
            "default_mimic_config",
        ),
        (
            "myoMimicFullbody-v0",
            "compile_mimic_fullbody_mjmodel",
            "default_mimic_fullbody_config",
        ),
    ),
)
def test_mimic_cpu_registration_matches_compiled_model(
    env_id: str,
    compile_fn: str,
    default_cfg_fn: str,
) -> None:
    """CPU env model dimensions should match direct model compile output."""
    require_musclemimic_models()
    import myosuite
    from myosuite.utils import gym

    from myosuite.integrations.musclemimic import (
        bimanual_model,
        fullbody_model,
    )

    module = bimanual_model if "Bimanual" in env_id else fullbody_model
    cfg = getattr(module, default_cfg_fn)()
    compiled_model, _, _ = getattr(module, compile_fn)(cfg)

    myosuite.register_all_envs()
    env = gym.make(env_id)
    inner = env.unwrapped
    try:
        assert inner.model.nq == compiled_model.nq
        assert inner.model.nv == compiled_model.nv
        assert inner.model.nu == compiled_model.nu
        np.testing.assert_allclose(
            inner.model.actuator_ctrlrange,
            compiled_model.actuator_ctrlrange,
            atol=1e-9,
        )
    finally:
        env.close()


def test_mimic_legacy_cpu_env_ids_remain_registered() -> None:
    """Legacy ``myoMuscleMimic*`` ids remain aliases of ``myoMimic*``."""
    require_musclemimic_models()
    import myosuite
    from myosuite.utils import gym

    myosuite.register_all_envs()
    for env_id in (
        "myoMuscleMimicBimanual-v0",
        "myoMuscleMimicFullbody-v0",
    ):
        env = gym.make(env_id)
        try:
            obs, _ = env.reset(seed=0)
            assert obs.size > 0
        finally:
            env.close()


def test_mimic_cpu_reset_is_seed_stable() -> None:
    """Same seed should produce identical reset observations on CPU."""
    require_musclemimic_models()
    import myosuite
    from myosuite.utils import gym

    myosuite.register_all_envs()
    env = gym.make("myoMimicBimanual-v0")
    try:
        obs1, _ = env.reset(seed=123)
        obs2, _ = env.reset(seed=123)
    finally:
        env.close()
    np.testing.assert_allclose(obs1, obs2, atol=1e-9)
