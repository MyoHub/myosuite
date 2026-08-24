# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for the additive chase-tag BC obs composer and warm-start utility.

Both are code-and-design-only utilities for a future chase-tag BC
fine-tuning pass (see ``myosuite/envs/myo/tasks/mimic/chasetag_obs.py`` and
``ActorCritic.load_expanded``) — no data collection or training happens here,
only shape/byte-identity/warm-start correctness checks.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.tier1


def test_chasetag_obs_prefix_is_byte_identical_to_directional_obs() -> None:
    """First 528 dims of ``chasetag_obs`` must exactly match ``_directional_obs``."""
    from myosuite.envs.myo.tasks.challenge.chase_tag_fb_model import (
        compile_fullbody_chasetag_model,
    )
    from myosuite.envs.myo.tasks.mimic.chasetag_obs import (
        CHASETAG_OBS_DIM,
        chasetag_obs,
    )
    from myosuite.integrations.musclemimic.bc_directional_collector import (
        _directional_obs,
    )

    model, _spec, _label = compile_fullbody_chasetag_model()
    data = mujoco.MjData(model)
    data.qpos[:] = model.key_qpos[0]
    mujoco.mj_forward(model, data)

    opponent_pos = np.array([1.5, -0.5, 0.9], dtype=np.float32)
    opponent_vel = np.array([0.2, 0.1, 0.0], dtype=np.float32)
    role_onehot = np.array([0.0, 1.0], dtype=np.float32)

    obs = chasetag_obs(model, data, opponent_pos, opponent_vel, role_onehot)
    assert obs.shape == (CHASETAG_OBS_DIM,)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()

    direct = _directional_obs(model, data, 0.0)
    assert np.array_equal(obs[:528], direct), "leading 528 dims must be byte-identical"

    opponent_block = obs[528:535]
    assert opponent_block.shape == (7,)
    role_block = obs[535:537]
    assert np.array_equal(role_block, role_onehot)


def test_load_expanded_warm_starts_shared_prefix_and_zero_inits_new_columns() -> None:
    """``ActorCritic.load_expanded`` must copy the pretrained prefix exactly."""
    from myosuite.envs.myo.tasks.mimic.policy import ActorCritic

    small = ActorCritic(obs_dim=528, act_dim=354, hidden=32, layers=2)
    with torch.no_grad():
        small.trunk[0].weight.uniform_(-1, 1)
        small.trunk[0].bias.uniform_(-1, 1)

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "dummy_directional.pt"
        torch.save(small.state_dict(), ckpt_path)

        big = ActorCritic.load_expanded(
            ckpt_path, new_obs_dim=537, act_dim=354, hidden=32, layers=2
        )

    assert big.trunk[0].weight.shape == (32, 537)
    assert torch.equal(big.trunk[0].weight[:, :528], small.trunk[0].weight)
    assert torch.equal(big.trunk[0].bias, small.trunk[0].bias)
    # Other (obs_dim-independent) layers must transfer unchanged.
    assert torch.equal(big.actor_mean.weight, small.actor_mean.weight)
    assert torch.equal(big.critic.weight, small.critic.weight)

    # Forward pass with the larger input must not error and must not be all
    # -zero (new columns keep default orthogonal init, not zero-init).
    obs = np.random.default_rng(0).normal(size=537).astype(np.float32)
    out = big.act(obs)
    assert out.shape == (354,)
    assert np.isfinite(out).all()
