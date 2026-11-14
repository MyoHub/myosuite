# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""``load_rslrl_policy`` checks the action head against the env and samples reproducibly."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")

from rsl_rl.modules import MLP  # noqa: E402

from myosuite.utils.rslrl_policy import load_rslrl_policy  # noqa: E402

pytestmark = pytest.mark.tier1


def _checkpoint(tmp_path: Path, out_dim: int, learned_std: bool) -> Path:
    state = {f"mlp.{k}": v for k, v in MLP(4, out_dim, [8], "elu").state_dict().items()}
    if learned_std:
        state["distribution.std_param"] = torch.full((out_dim,), 0.5)
    path = tmp_path / f"model_{out_dim}_{int(learned_std)}.pt"
    torch.save({"actor_state_dict": state}, path)
    return path


@pytest.mark.parametrize(
    ("out_dim", "learned_std", "action_dim"),
    [(6, True, 6), (6, False, 3), (3, False, 3)],
)
def test_matching_heads_load(
    tmp_path: Path, out_dim: int, learned_std: bool, action_dim: int
) -> None:
    policy = load_rslrl_policy(_checkpoint(tmp_path, out_dim, learned_std), action_dim)
    assert policy(torch.zeros(1, 4)).shape == (1, action_dim)


@pytest.mark.parametrize(
    ("out_dim", "learned_std", "action_dim"),
    [(6, True, 3), (6, False, 4), (6, True, 4)],
)
def test_a_policy_for_another_env_is_refused(
    tmp_path: Path, out_dim: int, learned_std: bool, action_dim: int
) -> None:
    with pytest.raises(ValueError, match="trained on another env"):
        load_rslrl_policy(_checkpoint(tmp_path, out_dim, learned_std), action_dim)


def test_seeded_sampling_is_reproducible_and_leaves_the_global_rng(
    tmp_path: Path,
) -> None:
    policy = load_rslrl_policy(_checkpoint(tmp_path, 6, True), 6)
    obs = np.zeros((2, 4), np.float32)
    runs = []
    for seed in (0, 0, 1):
        policy.seed(seed)
        state = torch.random.get_rng_state()
        runs.append(policy.act(obs, stochastic=True))
        assert torch.equal(torch.random.get_rng_state(), state)
    np.testing.assert_array_equal(runs[0], runs[1])
    assert not np.array_equal(runs[0], runs[2])
