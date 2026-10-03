# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for ``scripts/train_mjlab.py`` helpers that need no training run."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.tier1

torch = pytest.importorskip("torch")


class _SuccessStream:
    """Stand-in for the wrapped eval env: env *i* ends an episode every ``lengths[i]``
    steps, with final-step ``success`` metric ``successes[i]``."""

    def __init__(self, lengths: list[int], successes: list[float], horizon: int):
        self.num_envs = len(lengths)
        self.device = "cpu"
        self.max_episode_length = horizon
        self.unwrapped = self
        self.metrics_manager = self
        self._lengths = torch.tensor(lengths)
        self._successes = torch.tensor(successes)
        self.reset()

    def reset(self) -> None:
        self._t = torch.zeros(self.num_envs, dtype=torch.long)

    def get_observations(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 1)

    def step(self, _actions: torch.Tensor) -> tuple:
        self._t += 1
        dones = self._t >= self._lengths
        self._t[dones] = 0
        return self.get_observations(), torch.zeros(self.num_envs), dones.long(), {}

    def get_active_iterable_terms(self, env_idx: int) -> list:
        return [("success", [float(self._successes[env_idx])])]


def test_deterministic_success_counts_the_same_episodes_per_env() -> None:
    """96 envs succeed at the time limit, 4 fail at a tenth of it: 96 % per episode.

    Without a per-env cap, in the 2 x 100 evaluated steps the 4 fast failures add 20
    episodes each to the 2 of every other env (192 / 272 = 70.6 %), so
    ``--stop-on-success`` never fires.
    """
    pytest.importorskip("mjlab")
    from scripts.train_mjlab import deterministic_success

    horizon = 100
    env = _SuccessStream(
        lengths=[horizon // 10] * 4 + [horizon] * 96,
        successes=[0.0] * 4 + [1.0] * 96,
        horizon=horizon,
    )
    modes = []
    runner = SimpleNamespace(
        get_inference_policy=lambda device: lambda obs: torch.zeros(obs.shape[0], 1),
        alg=SimpleNamespace(train_mode=lambda: modes.append("train")),
    )

    assert deterministic_success(runner, env) == pytest.approx(0.96)
    assert modes == ["train"]
