# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Truncated episodes in the ``tutorials/files/5.2/train_mimic.py`` PPO tutorial.

Needs no Hugging Face motion clip: GAE is checked on hand-made rollouts and the
vector env on a synthetic clip.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.tier1

# ``tutorials/files/5.2`` is not a package (dots in the name): put it on sys.path.
sys.path.insert(
    0, str(pathlib.Path(__file__).resolve().parents[2] / "tutorials" / "files/5.2")
)

GAMMA, LAM = 0.9, 0.95


def _gae(terminated: list[int], truncated: list[int]) -> np.ndarray:
    """One env, 3 steps, r = V = 1; the episode cut at step 1 ends in a state of
    value 5 (``final_values``), the observation after the last step has value 2."""
    pytest.importorskip("torch")  # train_mimic imports it at module level
    from train_mimic import compute_gae

    def column(values: list[float]) -> np.ndarray:
        return np.array(values, dtype=np.float32)[:, None]

    return compute_gae(
        rewards=column([1, 1, 1]),
        values=column([1, 1, 1]),
        terminated=np.array(terminated, dtype=bool)[:, None],
        truncated=np.array(truncated, dtype=bool)[:, None],
        final_values=column([0, 5, 0]),
        last_value=np.array([2.0], dtype=np.float32),
        gamma=GAMMA,
        gae_lambda=LAM,
    )[:, 0]


def test_gae_without_episode_ends_is_the_lambda_return_advantage() -> None:
    """A_t = sum_l (gamma lambda)^l delta_{t+l}, delta_t = r_t + gamma V_{t+1} - V_t."""
    deltas = [1 + GAMMA * 1 - 1, 1 + GAMMA * 1 - 1, 1 + GAMMA * 2 - 1]
    expected = [
        sum((GAMMA * LAM) ** k * d for k, d in enumerate(deltas[t:])) for t in range(3)
    ]
    np.testing.assert_allclose(_gae([0, 0, 0], [0, 0, 0]), expected, rtol=1e-6)


def test_gae_bootstraps_a_truncated_episode_from_its_last_state() -> None:
    """Time-limit truncation is not terminal (Pardo et al., 2018): the cut step's
    target is r + gamma V(final obs), and the trace still stops at the boundary."""
    a2 = 1 + GAMMA * 2 - 1
    a1 = 1 + GAMMA * 5 - 1  # not 1 - 1 = 0, as when truncation counted as terminal
    a0 = (1 + GAMMA * 1 - 1) + GAMMA * LAM * a1
    np.testing.assert_allclose(_gae([0, 0, 0], [0, 1, 0]), [a0, a1, a2], rtol=1e-6)


def test_vec_env_returns_the_last_obs_of_a_truncated_episode(
    tmp_path: pathlib.Path,
) -> None:
    """At the time limit ``obs`` is already the next episode's first observation;
    ``info["final_obs"]`` keeps the state the episode was cut at (for bootstrapping)."""
    pytest.importorskip("torch")
    pytest.importorskip("musclemimic_models")
    from train_mimic import VecMimicEnv

    from myosuite.tests.test_mimic_partial_clip import _write_partial_clip

    clip = tmp_path / "clip.npz"
    _write_partial_clip(clip)
    env = VecMimicEnv(clip_path=clip, n_envs=1, max_episode_steps=1)
    env.reset_all()
    obs, rew, term, trunc, info = env.step(np.zeros((1, env.act_dim), np.float32))

    assert rew.shape == term.shape == trunc.shape == (1,)
    assert trunc[0] and not term[0]
    assert info["final_obs"].shape == obs.shape
    assert not np.allclose(info["final_obs"], obs)


def test_gae_bootstraps_nothing_after_a_terminal_state() -> None:
    """A terminated step (also one at the time limit) has target r_t: A_1 = 1 - 1."""
    for truncated in ([0, 0, 0], [0, 1, 0]):
        adv = _gae([0, 1, 0], truncated)
        np.testing.assert_allclose(adv[1], 0.0, atol=1e-7)
        np.testing.assert_allclose(adv[0], (1 + GAMMA - 1) + GAMMA * LAM * adv[1])
