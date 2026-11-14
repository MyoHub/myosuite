# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""``reset(seed)`` fully determines the episode, also the first one of an instance.

An evaluation script runs ``make_env(id); reset(seed=s)``; a training loop resets
a used env. Both must give the same episode. The tier-1 cases cover one id per
env class and every MyoChallenge id; the ``slow`` cases cover every registered id.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import myosuite  # noqa: F401  (registers the envs)
from myosuite import make_env
from myosuite.tests.support.optional_deps import skip_if_hf_gated
from myosuite.utils import gym

SEED = 123
N_STEPS = 30

# The first reset of these instances differs from later ones (fixes pending).
_KNOWN = {
    **dict.fromkeys(
        (
            "myoChallengeBimanual-v0",
            "myoFatiChallengeBimanual-v0",
            "myoSarcChallengeBimanual-v0",
        ),
        "the first reset keeps the nominal box size (ignore_first_scale)",
    ),
    **dict.fromkeys(
        (
            "myoChallengeDieReorientP2-v0",
            "myoFatiChallengeDieReorientP2-v0",
            "myoSarcChallengeDieReorientP2-v0",
        ),
        "reset takes a sign from the previous episode's geom_pos",
    ),
}


def _ids() -> list[str]:
    return sorted(i for i in gym.envs.registry if i.startswith(("myo", "motor")))


def _tier1_ids() -> list[str]:
    first: dict[str, str] = {}
    for env_id in _ids():
        entry = gym.spec(env_id).entry_point
        first.setdefault(entry if isinstance(entry, str) else repr(entry), env_id)
    return sorted(set(first.values()) | {i for i in _ids() if "Challenge" in i})


def _param(env_id: str, *marks: pytest.MarkDecorator) -> Any:
    known = _KNOWN.get(env_id)
    if known:
        marks = (
            *marks,
            pytest.mark.xfail(strict=True, raises=AssertionError, reason=known),
        )
    return pytest.param(env_id, marks=marks, id=env_id)


def _flat(x: Any) -> np.ndarray:
    if isinstance(x, dict):
        return np.concatenate([_flat(x[k]) for k in sorted(x)])
    return np.asarray(x, dtype=np.float64).ravel()


def _episode(env: gym.Env, seed: int) -> tuple[np.ndarray, np.ndarray]:
    space = env.action_space
    space.seed(7)
    obs, _ = env.reset(seed=seed)
    obs_traj, rwd_traj = [_flat(obs)], []
    for _ in range(N_STEPS):
        obs, rwd, terminated, truncated, _ = env.step(space.sample())
        obs_traj.append(_flat(obs))
        rwd_traj.append(_flat(rwd))
        if np.any(_flat(terminated)) or np.any(_flat(truncated)):
            break
    return np.stack(obs_traj), np.asarray(rwd_traj)


def _check(env_id: str) -> None:
    try:
        fresh = make_env(env_id)
    except Exception as exc:
        skip_if_hf_gated(env_id, exc)
        raise
    used = make_env(env_id)
    try:
        try:
            first = _episode(fresh, SEED)
            _episode(used, SEED + 1)
        except Exception as exc:
            skip_if_hf_gated(env_id, exc)
            raise
        later = _episode(used, SEED)
        np.testing.assert_array_equal(later[0], first[0], err_msg="obs differ")
        np.testing.assert_array_equal(later[1], first[1], err_msg="rewards differ")
    finally:
        fresh.close()
        used.close()


@pytest.mark.tier1
@pytest.mark.parametrize("env_id", [_param(i) for i in _tier1_ids()])
def test_first_episode_equals_a_later_episode(env_id: str) -> None:
    """A fresh instance and a used one replay ``reset(seed)`` + actions bit-exactly."""
    _check(env_id)


@pytest.mark.slow
@pytest.mark.parametrize(
    "env_id", [_param(i) for i in _ids() if i not in set(_tier1_ids())]
)
def test_first_episode_equals_a_later_episode_all_ids(env_id: str) -> None:
    """The same for every other registered id (nightly)."""
    _check(env_id)
