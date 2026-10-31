# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Muscle saturation metric and opt-in penalty of the mjlab tasks (MyoHub/myosuite#566)."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")

pytestmark = pytest.mark.tier2

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg  # noqa: E402
from mjlab.tasks.registry import list_tasks, load_env_cfg  # noqa: E402

import myosuite.envs.myo.backends.mjlab  # noqa: E402, F401
from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (  # noqa: E402
    _make_elbow_env_cfg,
)
from myosuite.envs.myo.backends.mjlab.tasks import mdp  # noqa: E402
from myosuite.terms.base_action import muscle_saturation_bounds  # noqa: E402

_SIGMOID = muscle_saturation_bounds(sigmoid=True)
_SIGMOID_CLAMPED = muscle_saturation_bounds(sigmoid=True, clip=(-1.0, 1.0))
_IDENTITY = muscle_saturation_bounds(sigmoid=False)
_EPS = 1e-3
_PENALTY = 2.0


def _builder_cfg(mode: str) -> Callable[[], ManagerBasedRlEnvCfg]:
    """The elbow config of ``MyoMuscleActivationAction`` with action mode *mode*."""

    def make() -> ManagerBasedRlEnvCfg:
        cfg = _make_elbow_env_cfg()
        cfg.actions["muscles"].action_mode = mode
        return cfg

    return make


def _registered(task_id: str) -> Callable[[], ManagerBasedRlEnvCfg]:
    return lambda: load_env_cfg(task_id)


# Every muscle action map: (config, saturation bounds of its policy actions).
_CASES = {
    "twin-sigmoid": (_registered("myoElbowPose1D6MRandom-v0"), _SIGMOID_CLAMPED),
    "twin-identity": (_registered("myoLegWalk-v0"), _IDENTITY),
    "builder-sigmoid": (_builder_cfg("sigmoid"), _SIGMOID),
    "builder-direct": (_builder_cfg("direct"), _IDENTITY),
    "builder-excitation": (_builder_cfg("excitation"), _IDENTITY),
    "table-tennis": (_registered("myoChallengeTableTennisP0-v0"), _SIGMOID_CLAMPED),
}


@pytest.fixture(scope="module", params=list(_CASES))
def case(
    request: pytest.FixtureRequest,
) -> Iterator[tuple[ManagerBasedRlEnv, tuple[float, float]]]:
    make_cfg, bounds = _CASES[request.param]
    cfg = make_cfg()
    cfg.scene.num_envs = 2
    mdp.add_muscle_saturation_penalty(cfg, _PENALTY)
    env = ManagerBasedRlEnv(cfg, device="cpu")
    env.reset()
    yield env, bounds
    env.close()


def _command(env: ManagerBasedRlEnv, value: float) -> None:
    """Pass the same action to every actuator (no physics step, so no resets)."""
    shape = (env.num_envs, env.action_manager.total_action_dim)
    env.action_manager.process_action(torch.full(shape, value, device=env.device))


def test_every_registered_muscle_task_logs_saturation() -> None:
    tasks = [t for t in list_tasks() if t.startswith("myo")]
    assert tasks
    missing = [t for t in tasks if mdp.SATURATION_TERM not in load_env_cfg(t).metrics]
    assert not missing


def test_metric_switches_at_the_map_bounds(
    case: tuple[ManagerBasedRlEnv, tuple[float, float]],
) -> None:
    env, (lo, hi) = case
    for value, expected in (
        (lo - _EPS, 1.0),
        (lo + _EPS, 0.0),
        (hi - _EPS, 0.0),
        (hi + _EPS, 1.0),
    ):
        _command(env, value)
        torch.testing.assert_close(
            mdp.muscle_saturation(env), torch.full((env.num_envs,), expected)
        )


def test_penalty_is_squared_excess_beyond_the_bounds(
    case: tuple[ManagerBasedRlEnv, tuple[float, float]],
) -> None:
    env, (lo, hi) = case
    assert env.cfg.rewards[mdp.SATURATION_TERM].weight == -_PENALTY
    for value in (lo - 0.5, hi + 0.5):
        _command(env, value)
        torch.testing.assert_close(
            mdp.muscle_saturation_l2(env), torch.full((env.num_envs,), 0.25)
        )
    _command(env, 0.5 * (lo + hi))
    assert not mdp.muscle_saturation_l2(env).any()


def test_metric_is_logged_per_episode(
    case: tuple[ManagerBasedRlEnv, tuple[float, float]],
) -> None:
    env, _ = case
    env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim))
    log = env.metrics_manager.reset()
    assert f"Episode_Metrics/{mdp.SATURATION_TERM}" in log


def test_penalty_is_opt_in() -> None:
    cfg = load_env_cfg("myoElbowPose1D6MRandom-v0")
    assert mdp.SATURATION_TERM not in cfg.rewards
    with pytest.raises(ValueError, match="penalty"):
        mdp.add_muscle_saturation_penalty(cfg, -1.0)
    mdp.add_muscle_saturation_penalty(cfg, 1.0)
    with pytest.raises(ValueError, match="already"):
        mdp.add_muscle_saturation_penalty(cfg, 1.0)
