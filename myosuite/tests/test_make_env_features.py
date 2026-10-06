# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""``make_env(EnvConfig(...))``: the same features and episode length on every backend."""

from __future__ import annotations

import functools

import gymnasium as gym
import numpy as np
import pytest

import myosuite  # noqa: F401
from myosuite.core.config import EnvConfig
from myosuite.core.registry import make_env
from myosuite.envs.muscle_stages import LowPassStage
from myosuite.envs.wrappers import (
    ExcitationStageWrapper,
    FatigueWrapper,
    MotorNoiseWrapper,
    wrapper_spec,
)

pytestmark = pytest.mark.tier2

_ID = "myoElbowPose1D6MRandom-v0"
_DIRECTIONAL = "myoLegDirectionalForward-v0"  # a ModularTaskEnv id with an mjlab twin
_NOISE = {"signal_dependent_std": 0.1, "constant_std": 0.02}


def _rollout(env: gym.Env, steps: int = 8) -> np.ndarray:
    env.reset(seed=3)
    rng = np.random.default_rng(0)
    out = [
        env.step(rng.uniform(-1, 1, env.action_space.shape))[0] for _ in range(steps)
    ]
    return np.asarray(out)


def test_cpu_features_equal_the_wrapper_stack() -> None:
    cfg = EnvConfig(
        _ID, features=(wrapper_spec(MotorNoiseWrapper, motor_noise=_NOISE),)
    )
    ref = MotorNoiseWrapper(gym.make(_ID), _NOISE)
    np.testing.assert_array_equal(_rollout(make_env(cfg)), _rollout(ref))


def test_a_plain_id_and_cpu_overrides_still_work() -> None:
    env = make_env(_ID, frame_skip=5)
    assert env.unwrapped.frame_skip == 5


def test_max_episode_steps_truncates_on_the_cpu() -> None:
    env = make_env(EnvConfig(_ID, max_episode_steps=4))
    env.reset(seed=0)
    truncated = [env.step(env.action_space.sample())[3] for _ in range(4)]
    assert truncated == [False, False, False, True]


def test_ctrl_dt_sets_the_substeps_on_the_cpu() -> None:
    host = make_env(_ID).unwrapped
    dt = host.model.opt.timestep
    half = make_env(EnvConfig(_ID, ctrl_dt=host.frame_skip * dt / 2)).unwrapped
    assert half.frame_skip == host.frame_skip // 2
    assert half.dt == pytest.approx(host.dt / 2)


def test_ctrl_dt_must_be_a_multiple_of_the_timestep() -> None:
    with pytest.raises(ValueError, match="Control step mismatch"):
        make_env(EnvConfig(_ID, ctrl_dt=0.0123))


def test_cpu_rejects_parallel_envs() -> None:
    with pytest.raises(ValueError, match="one env"):
        make_env(EnvConfig(_ID, num_envs=8))


def test_features_other_than_sarcopenia_are_not_supported_on_mjx() -> None:
    cfg = EnvConfig(_ID, features=(wrapper_spec(FatigueWrapper),))
    with pytest.raises(NotImplementedError, match="mjx"):
        make_env(cfg, backend="mjx")


def test_mjx_supports_sarcopenia_only() -> None:
    from myosuite.core.registry import mjx_feature_overrides
    from myosuite.envs.wrappers import SarcopeniaWrapper

    assert mjx_feature_overrides(()) == {}
    assert mjx_feature_overrides((wrapper_spec(SarcopeniaWrapper),)) == {
        "sarcopenia_force_scale": 0.5
    }
    assert mjx_feature_overrides(
        (wrapper_spec(SarcopeniaWrapper, force_scale=0.3),)
    ) == {"sarcopenia_force_scale": 0.3}
    assert mjx_feature_overrides((wrapper_spec(FatigueWrapper),)) is None


def test_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        make_env(_ID, backend="nope")


@pytest.mark.tier2
class TestMjlab:
    @pytest.fixture(autouse=True)
    def _need_mjlab(self) -> None:
        pytest.importorskip("mjlab")

    def _stages(self, env) -> tuple[str, ...]:
        return env.action_manager.get_term("muscles").stage_names

    def test_features_reach_the_twin(self) -> None:
        cfg = EnvConfig(
            _ID,
            num_envs=2,
            features=(
                wrapper_spec(MotorNoiseWrapper, motor_noise=_NOISE),
                wrapper_spec(FatigueWrapper),
                wrapper_spec(
                    ExcitationStageWrapper,
                    make_stage=functools.partial(LowPassStage, 0.5),
                ),
            ),
        )
        env = make_env(cfg, backend="mjlab", device="cpu")
        try:
            term = env.action_manager.get_term("muscles")
            assert env.num_envs == 2
            assert term.stage_names == ("noise", "fatigue", "lowpass")
            assert term.cfg.motor_noise.signal_dependent_std == 0.1
            assert term._fatigue is not None
        finally:
            env.close()

    def test_no_features_leaves_the_registration_untouched(self) -> None:
        env = make_env(EnvConfig(_ID, num_envs=1), backend="mjlab", device="cpu")
        try:
            assert self._stages(env) == ("noise",)
            assert not env.action_manager.get_term("muscles").cfg.motor_noise.enabled
        finally:
            env.close()

    def test_a_later_call_is_not_polluted(self) -> None:
        cfg = EnvConfig(_ID, num_envs=1, features=(wrapper_spec(FatigueWrapper),))
        make_env(cfg, backend="mjlab", device="cpu").close()
        env = make_env(_ID, backend="mjlab", num_envs=1, device="cpu")
        try:
            assert env.action_manager.get_term("muscles")._fatigue is None
        finally:
            env.close()

    def test_a_registered_wrapper_cannot_be_added_twice(self) -> None:
        cfg = EnvConfig(
            "myoFatiElbowPose1D6MRandom-v0",
            num_envs=1,
            features=(wrapper_spec(FatigueWrapper),),
        )
        with pytest.raises(ValueError, match="already has a FatigueWrapper"):
            make_env(cfg, backend="mjlab", device="cpu")

    def test_ctrl_dt_sets_the_twin_decimation(self) -> None:
        base = make_env(EnvConfig(_ID, num_envs=1), backend="mjlab", device="cpu")
        try:
            decimation, step_dt = base.cfg.decimation, base.step_dt
        finally:
            base.close()
        cfg = EnvConfig(_ID, num_envs=1, ctrl_dt=step_dt / 2)
        env = make_env(cfg, backend="mjlab", device="cpu")
        try:
            assert env.cfg.decimation == decimation // 2
            assert env.step_dt == pytest.approx(step_dt / 2)
        finally:
            env.close()

    def test_task_kwargs_are_not_applicable_to_the_twin(self) -> None:
        cfg = EnvConfig(_ID, task_kwargs={"frame_skip": 5})
        with pytest.raises(NotImplementedError, match="task_kwargs"):
            make_env(cfg, backend="mjlab", device="cpu")

    def test_max_episode_steps_sets_the_twin_horizon(self) -> None:
        cfg = EnvConfig(_ID, num_envs=1, max_episode_steps=37)
        env = make_env(cfg, backend="mjlab", device="cpu")
        try:
            assert env.max_episode_length == 37
        finally:
            env.close()

    def test_features_reach_a_taskconfig_twin(self) -> None:
        """The leg-directional twin (a ModularTaskEnv id) is rebuilt with the features too."""
        from myosuite.envs.myo.backends.mjlab.tasks.registration import (
            rebuild_twin_cfg,
        )

        features = (
            wrapper_spec(MotorNoiseWrapper, motor_noise=_NOISE),
            wrapper_spec(FatigueWrapper),
        )
        action = rebuild_twin_cfg(_DIRECTIONAL, features).actions["muscles"]
        assert action.motor_noise.signal_dependent_std == 0.1
        assert action.muscle_fatigue
        base = rebuild_twin_cfg(_DIRECTIONAL).actions["muscles"]
        assert not base.motor_noise.enabled and not base.muscle_fatigue

    def test_ctrl_dt_is_refused_for_taskconfig_ids_on_both_backends(self) -> None:
        """Their timing is part of the task config: no backend changes it silently."""
        with pytest.raises(ValueError, match="ctrl_dt cannot change"):
            make_env(EnvConfig(_DIRECTIONAL, ctrl_dt=0.01))
        cfg = EnvConfig(_DIRECTIONAL, num_envs=1, ctrl_dt=0.01)
        with pytest.raises(ValueError, match="ctrl_dt cannot change"):
            make_env(cfg, backend="mjlab", device="cpu")
