# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The distribution-matching reward on the CPU (``DistributionMatchingWrapper``) and on mjlab
(``mdp.DistributionMatchingReward``), on the elbow pose task."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from myosuite import make_env
from myosuite.envs.wrappers import DistributionMatchingWrapper
from myosuite.terms.distribution_reward import (
    DistributionMatchingCfg,
    GaussianMixtureSpec,
    density_ratio,
    mixture_log_density,
)
from myosuite.tests.support.optional_deps import require_mjlab, require_mujoco_warp
from myosuite.utils.density_estimator import KernelDensity

pytestmark = pytest.mark.tier2

ENV_ID = "myoElbowPose1D6MRandom-v0"
TARGET = GaussianMixtureSpec.from_gaussian(
    [1.0], [[0.09]], feature_names=("elbow angle (rad)",)
)


def _elbow(accessor: Any, task_state: dict[str, Any]) -> Any:
    return accessor.joint_pos()[..., :1]


def _cfg(**kwargs: Any) -> DistributionMatchingCfg:
    return DistributionMatchingCfg(feature=_elbow, target=TARGET, **kwargs)


def _rewards(env: Any, seed: int, steps: int = 8) -> list[float]:
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return [env.step(env.action_space.sample())[1] for _ in range(steps)]


def test_the_wrapper_adds_the_weighted_term() -> None:
    plain = _rewards(make_env(ENV_ID), seed=3)
    cfg = _cfg(estimator=KernelDensity(1, capacity=64))
    env = DistributionMatchingWrapper(make_env(ENV_ID), cfg, weight=0.5)
    env.reset(seed=3)
    env.action_space.seed(3)
    for base in plain:
        _, reward, _, _, info = env.step(env.action_space.sample())
        elbow = env.unwrapped.data.qpos[:1]
        assert reward == pytest.approx(base + 0.5 * info["distribution_reward"])
        assert info["distribution_log_p"] == pytest.approx(
            float(mixture_log_density(np, elbow, TARGET))
        )
    assert cfg.estimator.count == len(plain)


def test_the_wrapper_refuses_a_feature_of_several_frames() -> None:
    estimator = KernelDensity(1, capacity=8)
    cfg = DistributionMatchingCfg(
        feature=lambda accessor, state: np.zeros((2, 1)),
        target=TARGET,
        estimator=estimator,
    )
    env = DistributionMatchingWrapper(make_env(ENV_ID), cfg)
    env.reset(seed=0)
    with pytest.raises(ValueError, match="one frame per step"):
        env.step(env.action_space.sample())
    assert estimator.count == 0  # refused before it learnt


def test_a_diverged_step_pays_nothing_and_teaches_nothing() -> None:
    import gymnasium as gym

    class _Diverged(gym.Wrapper):
        def step(self, action: Any) -> tuple:
            obs, reward, terminated, truncated, info = self.env.step(action)
            return obs, reward, terminated, truncated, {**info, "mj_unstable": True}

    estimator = KernelDensity(1, capacity=8)
    plain = make_env(ENV_ID)
    plain.reset(seed=1)
    env = DistributionMatchingWrapper(
        _Diverged(make_env(ENV_ID)), _cfg(estimator=estimator)
    )
    env.reset(seed=1)
    action = env.action_space.sample()
    _, reward, _, _, info = env.step(action)
    assert reward == plain.step(action)[1]
    assert info["distribution_reward"] == 0.0 and estimator.count == 0
    assert "distribution_log_p" not in info


def test_a_frozen_estimator_gives_the_same_reward_after_each_reset() -> None:
    cfg = _cfg(estimator=KernelDensity(1, capacity=64))
    env = DistributionMatchingWrapper(make_env(ENV_ID), cfg)
    learning = _rewards(env, seed=1)
    assert _rewards(env, seed=1) != learning  # the estimate moved on
    cfg.estimator.freeze()
    frozen = _rewards(env, seed=1)
    assert _rewards(env, seed=1) == frozen
    assert cfg.estimator.count == 16


def test_the_mjlab_term_pays_the_same_as_numpy() -> None:
    require_mjlab()
    require_mujoco_warp()
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.tasks.registry import load_env_cfg

    from myosuite.envs.myo.backends.mjlab.tasks import mdp
    from myosuite.tests.support.host_sync import HostSyncCounter

    cfg = load_env_cfg(ENV_ID)
    cfg.scene.num_envs = 3
    cfg.rewards = {
        "distribution": RewardTermCfg(
            func=mdp.DistributionMatchingReward,
            weight=1.0,
            params={
                "distribution": _cfg(estimator=KernelDensity(1, capacity=64)),
                "entity_name": "robot",
            },
        )
    }
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        env.reset()
        # mjlab copies the cfg: the env's estimator is the one that learns
        estimator = (
            env.reward_manager.get_term_cfg("distribution")
            .params["distribution"]
            .estimator
        )
        mirror = KernelDensity(1, capacity=64)
        generator = torch.Generator().manual_seed(0)
        steps = 6
        for step in range(steps):
            action = torch.rand(
                3, env.action_manager.total_action_dim, generator=generator
            )
            with HostSyncCounter(package_only=True) as syncs:
                reward = env.step(action)[1]
            elbow = env.scene["robot"].data.joint_pos[:, :1].numpy().astype(np.float64)
            expected = density_ratio(np, elbow, TARGET, mirror)["dense"]
            mirror.update(np, elbow)
            if not env.reset_buf.any():
                np.testing.assert_allclose(reward.numpy(), expected, atol=1e-4)
            own = [
                site
                for (_, site) in syncs.sites
                if "density_estimator" in site
                or "distribution_reward" in site
                or "rewards.py" in site
            ]
            assert (
                step == 0 or not own
            ), syncs.report()  # the first step builds the buffer
        assert estimator.count == 3 * steps
    finally:
        env.close()


def test_the_mjlab_term_ignores_a_diverged_env() -> None:
    """mjlab scores a diverged env before it resets it: its NaN frame pays 0 and leaves the estimator finite."""
    require_mjlab()
    require_mujoco_warp()
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.tasks.registry import load_env_cfg

    from myosuite.envs.myo.backends.mjlab.tasks import mdp
    from myosuite.tests.support.host_sync import HostSyncCounter

    def elbow_with_env_1_diverged(accessor: Any, task_state: dict[str, Any]) -> Any:
        x = accessor.joint_pos()[..., :1].clone()
        x[1] = float("nan")
        return x

    cfg = load_env_cfg(ENV_ID)
    cfg.scene.num_envs = 3
    term = DistributionMatchingCfg(
        feature=elbow_with_env_1_diverged,
        target=TARGET,
        estimator=KernelDensity(1, capacity=64),
    )
    cfg.rewards = {
        "distribution": RewardTermCfg(
            func=mdp.DistributionMatchingReward,
            weight=1.0,
            params={"distribution": term, "entity_name": "robot"},
        )
    }
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        env.reset()
        estimator = (
            env.reward_manager.get_term_cfg("distribution")
            .params["distribution"]
            .estimator
        )
        generator = torch.Generator().manual_seed(0)
        for step in range(4):
            action = torch.rand(
                3, env.action_manager.total_action_dim, generator=generator
            )
            with HostSyncCounter(package_only=True) as syncs:
                reward = env.step(action)[1]
            assert reward[1] == 0.0 and torch.isfinite(reward).all()
            own = [s for (_, s) in syncs.sites if "distribution_reward" in s]
            assert step == 0 or not own, syncs.report()
        assert torch.isfinite(estimator.buffer[: estimator.count]).all()
    finally:
        env.close()


TARGET_4D = GaussianMixtureSpec.from_gaussian(
    [0.0, 0.2, 0.0, 0.3],
    np.diag([0.09, 0.09, 0.25, 0.04]),
    feature_names=(
        "elbow angle from the target (rad)",
        "0.1 x wrist angular speed (rad/s)",
        "world height of the wrist's x axis",
        "mean muscle activation",
    ),
)


def _wrist_features(accessor: Any, task_state: dict[str, Any]) -> Any:
    """The angle from the task's target, the wrist site's angular speed and orientation, and the mean activation:
    the reads a head feature needs, written alike for numpy and torch."""
    xp = accessor.array_module()
    site = accessor.site_id("wrist")
    spin = accessor.site_ang_vel(site)
    act = accessor.muscle_act()
    return xp.concatenate(
        [
            accessor.joint_pos()[..., :1] - task_state["target_angles"][..., :1],
            0.1 * ((spin * spin).sum(-1) ** 0.5)[..., None],
            accessor.site_xmat(site)[..., 2, 0][..., None],
            act.sum(-1)[..., None] / act.shape[-1],
        ],
        -1,
    )


def test_cpu_and_mjlab_score_matched_states_alike() -> None:
    """The CPU wrapper's rollout is replayed state by state into the mjlab twin, whose term scores each state; both
    sides' estimators learn from their own reads of the same states, so the two rewards agree at every step."""
    require_mjlab()
    require_mujoco_warp()
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.tasks.registry import load_env_cfg

    from myosuite.envs.myo.backends.mjlab.tasks import mdp

    def cfg() -> DistributionMatchingCfg:
        estimator = KernelDensity(4, capacity=64, bandwidth_scale=0.5)
        return DistributionMatchingCfg(
            feature=_wrist_features, target=TARGET_4D, estimator=estimator
        )

    cpu_cfg = cfg()
    cpu = DistributionMatchingWrapper(make_env(ENV_ID), cpu_cfg)
    cpu.reset(seed=4)
    cpu.action_space.seed(4)
    base = cpu.unwrapped
    states, cpu_rewards = [], []
    for _ in range(40):
        *_, info = cpu.step(cpu.action_space.sample())
        states.append(
            (
                base.data.qpos.copy(),
                base.data.qvel.copy(),
                base.data.act.copy(),
                base._task_state["target_angles"].copy(),
            )
        )
        cpu_rewards.append(info["distribution_reward"])
    cpu.close()
    assert np.ptp(cpu_rewards) > 1.0  # the states differ enough to test the term

    mj_cfg = load_env_cfg(ENV_ID)
    mj_cfg.scene.num_envs = 1
    mj_cfg.rewards = {
        "distribution": RewardTermCfg(
            func=mdp.DistributionMatchingReward,
            weight=1.0,
            params={
                "distribution": cfg(),
                "entity_name": "robot",
                # built as the pose task's own terms build theirs
                "task_state": lambda env: {
                    "target_angles": env.command_manager.get_command("pose")
                },
            },
        )
    }
    env = ManagerBasedRlEnv(cfg=mj_cfg, device="cpu")
    try:
        env.reset()
        ids = torch.zeros(1, dtype=torch.long)
        mj_rewards = []
        for qpos, qvel, act, target in states:
            as_row = lambda x: torch.as_tensor(x[None], dtype=torch.float32)  # noqa: E731
            mdp.write_cpu_state(env, "robot", ids, as_row(qpos), as_row(qvel))
            env.sim.data.act[:] = as_row(act)
            env.command_manager.get_term("pose")._target[:] = as_row(target)
            env.sim.forward()
            env.reward_manager.compute(env.step_dt)
            mj_rewards.append(
                dict(env.reward_manager.get_active_iterable_terms(0))["distribution"][0]
            )
        # float32 against float64 differs by up to 4.1e-4; a 1% error in the written velocity gives 1.8e-2, an
        # unsynced target 1.1, an unwritten activation or a state one step late 20
        np.testing.assert_allclose(mj_rewards, cpu_rewards, rtol=0, atol=2e-3)
        mj_estimator = (
            env.reward_manager.get_term_cfg("distribution")
            .params["distribution"]
            .estimator
        )
        np.testing.assert_allclose(
            mj_estimator.buffer.numpy(), cpu_cfg.estimator.buffer, rtol=0, atol=1e-5
        )
    finally:
        env.close()
