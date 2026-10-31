# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Shared PPO runner config for the MyoSuite mjlab tasks."""

from __future__ import annotations

from typing import Any

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from myosuite.envs.myo.backends.mjlab.tasks.distribution import (
    ProjectedGaussianDistribution,
    check_init_std,
)

# Bounds of the policy's action std (pre-mapping action units) for muscle tasks
# and clipped motor actions.
MUSCLE_STD_RANGE: tuple[float, float] = (0.05, 1.0)
# PPO entropy bonus for muscle tasks.
MUSCLE_ENTROPY_COEF: float = 0.001
# Import path that rsl_rl resolves for ``distribution_cfg["class_name"]``.
PROJECTED_GAUSSIAN_CLASS_NAME = (
    f"{ProjectedGaussianDistribution.__module__}:"
    f"{ProjectedGaussianDistribution.__qualname__}"
)


def muscle_gaussian_distribution_cfg(
    init_std: float = 0.5,
    std_range: tuple[float, float] = MUSCLE_STD_RANGE,
) -> dict[str, Any]:
    """rsl_rl Gaussian policy config for muscle actions, with a bounded std.

    The action mappings (``sigmoid(5 (a - 0.5))``, or a clip to the action range
    before a linear map to the control range) saturate within about one unit of
    their centre. Once a control saturates, its return no longer depends on its
    std, and the entropy bonus is the only systematic gradient left on it. Adam
    normalizes the gradient's scale, so the std then rises by up to the learning
    rate per update. The zero-mean noise of PPO's surrogate gradient makes the rise
    smaller: about ``c / sqrt(c**2 + 2 / N)`` of the learning rate for entropy
    coefficient ``c`` and ``N`` samples per minibatch. On ``myoMimicFullbody-v0``
    (``c = 0.01``) the std rose from 1 to 14 over 80k iterations
    (MyoHub/myosuite#566).

    :class:`~myosuite.envs.myo.backends.mjlab.tasks.distribution.ProjectedGaussianDistribution`
    projects the std parameter onto ``std_range`` before every forward pass, which
    bounds this drift. Unlike rsl_rl's own ``std_range`` clamp, it keeps the std
    trainable at the bounds, and a checkpoint with a larger std is projected to the
    bound on its first update. The bound does not by itself prevent saturated
    (bang-bang) controls.

    ``std_type`` stays ``"scalar"``: ``"log"`` renames the checkpoint parameter
    (``distribution.log_std_param``), and published checkpoints would not load.

    Args:
        init_std: Initial std, strictly inside ``std_range``.
        std_range: ``(min, max)`` bounds of the std.

    Returns:
        The ``distribution_cfg`` of an :class:`~mjlab.rl.RslRlModelCfg`.

    Raises:
        ValueError: If ``init_std`` is not strictly inside ``std_range``.
    """
    check_init_std(init_std, std_range)
    return {
        "class_name": PROJECTED_GAUSSIAN_CLASS_NAME,
        "init_std": init_std,
        "std_type": "scalar",
        "std_range": std_range,
    }


def myo_ppo_runner_cfg(
    experiment_name: str,
    hidden_dims: tuple[int, ...] = (128, 128, 128, 128),
    max_iterations: int = 1000,
    init_std: float = 0.5,
    entropy_coef: float = MUSCLE_ENTROPY_COEF,
    std_range: tuple[float, float] = MUSCLE_STD_RANGE,
    obs_normalization: bool = True,
    critic_hidden_dims: tuple[int, ...] = (256, 256, 256, 256),
    num_steps_per_env: int = 24,
    num_learning_epochs: int = 8,
    num_mini_batches: int = 8,
    clip_param: float = 0.3,
    gamma: float = 0.97,
) -> RslRlOnPolicyRunnerCfg:
    """PPO runner config shared by all pose and reach tasks.

    Algorithm settings and network sizes are those of the myoInteract pointing
    tasks, which train these muscle models much faster than the earlier defaults:
    8 epochs x 8 minibatches, clip 0.3, gamma 0.97, actor 4x128 and critic 4x256.
    Rollouts stay 24 steps (0.02 s control step); sampling rate, horizon and rewards
    stay the task's own.

    Defaults avoid a runaway-stochastic policy (observed: action std 1 -> 5.8,
    entropy 49 -> 102 and the adaptive learning rate collapsing on the arm-reach
    task): a moderate initial action std, a small entropy bonus, a bounded std
    (see :func:`muscle_gaussian_distribution_cfg`), and running observation
    normalization (raw observations mix radians, metres and [0, 1] activations).
    The motor-driven finger tasks (``motorFinger*``) use it too: their actions are
    clipped to ``[-1, 1]`` before the linear map to ``ctrlrange``, so the same
    saturation applies.

    The normalizer statistics are saved in the checkpoint and frozen at
    evaluation; ``load_rslrl_policy`` / ``export_rslrl_to_onnx`` fold them into
    the policy, which therefore still consumes the raw CPU observations. Tasks
    that must export to the browser runtime (which cannot load running
    statistics, see ``docs/wiki/cross-backend-contract.md``) pass
    ``obs_normalization=False`` and rely on fixed per-term ``scale`` instead.

    Args:
        experiment_name: Log directory name under ``logs/rsl_rl``.
        hidden_dims: Actor MLP sizes.
        max_iterations: Default PPO iterations (``--agent.max-iterations``).
        init_std: Initial std of the Gaussian policy (pre-mapping action units),
            strictly inside ``std_range``.
        entropy_coef: PPO entropy bonus coefficient.
        std_range: ``(min, max)`` bounds of the Gaussian policy's std.
        obs_normalization: Running mean/std normalization of actor and critic inputs.
        critic_hidden_dims: Critic MLP sizes.
        num_steps_per_env: Rollout length per env and PPO iteration.
        num_learning_epochs: PPO epochs per iteration.
        num_mini_batches: Minibatches per epoch.
        clip_param: PPO clipping epsilon.
        gamma: Discount factor.

    Returns:
        The runner config.
    """
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=hidden_dims,
            activation="elu",
            obs_normalization=obs_normalization,
            distribution_cfg=muscle_gaussian_distribution_cfg(init_std, std_range),
        ),
        critic=RslRlModelCfg(
            hidden_dims=critic_hidden_dims,
            activation="elu",
            obs_normalization=obs_normalization,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=clip_param,
            entropy_coef=entropy_coef,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=gamma,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name=experiment_name,
        save_interval=100,
        num_steps_per_env=num_steps_per_env,
        max_iterations=max_iterations,
    )
