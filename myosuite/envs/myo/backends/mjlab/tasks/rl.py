# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Shared PPO runner config for the MyoSuite mjlab tasks."""

from __future__ import annotations

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def myo_ppo_runner_cfg(
    experiment_name: str,
    hidden_dims: tuple[int, ...] = (128, 128, 128, 128),
    max_iterations: int = 1000,
    init_std: float = 0.5,
    entropy_coef: float = 0.001,
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
    task): a moderate initial action std, a small entropy bonus, and running
    observation normalization (raw observations mix radians, metres and [0, 1]
    activations).

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
        init_std: Initial std of the Gaussian policy (pre-sigmoid action units).
        entropy_coef: PPO entropy bonus coefficient.
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
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": init_std,
                "std_type": "scalar",
            },
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
