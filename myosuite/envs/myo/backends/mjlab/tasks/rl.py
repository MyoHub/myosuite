# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Shared PPO runner config for the MyoSuite mjlab tasks."""

from __future__ import annotations

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def myo_ppo_runner_cfg(
    experiment_name: str,
    hidden_dims: tuple[int, ...] = (256, 256),
    max_iterations: int = 1000,
) -> RslRlOnPolicyRunnerCfg:
    """PPO runner config shared by the basic-suite tasks.

    Observation normalization is off: the cross-backend contract requires
    policies to consume raw CPU observations (no running mean/std), so a policy
    trained here plays back unchanged in the CPU env.

    Args:
        experiment_name: Log directory name under ``logs/rsl_rl``.
        hidden_dims: Actor and critic MLP sizes.
        max_iterations: Default PPO iterations (``--agent.max-iterations``).

    Returns:
        The runner config.
    """
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=hidden_dims,
            activation="elu",
            obs_normalization=False,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=hidden_dims,
            activation="elu",
            obs_normalization=False,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name=experiment_name,
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=max_iterations,
    )
