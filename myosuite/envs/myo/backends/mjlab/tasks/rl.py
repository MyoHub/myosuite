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
    init_std: float = 0.5,
    entropy_coef: float = 0.001,
    obs_normalization: bool = True,
) -> RslRlOnPolicyRunnerCfg:
    """PPO runner config shared by the basic-suite tasks.

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
        hidden_dims: Actor and critic MLP sizes.
        max_iterations: Default PPO iterations (``--agent.max-iterations``).
        init_std: Initial std of the Gaussian policy (pre-sigmoid action units).
        entropy_coef: PPO entropy bonus coefficient.
        obs_normalization: Running mean/std normalization of actor and critic inputs.

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
            hidden_dims=hidden_dims,
            activation="elu",
            obs_normalization=obs_normalization,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=entropy_coef,
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
