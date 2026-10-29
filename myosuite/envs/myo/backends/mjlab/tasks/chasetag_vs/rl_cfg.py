# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""RL configuration of the ChaseTag train env (one full-body player, 354 muscles)."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def chasetag_vs_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO config of ``myoChallengeChaseTagFBVsTrain-v0`` (self-play).

    Settings of the full-body Halloween task: the action noise is capped
    (``std_range``) and gets no entropy bonus, because with excitations clipped
    to ``[0, 1]`` a large std acts as extra activation and the mean action then
    fails alone (MyoHub/myosuite#566); ``desired_kl`` 0.03 keeps the adaptive
    learning rate off its floor.
    """
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.2,
                "std_type": "scalar",
                "std_range": (0.01, 0.2),
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.0,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.03,
            max_grad_norm=1.0,
        ),
        experiment_name="myo_chasetag_fbvs_train",
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=10000,
        obs_groups={"actor": ("policy",), "critic": ("policy",)},
    )
