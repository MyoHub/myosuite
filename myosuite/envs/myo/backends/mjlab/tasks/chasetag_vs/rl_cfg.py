# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""RL configuration of the ChaseTag train env (one full-body player, 354 muscles)."""

from dataclasses import dataclass
from pathlib import Path

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

import myosuite

# The tutorial 5.4 directional walker the 5.6 ChaseTag Colab submits (in a checkout).
WALKER_CHECKPOINT = (
    Path(myosuite.__file__).resolve().parents[1]
    / "tutorials"
    / "files"
    / "5.6"
    / "policy_bc_best.pt"
)


@dataclass
class DirectionalWalkerActorCfg(RslRlModelCfg):
    """Actor = fixed 546 -> 528 adapter + the 5.4 walker network (``DirectionalWalkerActor``).

    Attributes:
        walker_checkpoint: 5.4 checkpoint the actor starts from; ``""`` starts
            the same network from a random init.
        hidden: Width of the 5.4 network.
        layers: Linear layers of its trunk.
    """

    class_name: str = "myosuite.integrations.musclemimic.directional_walker_actor:DirectionalWalkerActor"
    walker_checkpoint: str = str(WALKER_CHECKPOINT)
    hidden: int = 512
    layers: int = 6


def chasetag_vs_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """PPO config of ``myoChallengeChaseTagFBVsTrain-v0``: self-play from the 5.4 walker.

    Both players start as the walker the 5.6 Colab submits (it walks ~1.4 s in
    a mirror match; a random init falls in ~0.4 s). Fine-tuning settings: a
    small capped action noise without entropy bonus (with excitations clipped
    to ``[0, 1]`` a large std acts as extra activation, MyoHub/myosuite#566),
    and a low learning rate with ``desired_kl`` 0.01 so the first updates
    against a fresh critic do not undo the walker.
    """
    return RslRlOnPolicyRunnerCfg(
        actor=DirectionalWalkerActorCfg(
            obs_normalization=False,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.1,
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
            learning_rate=1.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="myo_chasetag_fbvs_train",
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=10000,
        obs_groups={"actor": ("policy",), "critic": ("policy",)},
    )
