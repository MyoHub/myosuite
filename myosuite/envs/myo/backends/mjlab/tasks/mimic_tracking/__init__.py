# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""mjlab twin of the motion-tracking task; importing registers it."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from myosuite.envs.myo.backends.mjlab.tasks.registration import register_cpu_twins
from myosuite.envs.myo.backends.mjlab.tasks.rl import myo_ppo_runner_cfg
from myosuite.envs.myo.tasks.mimic.tracking import MIMIC_ENV_ID

from .env_cfg import make_mimic_env_cfg


def mimic_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Generic PPO defaults (both observation groups to actor and critic).

    Not tuned for this task: a MuscleMimic PPO config is a planned method preset.
    """
    cfg = myo_ppo_runner_cfg(
        "myo_mimic",
        hidden_dims=(1024, 512, 256),
        critic_hidden_dims=(1024, 512, 256),
        gamma=0.99,
    )
    groups = ("policy", "goal")
    cfg.obs_groups = {"actor": groups, "critic": groups}
    return cfg


register_cpu_twins((MIMIC_ENV_ID,), make_mimic_env_cfg, mimic_ppo_runner_cfg)
