# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""ChaseTag train env (one FBVs player vs. an in-env opponent); importing registers the twin."""

from myosuite.envs.myo.backends.mjlab.tasks.registration import register_cpu_twins

from .env_cfg import make_chasetag_vs_env_cfg
from .rl_cfg import chasetag_vs_ppo_runner_cfg

CHASETAG_VS_IDS = ("myoChallengeChaseTagFBVsTrain-v0",)

register_cpu_twins(
    CHASETAG_VS_IDS, make_chasetag_vs_env_cfg, chasetag_vs_ppo_runner_cfg
)
