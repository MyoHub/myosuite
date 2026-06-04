# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""TaskSpec + registration for myoChallengeSaberP0-v0."""

from __future__ import annotations

from myosuite.core.registry import register
from myosuite.core.specs import EnvSpec, TaskSpec
from myosuite.envs.myo.tasks.challenge.saber_task_config import SaberP0Task

SABER_P0_ENV_ID = "myoChallengeSaberP0-v0"
SABER_P0_MIMIC_ENV_ID = "myoChallengeSaberP0Mimic-v0"


def get_saber_task_spec() -> TaskSpec:
    """Build the task spec for Saber P0."""
    return TaskSpec(task_config_factory=SaberP0Task, backends={"cpu"})


def get_saber_p0_env_spec() -> EnvSpec:
    """Build the env spec for Saber P0."""
    return EnvSpec(env_id=SABER_P0_ENV_ID, task_spec=get_saber_task_spec())


def register_saber_p0_env() -> str:
    """Register Saber P0 via EnvSpec adapter path."""
    return register(get_saber_p0_env_spec(), wrap_mj_instability_termination=False)
