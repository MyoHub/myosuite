# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""TaskSpec + registration for myoChallengeBoxingP0Mjlab-v0.

Follows the pattern established in ``saber_task_spec.py``.
"""

from __future__ import annotations

from myosuite.core.registry import register
from myosuite.core.specs import EnvSpec, TaskSpec
from myosuite.envs.myo.tasks.challenge.boxing_mannequin.boxing_mannequin_task_config import (
    BoxingMannequinTaskConfig,
)

BOXING_P0_MJLAB_ENV_ID = "myoChallengeBoxingP0Mjlab-v0"


def get_boxing_p0_task_spec() -> TaskSpec:
    """Build the task spec for BoxingP0 (6-pad static target variant)."""
    return TaskSpec(
        task_config_factory=lambda: BoxingMannequinTaskConfig(
            include_boxing_targets_scene=True
        ),
        backends={"cpu"},
    )


def get_boxing_p0_env_spec() -> EnvSpec:
    """Build the env spec for BoxingP0."""
    return EnvSpec(env_id=BOXING_P0_MJLAB_ENV_ID, task_spec=get_boxing_p0_task_spec())


def register_boxing_p0_env() -> str:
    """Register BoxingP0 via EnvSpec adapter path."""
    return register(get_boxing_p0_env_spec(), wrap_mj_instability_termination=False)
