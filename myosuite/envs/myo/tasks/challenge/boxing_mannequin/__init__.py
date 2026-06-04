# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Boxer-vs-mannequin challenge package."""

from myosuite.envs.myo.tasks.challenge.boxing_mannequin.boxing_mannequin_config import (
    BoxingMannequinConfig,
)
from myosuite.envs.myo.tasks.challenge.boxing_mannequin.boxing_mannequin_env import (
    ScriptedMannequinPolicy,
    make_boxing_6targets,
    make_boxing_mannequin_env,
)
from myosuite.envs.myo.tasks.challenge.boxing_mannequin.boxing_mannequin_model import (
    BoxingMannequinModelMeta,
)
from myosuite.envs.myo.tasks.challenge.boxing_mannequin.boxing_mannequin_task_config import (
    BoxingMannequinTaskConfig,
)

__all__ = [
    "BoxingMannequinConfig",
    "BoxingMannequinModelMeta",
    "BoxingMannequinTaskConfig",
    "ScriptedMannequinPolicy",
    "make_boxing_6targets",
    "make_boxing_mannequin_env",
]
