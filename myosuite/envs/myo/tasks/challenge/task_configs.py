# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""TaskConfig coverage for all MyoChallenge task variants.

These configs provide modular-task registration coverage for all public
MyoChallenge IDs. They are intentionally lightweight and can be refined as
challenge-specific term functions evolve.
"""

from __future__ import annotations

from dataclasses import dataclass, field


from myosuite.core.config import GoalSpec, RewardSpec, TaskConfig
from myosuite.core.registry import register_task


@dataclass
class SoccerP1Task(TaskConfig):
    model: str = "elbow_standard"
    max_episode_steps: int = 2000
    goal: GoalSpec = field(default_factory=lambda: GoalSpec(target_type="joint_angles"))
    reward: RewardSpec = field(
        default_factory=lambda: RewardSpec(terms=["act_reg"], weights={"act_reg": 1.0})
    )


@dataclass
class SoccerP2Task(SoccerP1Task):
    pass


@dataclass
class TableTennisP0Task(TaskConfig):
    model: str = "challenge_tabletennis"
    max_episode_steps: int = 300


@dataclass
class TableTennisP1Task(TableTennisP0Task):
    pass


@dataclass
class TableTennisP2Task(TableTennisP0Task):
    pass


@dataclass
class BimanualTask(TaskConfig):
    model: str = "full_arm"
    max_episode_steps: int = 1000


@dataclass
class OslRunFixedTask(TaskConfig):
    model: str = "elbow_standard"
    max_episode_steps: int = 1000


@dataclass
class OslRunRandomTask(TaskConfig):
    model: str = "elbow_standard"
    max_episode_steps: int = 60000


@dataclass
class ChaseTagP1Task(TaskConfig):
    model: str = "elbow_standard"
    max_episode_steps: int = 2000


@dataclass
class ChaseTagP2Task(ChaseTagP1Task):
    pass


@dataclass
class ChaseTagP2EvalTask(ChaseTagP1Task):
    pass


@dataclass
class DieReorientDemoTask(TaskConfig):
    model: str = "hand_standard"
    max_episode_steps: int = 150


@dataclass
class DieReorientP1Task(DieReorientDemoTask):
    pass


@dataclass
class DieReorientP2Task(DieReorientDemoTask):
    pass


@dataclass
class RelocateP1Task(TaskConfig):
    model: str = "challenge_relocate"
    max_episode_steps: int = 150


@dataclass
class RelocateP2Task(RelocateP1Task):
    pass


@dataclass
class RelocateP2EvalTask(RelocateP1Task):
    pass


@dataclass
class BaodingP1Task(TaskConfig):
    model: str = "challenge_baoding"
    max_episode_steps: int = 200


@dataclass
class BaodingP2Task(BaodingP1Task):
    pass


def register_myochallenge_modular_tasks() -> dict[str, str]:
    """Register modular MyoChallenge tasks and return env_id mapping."""
    mapping: dict[str, str] = {}
    mapping["myoChallengeSoccerP1Modular-v0"] = register_task(
        SoccerP1Task(), env_id="myoChallengeSoccerP1Modular-v0"
    )
    mapping["myoChallengeSoccerP2Modular-v0"] = register_task(
        SoccerP2Task(), env_id="myoChallengeSoccerP2Modular-v0"
    )
    mapping["myoChallengeTableTennisP0Modular-v0"] = register_task(
        TableTennisP0Task(), env_id="myoChallengeTableTennisP0Modular-v0"
    )
    mapping["myoChallengeTableTennisP1Modular-v0"] = register_task(
        TableTennisP1Task(), env_id="myoChallengeTableTennisP1Modular-v0"
    )
    mapping["myoChallengeTableTennisP2Modular-v0"] = register_task(
        TableTennisP2Task(), env_id="myoChallengeTableTennisP2Modular-v0"
    )
    mapping["myoChallengeBimanualModular-v0"] = register_task(
        BimanualTask(), env_id="myoChallengeBimanualModular-v0"
    )
    mapping["myoChallengeOslRunFixedModular-v0"] = register_task(
        OslRunFixedTask(), env_id="myoChallengeOslRunFixedModular-v0"
    )
    mapping["myoChallengeOslRunRandomModular-v0"] = register_task(
        OslRunRandomTask(), env_id="myoChallengeOslRunRandomModular-v0"
    )
    mapping["myoChallengeChaseTagP1Modular-v0"] = register_task(
        ChaseTagP1Task(), env_id="myoChallengeChaseTagP1Modular-v0"
    )
    mapping["myoChallengeChaseTagP2Modular-v0"] = register_task(
        ChaseTagP2Task(), env_id="myoChallengeChaseTagP2Modular-v0"
    )
    mapping["myoChallengeChaseTagP2evalModular-v0"] = register_task(
        ChaseTagP2EvalTask(), env_id="myoChallengeChaseTagP2evalModular-v0"
    )
    mapping["myoChallengeDieReorientDemoModular-v0"] = register_task(
        DieReorientDemoTask(), env_id="myoChallengeDieReorientDemoModular-v0"
    )
    mapping["myoChallengeDieReorientP1Modular-v0"] = register_task(
        DieReorientP1Task(), env_id="myoChallengeDieReorientP1Modular-v0"
    )
    mapping["myoChallengeDieReorientP2Modular-v0"] = register_task(
        DieReorientP2Task(), env_id="myoChallengeDieReorientP2Modular-v0"
    )
    mapping["myoChallengeRelocateP1Modular-v0"] = register_task(
        RelocateP1Task(), env_id="myoChallengeRelocateP1Modular-v0"
    )
    mapping["myoChallengeRelocateP2Modular-v0"] = register_task(
        RelocateP2Task(), env_id="myoChallengeRelocateP2Modular-v0"
    )
    mapping["myoChallengeRelocateP2evalModular-v0"] = register_task(
        RelocateP2EvalTask(), env_id="myoChallengeRelocateP2evalModular-v0"
    )
    mapping["myoChallengeBaodingP1Modular-v0"] = register_task(
        BaodingP1Task(), env_id="myoChallengeBaodingP1Modular-v0"
    )
    mapping["myoChallengeBaodingP2Modular-v0"] = register_task(
        BaodingP2Task(), env_id="myoChallengeBaodingP2Modular-v0"
    )
    return mapping
