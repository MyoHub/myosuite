# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""TaskConfig-based specs for elbow pose environments.

These specs demonstrate the target architecture: adding a new environment
requires only a TaskConfig subclass and a register_task() call — no new
class, no new registration file, no backend-specific code.

Registered environments
-----------------------
CPU (Gymnasium):
    ``myoElbowPoseTaskFixed-v0``   — fixed target (r_elbow_flex = 2.0 rad)
    ``myoElbowPoseTaskRandom-v0``  — random target in [0.0, 2.27] rad
    ``myoSarcElbowPoseTaskFixed-v0``   / ``myoSarcElbowPoseTaskRandom-v0``
    ``myoFatiElbowPoseTaskFixed-v0``   / ``myoFatiElbowPoseTaskRandom-v0``

These run via :class:`~myosuite.envs.modular_env.ModularTaskEnv` and are
therefore fully driven by the ``TaskConfig`` — no subclassing required.

Note: these IDs are separate from the legacy ``myoElbowPose1D6M*-v0`` envs
(which use ``PoseEnvV0`` with a raw XML path).  The two sets use different
model assembly paths (recipe vs raw XML) and are not required to be
numerically identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from myosuite.core.config import (
    ActuatorGroupSpec,
    BackendConfig,
    GoalSpec,
    ObsSpec,
    RewardSpec,
    TaskConfig,
    VariantSpec,
)
from myosuite.core.registry import register_task


@dataclass
class ElbowPoseFixedTask(TaskConfig):
    """Elbow pose task with a fixed target (r_elbow_flex = 2.0 rad).

    Uses the ``elbow_standard`` ModelBuilder recipe (fragment-based assembly).
    Registered as ``myoElbowPoseTaskFixed-v0`` on import.
    """

    model: str = "elbow_standard"
    max_episode_steps: int = 100
    backend: BackendConfig = field(
        default_factory=lambda: BackendConfig(n_substeps=10, ctrl_dt=0.02, sim_dt=0.002)
    )
    obs: ObsSpec = field(
        default_factory=lambda: ObsSpec(keys=["joint_pos", "joint_vel", "muscle_act"])
    )
    goal: GoalSpec = field(
        default_factory=lambda: GoalSpec(
            target_type="joint_angles",
            randomize=False,
            range={"r_elbow_flex": (2.0, 2.0)},
        )
    )
    reward: RewardSpec = field(
        default_factory=lambda: RewardSpec(
            terms=["pose", "act_reg"],
            weights={"pose": 1.0, "act_reg": 1.0},
        )
    )
    actuators: list[ActuatorGroupSpec] = field(
        default_factory=lambda: [
            ActuatorGroupSpec(name="elbow_muscles", normalize_actions=True)
        ]
    )

    variants: ClassVar[list[VariantSpec]] = [
        VariantSpec(
            suffix="Sarc",
            config_delta={
                "actuators": [
                    ActuatorGroupSpec(name="elbow_muscles", condition="sarcopenia")
                ]
            },
        ),
        VariantSpec(
            suffix="Fati",
            config_delta={
                "actuators": [
                    ActuatorGroupSpec(name="elbow_muscles", condition="fatigue")
                ]
            },
        ),
    ]


@dataclass
class ElbowPoseRandomTask(TaskConfig):
    """Elbow pose task with random target sampled from [0.0, 2.27] rad.

    Uses the ``elbow_standard`` ModelBuilder recipe (fragment-based assembly).
    Registered as ``myoElbowPoseTaskRandom-v0`` on import.
    """

    model: str = "elbow_standard"
    max_episode_steps: int = 100
    backend: BackendConfig = field(
        default_factory=lambda: BackendConfig(n_substeps=10, ctrl_dt=0.02, sim_dt=0.002)
    )
    obs: ObsSpec = field(
        default_factory=lambda: ObsSpec(keys=["joint_pos", "joint_vel", "muscle_act"])
    )
    goal: GoalSpec = field(
        default_factory=lambda: GoalSpec(
            target_type="joint_angles",
            randomize=True,
            range={"r_elbow_flex": (0.0, 2.27)},
        )
    )
    reward: RewardSpec = field(
        default_factory=lambda: RewardSpec(
            terms=["pose", "act_reg"],
            weights={"pose": 1.0, "act_reg": 1.0},
        )
    )
    actuators: list[ActuatorGroupSpec] = field(
        default_factory=lambda: [
            ActuatorGroupSpec(name="elbow_muscles", normalize_actions=True)
        ]
    )

    variants: ClassVar[list[VariantSpec]] = [
        VariantSpec(
            suffix="Sarc",
            config_delta={
                "actuators": [
                    ActuatorGroupSpec(name="elbow_muscles", condition="sarcopenia")
                ]
            },
        ),
        VariantSpec(
            suffix="Fati",
            config_delta={
                "actuators": [
                    ActuatorGroupSpec(name="elbow_muscles", condition="fatigue")
                ]
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Registration — runs once on import
# ---------------------------------------------------------------------------
register_task(ElbowPoseFixedTask(), env_id="myoElbowPoseTaskFixed-v0")
register_task(ElbowPoseRandomTask(), env_id="myoElbowPoseTaskRandom-v0")
