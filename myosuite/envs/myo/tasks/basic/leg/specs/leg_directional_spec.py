# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""TaskConfig-based specs for directional myoLeg locomotion.

A directional task commands the (single-legged-pair) myoLeg model to walk at
a fixed planar velocity ``target_speed * heading_dir``. It is intentionally
minimal — its only purpose is to pretrain a general locomotion policy whose
weights can be used to warm-start the two-agent ``myoChallengeChaseTagFBVs-v0``
task (see ``myosuite/envs/myo/tasks/challenge/chase_tag_vs/``).

Registered environments
------------------------
CPU (Gymnasium):
    ``myoLegDirectionalForward-v0``  — heading_dir=(0, 1), target_speed=1.2 m/s
    ``myoLegDirectionalBackward-v0`` — heading_dir=(0, -1), target_speed=1.0 m/s

These run via :class:`~myosuite.envs.modular_env.ModularTaskEnv` and are
therefore fully driven by the ``TaskConfig`` — no subclassing required.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from myosuite.core.config import (
    ActuatorGroupSpec,
    BackendConfig,
    GoalSpec,
    ObsSpec,
    RewardSpec,
    TaskConfig,
)
from myosuite.core.registry import register_task
from myosuite.envs.myo.assets._resolve import resolve_leg_xml


def _leg_model_path() -> str:
    """Resolve the bundled rigid-torso myoLeg host XML used by walk/directional tasks."""
    return str(resolve_leg_xml("myolegs_with_torso.xml"))


@dataclass
class LegDirectionalForwardTask(TaskConfig):
    """Directional myoLeg task: walk forward at a fixed commanded velocity.

    Uses the bundled ``myolegs_with_torso.xml`` host model (rigid torso,
    free-floating pelvis root joint). Registered as
    ``myoLegDirectionalForward-v0`` on import.
    """

    model: str = field(default_factory=_leg_model_path)
    max_episode_steps: int = 500
    backend: BackendConfig = field(
        default_factory=lambda: BackendConfig(n_substeps=5, ctrl_dt=0.01, sim_dt=0.002)
    )
    obs: ObsSpec = field(
        default_factory=lambda: ObsSpec(
            keys=[
                "joint_pos",
                "joint_vel",
                "muscle_act",
                "root_planar_vel",
                "heading_cmd",
            ],
            extra={"heading_dir": (0.0, 1.0)},
        )
    )
    goal: GoalSpec = field(default_factory=lambda: GoalSpec(target_type="joint_angles"))
    reward: RewardSpec = field(
        default_factory=lambda: RewardSpec(
            terms=["heading", "act_reg"],
            weights={"heading": 1.0, "act_reg": 0.1},
            extra={"heading_dir": (0.0, 1.0), "target_speed": 1.2},
        )
    )
    actuators: list[ActuatorGroupSpec] = field(
        default_factory=lambda: [
            ActuatorGroupSpec(name="leg_muscles", normalize_actions=True)
        ]
    )


@dataclass
class LegDirectionalBackwardTask(LegDirectionalForwardTask):
    """Directional myoLeg task: walk backward at a fixed commanded velocity.

    Registered as ``myoLegDirectionalBackward-v0`` on import.
    """

    obs: ObsSpec = field(
        default_factory=lambda: ObsSpec(
            keys=[
                "joint_pos",
                "joint_vel",
                "muscle_act",
                "root_planar_vel",
                "heading_cmd",
            ],
            extra={"heading_dir": (0.0, -1.0)},
        )
    )
    reward: RewardSpec = field(
        default_factory=lambda: RewardSpec(
            terms=["heading", "act_reg"],
            weights={"heading": 1.0, "act_reg": 0.1},
            extra={"heading_dir": (0.0, -1.0), "target_speed": 1.0},
        )
    )


@dataclass
class LegDirectionalRandomTask(LegDirectionalForwardTask):
    """Directional myoLeg task with the heading command randomized per episode.

    Each ``reset`` samples a fresh commanded direction from the full unit circle
    (``randomize_heading=True``), so the ``heading_cmd`` observation varies across
    episodes and the policy must learn the command->direction mapping. This is the
    fix for the forward-only checkpoint that ignored the command (see
    ``tasks/directional_policy_diagnosis.md``); a policy trained here is the
    intended warm-start for ``myoChallengeChaseTagFBVs-v0`` steering.

    Registered as ``myoLegDirectionalRandom-v0`` on import.
    """

    reward: RewardSpec = field(
        default_factory=lambda: RewardSpec(
            terms=["heading", "act_reg"],
            weights={"heading": 1.0, "act_reg": 0.1},
            extra={
                "heading_dir": (0.0, 1.0),  # fallback; overridden per episode
                "target_speed": 1.2,
                "randomize_heading": True,
            },
        )
    )


# ---------------------------------------------------------------------------
# Registration — runs once on import
# ---------------------------------------------------------------------------
register_task(LegDirectionalForwardTask(), env_id="myoLegDirectionalForward-v0")
register_task(LegDirectionalBackwardTask(), env_id="myoLegDirectionalBackward-v0")
register_task(LegDirectionalRandomTask(), env_id="myoLegDirectionalRandom-v0")
