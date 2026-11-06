# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Declarative athletic TaskSpec / PlantSpec / AthleticBundle.

The public construction arguments are the task, the skeleton, and the series
element. The skeleton defaults to ``myofullbody`` and the series element
defaults to rigid tendons. Elastic tendons are a registered env-id variant
(``…Elastic-v0``), not a silent constructor flag on the rigid id.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from myosuite.envs.myo.tasks.basic.full_body.athletic_contract import (
    ELASTIC_N_SUBSTEPS,
    ELASTIC_SIM_DT,
    AthleticSpec,
)

# Skeletons in the athletic catalog. ``myofullbody`` is the default.
ModelName = Literal["myofullbody", "myofullbody_simple", "myosprinter"]
# Series element. ``rigid`` is the default. ``elastic`` selects the plugin.
TendonKind = Literal["rigid", "elastic"]
# ``native_muscle`` is MuJoCo ``mjMUSCLE``. ``bolt_muscle`` is the plugin.
ActionKind = Literal["bolt_muscle", "native_muscle"]

DEFAULT_MODEL: ModelName = "myofullbody"
DEFAULT_TENDONS: TendonKind = "rigid"

TaskKind = Literal[
    "sprint",
    "backpedal",
    "side_shuffle",
    "hop",
    "hurdles",
    "sprint_blocks",
    "carioca",
    "curve",
    "vertical",
    "locomotion",
    "imitate",
    "uphill_sprint",
]


@dataclass(frozen=True)
class BodyMap:
    """Logical athletic names → plant-specific MuJoCo body / joint names.

    Terms must resolve through this map (or ``mj_name2id``) so qpos indices
    stay valid when myo_sim renames right-arm joints or inserts neck dofs.
    """

    root: str = "pelvis"
    toes_l: str = "toes_l"
    toes_r: str = "toes_r"
    hand_r: str = "hand_r"
    knee_l: str = "knee_angle_l"
    knee_r: str = "knee_angle_r"

    def resolve(self, logical: str) -> str:
        return getattr(self, logical)


@dataclass(frozen=True)
class TermRef:
    """Named MDP term with kwargs (shared by CPU ModularTaskEnv and mjlab)."""

    name: str
    weight: float = 1.0
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskSpec:
    """Athletic MDP declaration (msk_envs DerivedEnv analogue)."""

    key: str
    kind: TaskKind
    athletic: AthleticSpec = field(default_factory=AthleticSpec)
    # Relative heading in plant forward frame: +1 forward, -1 back, lateral via plant.side.
    heading_mode: Literal["forward", "backward", "lateral"] = "forward"
    facing_angle_deg: float | None = None
    reward_terms: Sequence[TermRef] = field(default_factory=tuple)
    terminations: Sequence[str] = field(
        default_factory=lambda: ("time_out", "fallen", "not_facing", "toes_out")
    )
    obs_keys: Sequence[str] = field(
        default_factory=lambda: (
            "joint_pos",
            "joint_vel",
            "muscle_act",
            "root_planar_vel",
        )
    )
    scene: str | None = None  # lanes | hurdles | blocks | curve | vertical | None
    pose_key: str = "starting_pose_run"
    episode_length_s: float = 10.0
    extra: Mapping[str, Any] = field(default_factory=dict)
    # Capability requirements evaluated against PlantSpec.capabilities.
    requires: frozenset[str] = field(default_factory=frozenset)

    def with_heading(self, heading_dir: tuple[float, float]) -> TaskSpec:
        return replace(
            self,
            athletic=replace(self.athletic, heading_dir=heading_dir),
        )


@dataclass(frozen=True)
class PlantSpec:
    """MSK model / actuation plant for the athletic matrix."""

    key: str
    entity_name: str
    model_recipe: str  # CPU ModularTaskEnv model key
    heading_dir: tuple[float, float]
    up_axis: Literal["y", "z"] = "z"
    fall_height: float = 0.6
    body_map: BodyMap = field(default_factory=BodyMap)
    # Actuator used when tendons="rigid". Elastic tendons use the plugin
    # (see AthleticBundle.action_kind), not this field.
    action_kind: ActionKind = "bolt_muscle"
    # Optional callable returning MjSpec (mjlab). None → use factory defaults per key.
    load_spec: Callable[[], Any] | None = None
    # Golden bolt path: factory must call make_sprint_env_cfg for sprint+this plant.
    use_golden_sprint_cfg: bool = False
    capabilities: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "lanes",
                "hop",
                "hurdles",
                "blocks",
                "curve",
                "vertical",
                "carioca",
                "locomotion",
                "imitate",
            }
        )
    )
    env_id_prefix: str = (
        ""  # e.g. "" → myoSprint; "FullbodySimple" → myoFullbodySimpleSprint
    )
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class AthleticBundle:
    task: TaskSpec
    plant: PlantSpec
    tendons: TendonKind = DEFAULT_TENDONS

    @property
    def action_kind(self) -> ActionKind:
        """Actuator for this skeleton and series element.

        Rigid tendons use :attr:`PlantSpec.action_kind`. On ``myofullbody``
        that is native ``mjMUSCLE``. Elastic tendons use the series-elastic
        plugin on whichever skeleton was requested.
        """
        if self.tendons == "elastic":
            return "bolt_muscle"
        return self.plant.action_kind

    @property
    def heading_dir(self) -> tuple[float, float]:
        hx, hy = self.plant.heading_dir
        mode = self.task.heading_mode
        if mode == "forward":
            return (hx, hy)
        if mode == "backward":
            return (-hx, -hy)
        # lateral: rotate plant forward 90° in planar frame (x,y) → (-y, x)
        return (-hy, hx)

    @property
    def athletic(self) -> AthleticSpec:
        facing = self.task.facing_angle_deg
        if facing is None:
            facing = self.task.athletic.facing_angle_deg
        spec = replace(
            self.task.athletic,
            heading_dir=self.heading_dir,
            fall_height=self.plant.fall_height,
            facing_angle_deg=facing,
        )
        if self.tendons == "elastic":
            # Own timestep so a checkpoint records which physics it trained on.
            return replace(
                spec,
                sim_dt=ELASTIC_SIM_DT,
                n_substeps=ELASTIC_N_SUBSTEPS,
            )
        return spec

    def env_id(self) -> str:
        """Gymnasium id for this task, skeleton, and tendon variant.

        Rigid ``myofullbody`` is ``myoFullbodySprint-v0``. Elastic is
        ``myoFullbodySprintElastic-v0`` so the checkpoint contract hash
        cannot load a rigid policy into an elastic env (or the reverse).
        ``myoSprint-v0`` stays the rigid sprinter.
        """
        prefix = self.plant.env_id_prefix
        task_title = "".join(p.capitalize() for p in self.task.key.split("_"))
        tendon_suffix = "Elastic" if self.tendons == "elastic" else ""
        if prefix:
            return f"myo{prefix}{task_title}{tendon_suffix}-v0"
        # Historical bolt sprinter sprint id (rigid only).
        if (
            self.task.key == "sprint"
            and self.plant.key == "myosprinter"
            and self.tendons == "rigid"
        ):
            return "myoSprint-v0"
        if self.plant.key == "myosprinter" and self.tendons == "elastic":
            return f"myoSprint{tendon_suffix}-v0"
        return f"myo{task_title}{tendon_suffix}-v0"


def ensure_tendons_applied(bundle: AthleticBundle) -> None:
    """Reject a series element the actuators do not apply yet.

    ``tendons="elastic"`` selects the plugin via :attr:`AthleticBundle.action_kind`
    and its own env id. Building that env before the series element is applied
    would silently run rigid tendons.
    """
    if bundle.tendons == "rigid":
        return
    raise NotImplementedError(
        "tendons='elastic' selects the series-elastic plugin "
        f"(model={bundle.plant.key!r}, env_id={bundle.env_id()!r}, "
        f"action={bundle.action_kind}) but that element is not applied yet. "
        "Use tendons='rigid'."
    )
