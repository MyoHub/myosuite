# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Declarative athletic TaskSpec / PlantSpec / AthleticBundle."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from myosuite.envs.myo.tasks.basic.full_body.athletic_contract import AthleticSpec

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
    """Logical athletic body names → plant-specific MuJoCo body names."""

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
        default_factory=lambda: ("joint_pos", "joint_vel", "muscle_act", "root_planar_vel")
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
    # "bolt_muscle" → BoltMuscleAction; "native_muscle" → MyoMuscleActivationActionCfg
    action_kind: Literal["bolt_muscle", "native_muscle"] = "bolt_muscle"
    # Optional callable returning MjSpec (mjlab). None → use factory defaults per key.
    load_spec: Callable[[], Any] | None = None
    # Golden bolt path: factory must call make_sprint_env_cfg for sprint+this plant.
    use_golden_sprint_cfg: bool = False
    capabilities: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {"lanes", "hop", "hurdles", "blocks", "curve", "vertical", "carioca", "locomotion", "imitate"}
        )
    )
    env_id_prefix: str = ""  # e.g. "" → myoSprint; "FullbodySimple" → myoFullbodySimpleSprint
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class AthleticBundle:
    task: TaskSpec
    plant: PlantSpec

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
        return replace(
            self.task.athletic,
            heading_dir=self.heading_dir,
            fall_height=self.plant.fall_height,
            facing_angle_deg=facing,
        )

    def env_id(self) -> str:
        prefix = self.plant.env_id_prefix
        task_title = "".join(p.capitalize() for p in self.task.key.split("_"))
        if prefix:
            return f"myo{prefix}{task_title}-v0"
        # Historical bolt sprinter sprint id.
        if self.task.key == "sprint" and self.plant.key == "myosprinter":
            return "myoSprint-v0"
        return f"myo{task_title}-v0"
