# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Shared metadata contract for multi-agent combat tasks.

This module defines a minimal, backend-agnostic cache shape for model indices
used by observation and reward terms. Concrete tasks can extend the base
dataclass with task-specific fields (for example pelvis/body ids).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol
from collections.abc import Mapping

import numpy as np

AgentId = str
Index = int


class CombatModelMeta(Protocol):
    """Minimal shared contract for combat task metadata caches."""

    n_act: int
    act_indices: Mapping[AgentId, list[Index]]
    jnt_ids: Mapping[AgentId, list[Index]]
    arm_jnt_ids: Mapping[AgentId, list[Index]]
    site_ids: Mapping[AgentId, Mapping[str, Index]]
    sensor_adr: Mapping[AgentId, Mapping[str, Index]]


@dataclass
class CombatModelMetaBase:
    """Default shared metadata container for combat tasks."""

    n_act: int
    act_indices: dict[AgentId, list[Index]] = field(default_factory=dict)
    jnt_ids: dict[AgentId, list[Index]] = field(default_factory=dict)
    arm_jnt_ids: dict[AgentId, list[Index]] = field(default_factory=dict)
    site_ids: dict[AgentId, dict[str, Index]] = field(default_factory=dict)
    sensor_adr: dict[AgentId, dict[str, Index]] = field(default_factory=dict)
    standing_qpos: np.ndarray | None = None
    """Full-length combined-model standing ``qpos``, reconstructed from each
    agent's source-model keyframe (see
    ``myosuite.integrations.musclemimic.two_agent_scene.two_agent_standing_qpos``).
    ``None`` if reconstruction wasn't possible (e.g. a source model shipped
    no keyframe); reset then falls back to ``mj_resetData``'s all-zero pose."""
