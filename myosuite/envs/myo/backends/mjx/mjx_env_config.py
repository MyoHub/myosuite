# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Typed dataclass configs for MyoSuite MJX environments.

These are the source-of-truth configs. ``make()`` in ``mjx/__init__.py``
converts them to ``ml_collections.ConfigDict`` at the boundary where
``mujoco_playground.MjxEnv`` requires it.  All user-facing code creates
and manipulates these dataclasses; never raw dicts or ConfigDict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MjxPoseRewardConfig:
    """Reward weights for :class:`MjxPoseEnv`."""

    angle_reward_weight: float = 1.0
    ctrl_cost_weight: float = 1.0
    pose_thd: float = 0.35
    far_th: float = 4 * math.pi / 2
    bonus_weight: float = 4.0


@dataclass
class MjxPoseConfig:
    """Full configuration for :class:`MjxPoseEnv`.

    Args:
        model_path: Path to the MJCF file.
        ctrl_dt: Control timestep in seconds.
        sim_dt: MuJoCo simulation timestep in seconds.
        num_envs: Number of parallel environments.
        mjx_impl: ``None`` for JAX/XLA backend; ``"warp"`` for GPU Warp.
        norm_actions: Whether to apply sigmoid action normalisation.
        max_episode_steps: Episode truncation length.
        target_jnt_range: Mapping of joint name → ``(lo, hi)`` JAX arrays.
        reward_config: Reward component weights.
    """

    model_path: Path
    ctrl_dt: float = 0.02
    sim_dt: float = 0.002
    num_envs: int = 4096
    mjx_impl: str | None = None
    norm_actions: bool = True
    max_episode_steps: int = 100
    target_jnt_range: dict[str, Any] = field(default_factory=dict)
    reward_config: MjxPoseRewardConfig = field(default_factory=MjxPoseRewardConfig)


@dataclass
class MjxReachRewardWeights:
    """Reward weights for :class:`MjxReachEnv`."""

    reach: float = 1.0
    bonus: float = 4.0
    penalty: float = 50.0


@dataclass
class MjxReachConfig:
    """Full configuration for :class:`MjxReachEnv`.

    Args:
        model_path: Path to the MJCF file.
        ctrl_dt: Control timestep in seconds.
        sim_dt: MuJoCo simulation timestep in seconds.
        num_envs: Number of parallel environments.
        mjx_impl: ``None`` for JAX/XLA backend; ``"warp"`` for GPU Warp.
        norm_actions: Whether to apply sigmoid action normalisation.
        max_episode_steps: Episode truncation length.
        far_th: Distance threshold (metres) triggering episode termination.
        target_reach_range: Mapping of site name → ``(lo, hi)`` JAX arrays.
        reward_weights: Reward component weights.
    """

    model_path: Path
    ctrl_dt: float = 0.02
    sim_dt: float = 0.002
    num_envs: int = 4096
    mjx_impl: str | None = None
    norm_actions: bool = True
    max_episode_steps: int = 100
    far_th: float = 0.35
    target_reach_range: dict[str, Any] = field(default_factory=dict)
    reward_weights: MjxReachRewardWeights = field(default_factory=MjxReachRewardWeights)


@dataclass
class MjxWalkConfig:
    """Full configuration for :class:`MjxWalkEnv`.

    Args:
        model_path: Path to the myolegs MJCF file.
        ctrl_dt: Control timestep in seconds.
        sim_dt: MuJoCo simulation timestep in seconds.
        num_envs: Number of parallel environments.
        mjx_impl: ``None`` for JAX/XLA backend; ``"warp"`` for GPU Warp.
        norm_actions: Whether to apply sigmoid action normalisation.
        min_height: Minimum pelvis height before termination (metres).
        max_rot: Maximum torso rotation magnitude before termination.
        hip_period: Desired cyclic hip period in steps.
        target_x_vel: Target x-axis COM velocity in m/s.
        target_y_vel: Target y-axis COM velocity in m/s.
        max_episode_steps: Episode truncation length.
    """

    model_path: Path
    ctrl_dt: float = 0.02
    sim_dt: float = 0.002
    num_envs: int = 1
    mjx_impl: str | None = None
    norm_actions: bool = True
    min_height: float = 0.8
    max_rot: float = 0.8
    hip_period: int = 100
    target_x_vel: float = 0.0
    target_y_vel: float = 1.2
    max_episode_steps: int = 1000
