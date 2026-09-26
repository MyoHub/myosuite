# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Task config for MyoChallenge TableTennis on the mjlab / MuJoCo Warp backend."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any

_ASSETS = pathlib.Path(__file__).parents[3] / "assets"


@dataclass
class TableTennisCfg:
    """Configuration for TableTennis mjlab tasks.

    Phase-specific Gymnasium kwargs (``myoChallengeTableTennisP{0,1,2}-v0``) are
    mirrored here so ``REGISTERED_TASKS`` and ``register_mjlab_task`` share one
    typed surface.

    Args:
        model_path: MJCF path (default: packaged ``myoarm_tabletennis.xml``).
        sim_dt: MuJoCo integration timestep (s).
        ctrl_dt: Control timestep; must be a multiple of ``sim_dt``.
        max_episode_steps: Truncation horizon in control steps.
        num_envs: Parallel environments on the device.
        phase: ``"p0"``, ``"p1"``, or ``"p2"`` — selects default randomisation.
        ball_xyz_range: Optional ``{"low": [...], "high": [...]}`` ball position box.
        ball_qvel: When True, sample initial ball linear velocity (P2-style).
        qpos_noise_range: Optional dict passed to ``uniform`` for joint noise.
        paddle_mass_range: Optional ``(low, high)`` paddle body mass (kg).
        ball_friction_range: Optional ``{"low": [...], "high": [...]}`` geom friction.
        rally_count: Successful rallies before episode can end on ``solved``.
    """

    model_path: pathlib.Path = field(
        default_factory=lambda: _ASSETS / "arm" / "myoarm_tabletennis.xml"
    )
    sim_dt: float = 0.002
    ctrl_dt: float = 0.01
    max_episode_steps: int = 300
    num_envs: int = 1
    phase: str = "p0"
    ball_xyz_range: dict[str, list[float]] | None = None
    ball_qvel: bool = False
    qpos_noise_range: dict[str, Any] | None = None
    paddle_mass_range: tuple[float, float] | None = None
    ball_friction_range: dict[str, list[float]] | None = None
    rally_count: int = 1

    @classmethod
    def p0(cls) -> TableTennisCfg:
        """Defaults matching ``myoChallengeTableTennisP0-v0``."""
        return cls(phase="p0")

    @classmethod
    def p1(cls) -> TableTennisCfg:
        """Defaults matching ``myoChallengeTableTennisP1-v0``."""
        return cls(
            phase="p1",
            ball_xyz_range={
                "high": [-1.20, -0.45, 1.5],
                "low": [-1.25, -0.5, 1.4],
            },
        )

    @classmethod
    def p2(cls) -> TableTennisCfg:
        """Defaults matching ``myoChallengeTableTennisP2-v0``."""
        return cls(
            phase="p2",
            ball_qvel=True,
            paddle_mass_range=(0.10, 0.15),
            qpos_noise_range=None,
            ball_xyz_range={
                "high": [-0.8, 0.5, 1.5],
                "low": [-1.25, -0.5, 1.4],
            },
            ball_friction_range={
                "high": [1.1, 0.006, 0.00003],
                "low": [0.9, 0.004, 0.00001],
            },
        )
