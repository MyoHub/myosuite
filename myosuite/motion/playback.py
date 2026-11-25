# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Backend-agnostic playback contracts.

These types define a stable interface for task-specific playback adapters
that can be reused across gym, mjx, and mjlab integrations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class PlaybackRequest:
    """User-facing playback request.

    Args:
        checkpoint_ref: Checkpoint reference (``hf://...`` or local path).
        motion_ref: Motion reference path or cache key.
        n_steps: Maximum rollout steps.
        seed: Random seed for stochastic playback paths.
        backend: Backend tag (e.g., ``gym``, ``mjx``, ``mjlab``).
        render_mode: Rendering mode identifier.
        options: Extra backend/task-specific options.
    """

    checkpoint_ref: str
    motion_ref: str
    n_steps: int = 1000
    seed: int = 0
    backend: str = "gym"
    render_mode: str = "mujoco_viewer"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlaybackArtifacts:
    """Resolved artifacts required for a playback run."""

    checkpoint_path: Path
    motion_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlaybackResult:
    """Summary of a playback run."""

    frames_executed: int
    mean_tracking_error: float | None
    warnings: tuple[str, ...] = ()


class PlaybackRunner(Protocol):
    """Protocol for backend-specific playback runners."""

    def run(
        self,
        request: PlaybackRequest,
        artifacts: PlaybackArtifacts,
    ) -> PlaybackResult:
        """Execute playback and return result summary."""


__all__ = [
    "PlaybackArtifacts",
    "PlaybackRequest",
    "PlaybackResult",
    "PlaybackRunner",
]
