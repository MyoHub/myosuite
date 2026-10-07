# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Task/Env specification dataclasses for declarative registration."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable

from myosuite.core.config import TaskConfig


@dataclass(frozen=True)
class TaskSpec:
    """Declarative specification for a task registration target."""

    task_config_factory: Callable[[], TaskConfig]
    backends: set[str] = field(default_factory=lambda: {"cpu"})

    def build_task_config(self) -> TaskConfig:
        """Create a concrete TaskConfig instance for registration."""
        return self.task_config_factory()


@dataclass(frozen=True)
class EnvSpec:
    """Environment spec binding an env ID to a TaskSpec."""

    env_id: str
    task_spec: TaskSpec
