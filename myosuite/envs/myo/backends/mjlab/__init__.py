# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""mjlab backend task discovery module.

.. note::
    This is the **supported GPU training path**. MJX (JAX) is experimental.
    Task IDs share the CPU ``env_id`` (see :func:`list_mjlab_tasks`).

This module is the entry-point for mjlab's auto-discovery mechanism, registered
in ``pyproject.toml`` as::

    [project.entry-points."mjlab.tasks"]
    myosuite = "myosuite.envs.myo.backends.mjlab"

When mjlab is installed, importing this module registers all MyoSuite mjlab
tasks into ``mjlab.tasks.registry``. Any registration failure raises immediately.

Public API::

    list_mjlab_tasks  — sorted MyoSuite task IDs registered in mjlab.tasks.registry
    MjlabEnvAccessor  — EnvAccessor implementation returning torch.Tensor
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from typing import Any

from myosuite.envs.myo.backends.mjlab.configs.baoding_cfg import BaodingCfg
from myosuite.envs.myo.backends.mjlab.configs.elbow_pose_cfg import ElbowPoseCfg
from myosuite.envs.myo.backends.mjlab.configs.finger_pose_cfg import FingerPoseCfg
from myosuite.envs.myo.backends.mjlab.configs.finger_reach_cfg import FingerReachCfg
from myosuite.envs.myo.backends.mjlab.configs.hand_pose_cfg import HandPoseCfg
from myosuite.envs.myo.backends.mjlab.configs.hand_reach_cfg import HandReachCfg
from myosuite.envs.myo.backends.mjlab.configs.musclemimic_bimanual_cfg import (
    MuscleMimicBimanualCfg,
)
from myosuite.envs.myo.backends.mjlab.configs.musclemimic_fullbody_cfg import (
    MuscleMimicFullbodyCfg,
)
from myosuite.envs.myo.backends.mjlab.configs.table_tennis_cfg import TableTennisCfg
from myosuite.envs.myo.backends.mjlab.configs.walk_cfg import WalkCfg
from myosuite.envs.myo.backends.mjlab.mjlab_env_base import MjlabEnvAccessor


def _is_mjlab_installed() -> bool:
    """Return True when a real ``mjlab`` distribution (not a test stub) is installed."""
    existing = sys.modules.get("mjlab")
    if existing is not None:
        return getattr(existing, "__spec__", None) is not None
    return importlib.util.find_spec("mjlab") is not None


# mjlab attaches each entity spec to its scene: the entity's own <option> is dropped (the
# physics options come from MujocoCfg) and attach reports differing compiler/visual settings.
# Masses, inertias and timestep are unaffected; hide exactly these two messages.
for _message in (
    r"Entity '.*' has non-default <option> fields",
    r"Attach conflict when attaching",
):
    warnings.filterwarnings("ignore", message=_message, category=UserWarning)

# Register with mjlab.tasks.registry when mjlab is installed. Failures raise so broken
# registrations never drop tasks silently.
if _is_mjlab_installed():
    from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
        bootstrap_myosuite_mjlab_registry,
    )

    bootstrap_myosuite_mjlab_registry()


def list_mjlab_tasks() -> list[str]:
    """Return sorted MyoSuite task IDs registered in ``mjlab.tasks.registry``."""
    if not _is_mjlab_installed():
        return []
    from myosuite.envs.myo.backends.mjlab.tasks.registration import (
        list_mjlab_tasks as _list_mjlab_tasks,
    )

    return _list_mjlab_tasks()


def __getattr__(name: str) -> Any:
    if name == "REGISTERED_TASKS":
        warnings.warn(
            "myosuite.envs.myo.backends.mjlab.REGISTERED_TASKS is deprecated; "
            "use list_mjlab_tasks() or mjlab.tasks.registry.list_tasks().",
            DeprecationWarning,
            stacklevel=2,
        )
        if not _is_mjlab_installed():
            return {}
        from mjlab.tasks.registry import load_env_cfg

        return {task_id: type(load_env_cfg(task_id)) for task_id in list_mjlab_tasks()}
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "list_mjlab_tasks",
    "MjlabEnvAccessor",
    "ElbowPoseCfg",
    "FingerPoseCfg",
    "FingerReachCfg",
    "HandPoseCfg",
    "HandReachCfg",
    "WalkCfg",
    "BaodingCfg",
    "TableTennisCfg",
    "MuscleMimicBimanualCfg",
    "MuscleMimicFullbodyCfg",
]
