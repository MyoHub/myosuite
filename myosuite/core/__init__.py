# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Core backend-agnostic abstractions for MyoSuite."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from myosuite.motion.hf_cache import (
        download_hf_dataset_file,
        setup_named_demo_cache,
    )
    from myosuite.motion.io import (
        MotionClip,
        expand_motion_clip_to_model,
        load_motion_clip,
        resolve_motion_path,
    )
    from myosuite.motion.playback import (
        PlaybackArtifacts,
        PlaybackRequest,
        PlaybackResult,
        PlaybackRunner,
    )

import myosuite.core.model_recipes  # noqa: F401

from myosuite.core.config import (
    ActuatorGroupSpec,
    BackendConfig,
    EnvConfig,
    GoalSpec,
    ObsSpec,
    RewardSpec,
    TaskConfig,
    VariantSpec,
)
from myosuite.core.hf_io import (
    HfRef,
    default_musclemimic_cache_root,
    parse_hf_ref,
    resolve_hf_snapshot,
)
from myosuite.core.model_builder import (
    ModelBuilder,
    build_from_recipe,
    list_recipes,
    model_recipe,
)
from myosuite.core.mujoco_playback import (
    MetricCallback,
    StepCallback,
    run_passive_viewer_loop,
)
from myosuite.core.protocols import EnvAccessor, PhysicsPath
from myosuite.core.registry import make_env, register_env, register_task
from myosuite.core.subprocess_orchestration import (
    build_python_command,
    prepend_env_path_var,
    run_command,
)

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "PlaybackArtifacts": ("myosuite.motion.playback", "PlaybackArtifacts"),
    "PlaybackRequest": ("myosuite.motion.playback", "PlaybackRequest"),
    "PlaybackResult": ("myosuite.motion.playback", "PlaybackResult"),
    "PlaybackRunner": ("myosuite.motion.playback", "PlaybackRunner"),
    "download_hf_dataset_file": (
        "myosuite.motion.hf_cache",
        "download_hf_dataset_file",
    ),
    "setup_named_demo_cache": (
        "myosuite.motion.hf_cache",
        "setup_named_demo_cache",
    ),
    "MotionClip": ("myosuite.motion.io", "MotionClip"),
    "expand_motion_clip_to_model": (
        "myosuite.motion.io",
        "expand_motion_clip_to_model",
    ),
    "load_motion_clip": ("myosuite.motion.io", "load_motion_clip"),
    "resolve_motion_path": ("myosuite.motion.io", "resolve_motion_path"),
}


_COMPAT_SUBMODULES: dict[str, str] = {
    "hf_dataset_cache": "myosuite.motion.hf_cache",
    "playback_contract": "myosuite.motion.playback",
    "retargeting": "myosuite.integrations.marker_retarget",
    "trajectory_io": "myosuite.motion.io",
    "trajectory_yaw": "myosuite.motion.transforms",
}


def __getattr__(name: str) -> Any:
    compat_target = _COMPAT_SUBMODULES.get(name)
    if compat_target is not None:
        mod = importlib.import_module(compat_target)
        globals()[name] = mod
        return mod
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr_name = target
    val = getattr(importlib.import_module(mod_name), attr_name)
    globals()[name] = val
    return val


__all__ = [
    "EnvAccessor",
    "PhysicsPath",
    "EnvConfig",
    "BackendConfig",
    "PlaybackRequest",
    "PlaybackArtifacts",
    "PlaybackResult",
    "PlaybackRunner",
    "HfRef",
    "parse_hf_ref",
    "resolve_hf_snapshot",
    "default_musclemimic_cache_root",
    "download_hf_dataset_file",
    "setup_named_demo_cache",
    "StepCallback",
    "MetricCallback",
    "run_passive_viewer_loop",
    "MotionClip",
    "expand_motion_clip_to_model",
    "resolve_motion_path",
    "load_motion_clip",
    "build_python_command",
    "prepend_env_path_var",
    "run_command",
    "ModelBuilder",
    "model_recipe",
    "build_from_recipe",
    "list_recipes",
    "TaskConfig",
    "ObsSpec",
    "GoalSpec",
    "RewardSpec",
    "ActuatorGroupSpec",
    "VariantSpec",
    "register_env",
    "register_task",
    "make_env",
]
