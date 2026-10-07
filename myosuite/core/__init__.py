# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Core backend-agnostic abstractions for MyoSuite."""

import myosuite.core.model_recipes  # noqa: F401

from myosuite.core.protocols import EnvAccessor, PhysicsPath
from myosuite.core.playback_contract import (
    PlaybackArtifacts,
    PlaybackRequest,
    PlaybackResult,
    PlaybackRunner,
)
from myosuite.core.config import (
    EnvConfig,
    BackendConfig,
    TaskConfig,
    ObsSpec,
    GoalSpec,
    RewardSpec,
    ActuatorGroupSpec,
    VariantSpec,
)
from myosuite.core.model_builder import (
    ModelBuilder,
    build_from_recipe,
    list_recipes,
    model_recipe,
)
from myosuite.core.registry import register_env, register_task, make_env
from myosuite.core.hf_io import (
    HfRef,
    default_musclemimic_cache_root,
    parse_hf_ref,
    resolve_hf_snapshot,
)
from myosuite.core.hf_dataset_cache import (
    download_hf_dataset_file,
    setup_named_demo_cache,
)
from myosuite.core.mujoco_playback import (
    MetricCallback,
    StepCallback,
    run_passive_viewer_loop,
)
from myosuite.core.trajectory_io import (
    MotionClip,
    expand_motion_clip_to_model,
    load_motion_clip,
    resolve_motion_path,
)
from myosuite.core.subprocess_orchestration import (
    build_python_command,
    prepend_env_path_var,
    run_command,
)

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
