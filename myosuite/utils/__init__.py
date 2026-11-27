# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Backward-compatible facade for :mod:`myosuite.utils`."""

from __future__ import annotations

import importlib
from typing import Any

import gymnasium as gym

from myosuite.core.seeding import seed_envs

__all__ = ["gym", "seed_envs"]

_COMPAT_SUBMODULES = frozenset(
    {
        "action_map",
        "asset_path_resolver",
        "checkpoint_manifest",
        "checkpoint_utils",
        "colored_noise",
        "deployment_manifest",
        "dict_utils",
        "env_check",
        "examine_env",
        "examine_sim",
        "export_onnx",
        "feature_cli",
        "html",
        "import_utils",
        "mujoco_geom_utils",
        "mujoco_joint_utils",
        "onnx_checkpoint",
        "onnxruntime_utils",
        "paired_replay",
        "path_plotting",
        "path_utils",
        "performance_card",
        "policy_utils",
        "prompt_utils",
        "push_metrics",
        "push_protocol",
        "push_recovery",
        "push_results",
        "push_scheduling",
        "reach_workspace",
        "rslrl_policy",
        "sb3_callbacks",
        "sb3_sweep",
        "spec_processing",
        "step_timing",
        "target_ranges",
        "tensor_utils",
        "video_io",
    }
)


def __getattr__(name: str) -> Any:
    if name in _COMPAT_SUBMODULES:
        mod = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
