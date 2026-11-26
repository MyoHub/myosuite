# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Reusable runtime policy loading, ONNX export, action maps, and manifests."""

from __future__ import annotations

import importlib
from typing import Any

_SUBMODULES = frozenset(
    {
        "action_map",
        "actor_tree",
        "checkpoint_manifest",
        "checkpoint_ref",
        "checkpoint_utils",
        "deployment_manifest",
        "export_onnx",
        "onnx_checkpoint",
        "onnxruntime_utils",
        "orbax_checkpoint",
        "rslrl_policy",
    }
)

_EXPORTS: dict[str, tuple[str, str]] = {
    "ActionMap": ("myosuite.policies.action_map", "ActionMap"),
    "ActionMapWrapper": ("myosuite.policies.action_map", "ActionMapWrapper"),
    "OnnxActorSession": ("myosuite.policies.onnxruntime_utils", "OnnxActorSession"),
    "find_checkpoint": ("myosuite.policies.checkpoint_utils", "find_checkpoint"),
    "load_onnx_session": ("myosuite.policies.onnxruntime_utils", "load_onnx_session"),
    "load_policy": ("myosuite.policies.checkpoint_utils", "load_policy"),
    "load_rslrl_policy": ("myosuite.policies.rslrl_policy", "load_rslrl_policy"),
    "require_onnxruntime": (
        "myosuite.policies.onnxruntime_utils",
        "require_onnxruntime",
    ),
}


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        mod = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr_name = target
    val = getattr(importlib.import_module(mod_name), attr_name)
    globals()[name] = val
    return val


__all__ = sorted(_EXPORTS)
