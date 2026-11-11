# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Load the optional ONNX inference runtime when a feature needs it."""

from __future__ import annotations

import importlib
from types import ModuleType


def require_onnxruntime() -> ModuleType:
    """Return ONNX Runtime or explain how to install the optional dependency.

    Raises:
        ImportError: If ONNX Runtime is not installed.
    """
    try:
        return importlib.import_module("onnxruntime")
    except ModuleNotFoundError as exc:
        if exc.name != "onnxruntime":
            raise
        raise ImportError(
            "ONNX inference requires onnxruntime. Install it with "
            "pip install 'myosuite[onnx]' (CPU), or install onnxruntime-gpu. "
            "If no wheel is available for your Python version, use an "
            "interpreter supported by ONNX Runtime or build it from source. "
            "Native MyoSuite CPU environments do not require ONNX Runtime."
        ) from exc
