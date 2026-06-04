#!/usr/bin/env python3
# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Backward-compatible shim for ONNX export utilities.

Canonical location:
    myosuite.utils.export_onnx
"""

from __future__ import annotations

import logging

from myosuite.utils.export_onnx import (
    compare_onnx_across_backends,
    export_jax_to_onnx,
    export_orbax_to_onnx,
    export_rslrl_to_onnx,
    export_sb3_to_onnx,
    main,
    verify_onnx_on_cpu,
)

logging.getLogger(__name__).warning(
    "benchmarks/sar_backends/export_onnx.py is deprecated; "
    "use myosuite.utils.export_onnx or `myosuite-export-onnx`."
)

__all__ = [
    "compare_onnx_across_backends",
    "export_jax_to_onnx",
    "export_orbax_to_onnx",
    "export_rslrl_to_onnx",
    "export_sb3_to_onnx",
    "main",
    "verify_onnx_on_cpu",
]


if __name__ == "__main__":
    main()
