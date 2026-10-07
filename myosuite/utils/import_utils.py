# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Optional-dependency guards for encoder and simulation packages."""

from __future__ import annotations

import importlib.util


def mujoco_isavailable() -> None:
    """Raise :exc:`ModuleNotFoundError` if ``mujoco`` is not installed."""
    if importlib.util.find_spec("mujoco") is None:
        raise ModuleNotFoundError(
            "mujoco is required. Install it with: pip install mujoco"
        )


def torch_isavailable() -> None:
    """Raise :exc:`ModuleNotFoundError` if ``torch`` is not installed."""
    if importlib.util.find_spec("torch") is None:
        raise ModuleNotFoundError(
            "torch is required for visual keys. Install it with: pip install torch"
        )


def torchvision_isavailable() -> None:
    """Raise :exc:`ModuleNotFoundError` if ``torchvision`` is not installed."""
    if importlib.util.find_spec("torchvision") is None:
        raise ModuleNotFoundError(
            "torchvision is required for visual keys. Install it with: pip install torchvision"
        )


def r3m_isavailable() -> None:
    """Raise :exc:`ModuleNotFoundError` if the R3M encoder is not installed."""
    if importlib.util.find_spec("r3m") is None:
        raise ModuleNotFoundError(
            "r3m is required for R3M visual keys. "
            "Install it with: pip install 'r3m@git+https://github.com/facebookresearch/r3m.git'"
        )


def vc_isavailable() -> None:
    """Raise :exc:`ModuleNotFoundError` if the VC1 encoder is not installed."""
    if importlib.util.find_spec("vc_models") is None:
        raise ModuleNotFoundError(
            "vc_models is required for VC1 visual keys. "
            "See https://eai-vc.github.io/ for installation instructions."
        )
