# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Optional-dependency guards for optional extras, encoders, and simulation packages."""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any


def require_extra(
    module_name: str,
    extra: str,
    feature: str | None = None,
    *,
    hint: str = "",
) -> Any:
    """Import ``module_name``, raising an :exc:`ImportError` naming ``myosuite[<extra>]`` when absent.

    Re-raises internal :exc:`ModuleNotFoundError` exceptions originating from a
    different top-level package and non-missing :exc:`ImportError` exceptions
    (such as broken shared-library loads) untouched so broken installations are
    not misreported as a missing MyoSuite extra.

    Args:
        module_name: Dotted module name to import (e.g. ``"onnxruntime"`` or ``"imageio.v3"``).
        extra: Optional-dependency group in ``pyproject.toml`` (e.g. ``"viz"``, ``"biomechanics"``).
        feature: Optional human-readable feature name prefixed to the error message.
        hint: Optional trailing guidance appended after the ``pip install 'myosuite[<extra>]'`` hint.

    Returns:
        The imported module object returned by :func:`importlib.import_module`.

    Raises:
        ModuleNotFoundError: Re-raised unchanged if an internal submodule or transitive dependency
            from a different top-level package is missing inside an installed ``module_name``.
        ImportError: Raised with a ``pip install 'myosuite[<extra>]'`` message when ``module_name``
            itself (or its top-level package) is not installed, or re-raised unchanged on broken
            shared-library / symbol-load failures.
    """
    top_pkg = module_name.split(".", 1)[0]
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = exc.name
        if (
            missing
            and missing != module_name
            and not module_name.startswith(missing + ".")
        ):
            raise
        target = f"{feature} requires" if feature else "Missing optional dependency"
        suffix = f" {hint}" if hint else ""
        raise ImportError(
            f"{target} `{top_pkg}`. Install with: pip install 'myosuite[{extra}]'{suffix}"
        ) from exc


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
