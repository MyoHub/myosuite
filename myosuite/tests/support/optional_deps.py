# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Pytest skip helpers for optional MyoSuite dependencies."""

from __future__ import annotations

import pytest


def require_musclemimic_models() -> None:
    """Skip when ``musclemimic_models`` is missing or only partially installed.

    A broken install may expose an empty namespace package (``model/`` only,
    no ``get_xml_path``). Full install: ``uv sync --extra musclemimic``.
    """
    pytest.importorskip("musclemimic_models")
    try:
        from musclemimic_models import get_xml_path  # noqa: F401
    except ImportError:
        pytest.skip(
            "musclemimic_models is incomplete (missing get_xml_path). "
            "Install with: uv sync --extra musclemimic  (or myosuite[dev])"
        )


def require_stable_baselines3() -> None:
    """Skip when Stable-Baselines3 is not installed (``myosuite[rl]`` extra)."""
    pytest.importorskip("stable_baselines3")


def require_mjlab() -> None:
    """Skip when mjlab is not installed (``myosuite[mjlab]`` extra)."""
    pytest.importorskip("mjlab")


def require_mujoco_warp(*, min_version: str = "3.7.0") -> None:
    """Skip when ``mujoco-warp`` is missing, too old, or fails to import."""
    from importlib.metadata import PackageNotFoundError, version

    from packaging.version import Version

    try:
        warp_ver = Version(version("mujoco-warp"))
    except PackageNotFoundError:
        pytest.skip("mujoco-warp not installed. Install with: uv sync --extra mjlab")
    if warp_ver < Version(min_version):
        pytest.skip(
            f"mujoco-warp {warp_ver} is older than {min_version}; "
            "run: uv sync --extra mjlab"
        )
    try:
        import mujoco_warp  # noqa: F401
    except AttributeError as exc:
        if "mjENBL_MULTICCD" in str(exc):
            pytest.fail(
                f"mujoco-warp {warp_ver} failed to import against installed "
                f"mujoco: {exc}. Run: uv sync --extra mjlab"
            )
        raise


def require_ipython() -> None:
    """Skip when IPython is not installed (notebook/tutorial helpers)."""
    pytest.importorskip("IPython")
