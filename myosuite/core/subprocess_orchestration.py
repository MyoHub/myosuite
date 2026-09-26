# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Backend-agnostic subprocess orchestration helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def build_python_command(
    *,
    exec_cwd: Path,
    py_args: list[str],
    override_env_var: str | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Build ``python`` command line for a target working directory.

    Resolution order:
    1. ``override_env_var`` executable if provided and set.
    2. ``uv run --directory`` when ``uv`` exists and ``pyproject.toml`` exists.
    3. ``exec_cwd/.venv/bin/python`` when present.
    4. current ``sys.executable``.
    """
    if override_env_var:
        override = os.environ.get(override_env_var)
        if override:
            return [override, *py_args], os.environ.copy()

    venv_py = exec_cwd / ".venv" / "bin" / "python"
    uv = shutil.which("uv")
    if uv and (exec_cwd / "pyproject.toml").is_file():
        return [
            uv,
            "run",
            "--directory",
            str(exec_cwd),
            "python",
            *py_args,
        ], os.environ.copy()

    if venv_py.is_file():
        return [str(venv_py), *py_args], os.environ.copy()

    return [sys.executable, *py_args], os.environ.copy()


def prepend_env_path_var(
    *,
    base_env: dict[str, str],
    env_var: str,
    prefix_value: str | None,
) -> dict[str, str]:
    """Prepend a path value to an environment path-like variable."""
    if not prefix_value:
        return base_env
    env = dict(base_env)
    prev = env.get(env_var, "")
    env[env_var] = prefix_value + (os.pathsep + prev if prev else "")
    return env


def run_command(
    *,
    cmd: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """Run subprocess and return integer exit code."""
    return int(
        subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            check=False,
        ).returncode
    )


__all__ = [
    "build_python_command",
    "prepend_env_path_var",
    "run_command",
]
