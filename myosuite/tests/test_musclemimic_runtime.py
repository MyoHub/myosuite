# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for :mod:`myosuite.integrations.musclemimic.runtime`."""

from __future__ import annotations

import builtins
import shutil
from pathlib import Path

import pytest

from myosuite.integrations.musclemimic.runtime import (
    build_demo_cache_command,
    resolve_musclemimic_exec_cwd,
    resolve_musclemimic_root_from_env,
)


def test_resolve_musclemimic_root_from_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unset MYOSUITE_MUSCLEMIMIC_ROOT returns None."""
    monkeypatch.delenv("MYOSUITE_MUSCLEMIMIC_ROOT", raising=False)
    assert resolve_musclemimic_root_from_env() is None


def test_resolve_musclemimic_exec_cwd_env_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MYOSUITE_MUSCLEMIMIC_ROOT should bypass imports."""
    monkeypatch.setenv("MYOSUITE_MUSCLEMIMIC_ROOT", str(tmp_path))
    assert resolve_musclemimic_exec_cwd() == tmp_path.resolve()
    assert resolve_musclemimic_root_from_env() == tmp_path.resolve()


def test_resolve_musclemimic_exec_cwd_env_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid MYOSUITE_MUSCLEMIMIC_ROOT raises FileNotFoundError."""
    monkeypatch.setenv(
        "MYOSUITE_MUSCLEMIMIC_ROOT",
        "/nonexistent_musclemimic_root_xyz",
    )
    with pytest.raises(FileNotFoundError, match="MYOSUITE_MUSCLEMIMIC_ROOT"):
        resolve_musclemimic_exec_cwd()


def test_resolve_musclemimic_exec_cwd_import_error_includes_hints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both imports missing: error lists env var and pip install."""
    monkeypatch.delenv("MYOSUITE_MUSCLEMIMIC_ROOT", raising=False)
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name in ("fullbody", "musclemimic"):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(
        ImportError,
        match="MYOSUITE_MUSCLEMIMIC_ROOT|fullbody.eval",
    ) as exc_info:
        resolve_musclemimic_exec_cwd()
    assert "Original errors" in str(exc_info.value)


def test_build_demo_cache_command_includes_snippet(tmp_path: Path) -> None:
    """Demo cache command should invoke ``setup_demo_for_myo_fullbody``."""
    cmd, _env = build_demo_cache_command(tmp_path)
    joined = " ".join(cmd)
    assert "setup_demo_for_myo_fullbody" in joined
    assert "-c" in cmd


def test_build_demo_cache_command_respects_python_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MYOSUITE_MUSCLEMIMIC_PYTHON should be the executable."""
    fake = tmp_path / "fakepy"
    fake.write_text("#!/bin/sh\necho\n")
    fake.chmod(0o755)
    monkeypatch.setenv("MYOSUITE_MUSCLEMIMIC_PYTHON", str(fake))
    cmd, _env = build_demo_cache_command(tmp_path)
    assert cmd[0] == str(fake)
    assert cmd[1] == "-c"
    monkeypatch.delenv("MYOSUITE_MUSCLEMIMIC_PYTHON", raising=False)


def test_build_demo_cache_command_prefers_venv_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no override and no ``uv``, use ``exec_cwd/.venv/bin/python``."""
    monkeypatch.delenv("MYOSUITE_MUSCLEMIMIC_PYTHON", raising=False)
    real_which = shutil.which

    def _which(name: str) -> str | None:
        if name == "uv":
            return None
        return real_which(name)

    monkeypatch.setattr(shutil, "which", _which)
    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    py = venv_bin / "python"
    py.write_text("#!/bin/sh\necho\n")
    py.chmod(0o755)
    cmd, _env = build_demo_cache_command(tmp_path)
    assert cmd[0] == str(py)
    assert cmd[1] == "-c"
