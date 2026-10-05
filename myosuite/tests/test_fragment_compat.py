# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the fragment version compatibility check (scripts/check_fragment_compat.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


pytestmark = pytest.mark.tier1

_SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "check_fragment_compat.py"


def _load_script() -> ModuleType:
    """Import scripts/check_fragment_compat.py as a module."""
    spec = importlib.util.spec_from_file_location("check_fragment_compat", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeRegistry:
    """Registry stub whose every fragment is installed at version 3."""

    class _Info:
        version = 3

    @classmethod
    def get(cls, name: str) -> _Info:
        return cls._Info()


def test_script_importable():
    """check_fragment_compat.py is importable as a module."""
    assert _SCRIPT.exists(), f"Script not found: {_SCRIPT}"
    assert hasattr(_load_script(), "main")


def test_no_myo_sim_registry_exits_zero(monkeypatch):
    """main() exits 0 (warning) when myo_sim.FragmentRegistry is absent."""
    mod = _load_script()
    monkeypatch.setattr(mod, "_load_myo_sim_registry", lambda: None)
    monkeypatch.setattr(mod, "_discover_task_configs", lambda: iter([]))

    result = mod.main()
    assert result == 0, "Should exit 0 when myo_sim registry is unavailable"


@pytest.mark.parametrize(("declared", "expected"), [(3, 0), (2, 1)])
def test_declared_version_against_installed(monkeypatch, declared, expected):
    """main() exits 1 only when the installed fragment is newer than declared."""
    mod = _load_script()
    monkeypatch.setattr(mod, "_load_myo_sim_registry", lambda: _FakeRegistry)
    monkeypatch.setattr(
        mod, "_discover_task_configs", lambda: iter([("TestTask", {"elbow": declared})])
    )

    assert mod.main() == expected


def test_no_task_config_discovered_exits_one(monkeypatch):
    """A scan that finds no TaskConfig subclass fails instead of passing."""
    mod = _load_script()
    monkeypatch.setattr(mod, "_load_myo_sim_registry", lambda: _FakeRegistry)
    monkeypatch.setattr(mod, "_discover_task_configs", lambda: iter([]))

    assert mod.main() == 1


# The tests below use the real discovery and the installed myo_sim registry.


def test_real_discovery_finds_the_task_configs():
    """The scan reaches the TaskConfig subclasses of the basic tasks."""
    names = {name for name, _ in _load_script()._discover_task_configs()}
    assert {
        "ElbowPoseFixedTask",
        "ElbowPoseRandomTask",
        "LegDirectionalForwardTask",
    } <= names


def test_real_check_passes_and_reports_what_it_checked(capsys):
    """The repo passes its own check, and says so only if something was checked."""
    mod = _load_script()
    declared = [versions for _, versions in mod._discover_task_configs() if versions]

    assert mod.main() == 0
    out = capsys.readouterr()
    if declared:
        assert "PASSED" in out.out
    else:
        # Today no task declares fragment_versions: no false "PASSED".
        assert "no fragment version was checked" in out.err
        assert "PASSED" not in out.out


@pytest.mark.parametrize(
    ("offset", "fragment", "expected"),
    [(0, "elbow", 0), (-1, "elbow", 1), (0, "no_such_fragment", 1)],
)
def test_real_declaration_is_checked_against_installed_registry(
    monkeypatch, offset, fragment, expected
):
    """A fragment_versions declaration on a real task config is discovered and checked."""
    import myo_sim

    from myosuite.envs.myo.tasks.basic.specs.elbow_pose_spec import ElbowPoseFixedTask

    installed = myo_sim.FragmentRegistry.get("elbow").version
    monkeypatch.setattr(
        ElbowPoseFixedTask, "fragment_versions", {fragment: installed + offset}
    )

    assert _load_script().main() == expected
