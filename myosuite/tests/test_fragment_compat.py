# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for fragment version compatibility check (Phase 1)."""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.tier1


def test_script_importable():
    """check_fragment_compat.py is importable as a module."""
    import importlib.util
    from pathlib import Path

    script = (
        Path(__file__).parent.parent.parent / "scripts" / "check_fragment_compat.py"
    )
    assert script.exists(), f"Script not found: {script}"

    spec = importlib.util.spec_from_file_location("check_fragment_compat", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "main")


def test_no_myo_sim_registry_exits_zero(monkeypatch):
    """main() exits 0 (warning) when myo_sim.FragmentRegistry is absent."""
    import importlib.util
    from pathlib import Path

    script = (
        Path(__file__).parent.parent.parent / "scripts" / "check_fragment_compat.py"
    )
    spec = importlib.util.spec_from_file_location("check_fragment_compat", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Patch _load_myo_sim_registry to return None
    monkeypatch.setattr(mod, "_load_myo_sim_registry", lambda: None)
    monkeypatch.setattr(mod, "_discover_task_configs", lambda: iter([]))

    result = mod.main()
    assert result == 0, "Should exit 0 when myo_sim registry is unavailable"


def test_version_mismatch_exits_one(monkeypatch):
    """main() exits 1 when an installed fragment version exceeds declared."""
    import importlib.util
    from pathlib import Path

    script = (
        Path(__file__).parent.parent.parent / "scripts" / "check_fragment_compat.py"
    )
    spec = importlib.util.spec_from_file_location("check_fragment_compat", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class FakeInfo:
        version = 5  # installed version > declared

    class FakeRegistry:
        @classmethod
        def get(cls, name):
            return FakeInfo()

        @classmethod
        def all_names(cls):
            return ["elbow"]

    # Task declares version 3, installed is 5 → mismatch
    monkeypatch.setattr(mod, "_load_myo_sim_registry", lambda: FakeRegistry)
    monkeypatch.setattr(
        mod, "_discover_task_configs", lambda: iter([("TestTask", {"elbow": 3})])
    )

    result = mod.main()
    assert result == 1, "Should exit 1 when fragment version mismatch detected"


def test_version_compatible_exits_zero(monkeypatch):
    """main() exits 0 when declared version matches installed."""
    import importlib.util
    from pathlib import Path

    script = (
        Path(__file__).parent.parent.parent / "scripts" / "check_fragment_compat.py"
    )
    spec = importlib.util.spec_from_file_location("check_fragment_compat", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class FakeInfo:
        version = 3

    class FakeRegistry:
        @classmethod
        def get(cls, name):
            return FakeInfo()

        @classmethod
        def all_names(cls):
            return ["elbow"]

    monkeypatch.setattr(mod, "_load_myo_sim_registry", lambda: FakeRegistry)
    monkeypatch.setattr(
        mod, "_discover_task_configs", lambda: iter([("TestTask", {"elbow": 3})])
    )

    result = mod.main()
    assert result == 0, "Should exit 0 when all versions are compatible"
