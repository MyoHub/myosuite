# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""The dependency rule of ``myosuite.biomechanics``, and the old import path of inverse kinematics."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.tier1

_PACKAGE = Path(__file__).parents[1]
# NumPy, SciPy and MuJoCo; OSQP (a core dependency) solves the static-optimization QPs,
# and absl comes with MuJoCo (inverse kinematics logs through it).
_METHOD_DEPENDENCIES = {"numpy", "scipy", "mujoco", "osqp", "absl", "myosuite"}
# Canonical differential IK uses the explicitly declared core Mink dependency.
_MODULE_DEPENDENCIES = {"inverse_kinematics.py": {"mink"}}


def _imports(path: Path) -> set[str]:
    """Top-level names of the modules *path* imports."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


def test_biomechanics_depends_only_on_approved_runtime_libraries() -> None:
    allowed = _METHOD_DEPENDENCIES | set(sys.stdlib_module_names)
    for path in sorted((_PACKAGE / "biomechanics").rglob("*.py")):
        module = path.relative_to(_PACKAGE / "biomechanics").as_posix()
        dependencies = allowed | _MODULE_DEPENDENCIES.get(module, set())
        extra = {n for n in _imports(path) if n.split(".")[0] not in dependencies}
        assert not extra, f"{path.relative_to(_PACKAGE)} imports {sorted(extra)}"


def test_inverse_kinematics_keeps_its_physics_path() -> None:
    from myosuite.biomechanics import inverse_kinematics as new
    from myosuite.physics import inverse_kinematics as old

    for name in ("IKResult", "nullspace_method", "qpos_from_site_pose"):
        assert getattr(old, name) is getattr(new, name)
