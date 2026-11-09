# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""The layering rules of ``myosuite.methods``, and the old import path of inverse kinematics."""

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


def _imports(path: Path) -> set[str]:
    """Top-level names of the modules *path* imports."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


def test_methods_depend_on_numpy_scipy_and_mujoco_only() -> None:
    allowed = _METHOD_DEPENDENCIES | set(sys.stdlib_module_names)
    for path in sorted((_PACKAGE / "methods").rglob("*.py")):
        extra = {n for n in _imports(path) if n.split(".")[0] not in allowed}
        assert not extra, f"{path.relative_to(_PACKAGE)} imports {sorted(extra)}"


@pytest.mark.parametrize("layer", ["envs", "terms", "core"])
def test_tasks_never_import_methods(layer: str) -> None:
    for path in sorted((_PACKAGE / layer).rglob("*.py")):
        uses = {n for n in _imports(path) if n.startswith("myosuite.methods")}
        assert not uses, f"{path.relative_to(_PACKAGE)} imports {sorted(uses)}"


def test_inverse_kinematics_keeps_its_physics_path() -> None:
    from myosuite.methods.biomech import inverse_kinematics as new
    from myosuite.physics import inverse_kinematics as old

    for name in ("IKResult", "nullspace_method", "qpos_from_site_pose"):
        assert getattr(old, name) is getattr(new, name)
