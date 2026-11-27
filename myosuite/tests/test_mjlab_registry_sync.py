# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests that REGISTERED_TASKS and deferred/entry-point mjlab registration stay in sync."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys

import gymnasium as gym
import pytest

from myosuite.envs.backends.mjlab import REGISTERED_TASKS

pytestmark = pytest.mark.tier2

_REPO = Path(__file__).resolve().parents[2]


def test_registered_tasks_non_empty() -> None:
    """REGISTERED_TASKS must have at least one entry."""
    assert len(REGISTERED_TASKS) > 0, "REGISTERED_TASKS is empty"


def test_registered_tasks_keys_in_gymnasium_registry() -> None:
    """Every key in REGISTERED_TASKS must be registered in the Gymnasium env registry."""
    gym_ids = set(gym.envs.registry.keys())
    missing = [env_id for env_id in REGISTERED_TASKS if env_id not in gym_ids]
    assert not missing, (
        f"The following env IDs are in REGISTERED_TASKS but not in gym.envs.registry: "
        f"{missing}. Either register them via myosuite.__init__ or remove them from "
        f"REGISTERED_TASKS."
    )


def test_every_mjlab_importing_module_calls_finish_deferred_bootstrap() -> None:
    """Every envs/{tasks,backends/mjlab} module importing mjlab or backends.mjlab at top level must call finish_deferred_mjlab_bootstrap."""
    missing: list[str] = []
    for root_rel in ("myosuite/envs/tasks", "myosuite/envs/backends/mjlab"):
        for py_path in sorted((_REPO / root_rel).rglob("*.py")):
            rel_str = py_path.relative_to(_REPO).as_posix()
            if rel_str == "myosuite/envs/backends/mjlab/bootstrap.py":
                continue
            src = py_path.read_text(encoding="utf-8")
            tree = ast.parse(src, filename=str(py_path))
            tc_nodes: set[ast.AST] = set()
            fn_nodes: set[ast.AST] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.If):
                    t = node.test
                    if (isinstance(t, ast.Name) and t.id == "TYPE_CHECKING") or (
                        isinstance(t, ast.Attribute) and t.attr == "TYPE_CHECKING"
                    ):
                        for s in node.body:
                            tc_nodes.update(ast.walk(s))
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for s in node.body:
                        fn_nodes.update(ast.walk(s))
            imports_mjlab = False
            for node in ast.walk(tree):
                if node in tc_nodes or node in fn_nodes:
                    continue
                if isinstance(node, ast.Import):
                    if any(
                        a.name == "mjlab"
                        or a.name.startswith("mjlab.")
                        or a.name == "myosuite.envs.backends.mjlab"
                        or a.name.startswith("myosuite.envs.backends.mjlab.")
                        for a in node.names
                    ):
                        imports_mjlab = True
                elif (
                    isinstance(node, ast.ImportFrom) and node.level == 0 and node.module
                ):
                    if (
                        node.module == "mjlab"
                        or node.module.startswith("mjlab.")
                        or node.module == "myosuite.envs.backends.mjlab"
                        or node.module.startswith("myosuite.envs.backends.mjlab.")
                    ):
                        imports_mjlab = True
            if imports_mjlab and "finish_deferred_mjlab_bootstrap(__name__)" not in src:
                missing.append(rel_str)
    assert not missing, (
        "Modules importing mjlab or myosuite.envs.backends.mjlab at module top level must call "
        f"finish_deferred_mjlab_bootstrap(__name__) at EOF: {missing}"
    )


@pytest.fixture(scope="module")
def _fresh_ep_reference(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, str], list[str]]:
    pytest.importorskip("mjlab")
    import json

    probe_root = tmp_path_factory.mktemp("mjlab_ep_probe")
    dist_info = probe_root / "myosuite_ep_probe-0.0.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: myosuite-ep-probe\nVersion: 0.0.0\n",
        encoding="utf-8",
    )
    (dist_info / "entry_points.txt").write_text(
        "[mjlab.tasks]\nmyosuite = myosuite.envs.backends.mjlab.bootstrap\n",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(probe_root), str(_REPO), os.environ.get("PYTHONPATH", "")]
        ),
    }
    ref_proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, mjlab; from mjlab.tasks.registry import list_tasks; print(json.dumps(sorted(list_tasks())))",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    expected_tasks = json.loads(ref_proc.stdout.strip().splitlines()[-1])
    assert "myoLegWalk-v0" in expected_tasks
    assert "myoChallengeTableTennisP0-v0" in expected_tasks
    return env, expected_tasks


@pytest.mark.parametrize(
    "first_module",
    [
        "myosuite.envs.backends.mjlab",
        "myosuite.envs.backends.mjlab.register_mjlab_tasks",
        "myosuite.envs.backends.mjlab.tasks.pose.mdp.commands",
        "myosuite.envs.backends.mjlab.tasks.reach.mdp.commands",
        "myosuite.envs.tasks.leg.walk.walk_mdp",
        "myosuite.envs.tasks.leg.walk.mjlab_cfg",
        "myosuite.envs.tasks.leg.walk.mjlab_register",
        "myosuite.envs.tasks.leg.stand.mjlab_cfg",
        "myosuite.envs.tasks.leg.directional.directional_mdp",
        "myosuite.envs.tasks.leg.directional.mjlab_cfg",
        "myosuite.envs.tasks.fullbody.waypoint.mdp",
        "myosuite.envs.tasks.fullbody.waypoint.mjlab_cfg",
        "myosuite.envs.tasks.fullbody.waypoint.mjlab_register",
        "myosuite.envs.tasks.arm.pose.mjlab_cfg",
        "myosuite.envs.tasks.arm.pose.mjlab_register",
        "myosuite.envs.tasks.arm.reach.mjlab_cfg",
        "myosuite.envs.tasks.arm.reach.mjlab_register",
        "myosuite.envs.tasks.hand.pose.finger_mjlab_register",
        "myosuite.envs.tasks.hand.pose.hand_mjlab_register",
        "myosuite.envs.tasks.hand.reach.finger_mjlab_register",
        "myosuite.envs.tasks.hand.reach.hand_mjlab_register",
        "myosuite.envs.tasks.torso.pose.mjlab_register",
        "myosuite.envs.tasks.mimic.mjlab_commands",
        "myosuite.envs.tasks.mimic.mjlab_cfg",
        "myosuite.envs.tasks.mimic.mjlab_register",
        "myosuite.envs.tasks.games.tabletennis.register_mjlab",
    ],
)
def test_colocated_mjlab_modules_import_first_with_fresh_entry_point(
    first_module: str, _fresh_ep_reference: tuple[dict[str, str], list[str]]
) -> None:
    """Importing any co-located mjlab module first (with fresh bootstrap entry-point metadata) registers the full twin set."""
    import json

    env, expected_tasks = _fresh_ep_reference

    code = (
        f"import importlib, json; importlib.import_module({first_module!r}); "
        "from mjlab.tasks.registry import list_tasks; "
        "print(json.dumps(sorted(list_tasks())))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"{first_module}: {proc.stderr}"
    assert "Failed to load task package" not in proc.stderr, proc.stderr
    assert "skipping" not in proc.stderr.lower(), proc.stderr

    got_tasks = json.loads(proc.stdout.strip().splitlines()[-1])
    assert got_tasks == expected_tasks, (
        f"{first_module} twin mismatch: missing={sorted(set(expected_tasks) - set(got_tasks))}, "
        f"extra={sorted(set(got_tasks) - set(expected_tasks))}"
    )


def test_legacy_mjlab_facade_registers_twins_and_exposes_task_children() -> None:
    """Legacy ``myosuite.envs.myo.backends.mjlab`` registers twins and exposes ``.leg``, ``.waypoint``, ``.mimic_tracking``, ``pose.config``."""
    pytest.importorskip("mjlab")
    code = (
        "import myosuite.envs.myo.backends.mjlab; "
        "import myosuite.envs.myo.backends.mjlab.tasks as t; "
        "from myosuite.envs.myo.backends.mjlab.tasks import pose, reach; "
        "from mjlab.tasks.registry import list_tasks; "
        "tasks = set(list_tasks()); "
        "assert 'myoLegWalk-v0' in tasks and 'myoElbowPose1D6MRandom-v0' in tasks, tasks; "
        "assert t.leg.walk_env_cfg is not None and t.leg.walk_mdp is not None and t.leg.rl_cfg is not None; "
        "assert t.waypoint.mdp is not None and t.mimic_tracking.env_cfg is not None; "
        "assert pose.config.elbow.env_cfgs is not None and reach.config.arm.env_cfgs is not None"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_legacy_mjlab_package_imports_without_torch_on_cpu() -> None:
    """On a CPU-only install without ``torch`` or ``mjlab``, ``import myosuite.envs.myo.backends.mjlab`` still succeeds."""
    code = (
        "import importlib.abc, sys\n"
        "class _BlockGpu(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path, target=None):\n"
        "        if fullname.split('.')[0] in {'torch', 'mjlab', 'warp', 'mujoco_warp'}:\n"
        "            raise ModuleNotFoundError(f'blocked {fullname}', name=fullname)\n"
        "        return None\n"
        "sys.meta_path.insert(0, _BlockGpu())\n"
        "import myosuite.envs.myo.backends.mjlab as m\n"
        "import myosuite.envs.myo.backends.mjlab.tasks as mt\n"
        "assert len(m.REGISTERED_TASKS) > 0 and mt is not None\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
