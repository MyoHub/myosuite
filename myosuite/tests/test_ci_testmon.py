# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache 2 license in the LICENSE file.
"""Execute CI's selection and cache scripts against controlled inputs."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.tier1

_WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/python-app.yml"
_STEPS = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))["jobs"]["core"]["steps"]
_SUITE = "Pytest — full suite (tier 1–3, all backends; excludes @slow)"
_REQUIRED = {
    "test_model_builder.py",
    "test_terms_cpu.py",
    "test_fragment_compat.py",
    "test_parity.py",
    "test_registry.py",
    "test_import_footprint.py",
    "test_mjlab_warning_filters.py",
    "test_train_mjlab.py",
}
_STUB = """import json, os, subprocess, sys
from pathlib import Path
tool, args = sys.argv[1], sys.argv[2:]
if tool == 'git':
    sys.stdout.buffer.write(Path(os.environ['CHANGED_FILE']).read_text().encode())
    sys.exit(0)
with open(os.environ['UV_LOG'], 'a') as log:
    log.write(json.dumps(args) + '\\n')
if '-m' in args and args[args.index('-m') + 1] == 'pytest':
    kind = 'SELECTED' if '--testmon-forceselect' in args else 'REQUIRED'
    sys.exit(int(os.environ.get('UV_' + kind + '_STATUS', '0')))
sys.exit(subprocess.call([sys.executable, *args[2:]]))
"""


def _step(name: str) -> dict:
    return next(step for step in _STEPS if step.get("name") == name)


@pytest.fixture
def shell_env(tmp_path: Path) -> dict[str, str]:
    stub = tmp_path / "stub.py"
    stub.write_text(_STUB, encoding="utf-8")
    for tool in ("git", "uv"):
        wrapper = tmp_path / tool
        wrapper.write_text(
            f'#!/usr/bin/env bash\nexec "$PYTHON_FOR_TEST" "$STUB_TOOL" {tool} "$@"\n',
            encoding="utf-8",
        )
        wrapper.chmod(0o755)
    paths = tmp_path / "changed.txt"
    paths.write_text("myosuite/terms/waypoint.py\n", encoding="utf-8")
    env = {
        **os.environ,
        "STUB_TOOL": stub.as_posix(),
        "PYTHON_FOR_TEST": Path(sys.executable).as_posix(),
        "CHANGED_FILE": paths.as_posix(),
        "UV_LOG": (tmp_path / "uv.jsonl").as_posix(),
        "GITHUB_OUTPUT": (tmp_path / "output").as_posix(),
        "GITHUB_EVENT_NAME": "pull_request",
        "TESTMON": "select",
        "SELECT": "not slow and not mjlab",
    }
    # Bash uses ':' even on Windows; prepend its own PATH inside the script.
    env["STUB_BIN"] = tmp_path.as_posix()
    return env


def _run(
    name: str, directory: Path, env: dict[str, str]
) -> subprocess.CompletedProcess:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("workflow script tests require bash")
    script = directory / "step.sh"
    script.write_text(
        'export PATH="$(cd "$STUB_BIN" && pwd):$PATH"\n' + _step(name)["run"],
        encoding="utf-8",
    )
    return subprocess.run(
        [bash, "--noprofile", "--norc", "-o", "pipefail", script.as_posix()],
        cwd=directory,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


@pytest.mark.parametrize("data_position", [None, 0, 10000])
def test_large_path_lists_keep_data_fallback(
    tmp_path: Path, shell_env: dict[str, str], data_position: int | None
) -> None:
    paths = [f"myosuite/terms/changed_module_{i:05}.py" for i in range(10000)]
    if data_position is not None:
        paths.insert(data_position, "myosuite/models/model.xml")
    Path(shell_env["CHANGED_FILE"]).write_text("\n".join(paths), encoding="utf-8")
    result = _run("Detect changed paths", tmp_path, shell_env)
    assert result.returncode == 0, result.stderr
    outputs = Path(shell_env["GITHUB_OUTPUT"]).read_text()
    assert f"only_python={'true' if data_position is None else 'false'}" in outputs


@pytest.mark.parametrize(
    "path",
    [
        "myosuite/__init__.py",
        "myosuite/envs/myo/tasks/basic/__init__.py",
        "myosuite/core/registry.py",
        "myosuite/core/config.py",
        "myosuite/envs/myo/backends/mjlab/register_mjlab_tasks.py",
        "myosuite/envs/myo/backends/mjlab/mimic_mjlab_env.py",
        "myosuite/envs/myo/backends/mjlab/tasks/registration.py",
        "myosuite/envs/myo/tasks/basic/specs/elbow_pose_spec.py",
        "myosuite/envs/myo/tasks/basic/leg/specs/leg_directional_spec.py",
    ],
)
def test_registration_changes_run_full_job(
    tmp_path: Path, shell_env: dict[str, str], path: str
) -> None:
    Path(shell_env["CHANGED_FILE"]).write_text(path, encoding="utf-8")
    result = _run("Detect changed paths", tmp_path, shell_env)
    assert result.returncode == 0, result.stderr
    assert "only_python=false" in Path(shell_env["GITHUB_OUTPUT"]).read_text()


@pytest.mark.parametrize(
    ("required_status", "selected_status", "expected_status", "count"),
    [
        (0, 0, 0, 2),
        (0, 5, 0, 2),
        (0, 1, 1, 2),
        (0, 2, 2, 2),
        (1, 0, 1, 1),
        (5, 0, 5, 1),
    ],
)
def test_required_tests_run_before_selection_and_failures_propagate(
    tmp_path: Path,
    shell_env: dict[str, str],
    required_status: int,
    selected_status: int,
    expected_status: int,
    count: int,
) -> None:
    shell_env.update(
        UV_REQUIRED_STATUS=str(required_status), UV_SELECTED_STATUS=str(selected_status)
    )
    result = _run(_SUITE, tmp_path, shell_env)
    assert result.returncode == expected_status, result.stderr
    calls = [
        json.loads(line) for line in Path(shell_env["UV_LOG"]).read_text().splitlines()
    ]
    assert len(calls) == count
    assert {Path(arg).name for arg in calls[0] if arg.endswith(".py")} == _REQUIRED
    assert "no:pytest-testmon" in calls[0] and "--testmon-forceselect" not in calls[0]
    if count == 2:
        assert "--testmon-forceselect" in calls[1]


@pytest.mark.parametrize("mode", ["record", "off"])
def test_full_runs_keep_their_recording_mode(
    tmp_path: Path, shell_env: dict[str, str], mode: str
) -> None:
    shell_env["TESTMON"] = mode
    result = _run(_SUITE, tmp_path, shell_env)
    assert result.returncode == 0, result.stderr
    calls = [
        json.loads(line) for line in Path(shell_env["UV_LOG"]).read_text().splitlines()
    ]
    assert len(calls) == 1 and ("--testmon" in calls[0]) == (mode == "record")


def test_checkpoint_makes_the_single_file_cache_complete(
    tmp_path: Path, shell_env: dict[str, str]
) -> None:
    path = tmp_path / ".testmondata"
    with sqlite3.connect(path) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("CREATE TABLE recorded (test TEXT)")
        writer.execute("INSERT INTO recorded VALUES ('parity')")
        writer.commit()
        assert path.with_name(".testmondata-wal").stat().st_size > 0
        result = _run("Checkpoint testmon data", tmp_path, shell_env)
        assert result.returncode == 0, result.stderr
        copied = tmp_path / "restored.db"
        shutil.copyfile(path, copied)
        with sqlite3.connect(copied) as restored:
            assert restored.execute("SELECT test FROM recorded").fetchall() == [
                ("parity",)
            ]
    checkpoint, save = _step("Checkpoint testmon data"), _step("Save testmon data")
    assert _STEPS.index(checkpoint) < _STEPS.index(save)
    assert checkpoint["if"] == save["if"] and save["with"]["path"] == ".testmondata"


def test_busy_wal_checkpoint_fails_instead_of_caching_partial_data(
    tmp_path: Path, shell_env: dict[str, str]
) -> None:
    path = tmp_path / ".testmondata"
    with sqlite3.connect(path) as writer, sqlite3.connect(path) as reader:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("CREATE TABLE recorded (test TEXT)")
        writer.commit()
        reader.execute("BEGIN")
        reader.execute("SELECT * FROM recorded").fetchall()
        writer.execute("INSERT INTO recorded VALUES ('parity')")
        writer.commit()
        result = _run("Checkpoint testmon data", tmp_path, shell_env)
        assert result.returncode != 0
        assert "checkpoint is busy" in result.stderr
        reader.rollback()
