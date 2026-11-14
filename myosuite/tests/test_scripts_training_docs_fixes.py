# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Regression tests for scripts, training helpers, docs, tutorials, and test markers."""

from __future__ import annotations

import ast
import importlib.util
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

pytestmark = pytest.mark.tier1

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, rel_path: str) -> Any:
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_dynsyn_length_similarity_short_trajectory_and_env_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sys.path.insert(0, str(REPO_ROOT / "training" / "dynsyn"))
    try:
        from myosuite_dynsyn import groups
    finally:
        sys.path.pop(0)

    rng = np.random.default_rng(0)
    lengths = rng.normal(size=(12, 5))
    sim = groups.length_similarity(lengths, segment=100)
    assert sim.shape == (5, 5)
    assert np.isfinite(sim).all()
    np.testing.assert_allclose(np.diag(sim), 1.0)

    with pytest.raises(ValueError, match="non-empty"):
        groups.length_similarity(np.zeros((0, 3)), segment=10)
    with pytest.raises(ValueError, match="segment"):
        groups.length_similarity(lengths, segment=0)

    closed = {"flag": False}

    class _FakeEnv:
        cfg = object()

        def close(self) -> None:
            closed["flag"] = True

    import types
    import myosuite

    fake_action_map = types.ModuleType(
        "myosuite.envs.myo.backends.mjlab.tasks.mdp.action_map"
    )
    fake_action_map.action_actuator_names = lambda *a, **kw: []  # type: ignore[attr-defined]

    def _boom(_cfg: Any) -> list[str]:
        raise RuntimeError("boom")

    fake_action_map.muscle_action_terms = _boom  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules,
        "myosuite.envs.myo.backends.mjlab.tasks.mdp.action_map",
        fake_action_map,
    )
    monkeypatch.setattr(myosuite, "make_env", lambda *a, **kw: _FakeEnv())
    with pytest.raises(RuntimeError, match="boom"):
        groups.task_muscles("myoElbowPose1D6MRandom-v0")
    assert closed["flag"] is True


def test_write_checkpoint_manifests_reads_utf8_table() -> None:
    mod = _load_module(
        "_test_write_checkpoint_manifests", "scripts/write_checkpoint_manifests.py"
    )
    table = (REPO_ROOT / "docs" / "baseline_checkpoints.md").read_text(encoding="utf-8")
    matches = mod.ROW.findall(table)
    assert len(matches) > 0


def test_retarget_amass_cross_platform_workers_and_posix_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load_module("_test_retarget_amass", "scripts/mimic/retarget_amass.py")
    monkeypatch.delattr(os, "getloadavg", raising=False)
    mem_gb = mod._mem_available_gb()
    assert mem_gb > 0.0
    workers = mod.choose_workers(2)
    assert 1 <= workers <= 2

    src_dir = tmp_path / "KIT" / "subject01"
    src_dir.mkdir(parents=True)
    clip_file = src_dir / "walk_poses.npz"
    np.savez(clip_file, poses=np.zeros((4, 3)))

    captured_jobs: list[tuple[str, str, str, dict[str, Any]]] = []

    class _FakePool:
        def __enter__(self) -> _FakePool:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def imap_unordered(
            self, fn: Any, jobs: list[Any], chunksize: int = 1
        ) -> list[dict[str, Any]]:
            captured_jobs.extend(jobs)
            return []

    class _FakeCtx:
        def Pool(self, *args: Any, **kwargs: Any) -> _FakePool:
            return _FakePool()

    monkeypatch.setattr(mod.mp, "get_context", lambda _: _FakeCtx())
    curation_path = tmp_path / "flags.json"
    curation_path.write_text(
        '{"AMASS/KIT/subject01/walk_poses.npz": {"quality_flags": ["ok"]}}',
        encoding="utf-8",
    )
    rc = mod.main(
        [
            str(tmp_path / "KIT"),
            "--body-models",
            str(tmp_path),
            "--out-root",
            str(tmp_path / "out"),
            "--report-dir",
            str(tmp_path / "report"),
            "--quality-flags",
            str(curation_path),
            "--workers",
            "1",
        ]
    )
    assert rc == 0
    assert len(captured_jobs) == 1
    _, _, label, cur = captured_jobs[0]
    assert label == "AMASS/KIT/subject01/walk_poses.npz"
    assert "\\" not in label
    assert cur == {"quality_flags": ["ok"]}


def test_generate_parity_baselines_dict_actions_zero_steps_and_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load_module(
        "_test_generate_parity_baselines", "scripts/generate_parity_baselines.py"
    )
    closed: list[str] = []

    class _DummyMultiAgentEnv:
        def __init__(self) -> None:
            self.np_random = np.random.default_rng(0)
            self.action_space = gym.spaces.Dict(
                {
                    "a0": gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
                    "a1": gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
                }
            )
            self.unwrapped = self

        def reset(
            self, seed: int | None = None
        ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
            obs = {
                "a0": np.zeros(3, dtype=np.float32),
                "a1": np.ones(3, dtype=np.float32),
            }
            return obs, {}

        def step(
            self, action: dict[str, np.ndarray]
        ) -> tuple[
            dict[str, np.ndarray],
            dict[str, float],
            dict[str, bool],
            dict[str, bool],
            dict[str, Any],
        ]:
            obs = {k: np.asarray(v, dtype=np.float32) for k, v in action.items()}
            return (
                obs,
                {"a0": 1.0, "a1": -1.0},
                {"a0": False, "a1": False},
                {"a0": False, "a1": False},
                {},
            )

        def close(self) -> None:
            closed.append("closed")

    monkeypatch.setattr(mod, "make_env", lambda env_id: _DummyMultiAgentEnv())
    monkeypatch.setattr(mod, "_get_all_env_ids", lambda: ["myoDummyMulti-v0"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_parity_baselines.py",
            "--env-id",
            "myoDummyMulti-v0",
            "--n-steps",
            "0",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert mod.main() == 0
    assert closed == ["closed"]
    with open(tmp_path / "myoDummyMulti-v0.pkl", "rb") as f:
        assert pickle.load(f) == []

    assert mod.generate_baseline(
        "myoDummyMulti-v0", n_steps=2, seed=7, output_dir=tmp_path
    )
    with open(tmp_path / "myoDummyMulti-v0.pkl", "rb") as f:
        records = pickle.load(f)
    assert len(records) == 2
    assert set(records[0]["action"].keys()) == {"a0", "a1"}


def test_render_sb3_solutions_inserts_repo_root_before_myosuite_import() -> None:
    source = (REPO_ROOT / "scripts" / "render_sb3_solutions.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    insert_line = None
    import_line = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "insert"
        ):
            insert_line = insert_line or node.lineno
        if isinstance(node, ast.ImportFrom) and node.module == "myosuite":
            import_line = import_line or node.lineno
    assert insert_line is not None and import_line is not None
    assert insert_line < import_line


def test_reflex_ctr_sensor_list_and_bfsh_param_mapping() -> None:
    mod = _load_module("_test_reflex_ctr", "tutorials/files/2.4/reflex_ctr.py")
    assert "dphi_knee" in mod.LocoCtrl.s_l_keys
    assert "F_RF" in mod.LocoCtrl.s_l_keys
    assert "dphi_kneeF_RF" not in mod.LocoCtrl.s_l_keys

    ctrl = mod.LocoCtrl(TIMESTEP=0.01, control_mode=1, control_dimension=3)
    params = np.ones(len(mod.LocoCtrl.cp_keys))
    params[mod.LocoCtrl.cp_map["BFSH_8_DG"]] = 2.0
    params[mod.LocoCtrl.cp_map["BFSH_8_PG"]] = 5.0
    ctrl.set_control_params(params)
    assert ctrl.cp["r_leg"]["BFSH_8_DG"] == pytest.approx(2.0)
    assert ctrl.cp["r_leg"]["BFSH_8_PG"] == pytest.approx(5.0)


def test_docs_links_and_conftest_mjlab_regex() -> None:
    pubs = (REPO_ROOT / "docs" / "source" / "publications.rst").read_text(
        encoding="utf-8"
    )
    assert "`<web " not in pubs
    assert "`<code " not in pubs
    assert "`<doi " not in pubs

    cdoc = (REPO_ROOT / "docs" / "source" / "challenge-doc.rst").read_text(
        encoding="utf-8"
    )
    assert "“myoChallengeOslRunRandom-v0”" not in cdoc

    suite = (REPO_ROOT / "docs" / "source" / "suite.rst").read_text(encoding="utf-8")
    assert "ElbowPose1D1MRandom-v0" not in suite
    assert "myoLeg<Rough|Hilly|Stair>Walk-v0" not in suite

    conftest = _load_module("_test_conftest_mjlab", "myosuite/tests/conftest.py")
    sample = 'env = make_env("myoElbowPose1D6MRandom-v0", ' + 'backend="mjlab")\n'
    assert conftest._MJLAB_SOURCE.search(sample) is not None
