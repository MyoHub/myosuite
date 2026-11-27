# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Unit and smoke tests for ``training/ppo`` (``myosuite_ppo``) and legacy wrappers."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

_PPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PPO_ROOT.parents[1]
for _root in (_REPO_ROOT, _PPO_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))


@pytest.mark.tier1
def test_presets_and_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    from myosuite_ppo import FOUR_BENCHMARK_TASKS, WinnerPpoConfig, budget_for
    from myosuite_ppo.cli import main as cli_main

    assert len(FOUR_BENCHMARK_TASKS) == 4
    assert "myoElbowPose1D6MRandom-v0" in FOUR_BENCHMARK_TASKS
    assert "myoMimic-v0" in FOUR_BENCHMARK_TASKS
    cfg = WinnerPpoConfig.for_n_envs(4)
    assert (4 * cfg.n_steps) % cfg.batch_size == 0
    assert budget_for(6, 100_000) == 100_000

    assert cli_main(["--help"]) == 0
    out = capsys.readouterr().out
    assert "sb3" in out and "mjlab" in out
    assert cli_main(["unknown-backend"]) == 2


@pytest.mark.tier1
def test_sb3_migration_stubs_and_solved_from_info_identity() -> None:
    import myosuite.benchmarks as benchmarks
    from myosuite.benchmarks import solved_from_info as bench_solved
    import myosuite.utils.sb3_callbacks as sb3_callbacks_stub
    import myosuite.utils.sb3_sweep as sb3_sweep_stub
    from myosuite_ppo.sb3_sweep import (
        WinnerPpoConfig,
        solved_from_info as ppo_solved,
    )

    assert benchmarks.solved_from_info is bench_solved
    assert sb3_sweep_stub.solved_from_info is bench_solved
    assert ppo_solved is bench_solved
    assert bench_solved({"solved": True}) is True
    assert bench_solved({"rwd_dict": {"solved": [0, 1]}}) is True
    assert bench_solved({}) is False

    # Dunders and unknown attributes must raise AttributeError (never ImportError)
    assert hasattr(sb3_callbacks_stub, "__path__") is False
    assert hasattr(sb3_sweep_stub, "__path__") is False
    assert hasattr(sb3_callbacks_stub, "nonexistent_symbol") is False
    assert hasattr(sb3_sweep_stub, "nonexistent_symbol") is False

    # Runtime myosuite never imports training/ packages; moved SB3 symbols always raise ImportError with migration guidance
    assert WinnerPpoConfig is not None
    with pytest.raises(ImportError, match="myosuite_ppo.sb3_callbacks"):
        _ = sb3_callbacks_stub.OnnxCheckpointCallback
    with pytest.raises(ImportError, match="myosuite_ppo.sb3_sweep"):
        _ = sb3_sweep_stub.WinnerPpoConfig


@pytest.mark.tier2
@pytest.mark.mjlab
def test_scripts_train_mjlab_exports_all_legacy_symbols_and_cli_help() -> None:
    pytest.importorskip("mjlab")
    from mjlab.tasks.registry import load_rl_cfg
    from myosuite_ppo.cli import main as cli_main
    from myosuite_ppo.mjlab_train import TrainConfig

    mod = importlib.import_module("scripts.train_mjlab")
    for name in (
        "TrainConfig",
        "SuccessReached",
        "deterministic_success",
        "launch_training",
        "run_train",
        "stop_on_success",
        "main",
    ):
        assert hasattr(mod, name), f"scripts/train_mjlab.py missing {name}"

    train_cfg = TrainConfig.from_task("myoElbowPose1D6MRandom-v0")
    rl_cfg = load_rl_cfg("myoElbowPose1D6MRandom-v0")
    assert rl_cfg.experiment_name == train_cfg.agent.experiment_name

    # `myosuite-ppo mjlab <task> --help` forwards argv cleanly to tyro (exits 0)
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["mjlab", "myoElbowPose1D6MRandom-v0", "--help"])
    assert exc_info.value.code == 0


@pytest.mark.tier2
def test_cli_sb3_ppo_smoke_and_legacy_script_exit_zero(tmp_path: Path) -> None:
    pytest.importorskip("stable_baselines3")
    from myosuite_ppo.cli import main as cli_main

    out_stem = tmp_path / "ppo_smoke"
    rc = cli_main(
        [
            "sb3",
            "myoFingerReachFixed-v0",
            "--algo",
            "ppo",
            "--timesteps",
            "64",
            "--eval-episodes",
            "1",
            "--out",
            str(out_stem),
        ]
    )
    assert rc == 0
    assert (tmp_path / "ppo_smoke.zip").is_file()

    # Subprocess execution of scripts/train_sb3.py must exit 0 (not 1 on returned Path)
    script_out = tmp_path / "script_smoke"
    import os

    env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT)}
    proc = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "train_sb3.py"),
            "myoFingerReachFixed-v0",
            "--algo",
            "ppo",
            "--timesteps",
            "64",
            "--eval-episodes",
            "1",
            "--out",
            str(script_out),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "script_smoke.zip").is_file()


@pytest.mark.tier1
@pytest.mark.parametrize("algo_name", ["PPO", "SAC"])
def test_onnx_checkpoint_callback_bundles_vec_normalize(
    algo_name: str, tmp_path: Path
) -> None:
    """A bundle acts on raw observations and resumes with its VecNormalize stats."""
    import numpy as np

    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    sb3 = pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.vec_env import DummyVecEnv

    from myosuite import make_env
    from myosuite.policies.onnx_checkpoint import (
        extract_checkpoint_from_onnx,
        read_onnx_checkpoint_metadata,
        vec_normalize_from_state,
    )
    from myosuite.tests.support.sb3_models import (
        ENV_ID as _ENV_ID,
        raw_observations,
        vec_normalized_model,
    )
    from myosuite_ppo.sb3_callbacks import OnnxCheckpointCallback

    model, venv = vec_normalized_model(algo_name)
    callback = OnnxCheckpointCallback(
        checkpoint_dir=tmp_path,
        task_id=_ENV_ID,
        obs_dim=venv.observation_space.shape[0],
        act_dim=venv.action_space.shape[0],
        save_freq=10**9,
    )
    try:
        model.learn(total_timesteps=16, callback=callback)
        raw = raw_observations(venv)
        expected, _ = model.predict(venv.normalize_obs(raw), deterministic=True)
    finally:
        venv.close()

    bundle = tmp_path / "model_final.onnx"
    sess = ort.InferenceSession(str(bundle), providers=["CPUExecutionProvider"])
    np.testing.assert_allclose(sess.run(None, {"obs": raw})[0], expected, atol=1e-5)

    meta = read_onnx_checkpoint_metadata(bundle)
    assert meta["framework"] == f"sb3-{algo_name.lower()}"
    restored = vec_normalize_from_state(
        meta["metadata"]["vec_normalize"], DummyVecEnv([lambda: make_env(_ENV_ID)])
    )
    try:
        for name in ("obs_rms", "ret_rms"):
            saved, live = getattr(restored, name), getattr(venv, name)
            np.testing.assert_array_equal(saved.mean, live.mean)
            np.testing.assert_array_equal(saved.var, live.var)
            assert saved.count == live.count
        for name in ("norm_obs", "norm_reward", "clip_obs", "clip_reward", "gamma"):
            assert getattr(restored, name) == getattr(venv, name)
        assert restored.epsilon == venv.epsilon
        checkpoint, _, temp_dir = extract_checkpoint_from_onnx(bundle)
        assert temp_dir is not None
        try:
            resumed = getattr(sb3, algo_name).load(checkpoint, env=restored)
        finally:
            temp_dir.cleanup()
        np.testing.assert_allclose(
            resumed.predict(restored.normalize_obs(raw), deterministic=True)[0],
            expected,
            atol=1e-6,
        )
    finally:
        restored.close()
