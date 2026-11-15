# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Smoke test of the CPU training script ``scripts/train_sb3.py``."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.tier2

pytest.importorskip("stable_baselines3")

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train_sb3.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("train_sb3", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_sb3"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("algo", ["ppo", "sac"])
def test_train_sb3_saves_a_loadable_policy(tmp_path: Path, algo: str) -> None:
    train_sb3 = _load()
    out = tmp_path / "model"
    zip_path = train_sb3.main(
        [
            "myoFingerReachFixed-v0",
            "--algo",
            algo,
            "--timesteps",
            "300" if algo == "sac" else "256",
            "--eval-episodes",
            "1",
            "--out",
            str(out),
        ]
    )
    assert zip_path == tmp_path / "model.zip" and zip_path.is_file()
    model = train_sb3.ALGOS[algo].load(zip_path)
    assert model.action_space.shape[0] > 0


@pytest.mark.parametrize("normalize", [False, True])
def test_cli_vectorizers_preserve_seeded_episodes(normalize: bool) -> None:
    script = _load()
    vectors = []
    try:
        for choice in ("dummy", "subproc"):
            args = script.parse_args(
                [
                    "myoElbowPose1D6MRandom-v0",
                    "--n-envs",
                    "2",
                    "--seed",
                    "17",
                    "--vec-env",
                    choice,
                ]
            )
            env = script._make_vec_env(
                args.env_id, args.n_envs, args.seed, args.vec_env
            )
            vectors.append(env)
            assert env.get_attr("np_random_seed") == [17, 18]
        workers = vectors[1].processes
        if normalize:
            vectors = [script.VecNormalize(v, norm_reward=False) for v in vectors]
        left, right = vectors
        np.testing.assert_array_equal(left.reset(), right.reset())
        action = np.zeros((2, *left.action_space.shape), dtype=np.float32)
        endings = 0
        for _ in range(105):
            lo, lr, ld, li = left.step(action)
            ro, rr, rd, ri = right.step(action)
            np.testing.assert_allclose(lo, ro, rtol=0, atol=1e-7)
            np.testing.assert_allclose(lr, rr, rtol=1e-6, atol=1e-6)
            np.testing.assert_array_equal(ld, rd)
            for i in np.flatnonzero(ld):
                endings += 1
                assert li[i]["TimeLimit.truncated"] == ri[i]["TimeLimit.truncated"]
                assert li[i]["episode"]["l"] == ri[i]["episode"]["l"]
                assert li[i]["episode"]["r"] == pytest.approx(ri[i]["episode"]["r"])
                np.testing.assert_allclose(
                    li[i]["terminal_observation"],
                    ri[i]["terminal_observation"],
                    rtol=0,
                    atol=1e-7,
                )
        assert endings >= 2
        if normalize:
            np.testing.assert_array_equal(left.obs_rms.mean, right.obs_rms.mean)
            np.testing.assert_array_equal(left.obs_rms.var, right.obs_rms.var)
    finally:
        for env in vectors:
            env.close()
    assert all(not worker.is_alive() for worker in workers)


def test_subprocess_training_saves_normalization_and_closes_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _load()
    created = []
    build = script._make_vec_env

    def tracked(*args: Any, **kwargs: Any) -> Any:
        env = build(*args, **kwargs)
        created.append(env)
        return env

    monkeypatch.setattr(script, "_make_vec_env", tracked)
    output = script.main(
        [
            "myoElbowPose1D6MRandom-v0",
            "--algo",
            "sac",
            "--timesteps",
            "8",
            "--n-envs",
            "2",
            "--vec-env",
            "subproc",
            "--normalize",
            "--eval-episodes",
            "1",
            "--out",
            str(tmp_path / "policy"),
        ]
    )
    model = script.SAC.load(output)
    assert model.action_space.shape == (6,)
    assert (tmp_path / "policy_vecnormalize.pkl").is_file()
    assert created[0].closed
    assert all(not worker.is_alive() for worker in created[0].processes)


def test_pending_worker_requests_are_reaped_on_training_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _load()
    created = []

    class FailingLearner:
        def __init__(self, policy: str, env: Any, **kwargs: Any) -> None:
            created.append(env)

        def learn(self, **kwargs: Any) -> None:
            env = created[0]
            env.step_async(np.zeros((env.num_envs, *env.action_space.shape)))
            raise RuntimeError("training failed with pending replies")

    monkeypatch.setitem(script.ALGOS, "sac", FailingLearner)
    with pytest.raises(RuntimeError, match="training failed with pending replies"):
        script.main(
            [
                "myoElbowPose1D6MRandom-v0",
                "--algo",
                "sac",
                "--n-envs",
                "2",
                "--vec-env",
                "subproc",
                "--out",
                str(tmp_path / "failed"),
            ]
        )
    assert created[0].closed
    assert all(not worker.is_alive() for worker in created[0].processes)


def test_failed_subprocess_startup_reaps_owned_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load()
    created = []
    terminate = script._terminate_workers

    def tracked(env: Any) -> None:
        created.extend(env.processes)
        terminate(env)

    monkeypatch.setattr(script, "_terminate_workers", tracked)
    with pytest.raises((EOFError, ConnectionError)):
        script._make_vec_env("MissingTrainingEnv-v0", 2, 0, "subproc")
    assert len(created) == 2
    assert all(not worker.is_alive() for worker in created)


def test_failed_evaluation_closes_its_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load()
    env = script.make_env("myoElbowPose1D6MRandom-v0")
    closed = []
    close = env.close

    def tracked_close() -> None:
        closed.append(True)
        close()

    class FailingPolicy:
        def predict(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("evaluation failed")

    monkeypatch.setattr(env, "close", tracked_close)
    monkeypatch.setattr(script, "make_env", lambda env_id: env)
    with pytest.raises(RuntimeError, match="evaluation failed"):
        script.evaluate(FailingPolicy(), "myoElbowPose1D6MRandom-v0", 1, 0)
    assert closed == [True]


@pytest.mark.parametrize("count", ["0", "-2"])
def test_nonpositive_environment_count_is_rejected(count: str) -> None:
    script = _load()
    with pytest.raises(SystemExit):
        script.parse_args(["--n-envs", count])
