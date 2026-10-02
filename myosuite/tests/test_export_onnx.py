# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""ONNX exports must reproduce the framework's own deterministic policy.

Every export is one self-contained ``.onnx`` file that takes the raw observation.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import myosuite
from myosuite.tests.support.sb3_models import (
    ENV_ID as _ENV_ID,
    raw_observations,
    vec_normalized_model,
)
from myosuite.utils import gym

pytestmark = pytest.mark.tier1


def _cp1252_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Emulate a Windows console whose stdout is piped (cp1252, strict)."""
    monkeypatch.setattr(
        sys, "stdout", io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    )


def _load_moved_onnx(onnx_path: Path, ort: Any) -> Any:
    """Move the export alone to another directory and open it there.

    An export whose weights live in an external ``.onnx.data`` sidecar fails here,
    as it does for a browser runtime, a W&B upload or a bundle written elsewhere.
    """
    assert sorted(p.name for p in onnx_path.parent.iterdir()) == [onnx_path.name]
    moved = onnx_path.parent.parent / "moved" / onnx_path.name
    moved.parent.mkdir()
    onnx_path.replace(moved)
    return ort.InferenceSession(str(moved), providers=["CPUExecutionProvider"])


@pytest.mark.parametrize("state_dependent_std", [False, True])
def test_rslrl_onnx_export_is_self_contained_and_matches_the_policy(
    state_dependent_std: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rsl_rl export loads on its own and equals ``load_rslrl_policy``."""
    pytest.importorskip("rsl_rl")
    ort = pytest.importorskip("onnxruntime")
    from rsl_rl.modules import MLP, EmpiricalNormalization

    from myosuite.utils.export_onnx import export_rslrl_to_onnx
    from myosuite.utils.rslrl_policy import load_rslrl_policy

    obs_dim, act_dim = 23, 6
    torch.manual_seed(0)
    mlp = MLP(
        obs_dim, [2, act_dim] if state_dependent_std else act_dim, [32, 16], "elu"
    )
    normalizer = EmpiricalNormalization(obs_dim)
    normalizer._mean.copy_(torch.randn(1, obs_dim) * 3)
    normalizer._var.copy_(torch.rand(1, obs_dim) * 5 + 0.01)
    normalizer._std.copy_(normalizer._var.sqrt())
    actor_state = {f"mlp.{k}": v for k, v in mlp.state_dict().items()}
    actor_state.update(
        {f"obs_normalizer.{k}": v for k, v in normalizer.state_dict().items()}
    )
    checkpoint = tmp_path / "model_0.pt"
    torch.save({"actor_state_dict": actor_state}, checkpoint)

    onnx_path = tmp_path / "export" / "policy.onnx"
    onnx_path.parent.mkdir()
    _cp1252_stdout(monkeypatch)
    export_rslrl_to_onnx(checkpoint, onnx_path, obs_dim, act_dim)
    monkeypatch.undo()

    sess = _load_moved_onnx(onnx_path, ort)
    obs = torch.randn(64, obs_dim) * 4
    with torch.no_grad():
        expected = load_rslrl_policy(checkpoint, act_dim)(obs).numpy()
    actual = sess.run(None, {"obs": obs.numpy()})[0]
    np.testing.assert_allclose(actual, expected, atol=1e-5)


def _orbax_actor_params(obs_dim: int, act_dim: int, rng: np.random.Generator) -> dict:
    """Random Flax-layout residual actor: a projected block, an identity block, a tail."""

    def dense(n_in: int, n_out: int) -> dict:
        return {
            "kernel": rng.normal(0.0, n_in**-0.5, (n_in, n_out)).astype(np.float32),
            "bias": rng.normal(0.0, 0.1, n_out).astype(np.float32),
        }

    def layer_norm(n: int) -> dict:
        return {
            "scale": rng.uniform(0.5, 1.5, n).astype(np.float32),
            "bias": rng.normal(0.0, 0.1, n).astype(np.float32),
        }

    actor: dict = {}
    for idx, (n_in, n_out) in enumerate([(obs_dim, 32), (32, 32)]):
        actor[f"block{idx}_layer0_dense"] = dense(n_in, 48)
        actor[f"block{idx}_layer0_ln"] = layer_norm(48)
        actor[f"block{idx}_layer1_dense"] = dense(48, n_out)
        actor[f"block{idx}_layer1_ln"] = layer_norm(n_out)
        actor[f"res_gate_{idx}"] = np.float32(rng.normal())
        if n_in != n_out:
            actor[f"block{idx}_proj"] = dense(n_in, n_out)
    actor["tail_dense"] = dense(32, 16)
    actor["tail_ln"] = layer_norm(16)
    actor["output"] = dense(16, act_dim)
    return {"actor": actor}


def test_orbax_onnx_export_is_self_contained_and_matches_the_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The MuscleMimic (Orbax) export equals the NumPy reference actor."""
    ort = pytest.importorskip("onnxruntime")
    from myosuite.integrations.musclemimic import fullbody_local_policy as local
    from myosuite.utils.export_onnx import export_orbax_to_onnx

    obs_dim, act_dim = 19, 7
    rng = np.random.default_rng(0)
    artifacts = local.LocalPolicyArtifacts(
        params=_orbax_actor_params(obs_dim, act_dim, rng),
        obs_mean=rng.normal(0.0, 2.0, obs_dim).astype(np.float32),
        obs_var=rng.uniform(0.01, 4.0, obs_dim).astype(np.float32),
        obs_count=np.float32(100.0),
        obs_dim=obs_dim,
        action_dim=act_dim,
    )
    monkeypatch.setattr(local, "has_local_policy_artifacts", lambda _root: True)
    monkeypatch.setattr(local, "load_local_policy_artifacts", lambda _root: artifacts)

    onnx_path = tmp_path / "export" / "policy.onnx"
    _cp1252_stdout(monkeypatch)
    export_orbax_to_onnx(tmp_path / "checkpoint", onnx_path)
    monkeypatch.undo()

    sess = _load_moved_onnx(onnx_path, ort)
    obs = rng.normal(0.0, 3.0, (32, obs_dim)).astype(np.float32)
    norm_obs = (obs - artifacts.obs_mean) / np.sqrt(artifacts.obs_var + 1e-8)
    expected = np.clip(local._actor_forward(artifacts.params, norm_obs), -1.0, 1.0)
    actual = sess.run(None, {"obs": obs})[0]
    np.testing.assert_allclose(actual, expected, atol=1e-5)


@pytest.mark.parametrize("algo_name", ["PPO", "SAC", "TD3"])
@pytest.mark.parametrize("action_low", [-1.0, 0.0], ids=["box-1to1", "box0to1"])
def test_sb3_onnx_export_matches_predict(
    algo_name: str, action_low: float, tmp_path: Path
) -> None:
    """ONNX actions equal model.predict(deterministic=True) on the env's Box.

    MyoSuite muscle envs expose Box(-1, 1); the [0, 1] case (an env
    without action normalisation) exercises the rescaling of squashed actions.
    """
    sb3 = pytest.importorskip("stable_baselines3")
    ort = pytest.importorskip("onnxruntime")
    from myosuite.utils.export_onnx import export_sb3_to_onnx

    myosuite.register_all_envs()
    env = gym.make(_ENV_ID)
    if action_low != -1.0:
        env = gym.wrappers.RescaleAction(
            env, min_action=np.float32(action_low), max_action=np.float32(1.0)
        )
    try:
        obs = np.stack([env.reset(seed=s)[0] for s in range(16)]).astype(np.float32)
        algo = getattr(sb3, algo_name)
        kwargs = {} if algo_name == "PPO" else {"buffer_size": 1}
        model = algo("MlpPolicy", env, seed=0, device="cpu", **kwargs)
    finally:
        env.close()

    # Spread the output bias so actions fall below, inside and above the bounds.
    head = model.policy.action_net if algo_name == "PPO" else model.policy.actor.mu
    out_layer = [m for m in head.modules() if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        out_layer.bias.copy_(torch.linspace(-2.0, 2.0, out_layer.out_features))
    expected, _ = model.predict(obs, deterministic=True)
    assert (expected < 0).any() == (action_low < 0)

    model.save(tmp_path / "model.zip")
    onnx_path = tmp_path / "model.onnx"
    export_sb3_to_onnx(
        tmp_path / "model.zip", onnx_path, obs.shape[1], expected.shape[1]
    )
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = sess.run(None, {"obs": obs})[0]
    np.testing.assert_allclose(actual, expected, atol=1e-5)


@pytest.mark.parametrize("algo_name", ["PPO", "SAC"])
def test_sb3_onnx_export_folds_vec_normalize(algo_name: str, tmp_path: Path) -> None:
    """On raw observations the export equals ``predict(normalize_obs(raw))``."""
    ort = pytest.importorskip("onnxruntime")
    from myosuite.utils.export_onnx import export_sb3_to_onnx

    model, venv = vec_normalized_model(algo_name)
    try:
        raw = raw_observations(venv)
        assert np.abs(venv.normalize_obs(raw)).max() == venv.clip_obs
        expected, _ = model.predict(venv.normalize_obs(raw), deterministic=True)
        model.save(tmp_path / "model.zip")
        venv.save(tmp_path / "vecnormalize.pkl")
    finally:
        venv.close()

    onnx_path = tmp_path / "model.onnx"
    export_sb3_to_onnx(
        tmp_path / "model.zip",
        onnx_path,
        raw.shape[1],
        expected.shape[1],
        vec_normalize=tmp_path / "vecnormalize.pkl",
    )
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    np.testing.assert_allclose(sess.run(None, {"obs": raw})[0], expected, atol=1e-5)


@pytest.mark.parametrize("algo_name", ["PPO", "SAC"])
def test_onnx_checkpoint_callback_bundles_vec_normalize(
    algo_name: str, tmp_path: Path
) -> None:
    """A bundle acts on raw observations and resumes with its VecNormalize stats."""
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    sb3 = pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.vec_env import DummyVecEnv

    from myosuite.utils.onnx_checkpoint import (
        extract_checkpoint_from_onnx,
        read_onnx_checkpoint_metadata,
        vec_normalize_from_state,
    )
    from myosuite.utils.sb3_callbacks import OnnxCheckpointCallback

    model, venv = vec_normalized_model(algo_name)
    callback = OnnxCheckpointCallback(
        checkpoint_dir=tmp_path,
        task_id=_ENV_ID,
        obs_dim=venv.observation_space.shape[0],
        act_dim=venv.action_space.shape[0],
        save_freq=10**9,
    )
    try:
        model.learn(total_timesteps=16, callback=callback)  # updates the statistics
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
        meta["metadata"]["vec_normalize"], DummyVecEnv([lambda: gym.make(_ENV_ID)])
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


def test_vec_normalize_state_round_trips_without_obs_normalization() -> None:
    """Reward-only normalization (no ``obs_rms``) survives the JSON round trip."""
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from myosuite.utils.onnx_checkpoint import (
        get_vec_normalize_state,
        vec_normalize_from_state,
    )

    myosuite.register_all_envs()
    venv = VecNormalize(
        DummyVecEnv([lambda: gym.make(_ENV_ID)]), norm_obs=False, gamma=0.9
    )
    venv.ret_rms.mean, venv.ret_rms.var, venv.ret_rms.count = 0.25, 3.5, 42.0
    state = json.loads(json.dumps(get_vec_normalize_state(venv)))
    venv.close()
    assert state["obs_rms"] is None

    restored = vec_normalize_from_state(state, DummyVecEnv([lambda: gym.make(_ENV_ID)]))
    try:
        assert not restored.norm_obs and restored.norm_reward
        assert restored.gamma == 0.9
        rms = restored.ret_rms
        assert (float(rms.mean), float(rms.var), rms.count) == (0.25, 3.5, 42.0)
    finally:
        restored.close()


def test_verify_onnx_on_cpu_feeds_raw_observations(tmp_path: Path) -> None:
    """``verify_onnx_on_cpu`` reproduces the VecNormalize -> predict rollout."""
    pytest.importorskip("onnxruntime")
    from myosuite.utils.export_onnx import export_sb3_to_onnx, verify_onnx_on_cpu

    model, venv = vec_normalized_model("PPO")
    try:
        model.save(tmp_path / "model.zip")
        export_sb3_to_onnx(
            tmp_path / "model.zip",
            tmp_path / "model.onnx",
            venv.observation_space.shape[0],
            venv.action_space.shape[0],
            vec_normalize=venv,
        )
        env = venv.envs[0]
        obs, _ = env.reset(seed=3)
        total_reward = 0.0
        for _ in range(10):
            action, _ = model.predict(venv.normalize_obs(obs), deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
    finally:
        venv.close()

    metrics = verify_onnx_on_cpu(
        tmp_path / "model.onnx", n_steps=10, seed=3, env_id=_ENV_ID
    )
    assert metrics["total_reward"] == pytest.approx(total_reward, rel=1e-4)
