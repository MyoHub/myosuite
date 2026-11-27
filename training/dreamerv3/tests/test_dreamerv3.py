# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Unit and smoke tests for myosuite-dreamerv3."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from myosuite_dreamerv3.benchmark import run_benchmark
from myosuite_dreamerv3.env import DreamerEnv, wrap_dreamer_env
from myosuite_dreamerv3.export import (
    evaluate_exported_onnx_stateful,
    export_dreamerv3_onnx,
    export_dreamerv3_torchscript,
    load_dreamerv3_torch_policy,
)
from myosuite_dreamerv3.hf import (
    DREAMERV3_HF_DIR_ENV,
    resolve_dreamerv3_checkpoint,
    stage_hf_release_bundle,
)
from myosuite_dreamerv3.mjlab_driver import MjlabDriver
from myosuite_dreamerv3.mjx_driver import MJXDriver
from myosuite_dreamerv3.train import train_cpu
from myosuite_dreamerv3.train_mjx import (
    build_dreamer_config,
    make_agent_and_factories,
    run_mjx_training_loop,
)


class _ToyMjxState(NamedTuple):
    obs: dict[str, jax.Array]
    reward: jax.Array
    done: jax.Array
    metrics: dict[str, jax.Array]


class _ToyMjxEnv:
    """Minimal JAX-traceable MJX-compatible environment for fast contract verification."""

    action_size = 4

    class _Cfg:
        max_episode_steps = 3

    _config = _Cfg()

    def reset(self, rng: jax.Array) -> _ToyMjxState:
        obs = jax.random.uniform(rng, (6,), minval=-0.1, maxval=0.1)
        return _ToyMjxState(
            obs={"state": obs},
            reward=jnp.float32(0.0),
            done=jnp.float32(0.0),
            metrics={"pose_err": jnp.float32(0.1)},
        )

    def step(self, state: _ToyMjxState, action: jax.Array) -> _ToyMjxState:
        next_obs = state.obs["state"] + 0.5 * jnp.pad(action, (0, 2))
        err = jnp.mean(jnp.square(next_obs))
        done = jnp.where(err > 0.8, jnp.float32(1.0), jnp.float32(0.0))
        return _ToyMjxState(
            obs={"state": next_obs},
            reward=-err,
            done=done,
            metrics={"pose_err": err},
        )


@pytest.mark.tier1
def test_dreamer_env_spaces_and_seeded_reproducibility() -> None:
    env_a = wrap_dreamer_env(DreamerEnv("myoElbowPose1D6MRandom-v0", seed=42))
    env_b = wrap_dreamer_env(DreamerEnv("myoElbowPose1D6MRandom-v0", seed=42))
    try:
        assert "proprio" in env_a.obs_space
        assert env_a.act_space["action"].low.tolist() == [-1.0] * 6
        assert env_a.act_space["action"].high.tolist() == [1.0] * 6

        zero_act = np.zeros(env_a.act_space["action"].shape, np.float32)
        obs_a0 = env_a.step({"action": zero_act, "reset": np.bool_(True)})
        obs_b0 = env_b.step({"action": zero_act, "reset": np.bool_(True)})
        assert bool(obs_a0["is_first"]) is True
        assert bool(obs_a0["is_last"]) is False
        np.testing.assert_allclose(obs_a0["proprio"], obs_b0["proprio"], atol=0.0)

        act = np.linspace(-0.5, 0.5, 6, dtype=np.float32)
        obs_a1 = env_a.step({"action": act, "reset": np.bool_(False)})
        obs_b1 = env_b.step({"action": act, "reset": np.bool_(False)})
        np.testing.assert_allclose(obs_a1["proprio"], obs_b1["proprio"], atol=0.0)
        assert obs_a1["reward"] == pytest.approx(obs_b1["reward"])
    finally:
        env_a.close()
        env_b.close()


@pytest.mark.tier1
def test_dreamer_env_features_and_offscreen_render() -> None:
    env = wrap_dreamer_env(
        DreamerEnv(
            "myoElbowPose1D6MRandom-v0",
            proprio=True,
            image=True,
            size=(64, 64),
            seed=7,
            features=("fatigue",),
        )
    )
    try:
        assert env.obs_space["image"].shape == (64, 64, 3)
        obs = env.step(
            {
                "action": np.zeros(env.act_space["action"].shape, np.float32),
                "reset": np.bool_(True),
            }
        )
        assert obs["image"].shape == (64, 64, 3)
        assert obs["image"].dtype == np.uint8
        assert int(obs["image"].max()) > 0
    finally:
        env.close()


@pytest.mark.tier1
def test_mjx_driver_preserves_terminal_observation_before_reset() -> None:
    driver = MJXDriver(_ToyMjxEnv(), num_envs=2, max_episode_steps=2, seed=0)
    transitions: list[tuple[int, dict[str, np.ndarray]]] = []
    driver.on_step(lambda trn, wid: transitions.append((wid, trn)))

    def _const_policy(carry, obs):
        n = len(obs["is_first"])
        return carry, {"action": np.ones((n, 4), dtype=np.float32)}, {}

    driver.reset()
    driver(_const_policy, steps=8)

    worker0 = [trn for wid, trn in transitions if wid == 0]
    assert [bool(t["is_first"]) for t in worker0] == [True, False, False, True]
    assert [bool(t["is_last"]) for t in worker0] == [False, False, True, False]
    assert np.linalg.norm(worker0[2]["state"]) > 0.5
    assert float(worker0[3]["reward"]) == 0.0


@pytest.mark.tier2
def test_train_cpu_and_mjx_smoke(tmp_path: Path) -> None:
    cpu_res = train_cpu(
        env_id="myoElbowPose1D6MRandom-v0",
        steps=40,
        envs=2,
        train_ratio=32.0,
        size="size1m",
        platform="cpu",
        seed=0,
        logdir=str(tmp_path / "cpu_run"),
        debug=True,
        eval_episodes=2,
    )
    assert "eval" in cpu_res
    assert np.isfinite(cpu_res["eval"]["mean_return"])

    mjx_cfg = build_dreamer_config(
        size="size1m",
        platform="cpu",
        seed=0,
        steps=40,
        envs=2,
        train_ratio=32.0,
        logdir=tmp_path / "mjx_run",
        debug=True,
    )
    (tmp_path / "mjx_run").mkdir(parents=True, exist_ok=True)
    driver = MJXDriver(_ToyMjxEnv(), num_envs=2, max_episode_steps=5, seed=0)
    agent, make_replay, make_stream, make_logger = make_agent_and_factories(
        mjx_cfg, driver.obs_space, driver.act_space
    )
    mjx_res = run_mjx_training_loop(
        driver, agent, make_replay, make_stream, make_logger, mjx_cfg
    )
    assert mjx_res["steps"] >= 40


@pytest.mark.tier2
def test_ppo_vs_dreamerv3_benchmark_smoke(tmp_path: Path) -> None:
    out_json = tmp_path / "bench.json"
    payload = run_benchmark(
        env_ids=("myoElbowPose1D6MRandom-v0",),
        seeds=(0,),
        steps=128,
        eval_episodes=2,
        size="size1m",
        platform="cpu",
        envs=2,
        train_ratio=16.0,
        out_path=out_json,
        logroot=tmp_path / "logs",
        debug=True,
    )
    assert out_json.is_file()
    assert len(payload["summary"]) == 2


@pytest.mark.tier1
def test_mjlab_driver_same_step_autoreset_contract() -> None:
    torch = pytest.importorskip("torch")

    class _ToyMjlabEnv:
        num_envs = 2
        device = "cpu"

        class _ActMgr:
            total_action_dim = 3

        action_manager = _ActMgr()

        def __init__(self) -> None:
            self._t = torch.zeros(2, dtype=torch.long)

        def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
            self._t.zero_()
            return {"actor": torch.full((2, 5), 10.0)}, {}

        def step(self, action: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
            self._t += 1
            term = torch.tensor([self._t[0] >= 2, False])
            trunc = torch.tensor([False, self._t[1] >= 3])
            done = term | trunc
            obs = self._t.float().unsqueeze(1).repeat(1, 5)
            obs[done] = 10.0
            self._t[done] = 0
            return {"actor": obs}, torch.ones(2), term, trunc, {}

        def close(self) -> None:
            pass

    driver = MjlabDriver(_ToyMjlabEnv())
    transitions: list[tuple[int, dict[str, np.ndarray]]] = []
    driver.on_step(lambda trn, wid: transitions.append((wid, trn)))

    def _policy(carry, obs):
        n = len(obs["is_first"])
        return carry, {"action": np.zeros((n, 3), dtype=np.float32)}, {}

    driver.reset()
    driver(_policy, steps=7)

    w0 = [t for wid, t in transitions if wid == 0]
    assert [bool(t["is_first"]) for t in w0[:4]] == [True, False, False, True]
    assert [bool(t["is_last"]) for t in w0[:4]] == [False, False, True, False]
    assert [bool(t["is_terminal"]) for t in w0[:4]] == [False, False, True, False]
    assert float(w0[3]["proprio"][0]) == pytest.approx(10.0)


@pytest.mark.tier1
def test_mjlab_driver_action_range_map_and_sanitize() -> None:
    """[-1, 1] DreamerV3 actions map linearly onto a [0, 1] twin action range (the
    leg-walk twins, like CPU ``NormalizeAction``), and NaN observations from a
    diverged world never reach the replay buffer."""
    torch = pytest.importorskip("torch")
    sent: list[np.ndarray] = []

    class _Term:
        action_dim = 2

        class cfg:  # noqa: N801
            action_range = (0.0, 1.0)
            normalize_act = True

    class _ActMgr:
        total_action_dim = 2
        active_terms = ("muscles",)

        def get_term(self, name: str) -> Any:
            return _Term()

    class _ToyWalkEnv:
        num_envs = 1
        device = "cpu"
        action_manager = _ActMgr()

        def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
            return {"actor": torch.zeros((1, 3))}, {}

        def step(self, action: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
            sent.append(action.numpy().copy())
            obs = torch.tensor([[float("nan"), 1e9, 0.5]])
            return (
                {"actor": obs},
                torch.ones(1),
                torch.zeros(1, dtype=torch.bool),
                torch.zeros(1, dtype=torch.bool),
                {},
            )

        def close(self) -> None:
            pass

    driver = MjlabDriver(_ToyWalkEnv())
    transitions: list[dict[str, np.ndarray]] = []
    driver.on_step(lambda trn, wid: transitions.append(trn))

    def _policy(carry, obs):
        n = len(obs["is_first"])
        return carry, {"action": np.tile([[-1.0, 0.0]], (n, 1)).astype(np.float32)}, {}

    driver.reset()
    driver(_policy, steps=2)
    np.testing.assert_allclose(sent[0], [[0.0, 0.5]])
    assert all(np.isfinite(t["proprio"]).all() for t in transitions)
    assert np.abs(transitions[-1]["proprio"]).max() <= 1e4
    assert driver.sanitized_values >= 2


@pytest.mark.tier1
def test_torch_policy_defaults_to_dreamerv3_silu_activations() -> None:
    """In-memory ``load_dreamerv3_torch_policy`` without disk ``policy_meta.json``
    must default ``enc_act``, ``dyn_act``, and ``pol_act`` to ``"silu"`` (matching
    DreamerV3's ``configs.yaml`` ``defaults``)."""
    pytest.importorskip("torch")
    from myosuite_dreamerv3.export import load_dreamerv3_torch_policy

    rng = np.random.default_rng(0)
    params = {
        "enc/mlp0/kernel": rng.standard_normal((5, 8), dtype=np.float32),
        "enc/mlp0norm/scale": np.ones((8,), dtype=np.float32),
        "dyn/dynin0/kernel": rng.standard_normal((8, 8), dtype=np.float32),
        "dyn/dynin0norm/scale": np.ones((8,), dtype=np.float32),
        "dyn/dynin1/kernel": rng.standard_normal((8, 8), dtype=np.float32),
        "dyn/dynin1norm/scale": np.ones((8,), dtype=np.float32),
        "dyn/dynin2/kernel": rng.standard_normal((2, 8), dtype=np.float32),
        "dyn/dynin2norm/scale": np.ones((8,), dtype=np.float32),
        "dyn/dyngru/kernel": rng.standard_normal((4, 6, 6), dtype=np.float32),
        "dyn/obslogit/kernel": rng.standard_normal((16, 8), dtype=np.float32),
        "pol/head/action/mean/kernel": rng.standard_normal((16, 2), dtype=np.float32),
    }
    pol = load_dreamerv3_torch_policy(params)
    assert (pol.enc_act, pol.dyn_act, pol.pol_act) == ("silu", "silu", "silu")


@pytest.mark.tier2
def test_vision_training_and_checkpoint_resume(tmp_path: Path) -> None:
    logdir = tmp_path / "vision_resume_run"
    phase1 = train_cpu(
        env_id="myoElbowPose1D6MRandom-v0",
        steps=32,
        envs=2,
        train_ratio=32.0,
        size="size1m",
        vision=True,
        platform="cpu",
        seed=0,
        logdir=str(logdir),
        debug=True,
        eval_episodes=0,
    )
    assert (logdir / "ckpt" / "latest").is_file()
    phase2 = train_cpu(
        env_id="myoElbowPose1D6MRandom-v0",
        steps=64,
        envs=2,
        train_ratio=32.0,
        size="size1m",
        vision=True,
        platform="cpu",
        seed=0,
        logdir=str(logdir),
        debug=True,
        eval_episodes=1,
    )
    assert phase1["steps"] == 32
    assert phase2["steps"] == 64
    assert np.isfinite(phase2["eval"]["mean_return"])


@pytest.mark.tier2
def test_torchscript_and_onnx_export_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    ort = pytest.importorskip("onnxruntime")
    from embodied.jax import outs as jax_outs
    from myosuite.utils.export_onnx import verify_onnx_on_cpu

    logdir = tmp_path / "export_run"
    onnx_path = tmp_path / "dreamer_policy.onnx"
    ts_path = tmp_path / "dreamer_policy.pt"
    train_cpu(
        env_id="myoElbowPose1D6MRandom-v0",
        steps=16,
        envs=2,
        train_ratio=16.0,
        size="size1m",
        platform="cpu",
        seed=0,
        logdir=str(logdir),
        export_onnx=str(onnx_path),
        export_torchscript=str(ts_path),
        debug=True,
    )
    assert onnx_path.is_file()
    assert Path(f"{onnx_path}.manifest.json").is_file()
    assert ts_path.is_file()

    pt_pol = load_dreamerv3_torch_policy(logdir)
    ts_mod = export_dreamerv3_torchscript(pt_pol)
    export_dreamerv3_onnx(pt_pol, onnx_path, env_id="myoElbowPose1D6MRandom-v0")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    monkeypatch.setattr(jax_outs.OneHot, "sample", lambda self, *a, **kw: self.pred())
    monkeypatch.setattr(jax_outs.Normal, "sample", lambda self, *a, **kw: self.pred())

    cfg = build_dreamer_config(
        size="size1m",
        platform="cpu",
        seed=0,
        steps=16,
        envs=2,
        logdir=str(logdir),
        debug=True,
    ).update({"jax.compute_dtype": "float32"})
    probe = wrap_dreamer_env(DreamerEnv("myoElbowPose1D6MRandom-v0", seed=0))
    agent, _, _, _ = make_agent_and_factories(
        cfg, dict(probe.obs_space), dict(probe.act_space)
    )
    probe.close()
    import elements

    ckpt_dir = logdir / "ckpt"
    latest_sub = ckpt_dir / (ckpt_dir / "latest").read_text(encoding="utf-8").strip()
    elements.checkpoint.load(
        str(latest_sub), dict(agent=lambda d: agent.load(d, regex=r".*"))
    )

    rng = np.random.default_rng(123)
    jax_carry = agent.init_policy(2)
    d_pt, s_pt, a_pt = pt_pol.initial_state(2)
    d_ox = np.zeros((2, pt_pol.deter_dim), dtype=np.float32)
    s_ox = np.zeros((2, pt_pol.stoch_dim, pt_pol.classes), dtype=np.float32)
    a_ox = np.zeros((2, pt_pol.act_dim), dtype=np.float32)

    for t in range(10):
        obs_np = rng.standard_normal((2, pt_pol.obs_dim)).astype(np.float32)
        first_bool = np.array([t in (0, 4), t in (0, 7)], dtype=bool)
        first_f32 = first_bool.astype(np.float32)
        jax_obs = {
            "proprio": obs_np,
            "is_first": first_bool,
            "is_last": np.zeros(2, dtype=bool),
            "is_terminal": np.zeros(2, dtype=bool),
            "reward": np.zeros(2, dtype=np.float32),
        }
        jax_carry, jax_acts, _ = agent.policy(jax_carry, jax_obs, mode="eval")
        act_ref = np.asarray(jax_acts["action"], dtype=np.float32)

        with torch.no_grad():
            act_pt, d_pt_next, s_pt_next = pt_pol(
                torch.from_numpy(obs_np),
                d_pt,
                s_pt,
                a_pt,
                torch.from_numpy(first_f32),
            )
            act_ts, _, _ = ts_mod(
                torch.from_numpy(obs_np),
                d_pt,
                s_pt,
                a_pt,
                torch.from_numpy(first_f32),
            )
        d_pt, s_pt, a_pt = d_pt_next, s_pt_next, act_pt
        act_ox, d_ox, s_ox = sess.run(
            ["action", "next_deter", "next_stoch"],
            {
                "obs": obs_np,
                "deter": d_ox,
                "stoch": s_ox,
                "prevact": a_ox,
                "is_first": first_f32,
            },
        )
        a_ox = act_ox

        np.testing.assert_allclose(act_pt.numpy(), act_ref, atol=1e-5)
        np.testing.assert_allclose(act_ts.numpy(), act_ref, atol=1e-5)
        np.testing.assert_allclose(act_ox, act_ref, atol=1e-5)

    stateless_ver = verify_onnx_on_cpu(
        onnx_path,
        n_steps=5,
        env_id="myoElbowPose1D6MRandom-v0",
        allow_provenance_mismatch=False,
    )
    stateful_ver = evaluate_exported_onnx_stateful(
        onnx_path, env_id="myoElbowPose1D6MRandom-v0", episodes=2
    )
    assert stateless_ver["n_steps"] == 5
    assert np.isfinite(stateful_ver["mean_return"])


@pytest.mark.tier1
def test_pinned_hf_checkpoint_resolution_and_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dummy_onnx = tmp_path / "dummy.onnx"
    dummy_onnx.write_bytes(b"onnx-test-bytes")
    dummy_log = tmp_path / "bench.json"
    dummy_log.write_text('{"ok": true}', encoding="utf-8")

    staged = tmp_path / "hf_bundle"
    manifest_path = stage_hf_release_bundle(
        staged,
        checkpoints={"myoElbowPose1D6MRandom-v0": dummy_onnx},
        eval_logs={"elbow_bench": dummy_log},
        metadata={"seed_count": 3},
    )
    assert manifest_path.is_file()

    with pytest.raises(ValueError, match="pinned"):
        resolve_dreamerv3_checkpoint(
            "hf://M-Colley/myosuite-dreamerv3/checkpoints/myoElbowPose1D6MRandom-v0"
        )
    with pytest.raises(ValueError, match="moving branch"):
        resolve_dreamerv3_checkpoint(
            "hf://M-Colley/myosuite-dreamerv3@main/checkpoints/myoElbowPose1D6MRandom-v0"
        )

    monkeypatch.setenv(DREAMERV3_HF_DIR_ENV, str(staged))
    resolved = resolve_dreamerv3_checkpoint(
        "hf://M-Colley/myosuite-dreamerv3@v0.1.0/checkpoints/myoElbowPose1D6MRandom-v0/dummy.onnx"
    )
    assert resolved.read_bytes() == b"onnx-test-bytes"


@pytest.mark.tier2
def test_benchmark_learning_curves_and_zero_deprecations(tmp_path: Path) -> None:
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        payload = run_benchmark(
            env_ids=("myoElbowPose1D6MRandom-v0",),
            seeds=(0,),
            steps=128,
            eval_episodes=1,
            size="size1m",
            platform="cpu",
            envs=2,
            train_ratio=16.0,
            curve_points=3,
            out_path=tmp_path / "curve_bench.json",
            logroot=tmp_path / "curve_logs",
            debug=True,
        )

    bad_deprecations = [
        str(w.message)
        for w in caught
        if "jax.set_mesh" in str(w.message) or "optax.global_norm" in str(w.message)
    ]
    assert bad_deprecations == []
    assert len(payload["learning_curves"]) == 2
    for series in payload["learning_curves"]:
        steps = [pt["steps"] for pt in series["points"]]
        assert steps == sorted(steps)
        assert steps[0] == 0
        assert steps[-1] == 128
