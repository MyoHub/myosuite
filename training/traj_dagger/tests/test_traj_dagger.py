# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Unit and smoke tests for multi-teacher trajectory-swap DAgger."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
import torch

from myosuite_traj_dagger import (
    DaggerConfig,
    DaggerRolloutState,
    OfflineBC,
    OfflineBCConfig,
    PersistentDaggerBuffer,
    StudentActorCritic,
    SwapPair,
    SyncEnvPool,
    TeacherRegistry,
    TeacherSpec,
    TrajSwapDagger,
    format_markdown_summary,
    repeat_pair,
    run_dagger_benchmark,
)
from myosuite_traj_dagger.train_mjlab import (
    distill_on_vec_env,
    parse_teacher_cli_specs,
    save_student_artifacts,
)


class _ModeTrackingEnv(gym.Env):
    """Lightweight continuous-control env recording `reset(options=...)` swaps."""

    def __init__(self, obs_dim: int = 6, act_dim: int = 3, horizon: int = 12) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.horizon = horizon
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(act_dim,), dtype=np.float32
        )
        self.mode: int = 0
        self.step_idx: int = 0
        self.state = np.zeros(obs_dim, dtype=np.float32)
        self.reset_modes_seen: list[int] = []
        self.executed_actions: list[np.ndarray] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.mode = int((options or {}).get("mode", 0))
        self.reset_modes_seen.append(self.mode)
        self.step_idx = 0
        center = -1.5 if self.mode == 0 else 1.5
        self.state = self.np_random.normal(
            loc=center, scale=0.25, size=(self.obs_dim,)
        ).astype(np.float32)
        self.state[0] = float(self.mode)
        return self.state.copy(), {"mode": self.mode}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        act = np.asarray(action, dtype=np.float32)
        self.executed_actions.append(act.copy())
        self.step_idx += 1
        self.state = (0.9 * self.state + 0.1 * np.resize(act, self.obs_dim)).astype(
            np.float32
        )
        self.state[0] = float(self.mode)
        reward = -float(np.mean(np.square(act)))
        truncated = self.step_idx >= self.horizon
        return self.state.copy(), reward, False, truncated, {"solved": True}


class _MockTorchVecEnv:
    """Minimal GPU-style PyTorch vectorized env matching `ManagerBasedRlEnv`."""

    def __init__(self, num_envs: int = 4, obs_dim: int = 6, act_dim: int = 3) -> None:
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self._step = 0
        self._obs = torch.zeros((num_envs, obs_dim), dtype=torch.float32)

    def reset(
        self, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        bias = float((options or {}).get("bias", 0.0))
        self._step = 0
        self._obs = torch.randn((self.num_envs, self.obs_dim)) * 0.2 + bias
        return {"actor": self._obs.clone()}, {}

    def step(
        self, actions: torch.Tensor
    ) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        self._step += 1
        self._obs = 0.85 * self._obs + 0.15 * actions.repeat(1, 2)[:, : self.obs_dim]
        rew = -actions.pow(2).mean(dim=-1)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        truncated = torch.full((self.num_envs,), self._step % 6 == 0, dtype=torch.bool)
        return {"actor": self._obs.clone()}, rew, terminated, truncated, {}


@pytest.mark.tier1
@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_rollout_state_sticky_horizon_and_reset_resampling(backend: str) -> None:
    """Sticky decisions hold for `h` steps and resample immediately on `done`."""
    state = DaggerRolloutState.create(
        num_envs=64,
        beta=0.5,
        h_min=4,
        h_max=4,
        backend=backend,  # type: ignore[arg-type]
        seed=42,
    )
    initial_choice = (
        state.using_teacher.clone()
        if isinstance(state.using_teacher, torch.Tensor)
        else state.using_teacher.copy()
    )
    no_dones = (
        torch.zeros(64, dtype=torch.bool)
        if backend == "torch"
        else np.zeros(64, dtype=bool)
    )

    # With h_min = h_max = 4 and no episode resets, steps 0, 1, 2 keep initial_choice.
    for expected_rem in (3, 2, 1):
        state.advance(no_dones, beta=0.5)
        rem_np = (
            state.steps_remaining.cpu().numpy()
            if isinstance(state.steps_remaining, torch.Tensor)
            else state.steps_remaining
        )
        cur_np = (
            state.using_teacher.cpu().numpy()
            if isinstance(state.using_teacher, torch.Tensor)
            else state.using_teacher
        )
        init_np = (
            initial_choice.cpu().numpy()
            if isinstance(initial_choice, torch.Tensor)
            else initial_choice
        )
        assert np.all(rem_np == expected_rem)
        assert np.array_equal(cur_np, init_np)

    # On the 4th advance, countdown hits 0 -> resamples to 4 with beta=0.0 (all False).
    state.advance(no_dones, beta=0.0)
    rem_after = (
        state.steps_remaining.cpu().numpy()
        if isinstance(state.steps_remaining, torch.Tensor)
        else state.steps_remaining
    )
    cur_after = (
        state.using_teacher.cpu().numpy()
        if isinstance(state.using_teacher, torch.Tensor)
        else state.using_teacher
    )
    assert np.all(rem_after == 4)
    assert not np.any(cur_after)

    # Mid-horizon episode done on env 0..7 resamples only those envs immediately.
    dones = np.zeros(64, dtype=bool)
    dones[:8] = True
    state.advance(
        torch.as_tensor(dones) if backend == "torch" else dones,
        beta=1.0,
    )
    cur_done = (
        state.using_teacher.cpu().numpy()
        if isinstance(state.using_teacher, torch.Tensor)
        else state.using_teacher
    )
    rem_done = (
        state.steps_remaining.cpu().numpy()
        if isinstance(state.steps_remaining, torch.Tensor)
        else state.steps_remaining
    )
    assert np.all(cur_done[:8])
    assert not np.any(cur_done[8:])
    assert np.all(rem_done[:8] == 4)
    assert np.all(rem_done[8:] == 3)


@pytest.mark.tier1
def test_persistent_dagger_buffer_ring_and_reservoir_across_swaps() -> None:
    """Short-term ring evicts old chunks while reservoir retains early teacher data."""
    buf = PersistentDaggerBuffer(
        obs_dim=4,
        act_dim=2,
        ring_buffer_size=32,
        long_term_buffer_size=64,
        min_include_prob=0.15,
        seed=7,
    )
    rng = np.random.default_rng(7)

    # Chunk 0 from specialist 0, followed by 6 chunks from specialist 1 that
    # completely overwrite the 32-slot ring buffer.
    obs0 = rng.normal(size=(48, 4)).astype(np.float32)
    act0 = np.full((48, 2), -0.75, dtype=np.float32)
    val0 = np.full((48,), 1.25, dtype=np.float32)
    buf.add(obs0, act0, val0, source_id=0)

    for _ in range(6):
        obs1 = rng.normal(size=(32, 4)).astype(np.float32)
        act1 = np.full((32, 2), 0.75, dtype=np.float32)
        val1 = np.full((32,), -2.5, dtype=np.float32)
        buf.add(obs1, act1, val1, source_id=1)

    assert buf.ring_size == 32
    assert buf.reservoir_size == 64
    assert buf.size == 96
    assert np.all(buf.ring_source_ids() == 1)
    assert 0 in set(buf.reservoir_source_ids().tolist())
    assert {0, 1}.issubset(set(buf.all_source_ids().tolist()))

    batch = buf.sample(40)
    assert batch.obs.shape == (40, 4)
    assert batch.teacher_action.shape == (40, 2)
    assert batch.teacher_value.shape == (40,)
    assert bool(batch.has_value.all())


@pytest.mark.tier1
def test_teacher_registry_deduplication_and_pair_repetition_weighting(
    tmp_path: Path,
) -> None:
    """Repeating a `(teacher, options)` pair `K` times weights sampling `K`x and loads once."""
    student_a = StudentActorCritic(obs_dim=6, act_dim=3, actor_hidden_dims=(16, 16))
    student_b = StudentActorCritic(obs_dim=6, act_dim=3, actor_hidden_dims=(16, 16))
    path_a = tmp_path / "teacher_a.ts"
    path_b = tmp_path / "teacher_b.ts"
    student_a.export_torchscript(path_a)
    student_b.export_torchscript(path_b)

    pairs = parse_teacher_cli_specs(
        [
            f'4*{path_a}::{{"mode": 0}}',
            f'1*{path_b}::{{"mode": 1}}',
        ],
        act_dim=3,
    )
    assert len(pairs) == 5

    registry = TeacherRegistry()
    registry.preload_schedule(pairs)
    assert len(registry) == 2
    assert registry.load_counts[pairs[0].teacher.key] == 1
    assert registry.load_counts[pairs[-1].teacher.key] == 1

    pool = SyncEnvPool([lambda: _ModeTrackingEnv(obs_dim=6, act_dim=3)], seed=0)
    try:
        student = StudentActorCritic(obs_dim=6, act_dim=3, actor_hidden_dims=(16, 16))
        trainer = TrajSwapDagger(
            pool,
            student,
            pairs,
            DaggerConfig(steps_per_chunk=8, epochs_per_chunk=1, seed=123),
            registry=registry,
        )
        counts = {pairs[0].teacher.key: 0, pairs[-1].teacher.key: 0}
        for _ in range(200):
            idx = trainer.select_pair_index()
            counts[pairs[idx].teacher.key] += 1
        # Expected 4:1 ratio (~160 vs ~40).
        assert 135 <= counts[pairs[0].teacher.key] <= 185
        assert registry.load_counts[pairs[0].teacher.key] == 1
    finally:
        pool.close()


@pytest.mark.tier1
def test_traj_swap_dagger_decreases_nll_and_value_loss_across_swaps() -> None:
    """Smaller student learns two mode-specific teachers + values while resetting env carry."""
    obs_dim, act_dim = 6, 3

    def _teacher0_act(obs: np.ndarray | torch.Tensor) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)
        return np.tanh(0.4 * x[..., 1:4] - 0.5).astype(np.float32)

    def _teacher0_val(obs: np.ndarray | torch.Tensor) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)
        return (2.0 + 0.5 * np.sum(x[..., 1:4], axis=-1)).astype(np.float32)

    def _teacher1_act(obs: np.ndarray | torch.Tensor) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)
        return np.tanh(-0.4 * x[..., 1:4] + 0.6).astype(np.float32)

    def _teacher1_val(obs: np.ndarray | torch.Tensor) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)
        return (-1.5 - 0.5 * np.sum(x[..., 1:4], axis=-1)).astype(np.float32)

    pairs = [
        SwapPair(
            TeacherSpec.from_callable(_teacher0_act, key="t0", value_fn=_teacher0_val),
            reset_options={"mode": 0},
        ),
        SwapPair(
            TeacherSpec.from_callable(_teacher1_act, key="t1", value_fn=_teacher1_val),
            reset_options={"mode": 1},
        ),
    ]

    envs = [_ModeTrackingEnv(obs_dim=obs_dim, act_dim=act_dim) for _ in range(2)]
    pool = SyncEnvPool([lambda e=e: e for e in envs], seed=11)
    try:
        torch.manual_seed(11)
        # Smaller student architecture (32,) than typical [128, 128] teacher networks.
        student = StudentActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            actor_hidden_dims=(32, 32),
            critic_hidden_dims=(32, 32),
        )
        cfg = DaggerConfig(
            steps_per_chunk=64,
            beta_start=1.0,
            beta_end=0.1,
            sticky_h_min=3,
            sticky_h_max=6,
            ring_buffer_size=256,
            long_term_buffer_size=256,
            min_include_prob=0.2,
            batch_size=32,
            epochs_per_chunk=8,
            learning_rate=3e-3,
            value_loss_coef=0.5,
            pair_sampling="round_robin",
            seed=11,
        )
        eval_obs_0 = np.full((16, obs_dim), -1.0, dtype=np.float32)
        eval_obs_0[:, 0] = 0.0
        eval_obs_1 = np.full((16, obs_dim), 1.0, dtype=np.float32)
        eval_obs_1[:, 0] = 1.0
        eval_obs = torch.as_tensor(np.concatenate([eval_obs_0, eval_obs_1], axis=0))
        target_act = torch.as_tensor(
            np.concatenate(
                [_teacher0_act(eval_obs_0), _teacher1_act(eval_obs_1)], axis=0
            )
        )
        target_val = torch.as_tensor(
            np.concatenate(
                [_teacher0_val(eval_obs_0), _teacher1_val(eval_obs_1)], axis=0
            )
        )

        with torch.no_grad():
            init_nll = float(student.policy_nll(eval_obs, target_act).item())
            init_val_mse = float(
                torch.mean((student.value(eval_obs) - target_val) ** 2).item()
            )

        trainer = TrajSwapDagger(pool, student, pairs, cfg)
        metrics = trainer.train(total_steps=640)

        with torch.no_grad():
            final_nll = float(student.policy_nll(eval_obs, target_act).item())
            final_val_mse = float(
                torch.mean((student.value(eval_obs) - target_val) ** 2).item()
            )

        assert len(metrics) == 10
        assert metrics[-1].policy_nll < metrics[0].policy_nll
        assert metrics[-1].policy_mse < metrics[0].policy_mse * 0.25
        assert final_nll < init_nll - 2.0
        assert final_val_mse < init_val_mse * 0.1

        # Verify optimizer state and buffer persisted across all 10 chunk swaps.
        first_param = next(student.parameters())
        assert int(trainer.optimizer.state[first_param]["step"].item()) > 10
        assert {0, 1} == set(trainer.buffer.all_source_ids().tolist())
        # Every chunk reset cleanly into alternating modes 0 and 1.
        assert {0, 1} == set(envs[0].reset_modes_seen)
    finally:
        pool.close()


@pytest.mark.tier1
def test_offline_bc_executes_only_teacher_actions_and_fits() -> None:
    """OfflineBC collects strictly with `beta=1.0` (never executing student actions)."""
    const_action = np.array([0.42, -0.31, 0.19], dtype=np.float32)
    pair = SwapPair(
        TeacherSpec.from_callable(
            lambda obs: np.broadcast_to(
                const_action, (*np.asarray(obs).shape[:-1], 3)
            ).copy(),
            key="const_teacher",
        ),
        reset_options={"mode": 1},
    )
    env = _ModeTrackingEnv(obs_dim=6, act_dim=3, horizon=50)
    pool = SyncEnvPool([lambda: env], seed=5)
    try:
        torch.manual_seed(5)
        student = StudentActorCritic(obs_dim=6, act_dim=3, actor_hidden_dims=(32, 32))
        bc = OfflineBC(
            pool,
            student,
            [pair],
            OfflineBCConfig(
                steps_per_chunk=32, batch_size=16, bc_epochs=12, learning_rate=5e-3
            ),
        )
        bc.train(total_steps=96)
        assert len(env.executed_actions) == 96
        for act in env.executed_actions:
            np.testing.assert_allclose(act, const_action, atol=1e-6)

        in_dist_obs = bc.buffer.sample(16).obs.numpy()
        pred = student.act(in_dist_obs, deterministic=True)
        np.testing.assert_allclose(
            pred, np.broadcast_to(const_action, (16, 3)), atol=0.1
        )
    finally:
        pool.close()


@pytest.mark.tier1
def test_vec_env_gpu_style_distill_and_artifact_saving(tmp_path: Path) -> None:
    """`distill_on_vec_env` distills on a Torch vectorized env and writes artifacts."""
    vec_env = _MockTorchVecEnv(num_envs=4, obs_dim=6, act_dim=3)
    teacher_net = StudentActorCritic(obs_dim=6, act_dim=3, actor_hidden_dims=(32, 32))
    pairs = repeat_pair(
        SwapPair(
            TeacherSpec.from_module(teacher_net, key="vec_teacher"),
            reset_options={"bias": 0.5},
        ),
        2,
    )
    student = StudentActorCritic(obs_dim=6, act_dim=3, actor_hidden_dims=(16, 16))
    cfg = DaggerConfig(
        steps_per_chunk=32,
        batch_size=16,
        epochs_per_chunk=4,
        learning_rate=3e-3,
        seed=9,
    )
    reg = TeacherRegistry()
    seen_steps: list[int] = []
    metrics = distill_on_vec_env(
        vec_env,
        student,
        pairs,
        cfg,
        total_steps=96,
        registry=reg,
        on_chunk=lambda cm, _m: seen_steps.append(cm.env_steps),
    )
    assert len(metrics) == 3
    assert seen_steps == [32, 64, 96]
    assert reg.load_counts["vec_teacher"] == 1
    written = save_student_artifacts(
        student, tmp_path / "run_out", metrics=metrics, export_onnx_file=False
    )
    assert written["checkpoint"].is_file()
    assert written["torchscript"].is_file()


@pytest.mark.tier1
def test_student_torchscript_export_matches_eager(tmp_path: Path) -> None:
    """TorchScript export embeds `RunningMeanStd` and matches eager deterministic actions."""
    torch.manual_seed(21)
    student = StudentActorCritic(obs_dim=9, act_dim=6, actor_hidden_dims=(32, 16))
    sample_obs = torch.randn(32, 9) * 3.5 + 1.2
    assert isinstance(student.obs_rms, torch.nn.Module)
    student.obs_rms.update(sample_obs)  # type: ignore[operator]
    student.eval()

    ts_path = student.export_torchscript(tmp_path / "policy.ts")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        loaded = torch.jit.load(str(ts_path)).eval()

    test_obs = torch.randn(11, 9) * 2.0 - 0.7
    with torch.no_grad():
        expected = student(test_obs).numpy()
        actual = loaded(test_obs).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-6)


@pytest.mark.tier1
def test_student_onnx_export_matches_eager(tmp_path: Path) -> None:
    """ONNX export loads standalone in ONNX Runtime and matches eager deterministic actions."""
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    torch.manual_seed(22)
    student = StudentActorCritic(obs_dim=9, act_dim=6, actor_hidden_dims=(32, 16))
    student.obs_rms.update(torch.randn(24, 9) * 2.0)  # type: ignore[operator]
    student.eval()

    onnx_path = student.export_onnx(tmp_path / "policy.onnx")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    test_obs = (np.random.default_rng(3).normal(size=(13, 9)) * 1.5).astype(np.float32)
    with torch.no_grad():
        expected = student(torch.as_tensor(test_obs)).numpy()
    actual = sess.run(None, {"obs": test_obs})[0]
    np.testing.assert_allclose(actual, expected, atol=1e-5)


@pytest.mark.tier1
def test_sb3_teacher_registry_caching_and_value_prediction(tmp_path: Path) -> None:
    """`TeacherRegistry` loads an SB3 `.zip` teacher once and predicts actions + values."""
    sb3 = pytest.importorskip("stable_baselines3")
    env = _ModeTrackingEnv(obs_dim=6, act_dim=3, horizon=10)
    try:
        model = sb3.PPO("MlpPolicy", env, n_steps=16, batch_size=8, seed=0, verbose=0)
        zip_path = tmp_path / "sb3_teacher.zip"
        model.save(str(zip_path))
    finally:
        env.close()

    spec = TeacherSpec.from_path(zip_path, kind="sb3", sb3_algo="PPO")
    pairs = repeat_pair(SwapPair(spec, reset_options={"mode": 0}), 3)
    reg = TeacherRegistry()
    reg.preload_schedule(pairs)
    assert reg.load_counts[spec.key] == 1

    loaded = reg.get_or_load(spec)
    assert loaded.supports_value
    obs = np.zeros((5, 6), dtype=np.float32)
    act, val = loaded.predict_with_value(obs)
    assert isinstance(act, np.ndarray) and act.shape == (5, 3)
    assert isinstance(val, np.ndarray) and val.shape == (5,)


@pytest.mark.tier2
def test_benchmark_smoke_run(tmp_path: Path) -> None:
    """Fast smoke check of `run_dagger_benchmark` on both pose tasks across all 3 arms."""
    out_json = tmp_path / "smoke_bench.json"
    report = run_dagger_benchmark(
        total_steps=512,
        eval_interval=256,
        seeds=(0,),
        n_envs=2,
        n_eval_episodes=2,
        teacher_steps=512,
        out_json=out_json,
    )
    assert out_json.is_file()
    reloaded = json.loads(out_json.read_text(encoding="utf-8"))

    for env_id in ("myoElbowPose1D6MRandom-v0", "myoHandPoseRandom-v0"):
        assert env_id in reloaded["environments"]
        env_entry = reloaded["environments"][env_id]
        assert len(env_entry["teacher_setup"]) == 1
        for arm in ("PPO", "BC", "TrajSwapDAgger"):
            summary = env_entry["arms"][arm]
            assert np.isfinite(summary["final_return_mean"])
            assert 0.0 <= summary["final_success_rate_mean"] <= 1.0
            assert summary["wall_clock_sec_mean"] > 0.0
            assert summary["peak_memory_mb_mean"] > 0.0
            steps = [pt["env_steps"] for pt in summary["curve_summary"]]
            assert steps == [0, 256, 512]

    md = format_markdown_summary(report)
    assert "TrajSwapDAgger" in md
    assert "myoElbowPose1D6MRandom-v0" in md
    assert "myoHandPoseRandom-v0" in md


@pytest.mark.tier1
@pytest.mark.parametrize("env_id", ["myoFingerReachRandom-v0", "myoLegStandRandom-v0"])
def test_target_half_space_wrapper_supports_reach_tasks(env_id: str) -> None:
    """`TargetHalfSpaceWrapper` partitions both workspace- and box-sampled 3D reach tasks."""
    from myosuite import make_env
    from myosuite_traj_dagger.benchmark import TargetHalfSpaceWrapper

    wrapped = TargetHalfSpaceWrapper(make_env(env_id))
    try:
        axis_idx = wrapped.split_index % 3
        site_idx = wrapped.split_index // 3
        target_sid = int(wrapped.unwrapped.target_sids[site_idx])
        low_coords = []
        high_coords = []
        for s in range(10):
            wrapped.reset(seed=s, options={"region": "low"})
            low_coords.append(
                float(wrapped.unwrapped.model.site_pos[target_sid, axis_idx])
            )
            wrapped.reset(seed=s + 100, options={"region": "high"})
            high_coords.append(
                float(wrapped.unwrapped.model.site_pos[target_sid, axis_idx])
            )
        assert max(low_coords) <= wrapped.split_midpoint + 1e-7
        assert min(high_coords) >= wrapped.split_midpoint - 1e-7
    finally:
        wrapped.close()
