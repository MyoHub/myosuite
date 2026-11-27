# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Side-by-side 3-arm benchmark: PPO vs. Offline BC vs. Multi-Teacher TrajSwapDAgger.

Follows `docs/wiki/training-guidelines.md`:
- Evaluates on unmodified registered environments (`myoElbowPose1D6MRandom-v0` and
  `myoHandPoseRandom-v0`) via `myosuite.make_env`.
- Compares 3 arms (`PPO`, `BC`, `TrajSwapDAgger`) at identical interactive/transition
  step budgets across seeds `(0, 1, 2)`, reporting `mean +- std` return and solved
  rate over 20 seeded deterministic evaluation episodes, wall-clock time, peak
  `tracemalloc` memory (MB), step-matched learning curves, and transparent teacher
  pre-training metadata.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
import tracemalloc
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import torch
from myosuite import make_env
from torch import nn
from torch.nn import functional as F

from myosuite_traj_dagger.dagger import (
    DaggerConfig,
    OfflineBC,
    OfflineBCConfig,
    RunningMeanStd,
    StudentActorCritic,
    SwapPair,
    SyncEnvPool,
    TeacherRegistry,
    TeacherSpec,
    TrajSwapDagger,
)

RegionName = Literal["full", "low", "high"]
DEFAULT_BENCHMARK_ENVS: tuple[str, str] = (
    "myoElbowPose1D6MRandom-v0",
    "myoHandPoseRandom-v0",
)


class TargetHalfSpaceWrapper(gym.Wrapper):
    """Partition a pose or reach task's target distribution into two disjoint half-spaces.

    Splits the widest target joint range (`_target_jnt_range`) or 3D site reach box
    (`target_reach_range`, plus `_workspace_points` when workspace-sampled) at its
    midpoint while leaving all other target coordinates unchanged, so a 50/50 mixture
    of `"low"` and `"high"` covers the unmodified `Random` environment without altering
    observations, actions, or rewards.
    """

    def __init__(self, env: gym.Env, *, default_region: RegionName = "full") -> None:
        super().__init__(env)
        base = env.unwrapped
        raw_jnt_range = getattr(base, "_target_jnt_range", None)
        raw_reach_range = getattr(base, "target_reach_range", None)
        self._reach_site_names: list[str] = []
        self._full_workspace_points: np.ndarray | None = None

        if raw_jnt_range is not None:
            self._mode = "pose"
            self.full_range = np.asarray(raw_jnt_range, dtype=np.float64).copy()
            spans = self.full_range[:, 1] - self.full_range[:, 0]
            self.split_index = int(np.argmax(spans))
            lo = float(self.full_range[self.split_index, 0])
            hi = float(self.full_range[self.split_index, 1])
            self.split_midpoint = 0.5 * (lo + hi)
            jnt_ids = getattr(base, "_target_jnt_ids", None)
            if jnt_ids and hasattr(base, "model"):
                self.split_joint_name = str(
                    base.model.joint(int(jnt_ids[self.split_index])).name
                )
            else:
                self.split_joint_name = f"joint_{self.split_index}"
        elif isinstance(raw_reach_range, dict) and raw_reach_range:
            self._mode = "reach"
            self._reach_site_names = list(raw_reach_range.keys())
            rows: list[list[float]] = []
            for sname in self._reach_site_names:
                lo_vec, hi_vec = raw_reach_range[sname]
                for lo_v, hi_v in zip(lo_vec, hi_vec):
                    rows.append([float(lo_v), float(hi_v)])
            self.full_range = np.asarray(rows, dtype=np.float64)
            ws_pts = getattr(base, "_workspace_points", None)
            if isinstance(ws_pts, np.ndarray) and ws_pts.ndim == 3 and len(ws_pts) > 1:
                self._full_workspace_points = ws_pts.copy()
                flat_ws = self._full_workspace_points.reshape(len(ws_pts), -1)
                spans = flat_ws.max(axis=0) - flat_ws.min(axis=0)
                self.split_index = int(np.argmax(spans))
                self.split_midpoint = float(np.median(flat_ws[:, self.split_index]))
            else:
                spans = self.full_range[:, 1] - self.full_range[:, 0]
                self.split_index = int(np.argmax(spans))
                lo = float(self.full_range[self.split_index, 0])
                hi = float(self.full_range[self.split_index, 1])
                self.split_midpoint = 0.5 * (lo + hi)
            site_idx = self.split_index // 3
            axis_name = "xyz"[self.split_index % 3]
            self.split_joint_name = f"{self._reach_site_names[site_idx]}.{axis_name}"
        else:
            raise ValueError(
                f"Environment {env} has neither `_target_jnt_range` nor `target_reach_range`."
            )
        self.default_region: RegionName = default_region

    def _apply_region(self, region: RegionName) -> None:
        if region not in ("full", "low", "high"):
            raise ValueError(f"Unknown target region {region!r}.")
        rng_arr = self.full_range.copy()
        if region == "low":
            rng_arr[self.split_index, 1] = self.split_midpoint
        elif region == "high":
            rng_arr[self.split_index, 0] = self.split_midpoint

        base = self.env.unwrapped
        if self._mode == "pose":
            base._target_jnt_range = rng_arr  # type: ignore[attr-defined]
            return

        new_reach: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {}
        for idx, sname in enumerate(self._reach_site_names):
            block = rng_arr[3 * idx : 3 * (idx + 1)]
            new_reach[sname] = (
                tuple(float(v) for v in block[:, 0]),
                tuple(float(v) for v in block[:, 1]),
            )
        base.target_reach_range = new_reach  # type: ignore[attr-defined]
        if self._full_workspace_points is not None:
            site_idx = self.split_index // 3
            axis_idx = self.split_index % 3
            coords = self._full_workspace_points[:, site_idx, axis_idx]
            if region == "low":
                mask = coords <= self.split_midpoint
            elif region == "high":
                mask = coords >= self.split_midpoint
            else:
                mask = np.ones(len(coords), dtype=bool)
            base._workspace_points = self._full_workspace_points[mask]  # type: ignore[attr-defined]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        region: RegionName = self.default_region
        forward_opts: dict[str, Any] | None = None
        if options is not None:
            opts_copy = dict(options)
            if "region" in opts_copy:
                region = opts_copy.pop("region")
            if opts_copy:
                forward_opts = opts_copy
        self._apply_region(region)
        return self.env.reset(seed=seed, options=forward_opts)


@dataclass(frozen=True)
class EvalSnapshot:
    """Evaluation checkpoint metrics on the unmodified task distribution."""

    env_steps: int
    elapsed_sec: float
    mean_return: float
    std_return: float
    success_rate: float
    any_step_success_rate: float


def evaluate_student_policy(
    env_id: str,
    policy: StudentActorCritic,
    *,
    n_episodes: int = 20,
    seed_base: int = 10_000,
    region: RegionName = "full",
) -> tuple[float, float, float, float]:
    """Evaluate deterministic `policy` over `n_episodes` seeded episodes.

    When `region == "full"` (the benchmark evaluation mode), runs on the bare,
    unwrapped `make_env(env_id)`.

    Returns:
        Tuple `(mean_return, std_return, final_solved_rate, any_step_solved_rate)`.
    """
    raw_env = make_env(env_id)
    env: gym.Env = (
        raw_env
        if region == "full"
        else TargetHalfSpaceWrapper(raw_env, default_region=region)
    )
    policy.eval()
    returns: list[float] = []
    final_solved: list[float] = []
    any_solved: list[float] = []

    try:
        for ep in range(int(n_episodes)):
            obs, _ = env.reset(seed=int(seed_base + ep))
            done = False
            ep_ret = 0.0
            last_solved = False
            ever_solved = False
            while not done:
                act = np.asarray(
                    policy.act(np.asarray(obs, dtype=np.float32), deterministic=True),
                    dtype=np.float32,
                )
                obs, rwd, terminated, truncated, info = env.step(act)
                ep_ret += float(rwd)
                step_solved = bool(np.asarray(info.get("solved", False)).item())
                last_solved = step_solved
                ever_solved = ever_solved or step_solved
                done = bool(terminated or truncated)
            returns.append(ep_ret)
            final_solved.append(1.0 if last_solved else 0.0)
            any_solved.append(1.0 if ever_solved else 0.0)
    finally:
        env.close()

    return (
        float(np.mean(returns)),
        float(np.std(returns)),
        float(np.mean(final_solved)),
        float(np.mean(any_solved)),
    )


@dataclass
class PPOConfig:
    """Hyperparameters for the matched-architecture online PPO trainer."""

    rollout_steps_per_env: int = 128
    ppo_epochs: int = 4
    batch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.005
    max_grad_norm: float = 0.5
    seed: int = 0
    device: str = "cpu"


def train_online_ppo(
    env_pool: SyncEnvPool,
    policy: StudentActorCritic,
    total_steps: int,
    cfg: PPOConfig,
    *,
    eval_interval: int | None = None,
    eval_callback: Callable[[int, StudentActorCritic], None] | None = None,
) -> StudentActorCritic:
    """Train `StudentActorCritic` with clipped PPO + GAE for `total_steps`."""
    dev = torch.device(cfg.device)
    policy = policy.to(dev)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate, eps=1e-5)
    rng = np.random.default_rng(cfg.seed)
    torch_rng = torch.Generator(device=dev)
    torch_rng.manual_seed(int(cfg.seed))

    obs = env_pool.reset_all()
    env_steps = 0
    next_eval = (
        int(eval_interval) if eval_interval is not None and eval_interval > 0 else None
    )
    if eval_callback is not None:
        eval_callback(0, policy)

    num_envs = env_pool.num_envs
    obs_dim = env_pool.obs_dim
    act_dim = env_pool.act_dim

    while env_steps < total_steps:
        rem = total_steps - env_steps
        if next_eval is not None:
            rem = min(rem, next_eval - env_steps)
        horizon = max(1, min(cfg.rollout_steps_per_env, math.ceil(rem / num_envs)))

        buf_obs = np.zeros((horizon, num_envs, obs_dim), dtype=np.float32)
        buf_act = np.zeros((horizon, num_envs, act_dim), dtype=np.float32)
        buf_logp = np.zeros((horizon, num_envs), dtype=np.float32)
        buf_rew = np.zeros((horizon, num_envs), dtype=np.float32)
        buf_done = np.zeros((horizon, num_envs), dtype=np.float32)
        buf_val = np.zeros((horizon, num_envs), dtype=np.float32)

        policy.eval()
        steps_recorded = 0
        actual_t = 0
        for t in range(horizon):
            if steps_recorded >= rem:
                break
            if isinstance(policy.obs_rms, RunningMeanStd):
                policy.obs_rms.update(obs)
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=dev)
            with torch.no_grad():
                mean, log_std = policy.distribution_params(obs_t)
                std = torch.exp(log_std)
                noise = torch.randn(
                    mean.shape,
                    dtype=mean.dtype,
                    device=dev,
                    generator=torch_rng,
                )
                raw_act = mean + std * noise
                dist = torch.distributions.Normal(mean, std)
                logp = dist.log_prob(raw_act).sum(dim=-1)
                val = policy.value(obs_t)

            raw_act_np = raw_act.cpu().numpy().astype(np.float32, copy=False)
            next_obs, rew, dones, _ = env_pool.step(raw_act_np)

            buf_obs[t] = obs
            buf_act[t] = raw_act_np
            buf_logp[t] = logp.cpu().numpy()
            buf_rew[t] = rew
            buf_done[t] = dones.astype(np.float32)
            buf_val[t] = val.cpu().numpy()

            obs = next_obs
            take = min(num_envs, rem - steps_recorded)
            steps_recorded += take
            env_steps += take
            actual_t = t + 1

        with torch.no_grad():
            next_val = (
                policy.value(torch.as_tensor(obs, dtype=torch.float32, device=dev))
                .cpu()
                .numpy()
            )

        adv = np.zeros((actual_t, num_envs), dtype=np.float32)
        last_gae = np.zeros(num_envs, dtype=np.float32)
        for t in reversed(range(actual_t)):
            nonterminal = 1.0 - buf_done[t]
            next_v = next_val if t == actual_t - 1 else buf_val[t + 1]
            delta = buf_rew[t] + cfg.gamma * next_v * nonterminal - buf_val[t]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * nonterminal * last_gae
            adv[t] = last_gae
        returns = adv + buf_val[:actual_t]

        flat_n = actual_t * num_envs
        b_obs = torch.as_tensor(buf_obs[:actual_t].reshape(flat_n, obs_dim), device=dev)
        b_act = torch.as_tensor(buf_act[:actual_t].reshape(flat_n, act_dim), device=dev)
        b_logp = torch.as_tensor(buf_logp[:actual_t].reshape(flat_n), device=dev)
        b_adv = torch.as_tensor(adv.reshape(flat_n), device=dev)
        b_ret = torch.as_tensor(returns.reshape(flat_n), device=dev)
        if flat_n > 1:
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std(unbiased=False) + 1e-8)

        policy.train()
        mb_size = min(cfg.batch_size, flat_n)
        for _ in range(cfg.ppo_epochs):
            perm = rng.permutation(flat_n)
            for start in range(0, flat_n, mb_size):
                idx = torch.as_tensor(
                    perm[start : start + mb_size], dtype=torch.long, device=dev
                )
                new_logp, entropy, new_val = policy.evaluate_actions(
                    b_obs[idx], b_act[idx]
                )
                ratio = torch.exp(new_logp - b_logp[idx])
                surr1 = ratio * b_adv[idx]
                surr2 = (
                    torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                    * b_adv[idx]
                )
                pg_loss = -torch.min(surr1, surr2).mean()
                v_loss = F.mse_loss(new_val, b_ret[idx])
                ent_loss = entropy.mean()
                loss = pg_loss + cfg.value_coef * v_loss - cfg.entropy_coef * ent_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                optimizer.step()

        policy.eval()
        if (
            eval_callback is not None
            and next_eval is not None
            and env_steps >= next_eval
        ):
            eval_callback(env_steps, policy)
            assert eval_interval is not None
            next_eval += int(eval_interval)

    return policy


def _measure_peak_mb(fn: Callable[[], Any]) -> tuple[Any, float, float]:
    """Run `fn()` recording `(result, wall_clock_sec, peak_tracemalloc_mb)`."""
    already_tracing = tracemalloc.is_tracing()
    if already_tracing:
        tracemalloc.reset_peak()
    else:
        tracemalloc.start()
    t0 = time.perf_counter()
    try:
        result = fn()
        elapsed = time.perf_counter() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        if not already_tracing:
            tracemalloc.stop()
    return result, float(elapsed), float(peak_bytes / (1024.0 * 1024.0))


def _train_region_specialists(
    env_id: str,
    *,
    seed: int,
    n_envs: int,
    teacher_steps: int,
    n_eval_episodes: int,
    hidden_dims: tuple[int, ...],
    device: str,
) -> tuple[StudentActorCritic, StudentActorCritic, dict[str, Any]]:
    """Train two disjoint workspace-half specialist teachers (`low` and `high`)."""
    t0 = time.perf_counter()
    probe = TargetHalfSpaceWrapper(make_env(env_id))
    obs_dim = int(probe.observation_space.shape[0])  # type: ignore[index]
    act_dim = int(probe.action_space.shape[0])  # type: ignore[index]
    split_info = {
        "split_joint_index": probe.split_index,
        "split_joint_name": probe.split_joint_name,
        "full_joint_range": [
            float(probe.full_range[probe.split_index, 0]),
            float(probe.full_range[probe.split_index, 1]),
        ],
        "split_midpoint": float(probe.split_midpoint),
        "teacher_steps_per_specialist": int(teacher_steps),
    }
    probe.close()

    specialists: dict[str, StudentActorCritic] = {}
    diagnostics: dict[str, Any] = {}
    for idx, region in enumerate(("low", "high")):
        region_lit: RegionName = "low" if region == "low" else "high"
        pool = SyncEnvPool(
            [
                (
                    lambda r=region_lit: TargetHalfSpaceWrapper(
                        make_env(env_id), default_region=r
                    )
                )
                for _ in range(n_envs)
            ],
            seed=seed * 100 + idx * 37 + 1,
        )
        try:
            torch.manual_seed(seed * 100 + idx * 37 + 1)
            teacher = StudentActorCritic(
                obs_dim=obs_dim,
                act_dim=act_dim,
                actor_hidden_dims=hidden_dims,
                critic_hidden_dims=hidden_dims,
            ).to(device)
            ppo_cfg = PPOConfig(
                rollout_steps_per_env=max(8, min(128, teacher_steps // max(1, n_envs))),
                seed=seed * 100 + idx * 37 + 1,
                device=device,
            )
            train_online_ppo(pool, teacher, teacher_steps, ppo_cfg)
        finally:
            pool.close()

        in_ret, _, in_succ, _ = evaluate_student_policy(
            env_id,
            teacher,
            n_episodes=n_eval_episodes,
            seed_base=20_000 + seed * 1000,
            region=region_lit,
        )
        full_ret, _, full_succ, _ = evaluate_student_policy(
            env_id,
            teacher,
            n_episodes=n_eval_episodes,
            seed_base=30_000 + seed * 1000,
            region="full",
        )
        specialists[region] = teacher.eval()
        diagnostics[f"teacher_{region}"] = {
            "in_region_return": in_ret,
            "in_region_success_rate": in_succ,
            "full_task_return": full_ret,
            "full_task_success_rate": full_succ,
        }

    split_info["pretrain_wall_clock_sec"] = float(time.perf_counter() - t0)
    split_info["diagnostics"] = diagnostics
    return specialists["low"], specialists["high"], split_info


def _aggregate_seed_runs(seed_runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    final_returns = [float(r["final_return"]) for r in seed_runs]
    final_succ = [float(r["final_success_rate"]) for r in seed_runs]
    any_succ = [float(r["final_any_step_success_rate"]) for r in seed_runs]
    wall_secs = [float(r["wall_clock_sec"]) for r in seed_runs]
    peak_mbs = [float(r["peak_memory_mb"]) for r in seed_runs]

    steps_grid = [int(pt["env_steps"]) for pt in seed_runs[0]["eval_curve"]]
    curve_summary: list[dict[str, float | int]] = []
    for idx, st in enumerate(steps_grid):
        rets = [float(r["eval_curve"][idx]["mean_return"]) for r in seed_runs]
        succs = [float(r["eval_curve"][idx]["success_rate"]) for r in seed_runs]
        times = [float(r["eval_curve"][idx]["elapsed_sec"]) for r in seed_runs]
        curve_summary.append(
            {
                "env_steps": st,
                "elapsed_sec_mean": float(np.mean(times)),
                "return_mean": float(np.mean(rets)),
                "return_std": float(np.std(rets)),
                "success_rate_mean": float(np.mean(succs)),
                "success_rate_std": float(np.std(succs)),
            }
        )

    return {
        "n_seeds": len(seed_runs),
        "final_return_mean": float(np.mean(final_returns)),
        "final_return_std": float(np.std(final_returns)),
        "final_success_rate_mean": float(np.mean(final_succ)),
        "final_success_rate_std": float(np.std(final_succ)),
        "final_any_step_success_rate_mean": float(np.mean(any_succ)),
        "final_any_step_success_rate_std": float(np.std(any_succ)),
        "wall_clock_sec_mean": float(np.mean(wall_secs)),
        "wall_clock_sec_std": float(np.std(wall_secs)),
        "peak_memory_mb_mean": float(np.mean(peak_mbs)),
        "peak_memory_mb_std": float(np.std(peak_mbs)),
        "curve_summary": curve_summary,
        "seeds": list(seed_runs),
    }


def run_dagger_benchmark(
    *,
    env_ids: Sequence[str] = DEFAULT_BENCHMARK_ENVS,
    total_steps: int = 32_768,
    eval_interval: int = 4_096,
    seeds: Sequence[int] = (0, 1, 2),
    n_envs: int = 8,
    n_eval_episodes: int = 20,
    teacher_steps: int = 32_768,
    steps_per_chunk: int | None = None,
    hidden_dims: tuple[int, ...] = (128, 128),
    device: str = "cpu",
    out_json: str | Path | None = None,
) -> dict[str, Any]:
    """Run the 3-arm (`PPO`, `BC`, `TrajSwapDAgger`) benchmark across `env_ids`.

    Args:
        env_ids: Registered MyoSuite environment IDs to benchmark.
        total_steps: Environment-step budget per arm (`PPO`, `BC`, `TrajSwapDAgger`).
        eval_interval: Step interval between full-distribution deterministic evals.
        seeds: Random seeds evaluated for every arm.
        n_envs: Parallel rollout environments per arm.
        n_eval_episodes: Deterministic episodes per evaluation checkpoint.
        teacher_steps: Pre-training step budget for each workspace specialist teacher.
        steps_per_chunk: Optional chunk length for `BC` and `TrajSwapDAgger`.
        hidden_dims: Hidden layer sizes shared by all arms.
        device: Torch execution device (`"cpu"` or `"cuda:0"`).
        out_json: Optional output path for JSON results.

    Returns:
        Structured benchmark report dictionary.
    """
    if total_steps < 1 or eval_interval < 1:
        raise ValueError("total_steps and eval_interval must be >= 1.")
    if not seeds:
        raise ValueError("seeds must be non-empty.")

    chunk_len = (
        int(steps_per_chunk)
        if steps_per_chunk is not None
        else max(n_envs * 4, min(1024, max(n_envs, eval_interval // 2)))
    )

    report: dict[str, Any] = {
        "benchmark": "myosuite-traj-dagger",
        "config": {
            "env_ids": list(env_ids),
            "arms": ["PPO", "BC", "TrajSwapDAgger"],
            "total_steps": int(total_steps),
            "eval_interval": int(eval_interval),
            "seeds": [int(s) for s in seeds],
            "n_envs": int(n_envs),
            "n_eval_episodes": int(n_eval_episodes),
            "teacher_steps": int(teacher_steps),
            "steps_per_chunk": chunk_len,
            "hidden_dims": list(hidden_dims),
            "device": device,
        },
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "environments": {},
    }

    for env_id in env_ids:
        probe_env = make_env(env_id)
        obs_dim = int(probe_env.observation_space.shape[0])  # type: ignore[index]
        act_dim = int(probe_env.action_space.shape[0])  # type: ignore[index]
        probe_env.close()

        arm_seed_runs: dict[str, list[dict[str, Any]]] = {
            "PPO": [],
            "BC": [],
            "TrajSwapDAgger": [],
        }
        teacher_setups: list[dict[str, Any]] = []

        for seed in seeds:
            seed_i = int(seed)
            teacher_low, teacher_high, t_info = _train_region_specialists(
                env_id,
                seed=seed_i,
                n_envs=n_envs,
                teacher_steps=teacher_steps,
                n_eval_episodes=n_eval_episodes,
                hidden_dims=hidden_dims,
                device=device,
            )
            t_info["seed"] = seed_i
            teacher_setups.append(t_info)

            spec_low = TeacherSpec.from_module(
                teacher_low, key=f"{env_id}_low_s{seed_i}", device=device
            )
            spec_high = TeacherSpec.from_module(
                teacher_high, key=f"{env_id}_high_s{seed_i}", device=device
            )
            swap_pairs = [
                SwapPair(teacher=spec_low, reset_options={"region": "low"}),
                SwapPair(teacher=spec_high, reset_options={"region": "high"}),
            ]

            # Arm 1: Online PPO from scratch on unmodified make_env(env_id).
            def _run_ppo_arm(s: int = seed_i) -> list[EvalSnapshot]:
                torch.manual_seed(s + 1000)
                pool = SyncEnvPool(
                    [lambda: make_env(env_id) for _ in range(n_envs)], seed=s + 1000
                )
                curve: list[EvalSnapshot] = []
                t_start = time.perf_counter()
                try:
                    student = StudentActorCritic(
                        obs_dim=obs_dim,
                        act_dim=act_dim,
                        actor_hidden_dims=hidden_dims,
                        critic_hidden_dims=hidden_dims,
                    ).to(device)

                    def _on_eval(step_n: int, pol: StudentActorCritic) -> None:
                        m_ret, s_ret, succ, any_s = evaluate_student_policy(
                            env_id,
                            pol,
                            n_episodes=n_eval_episodes,
                            seed_base=50_000 + s * 1000,
                        )
                        curve.append(
                            EvalSnapshot(
                                env_steps=step_n,
                                elapsed_sec=float(time.perf_counter() - t_start),
                                mean_return=m_ret,
                                std_return=s_ret,
                                success_rate=succ,
                                any_step_success_rate=any_s,
                            )
                        )

                    ppo_cfg = PPOConfig(
                        rollout_steps_per_env=max(
                            4, min(128, eval_interval // max(1, n_envs))
                        ),
                        seed=s + 1000,
                        device=device,
                    )
                    train_online_ppo(
                        pool,
                        student,
                        total_steps,
                        ppo_cfg,
                        eval_interval=eval_interval,
                        eval_callback=_on_eval,
                    )
                finally:
                    pool.close()
                return curve

            ppo_curve, ppo_sec, ppo_mb = _measure_peak_mb(_run_ppo_arm)
            arm_seed_runs["PPO"].append(
                _pack_seed_run(seed_i, ppo_curve, ppo_sec, ppo_mb)
            )

            # Arm 2: Offline BC (beta=1.0 teacher-only transitions, no student rollouts).
            def _run_bc_arm(
                s: int = seed_i, pairs_ref: list[SwapPair] = swap_pairs
            ) -> list[EvalSnapshot]:
                torch.manual_seed(s + 2000)
                pool = SyncEnvPool(
                    [
                        lambda: TargetHalfSpaceWrapper(make_env(env_id))
                        for _ in range(n_envs)
                    ],
                    seed=s + 2000,
                )
                reg = TeacherRegistry()
                curve: list[EvalSnapshot] = []
                t_start = time.perf_counter()
                try:
                    student = StudentActorCritic(
                        obs_dim=obs_dim,
                        act_dim=act_dim,
                        actor_hidden_dims=hidden_dims,
                        critic_hidden_dims=hidden_dims,
                    ).to(device)
                    bc_cfg = OfflineBCConfig(
                        steps_per_chunk=chunk_len,
                        batch_size=min(256, max(32, chunk_len)),
                        bc_epochs=6,
                        max_batches_per_epoch=32,
                        seed=s + 2000,
                        device=device,
                    )
                    bc = OfflineBC(
                        pool,
                        student,
                        pairs_ref,
                        bc_cfg,
                        registry=reg,
                        max_buffer_size=max(total_steps, chunk_len),
                    )

                    def _on_eval(step_n: int, pol: StudentActorCritic) -> None:
                        m_ret, s_ret, succ, any_s = evaluate_student_policy(
                            env_id,
                            pol,
                            n_episodes=n_eval_episodes,
                            seed_base=50_000 + s * 1000,
                        )
                        curve.append(
                            EvalSnapshot(
                                env_steps=step_n,
                                elapsed_sec=float(time.perf_counter() - t_start),
                                mean_return=m_ret,
                                std_return=s_ret,
                                success_rate=succ,
                                any_step_success_rate=any_s,
                            )
                        )

                    bc.train(
                        total_steps,
                        eval_interval=eval_interval,
                        eval_callback=_on_eval,
                    )
                finally:
                    pool.close()
                return curve

            bc_curve, bc_sec, bc_mb = _measure_peak_mb(_run_bc_arm)
            arm_seed_runs["BC"].append(_pack_seed_run(seed_i, bc_curve, bc_sec, bc_mb))

            # Arm 3: Online Multi-Specialist Trajectory-Swap DAgger.
            def _run_dagger_arm(
                s: int = seed_i, pairs_ref: list[SwapPair] = swap_pairs
            ) -> list[EvalSnapshot]:
                torch.manual_seed(s + 3000)
                pool = SyncEnvPool(
                    [
                        lambda: TargetHalfSpaceWrapper(make_env(env_id))
                        for _ in range(n_envs)
                    ],
                    seed=s + 3000,
                )
                reg = TeacherRegistry()
                curve: list[EvalSnapshot] = []
                t_start = time.perf_counter()
                try:
                    student = StudentActorCritic(
                        obs_dim=obs_dim,
                        act_dim=act_dim,
                        actor_hidden_dims=hidden_dims,
                        critic_hidden_dims=hidden_dims,
                    ).to(device)
                    d_cfg = DaggerConfig(
                        steps_per_chunk=chunk_len,
                        beta_start=1.0,
                        beta_end=0.05,
                        sticky_h_min=5,
                        sticky_h_max=15,
                        ring_buffer_size=max(total_steps // 2, chunk_len * 2),
                        long_term_buffer_size=max(total_steps // 4, chunk_len),
                        min_include_prob=0.05,
                        batch_size=min(256, max(32, chunk_len)),
                        epochs_per_chunk=4,
                        max_batches_per_epoch=32,
                        pair_sampling="round_robin",
                        seed=s + 3000,
                        device=device,
                    )
                    trainer = TrajSwapDagger(
                        pool, student, pairs_ref, d_cfg, registry=reg
                    )

                    def _on_eval(step_n: int, pol: StudentActorCritic) -> None:
                        m_ret, s_ret, succ, any_s = evaluate_student_policy(
                            env_id,
                            pol,
                            n_episodes=n_eval_episodes,
                            seed_base=50_000 + s * 1000,
                        )
                        curve.append(
                            EvalSnapshot(
                                env_steps=step_n,
                                elapsed_sec=float(time.perf_counter() - t_start),
                                mean_return=m_ret,
                                std_return=s_ret,
                                success_rate=succ,
                                any_step_success_rate=any_s,
                            )
                        )

                    trainer.train(
                        total_steps,
                        eval_interval=eval_interval,
                        eval_callback=_on_eval,
                    )
                finally:
                    pool.close()
                return curve

            dag_curve, dag_sec, dag_mb = _measure_peak_mb(_run_dagger_arm)
            arm_seed_runs["TrajSwapDAgger"].append(
                _pack_seed_run(seed_i, dag_curve, dag_sec, dag_mb)
            )

        report["environments"][env_id] = {
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "teacher_setup": teacher_setups,
            "arms": {
                arm_name: _aggregate_seed_runs(runs)
                for arm_name, runs in arm_seed_runs.items()
            },
        }

    if out_json is not None:
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def _pack_seed_run(
    seed: int,
    curve: Sequence[EvalSnapshot],
    wall_clock_sec: float,
    peak_memory_mb: float,
) -> dict[str, Any]:
    last = curve[-1]
    return {
        "seed": seed,
        "final_return": last.mean_return,
        "final_success_rate": last.success_rate,
        "final_any_step_success_rate": last.any_step_success_rate,
        "wall_clock_sec": wall_clock_sec,
        "peak_memory_mb": peak_memory_mb,
        "eval_curve": [asdict(pt) for pt in curve],
    }


def format_markdown_summary(report: dict[str, Any]) -> str:
    """Format a Markdown summary table of `run_dagger_benchmark` output."""
    lines = [
        "| Environment | Arm | Final Return (mean +- std) | Solved Rate (mean +- std) | Wall-Clock (s) | Peak Memory (MB) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for env_id, env_data in report["environments"].items():
        for arm_name, summary in env_data["arms"].items():
            ret_str = f"{summary['final_return_mean']:.2f} +- {summary['final_return_std']:.2f}"
            succ_str = (
                f"{100.0 * summary['final_success_rate_mean']:.1f}% +- "
                f"{100.0 * summary['final_success_rate_std']:.1f}%"
            )
            time_str = (
                f"{summary['wall_clock_sec_mean']:.2f} +- "
                f"{summary['wall_clock_sec_std']:.2f}"
            )
            mem_str = (
                f"{summary['peak_memory_mb_mean']:.2f} +- "
                f"{summary['peak_memory_mb_std']:.2f}"
            )
            lines.append(
                f"| `{env_id}` | `{arm_name}` | {ret_str} | {succ_str} | {time_str} | {mem_str} |"
            )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint for `traj-dagger-bench`."""
    parser = argparse.ArgumentParser(
        description="Benchmark PPO vs. Offline BC vs. Multi-Teacher TrajSwapDAgger."
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        default=list(DEFAULT_BENCHMARK_ENVS),
        help="Registered MyoSuite environment IDs.",
    )
    parser.add_argument("--total-steps", type=int, default=32_768)
    parser.add_argument("--teacher-steps", type=int, default=32_768)
    parser.add_argument("--eval-interval", type=int, default=4_096)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    parser.add_argument("--steps-per-chunk", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args(argv)

    report = run_dagger_benchmark(
        env_ids=args.envs,
        total_steps=args.total_steps,
        eval_interval=args.eval_interval,
        seeds=args.seeds,
        n_envs=args.n_envs,
        n_eval_episodes=args.n_eval_episodes,
        teacher_steps=args.teacher_steps,
        steps_per_chunk=args.steps_per_chunk,
        device=args.device,
        out_json=args.out_json,
    )
    print(format_markdown_summary(report))
    if args.out_json is not None:
        print(f"\nSaved JSON report to {args.out_json}")


if __name__ == "__main__":
    main()
