# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Like-for-like multi-seed benchmark comparing DreamerV3 against PPO on MyoSuite."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform as py_platform
import sys
import time
from typing import Any

import numpy as np
import psutil

from myosuite_dreamerv3.compat import import_myosuite
from myosuite_dreamerv3.mjlab_driver import load_myosuite_mjlab_env, train_mjlab
from myosuite_dreamerv3.train import train_cpu


def _make_cpu_eval_env(myosuite: Any, env_id: str, features: Sequence[Any] = ()) -> Any:
    if features:
        from myosuite.core.config import EnvConfig
        from myosuite.utils.feature_cli import feature_spec

        parsed = tuple(feature_spec(f) if isinstance(f, str) else f for f in features)
        return myosuite.make_env(EnvConfig(env_id=env_id, features=parsed))
    return myosuite.make_env(env_id)


def _default_curve_steps(total_steps: int, curve_points: int) -> list[int]:
    if curve_points <= 0:
        return []
    if curve_points == 1:
        return [int(total_steps)]
    return sorted(
        {round(i * int(total_steps) / (curve_points - 1)) for i in range(curve_points)}
    )


def evaluate_sb3_ppo(
    model: Any,
    env_id: str,
    episodes: int = 10,
    seed: int = 0,
    features: Sequence[Any] = (),
) -> dict[str, float]:
    """Evaluate a Stable-Baselines3 PPO model deterministically on ``env_id``."""
    myosuite = import_myosuite()
    returns: list[float] = []
    solved_flags: list[float] = []
    lengths: list[int] = []

    for ep in range(episodes):
        env = _make_cpu_eval_env(myosuite, env_id, features)
        try:
            obs, _ = env.reset(seed=int(seed + ep))
            done = False
            ep_ret = 0.0
            ep_solved = False
            ep_len = 0
            while not done:
                act, _ = model.predict(obs, deterministic=True)
                obs, rew, term, trunc, info = env.step(act)
                ep_ret += float(rew)
                rwd_dict = info.get("rwd_dict", {})
                solved = info.get("solved", rwd_dict.get("solved", 0.0))
                ep_solved = ep_solved or (float(np.asarray(solved).mean()) > 0.5)
                ep_len += 1
                done = bool(term or trunc)
            returns.append(ep_ret)
            solved_flags.append(1.0 if ep_solved else 0.0)
            lengths.append(ep_len)
        finally:
            env.close()

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "solved_rate": float(np.mean(solved_flags)),
        "mean_length": float(np.mean(lengths)),
    }


def evaluate_rslrl_checkpoint_cpu(
    ckpt_path: Path,
    env_id: str,
    episodes: int = 10,
    seed: int = 0,
    features: Sequence[Any] = (),
) -> dict[str, float]:
    """Evaluate a trained ``mjlab`` RSL-RL PPO checkpoint deterministically on CPU ``make_env``."""
    import torch

    myosuite = import_myosuite()
    from myosuite.utils.rslrl_policy import load_rslrl_policy

    probe = _make_cpu_eval_env(myosuite, env_id, features)
    act_dim = int(probe.action_space.shape[0])
    probe.close()
    policy = load_rslrl_policy(ckpt_path, act_dim).eval()

    returns: list[float] = []
    solved_flags: list[float] = []
    lengths: list[int] = []

    with torch.no_grad():
        for ep in range(episodes):
            env = _make_cpu_eval_env(myosuite, env_id, features)
            try:
                obs, _ = env.reset(seed=int(seed + ep))
                done = False
                ep_ret = 0.0
                ep_solved = False
                ep_len = 0
                while not done:
                    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                    act = policy(obs_t).squeeze(0).cpu().numpy()
                    obs, rew, term, trunc, info = env.step(act)
                    ep_ret += float(rew)
                    rwd_dict = info.get("rwd_dict", {})
                    solved = info.get("solved", rwd_dict.get("solved", 0.0))
                    ep_solved = ep_solved or (float(np.asarray(solved).mean()) > 0.5)
                    ep_len += 1
                    done = bool(term or trunc)
                returns.append(ep_ret)
                solved_flags.append(1.0 if ep_solved else 0.0)
                lengths.append(ep_len)
            finally:
                env.close()

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "solved_rate": float(np.mean(solved_flags)),
        "mean_length": float(np.mean(lengths)),
    }


def run_mjlab_ppo_arm(
    env_id: str,
    steps: int,
    seed: int,
    logroot: Path,
    num_envs: int = 256,
    device: str = "cuda:0",
    eval_episodes: int = 10,
    features: Sequence[Any] = (),
    curve_steps: Sequence[int] = (),
) -> dict[str, Any]:
    """Train MyoSuite's registered ``mjlab`` RSL-RL PPO baseline and evaluate on CPU ``make_env``."""
    import torch

    import_myosuite()
    import myosuite.envs.myo.backends.mjlab  # noqa: F401
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ["WANDB_MODE"] = "disabled"
    configure_torch_backends()

    rl_cfg = load_rl_cfg(env_id)
    rl_cfg.seed = int(seed)
    rl_cfg.logger = "tensorboard"
    steps_per_iter = int(num_envs) * int(rl_cfg.num_steps_per_env)
    max_iters = max(1, round(int(steps) / steps_per_iter))
    actual_steps = max_iters * steps_per_iter
    rl_cfg.max_iterations = max_iters

    run_dir = logroot / f"mjlab_ppo_{env_id}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    raw_env = load_myosuite_mjlab_env(
        env_id, num_envs=num_envs, seed=seed, device=device, features=features
    )
    vec_env = RslRlVecEnvWrapper(raw_env, clip_actions=rl_cfg.clip_actions)
    runner_cls = load_runner_cls(env_id) or MjlabOnPolicyRunner
    curve: list[dict[str, Any]] = []
    wall_time = 0.0
    try:
        t0 = time.perf_counter()
        runner = runner_cls(vec_env, asdict(rl_cfg), str(run_dir), device)
        wall_time += time.perf_counter() - t0

        if curve_steps:
            iter_targets = sorted(
                {
                    min(max_iters, max(0, round(int(s) / steps_per_iter)))
                    for s in curve_steps
                }
                | {max_iters}
            )
            done_iters = 0
            for idx, target_it in enumerate(iter_targets):
                delta = target_it - done_iters
                if delta > 0:
                    t_seg = time.perf_counter()
                    runner.learn(
                        num_learning_iterations=delta,
                        init_at_random_ep_len=(done_iters == 0),
                    )
                    wall_time += time.perf_counter() - t_seg
                    done_iters = target_it
                ckpt_it = run_dir / f"model_{done_iters}.pt"
                runner.save(str(ckpt_it))
                stats = evaluate_rslrl_checkpoint_cpu(
                    ckpt_it,
                    env_id=env_id,
                    episodes=eval_episodes,
                    seed=seed + 10_000,
                    features=features,
                )
                curve.append(
                    {
                        "steps": done_iters * steps_per_iter,
                        "wall_time_s": wall_time if idx > 0 or done_iters > 0 else 0.0,
                        **stats,
                    }
                )
            ckpt_path = run_dir / f"model_{max_iters}.pt"
            eval_stats = {
                k: curve[-1][k]
                for k in ("mean_return", "std_return", "solved_rate", "mean_length")
            }
        else:
            t_seg = time.perf_counter()
            runner.learn(num_learning_iterations=max_iters, init_at_random_ep_len=True)
            ckpt_path = run_dir / f"model_{max_iters}.pt"
            if not ckpt_path.is_file():
                runner.save(str(ckpt_path))
            wall_time += time.perf_counter() - t_seg
            eval_stats = evaluate_rslrl_checkpoint_cpu(
                ckpt_path,
                env_id=env_id,
                episodes=eval_episodes,
                seed=seed + 10_000,
                features=features,
            )
    finally:
        vec_env.close()

    rss_peak = max(rss_before, proc.memory_info().rss)
    vram_mb = (
        torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        if torch.cuda.is_available() and device.startswith("cuda")
        else 0.0
    )
    return {
        "method": "PPO (mjlab / RSL-RL)",
        "env_id": env_id,
        "seed": seed,
        "steps": actual_steps,
        "wall_time_s": wall_time,
        "peak_rss_mb": rss_peak / (1024.0 * 1024.0),
        "peak_vram_mb": vram_mb,
        "eval": eval_stats,
        "curve": curve,
    }


def run_ppo_arm(
    env_id: str,
    steps: int,
    seed: int,
    eval_episodes: int = 10,
    features: Sequence[Any] = (),
    curve_steps: Sequence[int] = (),
) -> dict[str, Any]:
    """Train and evaluate Stable-Baselines3 PPO on ``env_id`` for ``steps``."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    myosuite = import_myosuite()

    def _make_one() -> Any:
        env = _make_cpu_eval_env(myosuite, env_id, features)
        env.reset(seed=int(seed))
        return env

    vec_env = DummyVecEnv([_make_one])
    n_steps = min(512, max(64, steps // 2))
    if curve_steps:
        positive_deltas = [int(s) for s in curve_steps if int(s) > 0]
        if positive_deltas:
            n_steps = min(n_steps, max(64, min(positive_deltas)))
    batch_size = min(64, n_steps)
    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    curve: list[dict[str, Any]] = []
    wall_time = 0.0

    try:
        t0 = time.perf_counter()
        model = PPO(
            "MlpPolicy",
            vec_env,
            seed=int(seed),
            n_steps=n_steps,
            batch_size=batch_size,
            verbose=0,
            device="cpu",
        )
        wall_time += time.perf_counter() - t0

        if curve_steps:
            targets = sorted(
                {max(0, min(int(steps), int(s))) for s in curve_steps} | {int(steps)}
            )
            done_steps = 0
            for target_s in targets:
                delta = target_s - done_steps
                if delta > 0:
                    t_seg = time.perf_counter()
                    model.learn(
                        total_timesteps=delta,
                        reset_num_timesteps=(done_steps == 0),
                    )
                    wall_time += time.perf_counter() - t_seg
                    done_steps = target_s
                stats = evaluate_sb3_ppo(
                    model,
                    env_id=env_id,
                    episodes=eval_episodes,
                    seed=seed + 10_000,
                    features=features,
                )
                curve.append(
                    {
                        "steps": done_steps,
                        "wall_time_s": wall_time if done_steps > 0 else 0.0,
                        **stats,
                    }
                )
            eval_stats = {
                k: curve[-1][k]
                for k in ("mean_return", "std_return", "solved_rate", "mean_length")
            }
        else:
            t_seg = time.perf_counter()
            model.learn(total_timesteps=int(steps))
            wall_time += time.perf_counter() - t_seg
            eval_stats = evaluate_sb3_ppo(
                model,
                env_id=env_id,
                episodes=eval_episodes,
                seed=seed + 10_000,
                features=features,
            )
        rss_peak = max(rss_before, proc.memory_info().rss)
    finally:
        vec_env.close()

    return {
        "method": "PPO (SB3)",
        "env_id": env_id,
        "seed": seed,
        "steps": int(steps),
        "wall_time_s": wall_time,
        "peak_rss_mb": rss_peak / (1024.0 * 1024.0),
        "peak_vram_mb": 0.0,
        "eval": eval_stats,
        "curve": curve,
    }


def run_dreamerv3_arm(
    env_id: str,
    steps: int,
    seed: int,
    logroot: Path,
    size: str = "size1m",
    platform: str = "cpu",
    dreamer_backend: str = "cpu",
    envs: int = 4,
    train_ratio: float = 256.0,
    eval_episodes: int = 10,
    features: Sequence[Any] = (),
    curve_steps: Sequence[int] = (),
    export_onnx_dir: Path | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    """Train and evaluate DreamerV3 on ``env_id`` for ``steps``."""
    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    run_dir = logroot / f"dreamerv3_{dreamer_backend}_{env_id}_seed{seed}"
    onnx_out = (
        str(export_onnx_dir / f"{env_id}_seed{seed}.onnx")
        if export_onnx_dir is not None
        else ""
    )
    if dreamer_backend == "mjlab":
        res = train_mjlab(
            env_id=env_id,
            num_envs=envs,
            steps=steps,
            train_ratio=train_ratio,
            size=size,
            platform=platform,
            sim_device="cuda:0" if platform == "cuda" else "cpu",
            seed=seed,
            logdir=str(run_dir),
            outputs=("jsonl",),
            features=features,
            eval_episodes=eval_episodes,
            curve_steps=curve_steps,
            export_onnx=onnx_out,
            debug=debug,
        )
        label = f"DreamerV3 ({'debug' if debug else size}, mjlab)"
    else:
        res = train_cpu(
            env_id=env_id,
            steps=steps,
            envs=envs,
            train_ratio=train_ratio,
            size=size,
            platform=platform,
            seed=seed,
            logdir=str(run_dir),
            export_onnx=onnx_out,
            outputs=("jsonl",),
            script="train",
            features=features,
            debug=debug,
            eval_episodes=eval_episodes,
            curve_steps=curve_steps,
        )
        label = f"DreamerV3 ({'debug' if debug else size})"

    rss_peak = max(rss_before, proc.memory_info().rss)
    return {
        "method": label,
        "env_id": env_id,
        "seed": seed,
        "steps": res["steps"],
        "wall_time_s": res["wall_time_s"],
        "peak_rss_mb": rss_peak / (1024.0 * 1024.0),
        "peak_vram_mb": 0.0,
        "eval": res["eval"],
        "curve": res.get("curve", []),
        "logdir": res.get("logdir", str(run_dir)),
        "onnx": res.get("onnx", ""),
    }


def summarize_arms(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-seed runs into ``mean ± std`` rows grouped by ``(env_id, method)``."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for rec in records:
        grouped.setdefault((rec["env_id"], rec["method"]), []).append(rec)

    summary: list[dict[str, Any]] = []
    for (env_id, method), items in grouped.items():
        rets = [r["eval"]["mean_return"] for r in items]
        solves = [r["eval"]["solved_rate"] for r in items]
        times = [r["wall_time_s"] for r in items]
        mems = [r["peak_rss_mb"] for r in items]
        summary.append(
            {
                "env_id": env_id,
                "method": method,
                "seeds": [r["seed"] for r in items],
                "steps": items[0]["steps"],
                "return_mean": float(np.mean(rets)),
                "return_std": float(np.std(rets)),
                "solved_mean": float(np.mean(solves)),
                "solved_std": float(np.std(solves)),
                "wall_time_mean_s": float(np.mean(times)),
                "wall_time_std_s": float(np.std(times)),
                "peak_rss_mean_mb": float(np.mean(mems)),
            }
        )
    return summary


def summarize_learning_curves(
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate milestone learning curves across seeds by ``(env_id, method, steps)``."""
    grouped: dict[tuple[str, str], list[list[dict[str, Any]]]] = {}
    for rec in records:
        if rec.get("curve"):
            grouped.setdefault((rec["env_id"], rec["method"]), []).append(rec["curve"])

    out: list[dict[str, Any]] = []
    for (env_id, method), curves in grouped.items():
        n_pts = min(len(c) for c in curves)
        pts: list[dict[str, Any]] = []
        for idx in range(n_pts):
            step_val = int(curves[0][idx]["steps"])
            rets = [c[idx]["mean_return"] for c in curves]
            solves = [c[idx]["solved_rate"] for c in curves]
            walls = [c[idx]["wall_time_s"] for c in curves]
            pts.append(
                {
                    "steps": step_val,
                    "wall_time_mean_s": float(np.mean(walls)),
                    "wall_time_std_s": float(np.std(walls)),
                    "return_mean": float(np.mean(rets)),
                    "return_std": float(np.std(rets)),
                    "solved_mean": float(np.mean(solves)),
                    "solved_std": float(np.std(solves)),
                }
            )
        out.append({"env_id": env_id, "method": method, "points": pts})
    return out


def run_benchmark(
    env_ids: Sequence[str] = ("myoElbowPose1D6MRandom-v0", "myoHandPoseRandom-v0"),
    seeds: Sequence[int] = (0, 1, 2),
    steps: int = 8_000,
    eval_episodes: int = 10,
    size: str = "size1m",
    platform: str = "cpu",
    ppo_backend: str = "sb3",
    ppo_num_envs: int = 256,
    dreamer_backend: str = "cpu",
    envs: int = 4,
    train_ratio: float = 256.0,
    features: Sequence[Any] = (),
    curve_points: int = 0,
    export_onnx_dir: str | Path | None = None,
    out_path: str | Path = "dreamerv3_vs_ppo_benchmark.json",
    logroot: str | Path = "dreamerv3_bench_logs",
    debug: bool = False,
) -> dict[str, Any]:
    """Run multi-seed PPO vs. DreamerV3 benchmark across ``env_ids``."""
    logroot_path = Path(logroot)
    logroot_path.mkdir(parents=True, exist_ok=True)
    onnx_dir = Path(export_onnx_dir) if export_onnx_dir else None
    if onnx_dir is not None:
        onnx_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for env_id in env_ids:
        for seed in seeds:
            ppo_curve_steps = _default_curve_steps(steps, curve_points)
            if ppo_backend == "mjlab":
                ppo_rec = run_mjlab_ppo_arm(
                    env_id=env_id,
                    steps=steps,
                    seed=seed,
                    logroot=logroot_path,
                    num_envs=ppo_num_envs,
                    device="cuda:0" if platform == "cuda" else "cpu",
                    eval_episodes=eval_episodes,
                    features=features,
                    curve_steps=ppo_curve_steps,
                )
            else:
                ppo_rec = run_ppo_arm(
                    env_id=env_id,
                    steps=steps,
                    seed=seed,
                    eval_episodes=eval_episodes,
                    features=features,
                    curve_steps=ppo_curve_steps,
                )
            records.append(ppo_rec)

            dv3_curve_steps = (
                [int(pt["steps"]) for pt in ppo_rec["curve"]]
                if ppo_rec.get("curve")
                else _default_curve_steps(ppo_rec["steps"], curve_points)
            )
            dv3_rec = run_dreamerv3_arm(
                env_id=env_id,
                steps=ppo_rec["steps"],
                seed=seed,
                logroot=logroot_path,
                size=size,
                platform=platform,
                dreamer_backend=dreamer_backend,
                envs=envs,
                train_ratio=train_ratio,
                eval_episodes=eval_episodes,
                features=features,
                curve_steps=dv3_curve_steps,
                export_onnx_dir=onnx_dir,
                debug=debug,
            )
            records.append(dv3_rec)

    gpu_name = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    payload: dict[str, Any] = {
        "hardware": {
            "platform": py_platform.platform(),
            "processor": py_platform.processor(),
            "gpu": gpu_name,
            "python": py_platform.python_version(),
            "cpu_count": psutil.cpu_count(logical=True),
        },
        "features": [str(f) for f in features],
        "summary": summarize_arms(records),
        "learning_curves": summarize_learning_curves(records),
        "runs": records,
    }
    Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """CLI entrypoint for ``dreamerv3-bench``."""
    from myosuite.utils.feature_cli import parse_feature_args

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    features, rest = parse_feature_args(raw_argv)

    parser = argparse.ArgumentParser(
        description="Run multi-seed PPO vs. DreamerV3 benchmark on MyoSuite tasks."
    )
    parser.add_argument(
        "--envs-list",
        nargs="+",
        default=["myoElbowPose1D6MRandom-v0", "myoHandPoseRandom-v0"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=8_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--size", type=str, default="size1m")
    parser.add_argument(
        "--platform", type=str, default="cpu", choices=("cpu", "cuda", "tpu")
    )
    parser.add_argument(
        "--ppo-backend", type=str, default="sb3", choices=("sb3", "mjlab")
    )
    parser.add_argument("--ppo-num-envs", type=int, default=256)
    parser.add_argument(
        "--dreamer-backend", type=str, default="cpu", choices=("cpu", "mjlab")
    )
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=256.0)
    parser.add_argument(
        "--curve-points",
        type=int,
        default=0,
        help="Number of intermediate step/wall-time evaluation points (e.g. 4).",
    )
    parser.add_argument(
        "--export-onnx-dir",
        type=str,
        default="",
        help="Optional directory to export trained DreamerV3 .onnx policies.",
    )
    parser.add_argument("--out", type=str, default="dreamerv3_vs_ppo_benchmark.json")
    parser.add_argument("--logroot", type=str, default="dreamerv3_bench_logs")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(rest)

    result = run_benchmark(
        env_ids=args.envs_list,
        seeds=args.seeds,
        steps=args.steps,
        eval_episodes=args.eval_episodes,
        size=args.size,
        platform=args.platform,
        ppo_backend=args.ppo_backend,
        ppo_num_envs=args.ppo_num_envs,
        dreamer_backend=args.dreamer_backend,
        envs=args.num_envs,
        train_ratio=args.train_ratio,
        features=features,
        curve_points=args.curve_points,
        export_onnx_dir=args.export_onnx_dir or None,
        out_path=args.out,
        logroot=args.logroot,
        debug=args.debug,
    )
    for row in result["summary"]:
        print(
            f"{row['env_id']} | {row['method']} ({row['steps']} steps): "
            f"return={row['return_mean']:.2f}±{row['return_std']:.2f}, "
            f"solved={row['solved_mean']*100:.1f}±{row['solved_std']*100:.1f}%, "
            f"wall={row['wall_time_mean_s']:.1f}s"
        )
    return result


if __name__ == "__main__":
    main()
