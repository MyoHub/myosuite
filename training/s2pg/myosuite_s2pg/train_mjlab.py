# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""GPU training entrypoint running S2PG-PPO directly on mjlab ManagerBasedRlEnv tensors."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from myosuite import make_env
from myosuite.core.config import EnvConfig
from myosuite.utils.feature_cli import parse_feature_args

from myosuite_s2pg.networks import RNNType, VALID_RNN_TYPES, export_torchscript
from myosuite_s2pg.s2pg_ppo import (
    FeedforwardPPOTrainer,
    S2PGConfig,
    S2PGPPOTrainer,
)


def extract_mjlab_actor_obs(obs: Any) -> torch.Tensor:
    """Extract the actor observation tensor from an mjlab observation mapping or tensor."""
    if isinstance(obs, dict):
        if "actor" in obs:
            return obs["actor"]
        if "policy" in obs:
            return obs["policy"]
        raise KeyError(
            f"Expected 'actor' or 'policy' key in mjlab obs dict, got {list(obs)}."
        )
    if isinstance(obs, torch.Tensor):
        return obs
    raise TypeError(f"Unsupported mjlab observation type: {type(obs)!r}.")


class MjlabTensorVecEnv:
    """Zero-copy GPU tensor adapter around mjlab ``ManagerBasedRlEnv``."""

    def __init__(self, env: Any) -> None:
        """Wrap an initialized ``ManagerBasedRlEnv`` instance.

        Args:
            env: ``ManagerBasedRlEnv`` (or compatible GPU tensor environment).
        """
        self.env = env
        self.num_envs = int(env.num_envs)
        self.device = torch.device(env.device)
        self.action_dim = int(env.action_manager.total_action_dim)
        self.max_episode_steps = int(getattr(env, "max_episode_length", 200))

        obs_raw, _ = self.env.reset()
        actor_obs = extract_mjlab_actor_obs(obs_raw)
        self.obs_dim = int(actor_obs.shape[-1])
        self._cached_reset_obs: torch.Tensor | None = actor_obs.to(
            device=self.device, dtype=torch.float32
        )

        self._ep_returns = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._ep_lengths = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        metrics_mgr = getattr(self.env, "metrics_manager", None)
        active_terms = getattr(metrics_mgr, "active_terms", ())
        self._has_success = "success" in active_terms

    def reset(self, seed: int | None = None) -> torch.Tensor:
        """Return initial actor observations on ``self.device``."""
        del seed
        self._ep_returns.zero_()
        self._ep_lengths.zero_()
        if self._cached_reset_obs is not None:
            obs = self._cached_reset_obs
            self._cached_reset_obs = None
            return obs
        obs_raw, _ = self.env.reset()
        return extract_mjlab_actor_obs(obs_raw).to(
            device=self.device, dtype=torch.float32
        )

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Step ``ManagerBasedRlEnv`` directly with GPU tensors."""
        obs_raw, rew, terminated, truncated, _ = self.env.step(actions)
        actor_obs = extract_mjlab_actor_obs(obs_raw).to(
            device=self.device, dtype=torch.float32
        )
        rew_f = rew.to(device=self.device, dtype=torch.float32)
        term_b = terminated.to(device=self.device, dtype=torch.bool)
        trunc_b = truncated.to(device=self.device, dtype=torch.bool)

        self._ep_returns += rew_f
        self._ep_lengths += 1
        done = term_b | trunc_b

        completed: list[dict[str, float]] = []
        if bool(done.any()):
            done_indices = done.nonzero(as_tuple=False).flatten().tolist()
            done_returns = self._ep_returns[done].detach().cpu().tolist()
            done_lengths = self._ep_lengths[done].detach().cpu().tolist()
            for local_i, env_idx in enumerate(done_indices):
                record: dict[str, float] = {
                    "return": float(done_returns[local_i]),
                    "length": float(done_lengths[local_i]),
                }
                if self._has_success:
                    terms = dict(
                        self.env.metrics_manager.get_active_iterable_terms(env_idx)
                    )
                    if "success" in terms:
                        record["success"] = float(terms["success"][0])
                completed.append(record)
            self._ep_returns[done] = 0.0
            self._ep_lengths[done] = 0

        return actor_obs, rew_f, term_b, trunc_b, {"completed_episodes": completed}

    def close(self) -> None:
        """Close the wrapped ``ManagerBasedRlEnv``."""
        self.env.close()


@dataclass
class MjlabTrainArgs:
    """Configuration for ``s2pg-train-mjlab``."""

    env_id: str
    num_envs: int = 1024
    max_iterations: int = 500
    save_interval: int = 100
    device: str | None = None
    log_root: Path = Path("logs/s2pg")
    run_name: str = ""
    baseline_ppo: bool = False
    export_ts: bool = True
    s2pg: S2PGConfig = field(
        default_factory=lambda: S2PGConfig(
            num_steps_per_env=24,
            num_epochs=8,
            num_minibatches=8,
            gamma=0.97,
            clip_coef=0.3,
        )
    )


def build_mjlab_env(
    env_id: str,
    num_envs: int,
    seed: int,
    device: str,
    features: tuple[Any, ...] = (),
) -> Any:
    """Build an mjlab ``ManagerBasedRlEnv`` through :func:`myosuite.make_env`."""
    try:
        import mjlab  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            'mjlab is required for GPU training. Install with: pip install "myosuite-s2pg[mjlab]"'
        ) from exc

    cfg = EnvConfig(env_id=env_id, num_envs=num_envs, features=features)
    return make_env(cfg, backend="mjlab", device=device, seed=seed)


def run_s2pg_on_mjlab_env(
    raw_env: Any,
    args: MjlabTrainArgs,
    log_dir: Path | None = None,
) -> dict[str, Any]:
    """Run S2PG-PPO (or baseline PPO) on an initialized ``ManagerBasedRlEnv``.

    Args:
        raw_env: Initialized ``ManagerBasedRlEnv`` or compatible tensor environment.
        args: Training run arguments.
        log_dir: Optional directory for checkpoints and ``summary.json``.

    Returns:
        Summary dictionary with training history and final diagnostics.
    """
    vec_env = MjlabTensorVecEnv(raw_env)
    trainer: S2PGPPOTrainer | FeedforwardPPOTrainer
    if args.baseline_ppo:
        trainer = FeedforwardPPOTrainer(vec_env=vec_env, cfg=args.s2pg)
    else:
        trainer = S2PGPPOTrainer(vec_env=vec_env, cfg=args.s2pg)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "config.json").write_text(
            json.dumps(
                {
                    "env_id": args.env_id,
                    "num_envs": args.num_envs,
                    "max_iterations": args.max_iterations,
                    "baseline_ppo": args.baseline_ppo,
                    "s2pg": asdict(args.s2pg),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    history: list[dict[str, float]] = []
    recent_returns: list[float] = []
    recent_successes: list[float] = []
    t_start = time.perf_counter()

    for it in range(1, args.max_iterations + 1):
        iter_t0 = time.perf_counter()
        progress = (it - 1) / max(1, args.max_iterations)
        batch, metrics = trainer.train_step(progress=progress)
        iter_dt = max(1e-9, time.perf_counter() - iter_t0)

        for ep in batch.completed_episodes:
            recent_returns.append(ep["return"])
            if "success" in ep:
                recent_successes.append(ep["success"])
        recent_returns = recent_returns[-200:]
        recent_successes = recent_successes[-200:]

        steps_this_iter = args.s2pg.num_steps_per_env * vec_env.num_envs
        mean_ret = float(np.mean(recent_returns)) if recent_returns else float("nan")
        mean_succ = (
            float(np.mean(recent_successes)) if recent_successes else float("nan")
        )
        record = {
            "iteration": float(it),
            "env_steps": float(trainer.env_steps),
            "fps": float(steps_this_iter / iter_dt),
            "mean_return": mean_ret,
            "mean_success": mean_succ,
            **metrics,
        }
        history.append(record)

        if it == 1 or it % 10 == 0 or it == args.max_iterations:
            succ_txt = f"{mean_succ:.2%}" if not np.isnan(mean_succ) else "n/a"
            print(
                f"[iter {it:4d}/{args.max_iterations}] steps={trainer.env_steps:,} "
                f"fps={record['fps']:,.0f} return={mean_ret:.2f} success={succ_txt} "
                f"kl={metrics['approx_kl']:.4f}"
            )

        if (
            log_dir is not None
            and isinstance(trainer, S2PGPPOTrainer)
            and args.save_interval > 0
            and (it % args.save_interval == 0 or it == args.max_iterations)
        ):
            trainer.save_checkpoint(log_dir / f"model_{it}.pt")

    total_wall_s = time.perf_counter() - t_start
    if log_dir is not None and args.export_ts and isinstance(trainer, S2PGPPOTrainer):
        export_torchscript(
            trainer.policy,
            output=log_dir / "policy_stateful.pt",
            clip_actions=args.s2pg.clip_actions,
        )

    summary: dict[str, Any] = {
        "env_id": args.env_id,
        "algorithm": "ppo" if args.baseline_ppo else "s2pg_ppo",
        "env_steps": trainer.env_steps,
        "wall_clock_s": total_wall_s,
        "final_mean_return": (
            float(np.mean(recent_returns)) if recent_returns else None
        ),
        "final_mean_success": (
            float(np.mean(recent_successes)) if recent_successes else None
        ),
        "history": history,
    }
    if log_dir is not None:
        (log_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser for ``s2pg-train-mjlab``."""
    parser = argparse.ArgumentParser(
        prog="s2pg-train-mjlab",
        description="Train S2PG-PPO directly on MyoSuite mjlab GPU environments.",
    )
    parser.add_argument("env_id", type=str, help="Registered MyoSuite env_id.")
    parser.add_argument(
        "--num-envs",
        "--env.scene.num-envs",
        dest="num_envs",
        type=int,
        default=1024,
        help="Number of parallel mjlab environments.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=500,
        help="Number of rollout + PPO update iterations.",
    )
    parser.add_argument(
        "--steps-per-env",
        type=int,
        default=24,
        help="Rollout horizon per environment per iteration.",
    )
    parser.add_argument(
        "--rnn-type",
        choices=list(VALID_RNN_TYPES),
        default="gru",
        help="Recurrent transition cell type.",
    )
    parser.add_argument(
        "--hidden-state-dim",
        type=int,
        default=64,
        help="Latent recurrent cell width.",
    )
    parser.add_argument(
        "--ema-coef",
        type=float,
        default=1.0,
        help="Exponential smoothing coefficient on next_z_mean in [0, 1].",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (default: cuda:0 if available else cpu).",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("logs/s2pg"),
        help="Root directory for checkpoints and logs.",
    )
    parser.add_argument("--run-name", type=str, default="", help="Optional run suffix.")
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="Iterations between saved checkpoints (0 disables intermediate saves).",
    )
    parser.add_argument(
        "--baseline-ppo",
        action="store_true",
        help="Train the matched stateless feedforward PPO baseline instead of S2PG-PPO.",
    )
    parser.add_argument(
        "--no-export-ts",
        action="store_true",
        help="Skip TorchScript stateful policy export at completion.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for ``s2pg-train-mjlab``."""
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    features, rest = parse_feature_args(raw_argv)
    parsed = build_arg_parser().parse_args(rest)

    device = parsed.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    rnn_type: RNNType = parsed.rnn_type
    s2pg_cfg = S2PGConfig(
        hidden_state_dim=parsed.hidden_state_dim,
        rnn_type=rnn_type,
        ema_coef=parsed.ema_coef,
        lr=parsed.lr,
        num_steps_per_env=parsed.steps_per_env,
        num_epochs=8,
        num_minibatches=8,
        gamma=0.97,
        clip_coef=0.3,
        seed=parsed.seed,
    )
    train_args = MjlabTrainArgs(
        env_id=parsed.env_id,
        num_envs=parsed.num_envs,
        max_iterations=parsed.max_iterations,
        save_interval=parsed.save_interval,
        device=device,
        log_root=parsed.log_root,
        run_name=parsed.run_name,
        baseline_ppo=parsed.baseline_ppo,
        export_ts=not parsed.no_export_ts,
        s2pg=s2pg_cfg,
    )

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if train_args.run_name:
        stamp = f"{stamp}_{train_args.run_name}"
    log_dir = train_args.log_root / train_args.env_id / stamp

    raw_env = build_mjlab_env(
        env_id=train_args.env_id,
        num_envs=train_args.num_envs,
        seed=s2pg_cfg.seed,
        device=device,
        features=features,
    )
    try:
        run_s2pg_on_mjlab_env(raw_env, train_args, log_dir=log_dir)
    finally:
        raw_env.close()


if __name__ == "__main__":
    main()
