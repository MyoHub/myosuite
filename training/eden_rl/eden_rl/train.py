# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Runner configuration builder & CLI wiring ``eden_rl`` into MyoSuite ``mjlab`` tasks."""

from __future__ import annotations

import argparse
import copy
from collections.abc import Sequence
from typing import Any

from eden_rl.amp import AMP_PPO_CLASS_NAME
from eden_rl.muon_ppo import MUON_PPO_CLASS_NAME
from eden_rl.simba import SIMBA_MODEL_CLASS_NAME


def configure_eden_rsl_runner(
    base_runner_cfg: dict[str, Any],
    *,
    backbone: str = "simba",
    algo: str = "muon_ppo",
    hidden_dim: int = 512,
    num_blocks: int = 2,
    latent_dim: int = 128,
    amp_motion_file: str | None = None,
) -> dict[str, Any]:
    """Overlay ``SimBaModel`` and ``MuonPPO`` / ``AmpPPO`` onto an RSL-RL >= 5 runner dict."""
    cfg = copy.deepcopy(base_runner_cfg)

    if backbone == "simba":
        for role in ("actor", "critic"):
            role_cfg = dict(cfg.get(role, {}))
            role_cfg.pop("hidden_dims", None)
            role_cfg.update(
                {
                    "class_name": SIMBA_MODEL_CLASS_NAME,
                    "hidden_dims": [],
                    "hidden_dim": int(hidden_dim),
                    "num_blocks": int(num_blocks),
                    "latent_dim": int(latent_dim),
                    "activation": "swish",
                    "obs_normalization": True,
                }
            )
            cfg[role] = role_cfg
    elif backbone != "mlp":
        raise ValueError(f"Unsupported backbone {backbone!r}; choose 'simba' or 'mlp'.")

    alg_cfg = dict(cfg.get("algorithm", {}))
    if algo == "muon_ppo":
        alg_cfg["class_name"] = MUON_PPO_CLASS_NAME
    elif algo == "amp_ppo":
        if not amp_motion_file:
            raise ValueError("--algo amp_ppo requires --amp-motion <path>.")
        alg_cfg["class_name"] = AMP_PPO_CLASS_NAME
        alg_cfg["amp_motion_file"] = str(amp_motion_file)
    elif algo != "ppo":
        raise ValueError(
            f"Unsupported algo {algo!r}; choose 'ppo', 'muon_ppo', or 'amp_ppo'."
        )
    cfg["algorithm"] = alg_cfg
    return cfg


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="myosuite-eden-train",
        description="Train a MyoSuite mjlab task with SimBaV2 / MuonPPO / AMP-PPO.",
    )
    p.add_argument("--env", default="myoLegWalk-v0", help="MyoSuite task ID.")
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--max-iterations", type=int, default=2000)
    p.add_argument("--backbone", choices=["simba", "mlp"], default="simba")
    p.add_argument("--algo", choices=["muon_ppo", "amp_ppo", "ppo"], default="muon_ppo")
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument(
        "--amp-motion", default=None, help="Motion .npz bundle or myoMimic clip."
    )
    p.add_argument("--log-dir", default="logs/eden_rl")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--dry-run", action="store_true", help="Print resolved runner JSON and exit."
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for ``myosuite-eden-train``."""
    import json

    args = _build_parser().parse_args(argv)
    base_cfg: dict[str, Any] = {
        "num_steps_per_env": 24,
        "max_iterations": args.max_iterations,
        "save_interval": 100,
        "experiment_name": args.env,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "actor": {
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0}
        },
        "critic": {},
        "algorithm": {
            "learning_rate": 3e-4,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "desired_kl": 0.01,
        },
    }
    runner_cfg = configure_eden_rsl_runner(
        base_cfg,
        backbone=args.backbone,
        algo=args.algo,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        latent_dim=args.latent_dim,
        amp_motion_file=args.amp_motion,
    )
    if args.dry_run:
        print(json.dumps(runner_cfg, indent=2))
        return 0

    from pathlib import Path

    from rsl_rl.runners import OnPolicyRunner

    from myosuite.core.registry import make_env

    env = make_env(
        args.env, backend="mjlab", num_envs=args.num_envs, device=args.device
    )
    log_dir = Path(args.log_dir) / args.env
    log_dir.mkdir(parents=True, exist_ok=True)
    runner = OnPolicyRunner(env, runner_cfg, log_dir=str(log_dir), device=args.device)
    runner.learn(
        num_learning_iterations=args.max_iterations, init_at_random_ep_len=True
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
