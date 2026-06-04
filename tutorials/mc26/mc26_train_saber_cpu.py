#!/usr/bin/env python3
# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Train the CPU Saber env with PPO and ONNX-native checkpoints."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import myosuite
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from myosuite.utils.onnx_checkpoint import extract_checkpoint_from_onnx
from myosuite.utils.sb3_callbacks import OnnxCheckpointCallback

DEFAULT_TASK = "myoChallengeSaberP0-v0"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the CPU Saber env and persist checkpoints as ONNX bundles."
    )
    parser.add_argument("--task-id", default=DEFAULT_TASK)
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-freq", type=int, default=50_000)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument(
        "--log-root", type=Path, default=Path("logs") / "sb3_ppo" / "saber_cpu"
    )
    return parser.parse_args()


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cpu"
    return requested


def _build_model(args: argparse.Namespace, env: Any) -> PPO:
    device = _resolve_device(args.device)
    policy_kwargs = {
        "activation_fn": torch.nn.ELU,
        "net_arch": {"pi": [256, 256, 256, 256], "vf": [256, 256, 256, 256]},
    }
    if args.resume_from is not None:
        checkpoint_path, _, temp_dir = extract_checkpoint_from_onnx(args.resume_from)
        try:
            return PPO.load(checkpoint_path, env=env, device=device)
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()
    return PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        seed=args.seed,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        policy_kwargs=policy_kwargs,
    )


def main() -> None:
    args = _parse_args()
    myosuite.register_all_envs()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{timestamp}_{args.run_name}" if args.run_name else timestamp
    run_dir = args.log_root / run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(args.task_id, n_envs=args.n_envs, seed=args.seed)
    try:
        model = _build_model(args, env)
        obs_dim = int(env.observation_space.shape[0])
        act_dim = int(env.action_space.shape[0])
        callback = OnnxCheckpointCallback(
            checkpoint_dir=checkpoint_dir,
            task_id=args.task_id,
            obs_dim=obs_dim,
            act_dim=act_dim,
            save_freq=args.save_freq,
        )
        model.learn(total_timesteps=args.total_timesteps, callback=callback)
    finally:
        env.close()


if __name__ == "__main__":
    main()
