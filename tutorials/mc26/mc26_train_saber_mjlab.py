#!/usr/bin/env python3
# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Train the mjlab Saber env with ONNX-native checkpoints.

For training with mimic-checkpoint initialisation use
``mc26_train_saber_mjlab_mimic_init.py`` instead.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
import myosuite
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scripts.train import TrainConfig
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
    bootstrap_myosuite_mjlab_registry,
)
from myosuite.integrations.musclemimic.mjlab_policy_runner import (
    OnnxCheckpointingMjlabRunner,
    _maybe_freeze_actor_std,
)
from myosuite.utils.onnx_checkpoint import get_wandb_onnx_checkpoint_path

DEFAULT_TASK = "myoChallengeSaberP0-v0"
DEFAULT_MIMIC_TASK = "myoChallengeSaberP0Mimic-v0"
DEFAULT_MIMIC_CLIP_NQ = 108
DEFAULT_MIMIC_CLIP_NV = 106


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the mjlab Saber env and persist checkpoints as ONNX bundles."
    )
    parser.add_argument("--task-id", default=DEFAULT_TASK)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--num-steps-per-env", type=int, default=24)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=150)
    parser.add_argument(
        "--seed", type=int, default=42
    )  # TODO: figure out if this seed is ever used
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument(
        "--log-root", type=Path, default=Path("logs") / "rsl_rl" / "saber_mjlab"
    )
    parser.add_argument(
        "--mimic-clip",
        type=Path,
        default=None,
        help="Optional clip file to train the clip-mimic saber variant.",
    )
    parser.add_argument(
        "--mimic-expected-nq",
        type=int,
        default=DEFAULT_MIMIC_CLIP_NQ,
        help="Expected qpos width for the clip file used by the saber mimic path.",
    )
    parser.add_argument(
        "--mimic-expected-nv",
        type=int,
        default=DEFAULT_MIMIC_CLIP_NV,
        help="Expected qvel width for the clip file used by the saber mimic path.",
    )
    parser.add_argument(
        "--mimic-reward-mode",
        choices=("mimic", "env", "augmented"),
        default="augmented",
        help="Reward composition for clip-backed saber training.",
    )
    parser.add_argument(
        "--mimic-reward-weight",
        type=float,
        default=5.0,
        help="Global scale applied to the clip-mimic reward term.",
    )
    parser.add_argument(
        "--env-reward-weight",
        type=float,
        default=1.0,
        help="Global scale applied to the native saber reward terms.",
    )
    parser.add_argument(
        "--disable-lookahead",
        action="store_true",
        help="Disable mimic lookahead observations for clip-backed training.",
    )
    parser.add_argument(
        "--tracking-only",
        action="store_true",
        help="Use site-tracking reward instead of the DeepMimic composite term.",
    )
    parser.add_argument(
        "--init-std",
        type=float,
        default=0.5,
        help=(
            "Initial Gaussian action std. Keep ≤0.5 so that the effective stochastic "
            "excitation (post-clamp to [0,1]) stays close to the deterministic mean. "
            "With the default 1.0 std can grow to ~3 during training, making the "
            "deterministic policy useless."
        ),
    )
    parser.add_argument(
        "--freeze-std",
        action="store_true",
        default=True,
        help=(
            "Freeze the Gaussian std parameter so PPO cannot grow it away from "
            "--init-std. Strongly recommended for excitation-mode envs to keep "
            "deterministic and stochastic policies aligned."
        ),
    )
    parser.add_argument(
        "--no-freeze-std",
        dest="freeze_std",
        action="store_false",
        help="Allow PPO to learn the action std (not recommended for excitation mode).",
    )
    return parser.parse_args(argv)


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _resolve_task_id(args: argparse.Namespace) -> str:
    if args.mimic_clip is not None and args.task_id == DEFAULT_TASK:
        return DEFAULT_MIMIC_TASK
    return str(args.task_id)


def _register_requested_tasks(args: argparse.Namespace) -> None:
    bootstrap_myosuite_mjlab_registry()
    if args.mimic_clip is None:
        return

    from mjlab.tasks.registry import register_mjlab_task

    from myosuite.core.trajectory_io import load_motion_clip
    from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
        default_mimic_clip_on_policy_runner_cfg,
        register_mimic_mjlab_tasks_with_clip,
    )

    clip = load_motion_clip(
        args.mimic_clip,
        expected_nq=int(args.mimic_expected_nq),
        expected_nv=int(args.mimic_expected_nv),
    )
    register_mimic_mjlab_tasks_with_clip(
        register_mjlab_task=register_mjlab_task,
        rl_cfg_fn=default_mimic_clip_on_policy_runner_cfg,
        clip=clip,
        use_deepmimic_reward=not args.tracking_only,
        use_lookahead=not args.disable_lookahead,
        reward_mode=str(args.mimic_reward_mode),
        mimic_reward_weight=float(args.mimic_reward_weight),
        env_reward_weight=float(args.env_reward_weight),
    )


def _prepare_config(args: argparse.Namespace) -> TrainConfig:
    env_cfg = load_env_cfg(args.task_id)
    agent_cfg = load_rl_cfg(args.task_id)
    env_cfg.scene.num_envs = int(args.num_envs)
    agent_cfg.num_steps_per_env = int(args.num_steps_per_env)
    agent_cfg.max_iterations = int(args.max_iterations)
    agent_cfg.save_interval = int(args.save_interval)
    agent_cfg.seed = int(args.seed)
    # Override init_std so the stochastic and deterministic policies stay aligned.
    # action_mode="excitation" clamps to [0,1]: with std>>1 the clamped stochastic
    # mean drifts far from the raw MLP output, so the deterministic policy becomes
    # useless even though training metrics look good.
    if agent_cfg.actor.distribution_cfg is not None:
        agent_cfg.actor.distribution_cfg["init_std"] = float(args.init_std)
    if args.run_name:
        agent_cfg.run_name = args.run_name
    return replace(
        TrainConfig(env=env_cfg, agent=agent_cfg),
        gpu_ids=[0] if _resolve_device(args.device).startswith("cuda") else None,
        video=False,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    args.task_id = _resolve_task_id(args)
    myosuite.register_all_envs()
    _register_requested_tasks(args)
    configure_torch_backends()

    cfg = _prepare_config(args)
    device = _resolve_device(args.device)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{timestamp}_{cfg.agent.run_name}" if cfg.agent.run_name else timestamp
    log_dir = args.log_root / run_name
    dump_yaml(log_dir / "params" / "env.yaml", asdict(cfg.env))
    dump_yaml(log_dir / "params" / "agent.yaml", asdict(cfg.agent))

    env = ManagerBasedRlEnv(cfg=cfg.env, device=device)
    vec_env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
    runner = OnnxCheckpointingMjlabRunner(
        vec_env,
        asdict(cfg.agent),
        str(log_dir),
        device,
        task_id=args.task_id,
    )
    try:
        if args.resume_from is not None:
            try:
                runner.load_onnx(args.resume_from)
                print(f"Resumed from ONNX checkpoint: {args.resume_from}")
            except Exception:
                resume_path, was_cached = get_wandb_onnx_checkpoint_path(
                    args.log_root,
                    Path(args.resume_from),
                )
                runner.load_onnx(resume_path)
                print(f"Resumed from ONNX checkpoint: {resume_path}")
        _maybe_freeze_actor_std(runner, learn_actor_std=not args.freeze_std)
        runner.learn(
            num_learning_iterations=cfg.agent.max_iterations,
            # init_at_random_ep_len=True,
        )
        runner.save(str(log_dir / "checkpoints" / "model_final.pt"))
    finally:
        env.close()


if __name__ == "__main__":
    main()
