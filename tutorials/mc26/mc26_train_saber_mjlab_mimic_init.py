#!/usr/bin/env python3
# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Train the mjlab saber challenge env initialised from a MuscleMimic checkpoint.

Workflow overview
-----------------
Rather than training from scratch or using teacher-student distillation, this
script initialises the RSL-RL actor/critic weights directly from a full-body
MuscleMimic checkpoint (Dense/LayerNorm architecture).  The saber *environment*
wraps the native mjlab task in a :class:`CheckpointVecEnv` that exposes the
same full-body observation/action space as the checkpoint so that PPO rollouts
are collected in checkpoint space and the saber task reward signal can update
the same network end-to-end.

Usage
-----
::

    python tutorials/mc26/mc26_train_saber_mjlab_mimic_init.py \\
        --mimic-checkpoint Boxing/checkpoint_13114 \\
        --reference-motion Boxing/motions/Transitions_mocap/mazen_c3d/punchboxing_push_poses.npz \\
        --num-envs 256 \\
        --max-iterations 2000

Key library entry points
------------------------
* :class:`~myosuite.integrations.musclemimic.actor_torch.LayerNormSiLUMLPModel`
  — RSL-RL actor/critic whose Dense/LayerNorm architecture matches the mimic
  checkpoint so weights can be loaded directly.
* :class:`~myosuite.integrations.musclemimic.mjlab_policy_runner.CheckpointVecEnv`
  — RSL-RL VecEnv bridge that projects the native mjlab saber state into the
  full-body checkpoint observation/action space.
* :class:`~myosuite.integrations.musclemimic.mjlab_policy_runner.OnnxCheckpointingMjlabRunner`
  — RSL-RL runner that saves ONNX-bundled checkpoints instead of raw .pt files.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Sequence
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

# Orbax restore needs a CPU JAX device for the local mimic checkpoint.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import torch
import wandb
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts.train import TrainConfig
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

import myosuite
from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
    bootstrap_myosuite_mjlab_registry,
)
from myosuite.integrations.musclemimic.actor_torch import (
    LayerNormSiLUMLPModel,
    _load_dense_checkpoint_into_model,
)
from myosuite.integrations.musclemimic.fullbody_checkpoint_io import (
    resolve_checkpoint_ref,
)
from myosuite.integrations.musclemimic.fullbody_local_policy import (
    load_local_policy_artifacts,
)
from myosuite.integrations.musclemimic.mjlab_policy_runner import (
    CheckpointVecEnv,
    OnnxCheckpointingMjlabRunner,
    _initialize_runner_from_mimic_checkpoint,
    _maybe_freeze_actor_std,
    _run_sanity_rollouts,
)
from myosuite.integrations.musclemimic.model_bridge import (
    make_fullbody_checkpoint_bridged_policy,
)
from myosuite.utils.onnx_checkpoint import get_wandb_onnx_checkpoint_path

# ---------------------------------------------------------------------------
# Script-level constants
# ---------------------------------------------------------------------------

DEFAULT_TASK = "myoChallengeSaberP0-v0"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MIMIC_CHECKPOINT = REPO_ROOT / "Boxing/checkpoint_13114"
DEFAULT_REFERENCE_MOTION = (
    REPO_ROOT / "Boxing/motions/Transitions_mocap/mazen_c3d/punchboxing_push_poses.npz"
)
DEFAULT_LOG_ROOT = Path("logs") / "rsl_rl" / "saber_mjlab_mimic_init"
# Dotted import path used by RSL-RL to instantiate the actor/critic model class.
DEFAULT_MODEL_CLASS = (
    "myosuite.integrations.musclemimic.actor_torch:LayerNormSiLUMLPModel"
)
SABER_ENTITY_NAME = "saber_p0_robot"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train mjlab saber from the full-body mimic checkpoint used in the "
            "mc26 notebook by exposing the same full-body observation/action "
            "interface around the native saber task."
        )
    )
    parser.add_argument("--task-id", default=DEFAULT_TASK)
    parser.add_argument(
        "--num-envs",
        type=int,
        default=256,
        help=(
            "Number of parallel saber envs. Moderate values usually give much "
            "better aggregate throughput than a single env for the mimic-init path."
        ),
    )
    parser.add_argument("--num-steps-per-env", type=int, default=24)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--actor-init-std",
        type=float,
        default=0.05,
        help=(
            "Initial Gaussian action std used during PPO rollouts. Lower values keep "
            "the checkpoint-resumed policy close to deterministic behavior."
        ),
    )
    parser.add_argument(
        "--learn-actor-std",
        action="store_true",
        help=(
            "Let PPO update the Gaussian exploration std. By default the mimic-init "
            "path keeps the low resume std fixed so rollout behavior stays close to "
            "the imported checkpoint."
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="PPO learning rate used after mimic-checkpoint initialization.",
    )
    parser.add_argument(
        "--num-learning-epochs",
        type=int,
        default=6,
        help="PPO epochs per iteration for the checkpoint-resume path.",
    )
    parser.add_argument(
        "--clip-param",
        type=float,
        default=0.2,
        help="PPO clip parameter for the checkpoint-resume path.",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.01,
        help="Entropy regularization for the checkpoint-resume path.",
    )
    parser.add_argument(
        "--desired-kl",
        type=float,
        default=0.01,
        help="Adaptive-KL target used by PPO after checkpoint initialization.",
    )
    parser.add_argument("--run-name", default="")
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help=(
            "Optional native saber-checkpoint ONNX bundle created by this script. "
            "When omitted, weights are imported from --mimic-checkpoint."
        ),
    )
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument(
        "--mimic-checkpoint",
        type=Path,
        default=DEFAULT_MIMIC_CHECKPOINT,
        help="Full-body mimic Orbax checkpoint root used in the notebook path.",
    )
    parser.add_argument(
        "--reference-motion",
        type=Path,
        default=DEFAULT_REFERENCE_MOTION,
        help="Reference motion clip used to build the full-body observation.",
    )
    parser.add_argument(
        "--sanity-episodes",
        type=int,
        default=3,
        help="Deterministic zero-exploration episodes to run after initialization.",
    )
    parser.add_argument(
        "--sanity-min-length",
        type=int,
        default=950,
        help="Minimum acceptable episode length for every sanity rollout.",
    )
    parser.add_argument(
        "--skip-sanity",
        action="store_true",
        help="Skip the deterministic timeout sanity rollouts.",
    )
    return parser.parse_args(argv)


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda:0" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Config assembly
# ---------------------------------------------------------------------------


def _prepare_config(args: argparse.Namespace) -> TrainConfig:
    env_cfg = load_env_cfg(args.task_id)
    agent_cfg = load_rl_cfg(args.task_id)
    env_cfg.scene.num_envs = int(args.num_envs)
    agent_cfg.num_steps_per_env = int(args.num_steps_per_env)
    agent_cfg.max_iterations = int(args.max_iterations)
    agent_cfg.save_interval = int(args.save_interval)
    agent_cfg.seed = int(args.seed)
    # Both actor and critic use the Dense/LayerNorm/SiLU architecture that
    # matches the checkpoint so weights can be loaded directly after init.
    for model_cfg in (agent_cfg.actor, agent_cfg.critic):
        model_cfg.class_name = DEFAULT_MODEL_CLASS
        model_cfg.hidden_dims = (1024, 1024, 1024, 1024, 1024)
        model_cfg.activation = "silu"
        model_cfg.obs_normalization = True
    agent_cfg.actor.distribution_cfg = {
        "class_name": "GaussianDistribution",
        "init_std": float(args.actor_init_std),
        "std_type": "scalar",
    }
    agent_cfg.critic.distribution_cfg = None
    agent_cfg.algorithm.learning_rate = float(args.learning_rate)
    agent_cfg.algorithm.num_learning_epochs = int(args.num_learning_epochs)
    agent_cfg.algorithm.clip_param = float(args.clip_param)
    agent_cfg.algorithm.entropy_coef = float(args.entropy_coef)
    agent_cfg.algorithm.desired_kl = float(args.desired_kl)
    if args.run_name:
        agent_cfg.run_name = args.run_name
    return replace(
        TrainConfig(env=env_cfg, agent=agent_cfg),
        gpu_ids=[0] if _resolve_device(args.device).startswith("cuda") else None,
        video=False,
    )


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    """Run saber training with mimic-checkpoint initialization.

    This function is also imported by the tutorial notebook to reuse the
    config/checkpoint machinery in evaluation cells.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    if args.task_id != DEFAULT_TASK:
        raise ValueError(
            f"This mimic-init script only supports {DEFAULT_TASK!r}, got {args.task_id!r}."
        )

    myosuite.register_all_envs()
    bootstrap_myosuite_mjlab_registry()
    configure_torch_backends()

    cfg = _prepare_config(args)
    device = _resolve_device(args.device)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{timestamp}_{cfg.agent.run_name}" if cfg.agent.run_name else timestamp
    log_dir = args.log_root / run_name
    dump_yaml(log_dir / "params" / "env.yaml", asdict(cfg.env))
    dump_yaml(log_dir / "params" / "agent.yaml", asdict(cfg.agent))

    # --- Environment setup ---
    # CheckpointVecEnv wraps the native mjlab saber env and exposes the
    # full-body checkpoint observation/action space for RSL-RL.
    env = ManagerBasedRlEnv(cfg=cfg.env, device=device)
    vec_env = CheckpointVecEnv(
        env,
        checkpoint_root=args.mimic_checkpoint,
        motion_path=args.reference_motion,
    )
    runner = OnnxCheckpointingMjlabRunner(
        vec_env,
        asdict(cfg.agent),
        str(log_dir),
        device,
        task_id=args.task_id,
    )

    try:
        # --- Runner + checkpoint init / resume ---
        if args.resume_from is not None:
            try:
                runner.load_onnx(args.resume_from)
                logger.info("Resumed from ONNX checkpoint: %s", args.resume_from)
            except Exception:
                resume_path, _was_cached = get_wandb_onnx_checkpoint_path(
                    args.log_root, Path(args.resume_from),
                )
                runner.load_onnx(resume_path)
                logger.info("Resumed from ONNX checkpoint: %s", resume_path)
        else:
            # Import Dense/LayerNorm weights from the mimic checkpoint directly.
            _initialize_runner_from_mimic_checkpoint(runner, args.mimic_checkpoint)
            logger.info("Imported actor/critic weights from: %s", args.mimic_checkpoint)
            runner.save(str(log_dir / "checkpoints" / "model_init.pt"))
        _maybe_freeze_actor_std(runner, learn_actor_std=bool(args.learn_actor_std))

        # --- Sanity check (deterministic mimic rollouts) ---
        # Run a handful of zero-exploration episodes to verify that the imported
        # checkpoint produces sensible saber behavior before starting PPO.
        if not args.skip_sanity:
            sanity_lengths = _run_sanity_rollouts(
                runner.get_inference_policy(device=device),
                env,
                checkpoint_root=args.mimic_checkpoint,
                motion_path=args.reference_motion,
                device=device,
                num_episodes=int(args.sanity_episodes),
                min_episode_length=int(args.sanity_min_length),
            )
            logger.info("Deterministic sanity episode lengths: %s", sanity_lengths)

        # --- Training loop ---
        if int(cfg.agent.max_iterations) > 0:
            runner.learn(num_learning_iterations=cfg.agent.max_iterations)
        runner.save(str(log_dir / "checkpoints" / "model_final.pt"))
    finally:
        env.close()


if __name__ == "__main__":
    main()
