# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""GPU mjlab training entrypoint for AMP and GAIL on registered MyoSuite twins."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from myosuite_amp.discriminator import DiscriminatorMode
from myosuite_amp.ppo_amp import (
    AMPConfig,
    AMPTrainer,
    ExpertDataset,
    TrainStepStats,
    VecStepBatch,
    export_policy_onnx,
    export_policy_torchscript,
)


def _extract_actor_obs(obs: Any) -> torch.Tensor:
    """Extract policy observation tensor from mjlab / RSL-RL observation containers."""
    if isinstance(obs, torch.Tensor):
        return obs.float()
    if hasattr(obs, "__contains__"):
        if "actor" in obs:
            return obs["actor"].float()
        if "policy" in obs:
            return obs["policy"].float()
    raise TypeError(f"Unsupported mjlab observation container of type {type(obs)!r}.")


class MjlabVecEnvAdapter:
    """Zero-copy GPU tensor adapter wrapping ``RslRlVecEnvWrapper`` or ``ManagerBasedRlEnv``."""

    def __init__(self, env: Any, clip_actions: float = 1.0) -> None:
        """Wrap an mjlab environment for :class:`AMPTrainer`.

        Args:
            env: ``RslRlVecEnvWrapper`` or ``ManagerBasedRlEnv`` instance.
            clip_actions: Symmetric action bound ``[-clip_actions, clip_actions]``.
        """
        self._env = env
        base = getattr(env, "unwrapped", env)
        self.num_envs = int(base.num_envs)
        self.device = torch.device(base.device)

        if hasattr(env, "get_observations"):
            sample_raw = env.get_observations()
            if isinstance(sample_raw, tuple):
                sample_raw = sample_raw[0]
        else:
            sample_raw, _ = env.reset()
        sample_obs = _extract_actor_obs(sample_raw)
        self.obs_dim = int(sample_obs.shape[-1])

        if hasattr(env, "num_actions"):
            self.act_dim = int(env.num_actions)
        elif hasattr(base, "action_manager"):
            self.act_dim = int(base.action_manager.total_action_dim)
        else:
            raise AttributeError(
                "Could not determine action dimension from mjlab environment."
            )

        bound = float(clip_actions)
        self.action_low = np.full(self.act_dim, -bound, dtype=np.float32)
        self.action_high = np.full(self.act_dim, bound, dtype=np.float32)

    def reset(self, seed: int | None = None) -> torch.Tensor:
        """Reset wrapped mjlab environment and return actor observations on device."""
        base = getattr(self._env, "unwrapped", self._env)
        if seed is not None and hasattr(base, "seed"):
            base.seed(seed)
        if hasattr(self._env, "reset"):
            out = self._env.reset()
            raw_obs = out[0] if isinstance(out, tuple) else out
        else:
            raw_obs = self._env.get_observations()
            if isinstance(raw_obs, tuple):
                raw_obs = raw_obs[0]
        return _extract_actor_obs(raw_obs).to(self.device)

    def step(self, actions: torch.Tensor) -> VecStepBatch:
        """Advance mjlab simulation on batched GPU tensors."""
        clamped = torch.clamp(
            actions.to(self.device),
            float(self.action_low[0]),
            float(self.action_high[0]),
        )
        step_out = self._env.step(clamped)
        if len(step_out) == 4:
            raw_obs, rew, dones, extras = step_out
            done_bool = dones.to(device=self.device, dtype=torch.bool).reshape(-1)
            time_outs = extras.get("time_outs") if isinstance(extras, dict) else None
            if time_outs is not None:
                truncated = torch.as_tensor(
                    time_outs, device=self.device, dtype=torch.bool
                ).reshape(-1)
            else:
                truncated = torch.zeros_like(done_bool)
            terminated = done_bool & (~truncated)
        elif len(step_out) == 5:
            raw_obs, rew, term, trunc, _ = step_out
            terminated = term.to(device=self.device, dtype=torch.bool).reshape(-1)
            truncated = trunc.to(device=self.device, dtype=torch.bool).reshape(-1)
            done_bool = terminated | truncated
        else:
            raise RuntimeError(
                f"Unexpected mjlab step tuple of length {len(step_out)}."
            )

        next_obs = _extract_actor_obs(raw_obs).to(self.device)
        rewards = rew.to(device=self.device, dtype=torch.float32).reshape(-1)
        valid_transition = ~done_bool
        return VecStepBatch(
            obs=next_obs,
            transition_next_obs=next_obs,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            valid_transition=valid_transition,
        )

    def close(self) -> None:
        """Close the underlying mjlab environment."""
        self._env.close()


def load_motion_npz(
    dataset_path: str | Path, device: str | torch.device = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load state-only transitions ``(obs, next_obs)`` onto ``device`` from an ``.npz`` file.

    Any ``actions`` array present in the archive is ignored so AMP and GAIL consume
    only state transitions ``(s_t, s_{t+1})``.

    Args:
        dataset_path: Path to an ``.npz`` archive with ``obs`` and ``next_obs`` keys.
        device: Target torch device.

    Returns:
        Tuple ``(obs_tensor, next_obs_tensor)`` on ``device``.
    """
    ds = ExpertDataset.load_npz(dataset_path).without_actions()
    target = torch.device(device)
    return (
        torch.as_tensor(ds.obs, dtype=torch.float32, device=target),
        torch.as_tensor(ds.next_obs, dtype=torch.float32, device=target),
    )


@dataclass
class MjlabAMPTrainConfig:
    """CLI configuration for ``amp-train-mjlab``."""

    dataset: str
    mode: DiscriminatorMode = "amp"
    proportion_env_reward: float = 0.5
    num_envs: int = 1024
    total_steps: int = 2_000_000
    rollout_steps: int = 24
    ppo_epochs: int = 6
    num_minibatches: int = 4
    learning_rate: float = 3e-4
    disc_learning_rate: float = 3e-4
    disc_updates_per_iter: int = 4
    disc_batch_size: int = 512
    disc_grad_penalty_weight: float = 5.0
    feature_indices: tuple[int, ...] | None = None
    seed: int = 0
    device: str | None = None
    log_root: str = "logs/amp_mjlab"
    export_torchscript: str | None = None
    export_onnx: str | None = None


def train_mjlab(
    task_id: str,
    cfg: MjlabAMPTrainConfig,
    features: tuple = (),
    env_cfg_override: Any = None,
) -> tuple[Path, list[TrainStepStats]]:
    """Train ``AMPTrainer`` on a registered MyoSuite mjlab GPU twin.

    Args:
        task_id: Registered MyoSuite task identifier.
        cfg: Training and dataset configuration.
        features: Optional muscle-command wrapper feature specs.
        env_cfg_override: Optional pre-configured ``ManagerBasedRlEnvCfg``.

    Returns:
        Tuple ``(log_dir, history)``.
    """
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.utils.torch import configure_torch_backends

    import myosuite.envs.myo.backends.mjlab  # noqa: F401

    device_str = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    configure_torch_backends()

    if env_cfg_override is not None:
        env_cfg = env_cfg_override
    elif features:
        from myosuite.envs.myo.backends.mjlab.tasks.registration import rebuild_twin_cfg

        env_cfg = rebuild_twin_cfg(task_id, features)
    else:
        env_cfg = load_env_cfg(task_id)

    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.seed = cfg.seed

    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path(cfg.log_root) / task_id / f"{run_stamp}_{cfg.mode}"
    log_dir.mkdir(parents=True, exist_ok=True)

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device_str)
    wrapped = RslRlVecEnvWrapper(raw_env, clip_actions=1.0)
    adapter = MjlabVecEnvAdapter(wrapped, clip_actions=1.0)
    try:
        exp_obs, exp_next = load_motion_npz(cfg.dataset, device=device_str)
        amp_cfg = AMPConfig(
            mode=cfg.mode,
            use_discriminator=True,
            proportion_env_reward=cfg.proportion_env_reward,
            feature_indices=cfg.feature_indices,
            rollout_steps=cfg.rollout_steps,
            ppo_epochs=cfg.ppo_epochs,
            num_minibatches=cfg.num_minibatches,
            learning_rate=cfg.learning_rate,
            disc_learning_rate=cfg.disc_learning_rate,
            disc_updates_per_iter=cfg.disc_updates_per_iter,
            disc_batch_size=cfg.disc_batch_size,
            disc_grad_penalty_weight=cfg.disc_grad_penalty_weight,
            seed=cfg.seed,
            device=device_str,
        )
        trainer = AMPTrainer(adapter, amp_cfg, expert_transitions=(exp_obs, exp_next))
        history = trainer.train(total_steps=cfg.total_steps)
        if cfg.export_torchscript:
            export_policy_torchscript(trainer.actor_critic, cfg.export_torchscript)
        if cfg.export_onnx:
            export_policy_onnx(trainer.actor_critic, cfg.export_onnx)
    finally:
        adapter.close()

    summary_path = log_dir / "metrics.json"
    summary_path.write_text(
        json.dumps(
            {
                "task_id": task_id,
                "config": asdict(cfg),
                "history": [asdict(step) for step in history],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return log_dir, history


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train AMP or GAIL on a MyoSuite mjlab GPU twin from an .npz motion dataset."
    )
    parser.add_argument(
        "task", help="Registered MyoSuite task ID (e.g., myoElbowPose1D6MRandom-v0)"
    )
    parser.add_argument(
        "--dataset", required=True, help="Path to .npz transition dataset"
    )
    parser.add_argument("--mode", choices=("amp", "gail"), default="amp")
    parser.add_argument("--proportion-env-reward", type=float, default=0.5)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--rollout-steps", type=int, default=24)
    parser.add_argument("--ppo-epochs", type=int, default=6)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--disc-learning-rate", type=float, default=3e-4)
    parser.add_argument("--disc-updates-per-iter", type=int, default=4)
    parser.add_argument("--disc-batch-size", type=int, default=512)
    parser.add_argument("--disc-grad-penalty-weight", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-root", default="logs/amp_mjlab")
    parser.add_argument("--export-torchscript", default=None)
    parser.add_argument("--export-onnx", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for ``amp-train-mjlab``."""
    raw_args = list(sys.argv[1:] if argv is None else argv)
    from myosuite.utils.feature_cli import parse_feature_args

    features, clean_args = parse_feature_args(raw_args)
    ns = _build_arg_parser().parse_args(clean_args)
    cfg = MjlabAMPTrainConfig(
        dataset=ns.dataset,
        mode=ns.mode,
        proportion_env_reward=ns.proportion_env_reward,
        num_envs=ns.num_envs,
        total_steps=ns.total_steps,
        rollout_steps=ns.rollout_steps,
        ppo_epochs=ns.ppo_epochs,
        num_minibatches=ns.num_minibatches,
        learning_rate=ns.learning_rate,
        disc_learning_rate=ns.disc_learning_rate,
        disc_updates_per_iter=ns.disc_updates_per_iter,
        disc_batch_size=ns.disc_batch_size,
        disc_grad_penalty_weight=ns.disc_grad_penalty_weight,
        seed=ns.seed,
        device=ns.device,
        log_root=ns.log_root,
        export_torchscript=ns.export_torchscript,
        export_onnx=ns.export_onnx,
    )
    log_dir, _ = train_mjlab(ns.task, cfg, features=features)
    print(f"[INFO] Completed {cfg.mode.upper()} mjlab run in {log_dir}")


if __name__ == "__main__":
    main()
