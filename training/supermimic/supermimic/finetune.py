# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Fine-tune a trained baseline on a few clips, and evaluate it on them before and after.

The baseline checkpoint (a ``model_*.pt`` of the fast runner, e.g. trained on MuscleMimic's 972 KIT clips) brings
its policy (live and averaged), world model, critic, every normalizer and the optimizer states. Fine-tuning trains
on the given clips with the baseline's runner settings and env settings (control rate, goal stride, observation),
restarting the iteration count, the learning-rate schedule and the training clock, without the world-model
warm-up. Evaluation plays every given clip once from its first frame with the deployed (averaged) policy.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass
class FinetuneCfg:
    """Fine-tuning settings."""

    checkpoint: str
    """The baseline checkpoint (``model_*.pt``)."""
    motion_file: str
    """The clips to fine-tune on and evaluate: a clip, a directory or a ``.txt`` list."""
    motion_root: str = ""
    """The directory a list's paths start from."""
    out: str = "logs/supermimic_finetune"
    """Run directory: checkpoints, ``train_log.csv`` and ``finetune.json`` (the evaluations)."""
    iterations: int = 2000
    max_hours: float = 0.0
    """Stop after this much training time (0: no limit)."""
    num_envs: int = 1024
    pi_lr_scale: float = 1.0
    """Scales the baseline's learning rate for the fine-tuning (world model, policy and critic alike)."""
    save_interval: int = 250
    seed: int = 0
    device: str = "cuda:0"
    evaluate_before: bool = True


def finetune(cfg: FinetuneCfg) -> dict:
    """Fine-tune ``cfg.checkpoint`` on ``cfg.motion_file``.

    Args:
        cfg: The settings.

    Returns:
        The evaluations on the clips (``"before"`` if asked, ``"after"``) and the final checkpoint's path.
    """
    from supermimic.env import make_task_env, settings_cfg  # noqa: PLC0415
    from supermimic.evaluate import build, evaluate  # noqa: PLC0415
    from mjlab.utils.torch import configure_torch_backends  # noqa: PLC0415

    from supermimic.runner import SuperMimicRunner  # noqa: PLC0415

    configure_torch_backends()  # the training's flags (TF32), also the compile caches' keys
    ck = torch.load(cfg.checkpoint, map_location="cpu", weights_only=True)
    settings = ck.get("env") or {}
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    results: dict = {"checkpoint": cfg.checkpoint, "motion_file": cfg.motion_file}

    def evaluate_checkpoint(path: str) -> dict:
        env, runner = build(
            cfg.motion_file, cfg.device, motion_root=cfg.motion_root, settings=settings
        )
        runner.load(path, load_cfg={"actor": True})
        res = evaluate(env, runner)
        env.close()
        return res

    if cfg.evaluate_before:
        results["before"] = evaluate_checkpoint(cfg.checkpoint)
    env_cfg = settings_cfg(settings, motion_file=cfg.motion_file, motion_root=cfg.motion_root,
                           num_envs=cfg.num_envs)  # fmt: skip
    torch.manual_seed(cfg.seed)
    env = make_task_env(env_cfg, cfg.device, seed=cfg.seed)
    run_cfg = {
        **ck["cfg"],
        "finetune": True,
        "max_iterations": cfg.iterations,
        "max_hours": cfg.max_hours,
        "save_interval": cfg.save_interval,
        "seed": cfg.seed,
        "lr": ck["cfg"]["lr"] * cfg.pi_lr_scale,
    }
    runner = SuperMimicRunner(env, run_cfg, log_dir=str(out), device=cfg.device)
    runner.load(cfg.checkpoint)
    runner.learn(cfg.iterations)
    final = str(out / f"model_{runner.it:06d}.pt")
    env.close()
    results["after"] = evaluate_checkpoint(final)
    results["final_checkpoint"] = final
    results["cfg"] = asdict(cfg)
    (out / "finetune.json").write_text(json.dumps(results, indent=1))
    return results
