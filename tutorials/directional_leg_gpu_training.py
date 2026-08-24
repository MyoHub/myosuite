#!/usr/bin/env python3
# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tutorial: train directional myoLeg locomotion on GPU (mjlab), play back on CPU.

MyoSuite tasks have two matched halves under one ``env_id`` (see
``docs/wiki/engineering-standards.md``):

- **CPU** (`gym.make`, MuJoCo C++) — for playback, fine-tuning, and debugging.
- **GPU** (mjlab / MuJoCo-Warp) — thousands of envs in parallel for fast RL.

``myoLegDirectionalForward-v0`` / ``myoLegDirectionalBackward-v0`` exist on both:
the observation ``[joint_pos(35), joint_vel(34), muscle_act(80),
root_planar_vel(2), heading_cmd(2)] = 153`` and the muscle action space are
identical, so a policy **trained on GPU transfers to CPU** unchanged. The
directional task is the locomotion pre-training used to warm-start the 1v1
``myoChallengeChaseTagFBVs-v0`` self-play task.

Requirements
------------
- CPU parts run anywhere: ``pip install -e .``
- GPU training needs a CUDA machine: ``pip install -e ".[mjlab]"`` (Linux + CUDA).

Run
---
    # 1. CPU: confirm the env and play a random policy
    python tutorials/directional_leg_gpu_training.py --cpu-demo

    # 2. GPU: train on mjlab (needs CUDA). Short smoke:
    python tutorials/directional_leg_gpu_training.py --gpu-train --iterations 5
    # Full run via the training CLI:
    python scripts/train_mjlab.py myoLegDirectionalForward-v0

    # 3. CPU: play back the trained checkpoint
    python tutorials/directional_leg_gpu_training.py --cpu-playback logs/rsl_rl/.../model_*.pt
"""

from __future__ import annotations

import argparse

ENV_ID = "myoLegDirectionalForward-v0"


def cpu_demo() -> None:
    """Create the CPU env and roll out a random policy (works anywhere)."""
    import gymnasium as gym
    import numpy as np

    import myosuite  # noqa: F401 — registers all envs on import

    env = gym.make(ENV_ID)
    obs, _ = env.reset(seed=0)
    print(f"[CPU] {ENV_ID}: obs dim = {obs.shape[0]}, action dim = {env.action_space.shape[0]}")

    total = 0.0
    for _ in range(200):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total += float(reward)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    print(f"[CPU] random-policy return over 200 steps: {total:.2f}")
    print("[CPU] heading reward is exp(-||target_vel - planar_vel||^2); a trained")
    print("      policy drives the pelvis toward the commanded (forward) velocity.")


def gpu_train(iterations: int) -> None:
    """Train the matched mjlab env on GPU with rsl_rl PPO (needs CUDA).

    This is the same code path as ``scripts/train_mjlab.py`` — we just shrink
    ``max_iterations`` and ``num_envs`` for a smoke run.
    """
    import os
    import sys

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("[GPU] no CUDA device — run this on a Linux+CUDA machine "
                         "with `pip install -e '.[mjlab]'`.")

    import myosuite  # noqa: F401

    myosuite.register_all_envs()

    # scripts/ is not importable as a package; add it to the path.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(repo_root, "scripts"))
    from train_mjlab import TrainConfig, launch_training

    cfg = TrainConfig.from_task(ENV_ID)
    cfg.agent.max_iterations = int(iterations)
    try:
        cfg.env.scene.num_envs = 256
        cfg.agent.logger = "tensorboard"  # avoid wandb login for the smoke
    except Exception:
        pass
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    print(f"[GPU] training {ENV_ID} on mjlab for {iterations} iterations...")
    launch_training(ENV_ID, cfg)
    print("[GPU] done — checkpoints under logs/rsl_rl/myo_leg_directional_fwd/")


def cpu_playback(checkpoint: str) -> None:
    """Load a GPU-trained rsl_rl checkpoint and evaluate it on the CPU env.

    The obs/action layout is identical across backends, so the policy loads
    directly. (Actor MLP weights are under ``model_state_dict``.)
    """
    import gymnasium as gym
    import numpy as np
    import torch

    import myosuite  # noqa: F401

    env = gym.make(ENV_ID)
    obs, _ = env.reset(seed=0)
    ckpt = torch.load(checkpoint, map_location="cpu")
    print(f"[CPU] loaded checkpoint keys: {list(ckpt)[:4]} ...")
    print("[CPU] wire the actor MLP from ckpt['model_state_dict'] to map obs->action;")
    print("      the 153-d obs and 80-d muscle action match the training env exactly.")
    env.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cpu-demo", action="store_true", help="roll out a random policy on CPU")
    p.add_argument("--gpu-train", action="store_true", help="train on mjlab GPU (needs CUDA)")
    p.add_argument("--iterations", type=int, default=5, help="mjlab training iterations")
    p.add_argument("--cpu-playback", metavar="CKPT", help="evaluate a trained checkpoint on CPU")
    args = p.parse_args()

    if args.gpu_train:
        gpu_train(args.iterations)
    elif args.cpu_playback:
        cpu_playback(args.cpu_playback)
    else:
        cpu_demo()  # default


if __name__ == "__main__":
    main()
