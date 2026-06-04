#!/usr/bin/env python3
# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Render rollout of an mjlab-trained SAR policy using ONNX on CPU.

This script:
  1. Loads the trained checkpoint (ppo_model.pt) from a Phase B run directory.
  2. Reconstructs the *actor network only* using RSL-RL's ``MLPModel`` and the
     saved ``actor_state_dict`` (no mjlab import needed).
  3. Exports that actor network to ONNX (deterministic output).
  4. Runs a rollout on the CPU ``myoLegWalk-v0`` gymnasium env, applying the
     SAR inverse transform on CPU and saving an MP4 video.

It is intentionally decoupled from mjlab's Warp backend during rendering:
the ONNX policy runs on CPU via onnxruntime, and physics/rendering are done
entirely in the standard MyoSuite gymnasium environment.

Usage (from repo root):

    python benchmarks/sar_backends/render_mjlab_onnx.py \\
        --run-dir sar_benchmark_results/phase_b_training/mjlab \\
        --sar-dir myosuite/agents/SAR_pretrained/locomotion \\
        --output eval_rollout_onnx.mp4 \\
        --steps 500
"""

from __future__ import annotations

import argparse
import os
import pathlib

import numpy as np

# Ensure repo root is on sys.path when running as a script
import pathlib as _pathlib
import sys as _sys

_REPO_ROOT = _pathlib.Path(__file__).parents[2]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

# Headless rendering defaults (can be overridden by user env vars)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def _export_policy_to_onnx(
    run_dir: pathlib.Path,
    sar_dir: pathlib.Path,
    device: str = "cuda:0",
) -> pathlib.Path:
    """Export an ONNX policy from an mjlab PPO checkpoint.

    Args:
        run_dir: Phase B run directory containing ``ppo_model.pt``.
        sar_dir: Directory with ``ica.pkl``, ``pca.pkl``, ``normalizer.pkl``.
        device: Torch device for exporting the model (default: cuda:0 if available).

    Returns:
        Path to the exported ONNX file.
    """
    import torch
    import torch.nn as nn

    checkpoint_path = run_dir / "ppo_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"mjlab checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "actor_state_dict" not in ckpt:
        raise ValueError(f"Unexpected checkpoint format in {checkpoint_path}")

    actor_sd: dict[str, torch.Tensor] = ckpt["actor_state_dict"]
    # Infer (obs_dim, hidden_dims, output_dim) from MLP weights.
    w0 = actor_sd.get("mlp.0.weight")
    w4 = actor_sd.get("mlp.4.weight")
    w6 = actor_sd.get("mlp.6.weight")
    if w0 is None or w4 is None or w6 is None:
        raise ValueError("Checkpoint missing expected MLP weights (mlp.0/mlp.4/mlp.6).")
    obs_dim = int(w0.shape[1])
    h1 = int(w0.shape[0])
    h3 = int(w4.shape[0])
    # hidden dims are (h1, h2, h3); infer h2 from mlp.2.weight
    w2 = actor_sd.get("mlp.2.weight")
    if w2 is None:
        raise ValueError("Checkpoint missing expected weight mlp.2.weight.")
    h2 = int(w2.shape[0])
    out_dim = int(w6.shape[0])

    mean = actor_sd.get("obs_normalizer._mean")
    std = actor_sd.get("obs_normalizer._std")
    if mean is None or std is None:
        raise ValueError("Checkpoint missing obs_normalizer stats (_mean/_std).")

    class _ActorDeterministic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("mean", mean.reshape(1, -1).to(dtype=torch.float32))
            self.register_buffer("std", std.reshape(1, -1).to(dtype=torch.float32))
            self.fc0 = nn.Linear(obs_dim, h1)
            self.fc1 = nn.Linear(h1, h2)
            self.fc2 = nn.Linear(h2, h3)
            self.fc3 = nn.Linear(h3, out_dim)
            self.act = nn.ELU()

            # Load weights/biases from checkpoint
            self.fc0.weight.data.copy_(actor_sd["mlp.0.weight"].to(dtype=torch.float32))
            self.fc0.bias.data.copy_(actor_sd["mlp.0.bias"].to(dtype=torch.float32))
            self.fc1.weight.data.copy_(actor_sd["mlp.2.weight"].to(dtype=torch.float32))
            self.fc1.bias.data.copy_(actor_sd["mlp.2.bias"].to(dtype=torch.float32))
            self.fc2.weight.data.copy_(actor_sd["mlp.4.weight"].to(dtype=torch.float32))
            self.fc2.bias.data.copy_(actor_sd["mlp.4.bias"].to(dtype=torch.float32))
            self.fc3.weight.data.copy_(actor_sd["mlp.6.weight"].to(dtype=torch.float32))
            self.fc3.bias.data.copy_(actor_sd["mlp.6.bias"].to(dtype=torch.float32))

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            # Deterministic action mean in synergy space.
            x = (obs - self.mean) / (self.std + 1e-8)
            x = self.act(self.fc0(x))
            x = self.act(self.fc1(x))
            x = self.act(self.fc2(x))
            return self.fc3(x)

    actor = _ActorDeterministic().to(device)
    actor.eval()

    onnx_dir = run_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / "policy.onnx"

    dummy = torch.zeros(1, obs_dim, dtype=torch.float32, device=device)
    torch.onnx.export(
        actor,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=18,
        verbose=False,
        input_names=["obs"],
        output_names=["actions"],
    )
    return onnx_path


def _render_with_onnx_policy(
    onnx_path: pathlib.Path,
    sar_dir: pathlib.Path,
    output_path: pathlib.Path,
    n_steps: int = 500,
) -> None:
    """Render a rollout on CPU using an ONNX policy and CPU gym env.

    Args:
        onnx_path: Path to exported ONNX policy file.
        sar_dir: Path to SAR pkl files (ica.pkl, pca.pkl, normalizer.pkl).
        output_path: Where to write the MP4 video.
        n_steps: Number of environment steps to render.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "onnxruntime is required to render mjlab ONNX rollouts. "
            "Install with: pip install onnxruntime"
        ) from exc

    import gymnasium as gym
    import imageio
    import myosuite

    myosuite.register_all_envs()

    from benchmarks.sar_backends.run_benchmark import (
        load_sar,
        synergy_inverse_transform,
    )

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX policy not found: {onnx_path}")

    ica, pca, normalizer = load_sar(sar_dir)

    sess = ort.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    env = gym.make("myoLegWalk-v0", reset_type="init", render_mode="rgb_array")
    obs, _ = env.reset(seed=42)

    frames: list[np.ndarray] = []
    for _ in range(n_steps):
        obs_batch = obs.astype(np.float32)[None, :]
        syn = sess.run([output_name], {input_name: obs_batch})[0]  # (1, n_syn)
        muscle = synergy_inverse_transform(syn[0], ica, pca, normalizer)  # (n_muscles,)

        obs, _, terminated, truncated, _ = env.step(muscle)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()

    if not frames:
        raise RuntimeError("No frames captured during ONNX rollout rendering.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(output_path), frames, fps=50, codec="libx264")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render mjlab SAR policy rollout via ONNX on CPU."
    )
    parser.add_argument(
        "--run-dir",
        type=pathlib.Path,
        default=pathlib.Path("sar_benchmark_results/phase_b_training/mjlab"),
        help="Phase B run directory containing ppo_model.pt (default: sar_benchmark_results/phase_b_training/mjlab).",
    )
    parser.add_argument(
        "--sar-dir",
        type=pathlib.Path,
        default=pathlib.Path("myosuite/agents/SAR_pretrained/locomotion"),
        help="Directory with SAR pkl files (ica.pkl, pca.pkl, normalizer.pkl).",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Output MP4 path (default: <run-dir>/eval_rollout_onnx.mp4).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of environment steps to render (default: 500).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help='Torch device used for reconstructing the mjlab runner (default: "cuda:0").',
    )
    args = parser.parse_args()

    run_dir: pathlib.Path = args.run_dir
    sar_dir: pathlib.Path = args.sar_dir
    output_path: pathlib.Path = args.output or (run_dir / "eval_rollout_onnx.mp4")

    onnx_path = run_dir / "onnx" / "policy.onnx"
    if not onnx_path.exists():
        onnx_path = _export_policy_to_onnx(
            run_dir=run_dir, sar_dir=sar_dir, device=args.device
        )

    _render_with_onnx_policy(
        onnx_path=onnx_path,
        sar_dir=sar_dir,
        output_path=output_path,
        n_steps=args.steps,
    )


if __name__ == "__main__":
    main()
