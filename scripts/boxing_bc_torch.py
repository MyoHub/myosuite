#!/usr/bin/env python3
# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""PyTorch BC policy cloned from a Flax/JAX MJX boxing checkpoint.

Workflow
--------
1. **export** – load Orbax checkpoint, port weights into a PyTorch model, save
   as ``<policy-out>.pt``.
2. **validate** – roll out the PyTorch policy in the CPU env, compare MAE
   against the original teacher, write a video and a JSON report.

The PyTorch model is SB3-compatible: wrap it in ``TorchBCPolicy`` and use
``policy.predict(obs)`` directly, or subclass ``stable_baselines3.common.policies.BasePolicy``
for full SB3 training integration.

Example::

    # export + validate in one shot (default paths)
    python scripts/boxing_bc_torch.py

    # validate only from a saved .pt
    python scripts/boxing_bc_torch.py --policy-in outputs/bc_torch/boxing_bc.pt
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any  # used in build_torch_policy type hints

import gymnasium as gym
import mujoco
import numpy as np
import torch
import torch.nn as nn

from myosuite import register_all_envs
from myosuite.integrations.musclemimic.actor_torch import BCActorNet
from myosuite.integrations.musclemimic.fullbody_checkpoint_io import (
    resolve_checkpoint_ref,
)
from myosuite.integrations.musclemimic.fullbody_local_policy import (
    FullbodyObsAdapter,
    LocalPolicyArtifacts,
    StandaloneBCPolicy,
    load_local_policy_artifacts,
    read_checkpoint_config_metadata,
)
from myosuite.integrations.musclemimic.fullbody_model import (
    compile_mimic_fullbody_mjmodel,
    default_mimic_fullbody_config,
)
from myosuite.integrations.musclemimic.trajectory_io import load_motion_clip
from myosuite.utils.video_io import write_video

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weight transfer: Flax → PyTorch
# ---------------------------------------------------------------------------


def _sorted_dense_indices(actor: dict[str, Any]) -> list[int]:
    indices: list[int] = []
    for key in actor:
        if key.startswith("Dense_") and key[6:].isdigit():
            indices.append(int(key[6:]))
    return sorted(indices)


def build_torch_policy(artifacts: LocalPolicyArtifacts) -> BCActorNet:
    """Build and weight-initialise a :class:`BCActorNet` from Orbax artifacts.

    Flax stores kernels as ``(in_features, out_features)``; PyTorch expects
    ``(out_features, in_features)``, so all kernels are transposed.

    Args:
        artifacts: Loaded checkpoint artifacts (params + normalisation stats).

    Returns:
        ``BCActorNet`` with weights transferred from the checkpoint, in eval
        mode on CPU.

    Raises:
        ValueError: If the checkpoint actor architecture is not the expected
            ``Dense_* / LayerNorm_*`` MLP format.
    """
    actor = artifacts.params["actor"]
    if "output" in actor:
        raise ValueError(
            "ResNet-block checkpoint architecture is not supported by this script. "
            "Use the simple Dense_* MLP checkpoint."
        )

    dense_indices = _sorted_dense_indices(actor)
    if not dense_indices:
        raise ValueError("No Dense_* layers found in checkpoint actor params.")

    hidden_indices = dense_indices[:-1]
    out_idx = dense_indices[-1]

    hidden_dims: list[int] = [
        int(np.asarray(actor[f"Dense_{i}"]["bias"]).shape[0]) for i in hidden_indices
    ]
    obs_std = np.sqrt(np.asarray(artifacts.obs_var, dtype=np.float32) + 1e-8)

    model = BCActorNet(
        obs_dim=artifacts.obs_dim,
        action_dim=artifacts.action_dim,
        hidden_dims=hidden_dims,
        obs_mean=artifacts.obs_mean,
        obs_std=obs_std,
    )

    # Transfer weights: iterate net children in insertion order.
    # net layout: [Linear, LayerNorm, SiLU] × N_hidden + [Linear_out]
    dense_layer_idx = 0
    ln_layer_idx = 0

    with torch.no_grad():
        for module in model.net:
            if isinstance(module, nn.Linear):
                if dense_layer_idx < len(hidden_indices):
                    flax_key = f"Dense_{hidden_indices[dense_layer_idx]}"
                else:
                    flax_key = f"Dense_{out_idx}"
                kernel = np.asarray(actor[flax_key]["kernel"], dtype=np.float32)
                bias = np.asarray(actor[flax_key]["bias"], dtype=np.float32)
                # Flax kernel: (in, out) → PyTorch weight: (out, in)
                module.weight.copy_(torch.as_tensor(kernel.T))
                module.bias.copy_(torch.as_tensor(bias))
                dense_layer_idx += 1

            elif isinstance(module, nn.LayerNorm):
                flax_key = f"LayerNorm_{ln_layer_idx}"
                scale = np.asarray(actor[flax_key]["scale"], dtype=np.float32)
                bias = np.asarray(actor[flax_key]["bias"], dtype=np.float32)
                module.weight.copy_(torch.as_tensor(scale))
                module.bias.copy_(torch.as_tensor(bias))
                ln_layer_idx += 1

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Validation rollout
# ---------------------------------------------------------------------------


def _validate(
    model: BCActorNet,
    teacher: StandaloneBCPolicy,
    env: gym.Env,
    fb_model: mujoco.MjModel,
    clip,
    adapter: FullbodyObsAdapter,
    horizon: int,
    cam: mujoco.MjvCamera,
    render_w: int,
    render_h: int,
) -> tuple[list[np.ndarray], list[float], dict]:
    """Roll out the PyTorch policy and compare against the teacher.

    Returns:
        Tuple of ``(frames, step_mae_list, info_last)``.
    """
    fb_data = mujoco.MjData(fb_model)
    mujoco.mj_resetDataKeyframe(fb_model, fb_data, 0)
    mujoco.mj_forward(fb_model, fb_data)
    env.reset(seed=0)

    mj_data = env.unwrapped.data
    mj_model = env.unwrapped.model
    renderer = mujoco.Renderer(mj_model, height=render_h, width=render_w)

    frames: list[np.ndarray] = []
    step_mae: list[float] = []
    info: dict = {}
    executed = 0

    for t in range(horizon):
        fb_data.qpos[:] = mj_data.qpos[: fb_model.nq]
        fb_data.qvel[:] = mj_data.qvel[: fb_model.nv]
        fb_data.act[:] = mj_data.act[: fb_model.nu]
        fb_data.ctrl[:] = mj_data.ctrl[: fb_model.nu]
        mujoco.mj_forward(fb_model, fb_data)

        frame_idx = t % int(clip.qpos.shape[0])
        full_obs = adapter.build(fb_data, frame_idx).astype(np.float32)

        torch_act = model.predict(full_obs)
        teacher_act = teacher.action(full_obs)
        step_mae.append(float(np.mean(np.abs(teacher_act - torch_act))))

        # BC actor in [-1, 1]; boxing env expects muscle activations in [0, 1].
        muscle = np.clip(0.5 * (torch_act + 1.0), 0.0, 1.0)
        _, _, term, trunc, info = env.step(muscle)
        executed += 1

        renderer.update_scene(mj_data, camera=cam)
        frames.append(np.asarray(renderer.render(), dtype=np.uint8).copy())
        if term or trunc:
            break

    renderer.close()
    return frames, step_mae, info, executed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-id", default="myoChallengeBoxingP0-v0")
    p.add_argument("--checkpoint", default="Boxing/checkpoint_13114")
    p.add_argument(
        "--motion",
        default="Boxing/motions/Transitions_mocap/mazen_c3d/punchboxing_push_poses.npz",
    )
    p.add_argument(
        "--policy-in",
        default="",
        help="Optional pre-exported .pt to validate directly (skip export).",
    )
    p.add_argument("--policy-out", default="outputs/bc_torch/boxing_bc.pt")
    p.add_argument("--horizon", type=int, default=300)
    p.add_argument("--video-out", default="outputs/bc_torch/boxing_bc_torch.mp4")
    p.add_argument(
        "--report-out", default="outputs/bc_torch/boxing_bc_torch_report.json"
    )
    p.add_argument("--cam-azimuth", type=float, default=90.0)
    p.add_argument("--cam-elevation", type=float, default=-8.0)
    p.add_argument("--cam-distance", type=float, default=1.0)
    p.add_argument("--cam-lookat-x", type=float, default=-0.25)
    p.add_argument("--cam-lookat-y", type=float, default=-0.10)
    p.add_argument("--cam-lookat-z", type=float, default=1.05)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    return p


def main() -> None:  # noqa: C901
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    register_all_envs()

    # --- load or export PyTorch policy ---
    policy_in = args.policy_in.strip()
    if policy_in:
        logger.info("Loading PyTorch policy from %s", policy_in)
        model: BCActorNet = torch.load(
            policy_in, map_location="cpu", weights_only=False
        )
        model.eval()
        # still need artifacts for teacher + adapter
        ckpt = resolve_checkpoint_ref(args.checkpoint)
        artifacts = load_local_policy_artifacts(ckpt.local_path)
        goal_params = (
            read_checkpoint_config_metadata(ckpt.local_path)
            .get("experiment", {})
            .get("env_params", {})
            .get("goal_params", {})
        )
    else:
        logger.info("Exporting PyTorch policy from checkpoint: %s", args.checkpoint)
        ckpt = resolve_checkpoint_ref(args.checkpoint)
        artifacts = load_local_policy_artifacts(ckpt.local_path)
        goal_params = (
            read_checkpoint_config_metadata(ckpt.local_path)
            .get("experiment", {})
            .get("env_params", {})
            .get("goal_params", {})
        )
        model = build_torch_policy(artifacts)
        out_pt = Path(args.policy_out)
        out_pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model, str(out_pt))
        logger.info("Saved PyTorch policy: %s", out_pt)

    teacher = StandaloneBCPolicy(artifacts)

    # --- quick parity check before rollout ---
    rng = np.random.default_rng(0)
    test_obs = (
        rng.standard_normal(artifacts.obs_dim).astype(np.float32)
        * np.sqrt(artifacts.obs_var + 1e-8)
        + artifacts.obs_mean
    )
    np_act = teacher.action(test_obs)
    torch_act = model.predict(test_obs)
    parity_mae = float(np.mean(np.abs(np_act - torch_act)))
    logger.info("Parity MAE on random obs (should be ~0): %.6f", parity_mae)
    if parity_mae > 1e-4:
        logger.warning(
            "Parity MAE %.6f exceeds 1e-4 — weight transfer may have failed.",
            parity_mae,
        )

    # --- env + motion clip setup ---
    env = gym.make(args.env_id, disable_env_checker=True, render_mode="rgb_array")
    fb_model, _spec, _xml = compile_mimic_fullbody_mjmodel(
        default_mimic_fullbody_config()
    )
    clip = load_motion_clip(
        Path(args.motion),
        expected_nq=fb_model.nq,
        expected_nv=fb_model.nv,
    )
    adapter = FullbodyObsAdapter(fb_model, clip, goal_params)

    mj_model = env.unwrapped.model
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = float(args.cam_azimuth)
    cam.elevation = float(args.cam_elevation)
    cam.distance = float(args.cam_distance)
    cam.lookat[:] = [args.cam_lookat_x, args.cam_lookat_y, args.cam_lookat_z]
    render_w = min(
        args.width, int(getattr(mj_model.vis.global_, "offwidth", args.width))
    )
    render_h = min(
        args.height, int(getattr(mj_model.vis.global_, "offheight", args.height))
    )

    logger.info("Running validation rollout (horizon=%d)…", args.horizon)
    frames, step_mae, last_info, executed = _validate(
        model=model,
        teacher=teacher,
        env=env,
        fb_model=fb_model,
        clip=clip,
        adapter=adapter,
        horizon=args.horizon,
        cam=cam,
        render_w=render_w,
        render_h=render_h,
    )
    env.close()

    # --- write video ---
    out_video = Path(args.video_out)
    out_video.parent.mkdir(parents=True, exist_ok=True)
    write_video(
        str(out_video),
        np.asarray(frames, dtype=np.uint8),
        outputdict={"-r": "30"},
    )
    logger.info("Wrote video: %s", out_video)

    # --- write report ---
    report = {
        "steps": int(executed),
        "parity_mae_random_obs": float(parity_mae),
        "teacher_torch_action_mae_mean": float(np.mean(step_mae)) if step_mae else None,
        "teacher_torch_action_mae_p95": float(np.percentile(step_mae, 95))
        if step_mae
        else None,
        "fell": bool(last_info.get("fell/agent_0", False)),
        "damage_last": float(last_info.get("damage_delivered/agent_0", 0.0)),
        "env_id": args.env_id,
        "policy_out": args.policy_out if not policy_in else None,
        "video": str(out_video.resolve()),
    }
    out_report = Path(args.report_out)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Report:\n%s", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
