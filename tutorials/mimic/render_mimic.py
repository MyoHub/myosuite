#!/usr/bin/env python3
# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Render a trained MuscleMimic policy alongside the ghost reference motion.

The passive MuJoCo viewer shows:
  - Policy-driven body (rendered by physics)
  - Semi-transparent ghost skeleton from motion clip (reference target)

Usage::

    # Interactive viewer
    python render_mimic.py \\
        --clip ~/.cache/huggingface/.../walking_medium06_poses.npz \\
        --checkpoint tutorials/mimic/mimic_policy_demo.pt

    # Export to MP4
    python render_mimic.py \\
        --clip ... --checkpoint ... --output render.mp4 --n-frames 500

The script falls back to running without a policy (replay mode) if
--checkpoint is omitted.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import mujoco
import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────


def _load_policy(checkpoint_path: pathlib.Path, obs_dim: int, act_dim: int):
    """Load trained ActorCritic policy from checkpoint."""
    import torch

    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from train_mimic import ActorCritic

    policy = ActorCritic(obs_dim, act_dim)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    policy.load_state_dict(state)
    policy.eval()
    return policy


def _make_env(clip_path: pathlib.Path, max_episode_steps: int = 1000):
    """Create MuscleMimicClipEnvV0 for the given clip."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from myosuite.envs.myo.tasks.mimic.clip_env import MuscleMimicClipEnvV0

    return MuscleMimicClipEnvV0(
        clip_path=clip_path,
        max_episode_steps=max_episode_steps,
        frame_skip=5,
        random_start=False,
    )


# ── interactive rendering ─────────────────────────────────────────────────────


def _run_viewer(
    env,
    policy,
    ghost_fn,
    *,
    playback_speed: float = 1.0,
) -> None:
    """Run the MuJoCo passive viewer with ghost overlay."""
    try:
        from mujoco import viewer as mj_viewer
    except ImportError:
        print("mujoco.viewer not available — cannot open interactive window.")
        return

    import torch

    obs, _ = env.reset()
    step_idx = 0
    done = False

    def _key_callback(keycode):
        nonlocal obs, step_idx, done
        if keycode == ord("R"):
            obs, _ = env.reset()
            step_idx = 0
            done = False

    with mj_viewer.launch_passive(
        env.model, env.data, key_callback=_key_callback
    ) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        while viewer.is_running():
            t0 = time.time()

            if done:
                obs, _ = env.reset()
                step_idx = 0
                done = False

            # Step policy
            if policy is not None:
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action, _, _, _ = policy.get_action_and_value(obs_t)
                action_np = action.squeeze(0).numpy()
                obs, _, term, trunc, _ = env.step(action_np)
                done = bool(term or trunc)
            else:
                # Pure replay: advance clip frame, set qpos/qvel directly
                T = env._clip_qpos.shape[0]
                frame = step_idx % T
                env.data.qpos[:] = env._clip_qpos[frame]
                env.data.qvel[:] = env._clip_qvel[frame]
                mujoco.mj_forward(env.model, env.data)
                obs = env._build_obs()

            # Draw ghost
            with viewer.lock():
                ghost_fn(step_idx, env.model, env.data, viewer.user_scn)

            viewer.sync()
            step_idx += 1

            # Throttle to real-time
            elapsed = time.time() - t0
            target = env.model.opt.timestep * env.frame_skip / playback_speed
            if elapsed < target:
                time.sleep(target - elapsed)


# ── offline MP4 export ────────────────────────────────────────────────────────


def _export_mp4(
    env,
    policy,
    ghost_fn,
    output_path: pathlib.Path,
    *,
    n_frames: int = 500,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
) -> None:
    """Render n_frames to an MP4 file using mujoco.Renderer."""
    import torch

    try:
        import mediapy
    except ImportError:
        print("mediapy not installed — cannot export MP4.  Run: pip install mediapy")
        return

    renderer = mujoco.Renderer(env.model, height=height, width=width)
    frames: list[np.ndarray] = []

    obs, _ = env.reset()
    for step_idx in range(n_frames):
        if policy is not None:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = policy.get_action_and_value(obs_t)
            action_np = action.squeeze(0).numpy()
            obs, _, term, trunc, _ = env.step(action_np)
            if term or trunc:
                obs, _ = env.reset()
        else:
            T = env._clip_qpos.shape[0]
            frame = step_idx % T
            env.data.qpos[:] = env._clip_qpos[frame]
            env.data.qvel[:] = env._clip_qvel[frame]
            mujoco.mj_forward(env.model, env.data)

        # Ghost geoms into scene
        renderer.update_scene(env.data)
        ghost_fn(step_idx, env.model, env.data, renderer.scene)

        frames.append(renderer.render())

    mediapy.write_video(str(output_path), frames, fps=fps)
    print(f"Saved {len(frames)} frames → {output_path}")
    renderer.close()


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--clip",
        required=True,
        type=pathlib.Path,
        help="Path to retargeted .npz motion clip",
    )
    p.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        default=None,
        help="Path to saved ActorCritic state_dict (.pt). "
        "Omit for pure replay mode.",
    )
    p.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="If set, render to this MP4 file instead of viewer",
    )
    p.add_argument(
        "--n-frames", type=int, default=500, help="Number of frames for --output mode"
    )
    p.add_argument("--fps", type=int, default=30, help="FPS for exported video")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="Viewer playback speed multiplier",
    )
    p.add_argument(
        "--skeleton-alpha", type=float, default=0.35, help="Ghost skeleton opacity"
    )
    p.add_argument("--capsule-radius", type=float, default=0.018)
    p.add_argument("--joint-radius", type=float, default=0.022)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if not args.clip.is_file():
        sys.exit(f"Clip not found: {args.clip}")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        sys.exit(f"Checkpoint not found: {args.checkpoint}")

    print(f"Loading env from clip: {args.clip}")
    env = _make_env(args.clip)

    print("Building ghost viz...")
    from myosuite.viz.ghost_body_viz import GhostBodyViz

    ghost_viz = GhostBodyViz.from_clip_and_model(
        args.clip,
        env.model,
        skeleton_rgba=(0.3, 0.6, 1.0, args.skeleton_alpha),
        joint_rgba=(0.5, 0.8, 1.0, min(1.0, args.skeleton_alpha + 0.15)),
        capsule_radius=args.capsule_radius,
        joint_radius=args.joint_radius,
    )
    ghost_fn = ghost_viz.as_viz_fn()

    policy = None
    if args.checkpoint is not None:
        print(f"Loading policy from: {args.checkpoint}")
        policy = _load_policy(args.checkpoint, env.obs_dim, env.act_dim)
    else:
        print("No checkpoint — running in replay mode (reference motion only)")

    if args.output is not None:
        _export_mp4(
            env,
            policy,
            ghost_fn,
            args.output,
            n_frames=args.n_frames,
            fps=args.fps,
            width=args.width,
            height=args.height,
        )
    else:
        _run_viewer(env, policy, ghost_fn, playback_speed=args.playback_speed)


if __name__ == "__main__":
    main()
