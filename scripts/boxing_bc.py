#!/usr/bin/env python3
# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Export and run a reusable boxing BC policy from a checkpoint.

This script provides one clean workflow for users:
1) export standalone BC policy from a checkpoint, and
2) run that standalone policy in a boxing mannequin env.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from myosuite import register_all_envs
from myosuite.integrations.musclemimic.fullbody_checkpoint_io import (
    resolve_checkpoint_ref,
)
from myosuite.integrations.musclemimic.fullbody_local_policy import (
    FullbodyObsAdapter,
    LocalPolicyRunner,
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

# pylint: disable=no-member

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="myoChallengeBoxingP0-v0")
    parser.add_argument("--checkpoint", default="Boxing/checkpoint_13114")
    parser.add_argument(
        "--motion",
        default=(
            "Boxing/motions/Transitions_mocap/mazen_c3d/" "punchboxing_push_poses.npz"
        ),
    )
    parser.add_argument(
        "--policy-in",
        default="",
        help="Optional standalone policy .npz to run directly.",
    )
    parser.add_argument(
        "--policy-out",
        default="outputs/debug/mannequin_exact_clone_standalone.npz",
        help="Standalone policy output path when exporting from checkpoint.",
    )
    parser.add_argument("--skip-teacher-compare", action="store_true")
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument(
        "--video-out",
        default="outputs/debug/mannequin_exact_clone_standalone_only_p0.mp4",
    )
    parser.add_argument(
        "--report-out",
        default=(
            "outputs/debug/" "mannequin_exact_clone_standalone_only_p0_report.json"
        ),
    )
    parser.add_argument("--cam-azimuth", type=float, default=90.0)
    parser.add_argument("--cam-elevation", type=float, default=-8.0)
    parser.add_argument("--cam-distance", type=float, default=1.0)
    parser.add_argument("--cam-lookat-x", type=float, default=-0.25)
    parser.add_argument("--cam-lookat-y", type=float, default=-0.10)
    parser.add_argument("--cam-lookat-z", type=float, default=1.05)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    register_all_envs()

    policy_in = args.policy_in.strip()
    need_checkpoint = (not policy_in) or (not args.skip_teacher_compare)
    goal_params: dict = {}
    artifacts = None
    if need_checkpoint:
        ckpt = resolve_checkpoint_ref(args.checkpoint)
        artifacts = load_local_policy_artifacts(ckpt.local_path)
        goal_params = (
            read_checkpoint_config_metadata(ckpt.local_path)
            .get("experiment", {})
            .get("env_params", {})
            .get("goal_params", {})
        )

    if policy_in:
        policy, stored_goal_params = StandaloneBCPolicy.load(policy_in)
        if not goal_params:
            goal_params = stored_goal_params
        logger.info("Loaded standalone policy: %s", policy_in)
    else:
        if artifacts is None:
            raise ValueError("Checkpoint artifacts missing for export.")
        policy = StandaloneBCPolicy(artifacts)
        policy.save(args.policy_out, goal_params=goal_params)
        logger.info("Exported standalone policy: %s", args.policy_out)

    teacher = None
    if not args.skip_teacher_compare:
        if artifacts is None:
            raise ValueError("Teacher compare needs checkpoint artifacts.")
        teacher = LocalPolicyRunner(
            artifacts=artifacts,
            stochastic=False,
            seed=0,
            obs_adapter=None,
        )

    env = gym.make(
        args.env_id,
        disable_env_checker=True,
        render_mode="rgb_array",
    )
    fb_model, _spec, _xml = compile_mimic_fullbody_mjmodel(
        default_mimic_fullbody_config()
    )
    clip = load_motion_clip(
        Path(args.motion),
        expected_nq=fb_model.nq,
        expected_nv=fb_model.nv,
    )
    adapter = FullbodyObsAdapter(fb_model, clip, goal_params)
    fb_data = mujoco.MjData(fb_model)
    mujoco.mj_resetDataKeyframe(fb_model, fb_data, 0)
    mujoco.mj_forward(fb_model, fb_data)
    _, _ = env.reset(seed=0)

    model = env.unwrapped.model
    data = env.unwrapped.data
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = float(args.cam_azimuth)
    cam.elevation = float(args.cam_elevation)
    cam.distance = float(args.cam_distance)
    cam.lookat[:] = [
        float(args.cam_lookat_x),
        float(args.cam_lookat_y),
        float(args.cam_lookat_z),
    ]
    render_w = min(
        int(args.width),
        int(getattr(model.vis.global_, "offwidth", args.width)),
    )
    render_h = min(
        int(args.height),
        int(getattr(model.vis.global_, "offheight", args.height)),
    )
    renderer = mujoco.Renderer(model, height=render_h, width=render_w)

    frames: list[np.ndarray] = []
    step_mae: list[float] = []
    executed_steps = 0
    info = {}
    for t in range(args.horizon):
        fb_data.qpos[:] = data.qpos[: fb_model.nq]
        fb_data.qvel[:] = data.qvel[: fb_model.nv]
        fb_data.act[:] = data.act[: fb_model.nu]
        fb_data.ctrl[:] = data.ctrl[: fb_model.nu]
        mujoco.mj_forward(fb_model, fb_data)
        full_obs = np.asarray(
            adapter.build(fb_data, t % int(clip.qpos.shape[0])),
            dtype=np.float32,
        )
        clone_act = policy.action(full_obs)
        if teacher is not None:
            teacher_act = teacher.action_for(
                fb_data,
                clip,
                t % int(clip.qpos.shape[0]),
                obs_adapter=adapter,
            )
            step_mae.append(float(np.mean(np.abs(teacher_act - clone_act))))
        # BC actor is [-1, 1]; boxing envs expect muscle activations in [0, 1].
        muscle = np.clip(0.5 * (clone_act + 1.0), 0.0, 1.0)
        _, _, term, trunc, info = env.step(muscle)
        executed_steps += 1
        renderer.update_scene(data, camera=cam)
        frames.append(np.asarray(renderer.render(), dtype=np.uint8).copy())
        if term or trunc:
            break

    renderer.close()
    out_video = Path(args.video_out)
    out_video.parent.mkdir(parents=True, exist_ok=True)
    write_video(
        str(out_video),
        np.asarray(frames, dtype=np.uint8),
        outputdict={"-r": "30"},
    )
    report = {
        "steps": int(executed_steps),
        "teacher_compare_enabled": teacher is not None,
        "teacher_clone_action_mae_mean": (
            float(np.mean(step_mae)) if step_mae else None
        ),
        "teacher_clone_action_mae_p95": (
            float(np.percentile(step_mae, 95)) if step_mae else None
        ),
        "fell": bool(info.get("fell/agent_0", False)),
        "damage_last": float(info.get("damage_delivered/agent_0", 0.0)),
        "env_id": args.env_id,
        "policy_in": policy_in or None,
        "policy_out": None if policy_in else args.policy_out,
        "video": str(out_video.resolve()),
    }
    out_report = Path(args.report_out)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote report: %s", out_report)
    env.close()


if __name__ == "__main__":
    main()
