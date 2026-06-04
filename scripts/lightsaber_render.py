# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Light-saber CPU prototype renderer.

Rolls out ``myoChallengeSaberP0-v0`` with random muscle actions and captures
RGB frames. The env uses a **100-cube mocap pool** with env-driven spawn and
kinematics (no scripted two-block paths).

Usage::

    uv run python scripts/lightsaber_render.py --output lightsaber_demo.mp4
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import imageio
import mujoco
import numpy as np

from myosuite.viz.mj_renderer import _tune_mjv_scene_for_rgb
from scripts.saber_pose_utils import (
    build_finger_flexor_mask,
    build_non_finger_qpos_mask,
    reset_non_arm_joints_to_upright,
)

logger = logging.getLogger(__name__)
# MuJoCo Python bindings expose attributes dynamically at runtime.
# pylint: disable=no-member


def _add_pool_overlay(
    frame: np.ndarray,
    hits_total: int,
    spawned: int,
    hit_this_step: float,
    d_min: float,
    hit_thr: float,
) -> np.ndarray:
    """Draw pool statistics with OpenCV if available."""
    try:
        import cv2  # type: ignore[import-untyped]
    except ImportError:
        return frame

    frame = frame.copy()
    cv2.rectangle(frame, (8, 8), (520, 92), (20, 20, 20), -1)
    cv2.putText(
        frame,
        f"hits_total={hits_total}  spawned={spawned}",
        (14, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (140, 220, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"hit_step={hit_this_step:.0f}  d_min={d_min:.3f}m  thr={hit_thr:.3f}m",
        (14, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (120, 255, 180),
        1,
        cv2.LINE_AA,
    )
    return frame


def _grasp_hold_error(uw: object) -> float:
    """Saber body COM distance to corresponding grasp sites (sum of both hands)."""
    model = uw.model
    data = uw.data
    lsid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "S_grasp_left")
    rsid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "S_grasp")
    lbid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_lightsaber")
    rbid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_lightsaber")
    d_l = float(np.linalg.norm(data.xpos[lbid] - data.site_xpos[lsid]))
    d_r = float(np.linalg.norm(data.xpos[rbid] - data.site_xpos[rsid]))
    return d_l + d_r


def run_demo(
    output: str = "lightsaber_demo.mp4",
    fps: int = 30,
    width: int = 640,
    height: int = 480,
    steps: int = 420,
    cam_azimuth: float = 165.0,
    cam_elevation: float = -14.0,
    cam_distance: float = 2.5,
    cam_lookat: tuple[float, float, float] = (-0.35, -0.25, 1.1),
    static_posture: bool = False,
    use_grasp_q0: bool = False,
    grasp_warmup_steps: int = 120,
    grasp_action_level: float = 0.85,
    q0_output: str | None = None,
    debug_saber_color: bool = False,
    deterministic_hold_baseline: bool = False,
    baseline_finger_level: float = 0.6,
    ppo_model_path: str | None = None,
    policy_assist_baseline: bool = False,
    assist_clamp_alpha: float = 0.9,
    assist_prior_mix: float = 0.7,
    assist_prior_level: float = 0.6,
) -> Path:
    """Roll out the saber P0 env and save an MP4 video."""
    from myosuite import register_all_envs

    register_all_envs()
    import gymnasium as gym

    env = gym.make("myoChallengeSaberP0-v0")
    env.reset(seed=0)

    uw = env.unwrapped
    model = uw.model
    data = uw.data

    pool_geom_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"saber_target_geom_{i}")
        for i in (0, 50, 99)
    ]
    if min(*pool_geom_ids) < 0:
        raise RuntimeError("Missing pool target geoms in model")

    # Matches ``SaberP0Task`` default hit threshold (site-based hit test).
    hit_threshold = 0.03

    if debug_saber_color:
        for gname in (
            "left_saber_geom",
            "right_saber_geom",
            "left_blade_aura",
            "right_blade_aura",
        ):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
            mid = int(model.geom_matid[gid])
            mname = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, mid)
                if mid >= 0
                else None
            )
            logger.info(
                "%s geom_rgba=%s mat=%s mat_rgba=%s",
                gname,
                np.array2string(model.geom_rgba[gid], precision=3),
                mname,
                np.array2string(model.mat_rgba[mid], precision=3) if mid >= 0 else None,
            )

    # Save upright reset root pose (freejoint qpos: x,y,z, qw,qx,qy,qz).
    reset_qpos = data.qpos.copy()

    # Optional: build a stable grasp pose before rendering.
    if use_grasp_q0 and grasp_warmup_steps > 0:
        grasp_action = np.full(
            env.action_space.shape,
            float(np.clip(grasp_action_level, 0.0, 1.0)),
            dtype=np.float32,
        )
        for _ in range(int(grasp_warmup_steps)):
            env.step(grasp_action)

        # Keep grasped arm/hand joints but pin torso/root to upright reset pose.
        reset_non_arm_joints_to_upright(model, data, reset_qpos)
        # Free sabers do not follow hands automatically; re-snap at post-warmup grasp.
        snap_fn = getattr(uw, "_snap_free_sabers_to_grasp_sites", None)
        if callable(snap_fn):
            snap_fn()

    q0 = data.qpos.copy()
    qd0 = data.qvel.copy()
    if q0_output:
        np.save(Path(q0_output), q0)
        logger.info("Saved q0 to %s", q0_output)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = list(cam_lookat)
    cam.distance = float(cam_distance)
    cam.azimuth = float(cam_azimuth)
    cam.elevation = float(cam_elevation)
    renderer = mujoco.Renderer(model, height=height, width=width)
    _tune_mjv_scene_for_rgb(model, renderer.scene)
    scene_opt = mujoco.MjvOption()
    scene_opt.flags[int(mujoco.mjtVisFlag.mjVIS_TRANSPARENT)] = False

    rng = np.random.default_rng(0)
    policy = None
    if ppo_model_path:
        from stable_baselines3 import PPO

        policy = PPO.load(ppo_model_path)
    frames: list[np.ndarray] = []
    peak_hits = 0
    hold_err_max = 0.0
    hold_err_mean = 0.0
    hold_err_count = 0

    finger_mask = build_finger_flexor_mask(model)
    clamp_non_finger_qpos = build_non_finger_qpos_mask(model)
    q0_clamp = q0.copy()

    for _ in range(steps):
        if deterministic_hold_baseline:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            action += float(np.clip(baseline_finger_level, 0.0, 1.0)) * finger_mask
            _, _, _, _, info = env.step(action)
            # Deterministic posture controller: keep non-finger posture near q0.
            data.qpos[clamp_non_finger_qpos] = q0_clamp[clamp_non_finger_qpos]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
        elif static_posture:
            data.qpos[:] = q0
            data.qvel[:] = qd0
            data.ctrl[:] = 0.0
            mujoco.mj_forward(model, data)
            info = {}
        else:
            if policy is not None:
                obs_now = uw._obs_dict_to_vec(uw._get_obs_dict(uw._accessor)).astype(  # pylint: disable=protected-access
                    np.float32
                )
                action, _ = policy.predict(obs_now, deterministic=True)
                action = np.asarray(action, dtype=np.float32)
                if policy_assist_baseline:
                    prior = float(np.clip(assist_prior_level, 0.0, 1.0)) * finger_mask
                    mix = float(np.clip(assist_prior_mix, 0.0, 1.0))
                    action = (1.0 - mix) * action + mix * prior
            else:
                action = rng.uniform(0.0, 1.0, size=env.action_space.shape).astype(
                    np.float32
                )
            _, _, _, _, info = env.step(action)
            model = uw.model
            data = uw.data
            if policy_assist_baseline:
                a = float(np.clip(assist_clamp_alpha, 0.0, 1.0))
                data.qpos[clamp_non_finger_qpos] = (1.0 - a) * data.qpos[
                    clamp_non_finger_qpos
                ] + a * q0_clamp[clamp_non_finger_qpos]
                data.qvel[:] *= 1.0 - a
            mujoco.mj_forward(model, data)

        hold_err = _grasp_hold_error(uw)
        hold_err_max = max(hold_err_max, hold_err)
        hold_err_mean += hold_err
        hold_err_count += 1

        htot = int(info.get("saber_target_pool/hits_total", 0))
        spawned = int(info.get("saber_target_pool/total_spawned", 0))
        hit_step = float(info.get("saber_target_pool/hit_this_step", 0.0))
        d_min = float(info.get("saber_target_pool/d_min_closest", 0.0))
        peak_hits = max(peak_hits, htot)

        renderer.update_scene(data, camera=cam, scene_option=scene_opt)
        _tune_mjv_scene_for_rgb(model, renderer.scene)
        frame = np.asarray(renderer.render(), dtype=np.uint8).copy()
        frame = _add_pool_overlay(frame, htot, spawned, hit_step, d_min, hit_threshold)
        frames.append(frame)

    renderer.close()
    env.close()

    out_path = Path(output)

    def safe_write_video(video_path, frames, fps):
        try:
            with imageio.get_writer(video_path, fps=fps) as video:
                for frame in frames:
                    video.append_data(frame)
            return True
        except Exception as e:
            print(f"Skipping video export for {video_path}: {e}")
            return False

    safe_write_video(out_path, frames, fps)

    logger.info(
        (
            "Saved %d frames to %s (peak_hits_total=%d, "
            "hold_err_mean=%.6f, hold_err_max=%.6f)"
        ),
        len(frames),
        out_path,
        peak_hits,
        hold_err_mean / max(1, hold_err_count),
        hold_err_max,
    )
    return out_path


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="lightsaber_demo.mp4", help="Output MP4 path"
    )
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--width", type=int, default=640, help="Render width")
    parser.add_argument("--height", type=int, default=480, help="Render height")
    parser.add_argument("--steps", type=int, default=600, help="Number of env steps")
    parser.add_argument("--cam-azimuth", type=float, default=165.0)
    parser.add_argument("--cam-elevation", type=float, default=-14.0)
    parser.add_argument("--cam-distance", type=float, default=2.5)
    parser.add_argument("--cam-lookat-x", type=float, default=-0.35)
    parser.add_argument("--cam-lookat-y", type=float, default=-0.25)
    parser.add_argument("--cam-lookat-z", type=float, default=1.1)
    parser.add_argument(
        "--static-posture",
        action="store_true",
        help="Disable muscle control and render a fixed pose (no dynamics).",
    )
    parser.add_argument(
        "--use-grasp-q0",
        action="store_true",
        help="Apply a grasp warmup before capture (can tilt torso).",
    )
    parser.add_argument(
        "--grasp-warmup-steps",
        type=int,
        default=120,
        help="Warmup steps with a constant action before capture (if enabled).",
    )
    parser.add_argument(
        "--grasp-action-level",
        type=float,
        default=0.85,
        help="Constant action value in [0,1] used during grasp warmup.",
    )
    parser.add_argument(
        "--q0-output",
        type=str,
        default=None,
        help="Optional .npy path to save the post-warmup q0.",
    )
    parser.add_argument(
        "--debug-saber-color",
        action="store_true",
        help="Log saber geom/material colors at startup.",
    )
    parser.add_argument(
        "--deterministic-hold-baseline",
        action="store_true",
        help="No-policy baseline: only finger flexors active + non-finger posture clamp.",
    )
    parser.add_argument(
        "--baseline-finger-level",
        type=float,
        default=0.6,
        help="Constant activation for finger flexors in deterministic baseline mode.",
    )
    parser.add_argument(
        "--ppo-model-path",
        type=str,
        default=None,
        help="Optional SB3 PPO checkpoint path for policy-driven rendering.",
    )
    parser.add_argument(
        "--policy-assist-baseline",
        action="store_true",
        help="Blend PPO actions with flexor prior and apply posture clamp each step.",
    )
    parser.add_argument(
        "--assist-clamp-alpha",
        type=float,
        default=0.9,
        help="Clamp strength for non-finger posture in policy-assist mode.",
    )
    parser.add_argument(
        "--assist-prior-mix",
        type=float,
        default=0.7,
        help="Blend weight of flexor prior in policy-assist mode.",
    )
    parser.add_argument(
        "--assist-prior-level",
        type=float,
        default=0.6,
        help="Constant activation level for flexor prior in policy-assist mode.",
    )
    args = parser.parse_args()

    out = run_demo(
        output=args.output,
        fps=args.fps,
        width=args.width,
        height=args.height,
        steps=args.steps,
        cam_azimuth=args.cam_azimuth,
        cam_elevation=args.cam_elevation,
        cam_distance=args.cam_distance,
        cam_lookat=(args.cam_lookat_x, args.cam_lookat_y, args.cam_lookat_z),
        static_posture=args.static_posture,
        use_grasp_q0=args.use_grasp_q0,
        grasp_warmup_steps=args.grasp_warmup_steps,
        grasp_action_level=args.grasp_action_level,
        q0_output=args.q0_output,
        debug_saber_color=args.debug_saber_color,
        deterministic_hold_baseline=args.deterministic_hold_baseline,
        baseline_finger_level=args.baseline_finger_level,
        ppo_model_path=args.ppo_model_path,
        policy_assist_baseline=args.policy_assist_baseline,
        assist_clamp_alpha=args.assist_clamp_alpha,
        assist_prior_mix=args.assist_prior_mix,
        assist_prior_level=args.assist_prior_level,
    )
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
