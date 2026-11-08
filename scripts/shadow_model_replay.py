# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Replay a motion through the live shadow model, as a tracker would feed it.

Streams poses one at a time (with optional tracking noise and real-time pacing)
through :class:`myosuite.physics.shadow_model.ShadowModel` and writes its
per-frame log. Without ``--clip`` it plays random minimum-jerk reaches on a
fixed-base model, so the pipeline can be tried without data::

    python scripts/shadow_model_replay.py --log-dir logs/demo
    python scripts/shadow_model_replay.py --env myoLegWalk-v0 --clip walk.npz --contacts --log-dir logs/walk
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

from myosuite import make_env
from myosuite.core.trajectory_io import expand_motion_clip_to_model, load_motion_clip
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings
from myosuite.physics.shadow_model import ShadowModel, ShadowModelConfig
from myosuite.physics.static_optimization import StaticOptimizationConfig


def reaching_motion(
    model: mujoco.MjModel, rate: float, duration: float, seed: int
) -> np.ndarray:
    """Random minimum-jerk reaches (0.5-0.9 s, 0.3 s holds) of every independent hinge."""
    c = JointCouplings.from_model(model)
    coupled = set(c.dep_dof.tolist()) | set(c.locked_dof.tolist())
    hinge = mujoco.mjtJoint.mjJNT_HINGE
    joints = [
        j
        for j in range(model.njnt)
        if model.jnt_type[j] == hinge and model.jnt_dofadr[j] not in coupled
    ]
    lo, hi = model.jnt_range[joints].T
    lo, hi = lo + 0.2 * (hi - lo), hi - 0.2 * (hi - lo)
    rng = np.random.default_rng(seed)
    start = model.key_qpos[0] if model.nkey else model.qpos0
    cur, poses = np.clip(start[model.jnt_qposadr[joints]], lo, hi), []
    while sum(len(p) for p in poses) < duration * rate:
        goal, n = rng.uniform(lo, hi), int(rng.uniform(0.5, 0.9) * rate)
        s = np.arange(n) / n
        poses.append(cur + (10 * s**3 - 15 * s**4 + 6 * s**5)[:, None] * (goal - cur))
        poses.append(np.repeat(goal[None], int(0.3 * rate), 0))
        cur = goal
    qpos = np.tile(start, (int(duration * rate), 1))
    qpos[:, model.jnt_qposadr[joints]] = np.concatenate(poses)[: len(qpos)]
    project_joint_couplings(c, qpos)
    return qpos


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, stream the motion and print a summary."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--env",
        default="myoArmReachRandom-v0",
        help="registered env whose model to use",
    )
    p.add_argument(
        "--clip", type=Path, help="motion NPZ (qpos, joint names, frequency)"
    )
    p.add_argument(
        "--rate", type=float, default=90.0, help="pose rate without --clip [Hz]"
    )
    p.add_argument(
        "--duration", type=float, default=10.0, help="length without --clip [s]"
    )
    p.add_argument(
        "--noise-deg", type=float, default=0.5, help="tracking noise on hinges [deg]"
    )
    p.add_argument("--window", type=int, default=15, help="poses per kinematic fit")
    p.add_argument("--contacts", action="store_true", help="estimate ground reactions")
    p.add_argument(
        "--realtime", action="store_true", help="pace poses at their timestamps"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-dir", type=Path, required=True)
    args = p.parse_args(argv)

    env = make_env(args.env)
    model = env.unwrapped.model
    if args.clip is not None:
        clip = expand_motion_clip_to_model(
            load_motion_clip(args.clip, model.nq, model.nv), model
        )
        qpos, rate = clip.qpos, clip.frequency_hz or args.rate
    elif (model.jnt_type == mujoco.mjtJoint.mjJNT_FREE).any():
        raise SystemExit(
            "Synthetic reaches need a fixed-base model; pass --clip for this env."
        )
    else:
        qpos, rate = (
            reaching_motion(model, args.rate, args.duration, args.seed),
            args.rate,
        )
    hinges = model.jnt_qposadr[model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE]
    noisy = qpos.copy()
    noisy[:, hinges] += np.radians(args.noise_deg) * np.random.default_rng(
        args.seed
    ).standard_normal((len(qpos), len(hinges)))

    config = ShadowModelConfig(
        window=args.window,
        optimization=StaticOptimizationConfig(contact_forces=args.contacts),
    )
    frames, wall0 = [], time.perf_counter()
    with ShadowModel(
        model,
        config,
        log_dir=args.log_dir,
        metadata={"env": args.env, "clip": str(args.clip)},
    ) as shadow:
        for k, q in enumerate(noisy):
            if args.realtime:
                time.sleep(max(0.0, k / rate - (time.perf_counter() - wall0)))
            frame = shadow.push(k / rate, q)
            if frame is not None:
                frames.append(frame)
    env.close()

    ms = np.array([f.compute_ms for f in frames])
    solved = sum(f.optimization.status == "solved" for f in frames)
    delay = 1e3 * (frames[0].latest_time - frames[0].state.time)
    mf = (
        max(float(f.fatigue.max()) for f in frames)
        if frames[0].fatigue is not None
        else 0.0
    )
    print(
        f"{len(frames)} frames at {rate:g} Hz, delay {delay:.0f} ms, update {np.median(ms):.2f} ms "
        f"(p99 {np.percentile(ms, 99):.2f}), solved {solved}/{len(frames)}, peak MF {mf:.3f}; "
        f"log in {args.log_dir}"
    )


if __name__ == "__main__":
    main()
