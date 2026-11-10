# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Collect (obs, action) pairs for BC-training a directional locomotion policy.

Rolls the ``amathislab/mm-10m-2`` MuscleMimic teacher (via
:class:`~myosuite.integrations.musclemimic.fullbody_local_policy.LocalPolicyRunner`)
through circular walking clips that continuously sweep every compass
direction.  At each simulation step records:

- the 528-dim *directional-task* observation
  (``qpos_local(82) + qvel_local(82) + act(354) + root_vel_body(2)
  + heading_cmd(2) + orientation(6)``) as consumed by
  ``MuscleMimicFullbodyDirectionalEnv`` / ``myoFullBodyDirectional-v0``, with
  ``heading_cmd`` set to the clip's *actual* instantaneous walking direction
  (so a single continuous rollout yields balanced coverage of all 8 sectors);
- the teacher's muscle-activation action for that step.

This mirrors the collection pattern in
:mod:`myosuite.integrations.musclemimic.activation_collector` (same
``LocalPolicyRunner`` + clip-rollout idiom), but records ``(obs, action)``
teacher-imitation pairs instead of filtered activation-only sequences.

Requires ``orbax-checkpoint`` (Linux only per this repo's ``pyproject.toml``
platform markers) to load the teacher checkpoint's Orbax artifacts. Run this
module on Linux (e.g. a Modal CPU container) — it cannot run on macOS/darwin
dev machines.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from myosuite.core.trajectory_io import MotionClip
from myosuite.envs.myo.tasks.mimic.directional_obs import (
    DIRECTIONAL_OBS_DIM,
    directional_obs,
)
from myosuite.integrations.musclemimic.fullbody_local_policy import load_teacher_runner

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BcCollectionConfig:
    """Hyperparameters for BC teacher-rollout data collection.

    Args:
        samples_per_clip: Number of (obs, action) transitions to record per
            circular clip. Total dataset size is ``samples_per_clip *
            len(clips)``.
        episode_len: Steps per rollout episode before resetting to a fresh
            (random) clip start frame — keeps state-visitation diverse and
            avoids drift accumulating unboundedly within one continuous
            rollout.
        seed: RNG seed for reproducible random episode starts.
    """

    samples_per_clip: int = 25_000
    episode_len: int = 400
    seed: int = 0


def _clip_walk_theta(clip: MotionClip, frame: int) -> float:
    """Instantaneous horizontal walking direction (radians) at a clip frame."""
    qvel = clip.qvel[frame]
    return float(np.arctan2(float(qvel[1]), float(qvel[0])))


def collect_bc_dataset(
    checkpoint_root: Path,
    model: mujoco.MjModel,
    clips: list[MotionClip],
    config: BcCollectionConfig | None = None,
) -> dict[str, np.ndarray]:
    """Roll the teacher through circular clips, recording (obs, action) pairs.

    Args:
        checkpoint_root: Local Orbax checkpoint directory for the teacher.
        model: Compiled MuscleMimic full-body model.
        clips: Circular walking clips (continuously sweep all headings).
        config: Collection hyperparameters.

    Returns:
        Dict with ``obs`` ``(N, 528)`` float32 and ``actions`` ``(N, nu)``
        float32 arrays, plus ``theta`` ``(N,)`` float32 (the heading label
        used per transition, for balance diagnostics).
    """
    cfg = config or BcCollectionConfig()
    rng = np.random.default_rng(cfg.seed)
    data = mujoco.MjData(model)

    obs_chunks: list[np.ndarray] = []
    act_chunks: list[np.ndarray] = []
    theta_chunks: list[np.ndarray] = []

    ctrl_range = np.asarray(model.actuator_ctrlrange, dtype=np.float32)

    for clip_idx, clip in enumerate(clips):
        n_frames = int(clip.qpos.shape[0])
        runner = load_teacher_runner(
            checkpoint_root, model, clip, seed=cfg.seed + clip_idx
        )

        obs_buf = np.empty((cfg.samples_per_clip, DIRECTIONAL_OBS_DIM), dtype=np.float32)
        act_buf = np.empty((cfg.samples_per_clip, model.nu), dtype=np.float32)
        theta_buf = np.empty((cfg.samples_per_clip,), dtype=np.float32)

        collected = 0
        frame_idx = int(rng.integers(0, n_frames))
        data.qpos[:] = clip.qpos[frame_idx]
        data.qvel[:] = clip.qvel[frame_idx]
        mujoco.mj_forward(model, data)
        runner.reset()
        ep_step = 0

        while collected < cfg.samples_per_clip:
            theta = _clip_walk_theta(clip, frame_idx)
            action = runner.action_for(data, clip, frame_idx)
            # Record the action in the directional env's action-space units
            # (raw actuator ctrlrange, e.g. [0, 1] for muscles) rather than
            # the teacher's internal [-1, 1] pre-clip range — this is what
            # MuscleMimicFullbodyDirectionalEnv.action_space expects as a BC
            # target, and what env.step() consumes directly.
            clipped_action = np.clip(action, ctrl_range[:, 0], ctrl_range[:, 1])
            obs_buf[collected] = directional_obs(model, data, theta)
            act_buf[collected] = clipped_action
            theta_buf[collected] = theta
            collected += 1

            runner.step(model, data, action)
            frame_idx = min(frame_idx + 1, n_frames - 1)
            ep_step += 1

            fell = float(data.qpos[2]) < 0.6
            done = fell or ep_step >= cfg.episode_len or frame_idx >= n_frames - 1
            if done and collected < cfg.samples_per_clip:
                frame_idx = int(rng.integers(0, n_frames))
                data.qpos[:] = clip.qpos[frame_idx]
                data.qvel[:] = clip.qvel[frame_idx]
                data.act[:] = 0.05
                mujoco.mj_forward(model, data)
                runner.reset()
                ep_step = 0

        obs_chunks.append(obs_buf)
        act_chunks.append(act_buf)
        theta_chunks.append(theta_buf)
        logger.info(
            "Clip %d/%d (%s): collected %d transitions",
            clip_idx + 1,
            len(clips),
            getattr(clip, "source_path", "?"),
            collected,
        )

    return {
        "obs": np.concatenate(obs_chunks, axis=0),
        "actions": np.concatenate(act_chunks, axis=0),
        "theta": np.concatenate(theta_chunks, axis=0),
    }


__all__ = [
    "BcCollectionConfig",
    "collect_bc_dataset",
]
