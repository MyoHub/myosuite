# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Solvability-evidence probe for ``myoChallengeChaseTagFBVs-v0`` (1v1 self-play).

This env exposes a ``Dict`` observation/action space (per-agent), so it is
out of scope for both ``scripts/generate_parity_baselines.py`` (assumes a
flat ``Box`` obs/action contract) and ``scripts/train_sb3_all_envs.py``
(records ``Dict`` spaces as ``unsupported_space``, not trainable by
off-the-shelf single-agent SB3 PPO). Instead of building new multi-agent
training infra (out of scope), this script reuses the existing
``bc_directional_v2`` checkpoint plus the heading-reuse inference trick from
``myosuite/envs/myo/tasks/mimic/chasetag_obs.py::chasetag_heading_directional_obs``
(already the recommended path for the single-agent chase-tag envs) to drive
*both* agents and reports survival steps / tag rate as a baseline data point.

Usage::

    python scripts/eval_chasetag_fbvs_bc.py \\
        --ckpt runs/bc_directional_v2/policy_bc_best.pt --n-episodes 20
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

ENV_ID = "myoChallengeChaseTagFBVs-v0"
OBS_DIM = 528
ACT_DIM = 354
# Fixed by myosuite/terms/multiplayer/chase_tag_vs_reward.py -- agent_0 always
# plays CHASER, agent_1 always plays RUNNER for the default "CHASE" task.
CHASER_ID = "agent_0"
RUNNER_ID = "agent_1"


def _pelvis_xy(data, meta, agent_id: str) -> np.ndarray:
    site_id = meta.site_ids[agent_id]["pelvis_site"]
    return np.asarray(data.site_xpos[site_id][:2], dtype=np.float64)


def _agent_directional_obs(model, data, meta, agent_id: str, heading_theta: float):
    """528-dim directional obs for one agent of the shared two-agent model.

    ``chasetag_heading_directional_obs`` (chasetag_obs.py) assumes a
    single-agent model layout (``model.nq`` = one body), which does not hold
    for FBVs' merged two-agent ``MjModel``. This mirrors its per-field
    layout (``qpos_local(82) + qvel_local(82) + act(354) + root_vel_body(2) +
    heading_cmd(2) + orientation(6)``) but slices state for a single agent
    out of the shared model/data, the same way
    ``scripts/bc_directional_render_1v1.py::_bc_obs_for_agent`` does.
    """
    jnt_ids = meta.jnt_ids[agent_id]
    root_qposadr = int(model.jnt_qposadr[jnt_ids[0]])
    root_qveladr = int(model.jnt_dofadr[jnt_ids[0]])
    local_qpos_start = int(model.jnt_qposadr[jnt_ids[1]])
    local_qvel_start = int(model.jnt_dofadr[jnt_ids[1]])
    nq_local = model.nq // 2 - 7
    nv_local = model.nv // 2 - 6
    local_qpos = data.qpos[local_qpos_start : local_qpos_start + nq_local].astype(
        np.float32
    )
    local_qvel = data.qvel[local_qvel_start : local_qvel_start + nv_local].astype(
        np.float32
    )
    act = data.act[meta.act_indices[agent_id]].astype(np.float32)

    rq = data.qpos[root_qposadr + 3 : root_qposadr + 7]
    w, x, y, z = rq[0], rq[1], rq[2], rq[3]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    vx, vy = data.qvel[root_qveladr], data.qvel[root_qveladr + 1]
    c, s = math.cos(-yaw), math.sin(-yaw)
    vel_body = np.array([c * vx - s * vy, s * vx + c * vy], dtype=np.float32)

    heading_dir = np.array(
        [math.cos(heading_theta), math.sin(heading_theta)], dtype=np.float32
    )

    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(float(np.clip(2 * (w * y - z * x), -1.0, 1.0)))
    wx_w = data.qvel[root_qveladr + 3]
    wy_w = data.qvel[root_qveladr + 4]
    wz_w = data.qvel[root_qveladr + 5]
    wx_b = c * wx_w - s * wy_w
    wy_b = s * wx_w + c * wy_w
    vz = data.qvel[root_qveladr + 2]
    orientation = np.array([roll, pitch, wx_b, wy_b, wz_w, vz], dtype=np.float32)

    return np.concatenate(
        [local_qpos, local_qvel, act, vel_body, heading_dir, orientation]
    )


def run_episode(env, policy, seed: int, max_steps: int) -> dict:
    """Roll out one episode with the BC policy driving both agents.

    Each agent gets the native 528-dim directional obs with its heading
    channel pointed at (chaser) or away from (runner) the opponent's live
    pelvis position, recomputed every step -- the "heading-reuse" trick from
    ``chasetag_heading_directional_obs``, adapted for the shared two-agent
    model via :func:`_agent_directional_obs`.
    """
    obs, _ = env.reset(seed=seed)
    raw = env.unwrapped
    model, data, meta = raw.model, raw.data, raw._meta

    tagged = False
    step = 0
    for step in range(1, max_steps + 1):
        chaser_xy = _pelvis_xy(data, meta, CHASER_ID)
        runner_xy = _pelvis_xy(data, meta, RUNNER_ID)
        chaser_delta = runner_xy - chaser_xy  # chase: toward the runner
        runner_delta = runner_xy - chaser_xy  # evade: away from the chaser (self - opp)
        theta_chaser = math.atan2(chaser_delta[1], chaser_delta[0])
        theta_runner = math.atan2(runner_delta[1], runner_delta[0])
        obs_chaser = _agent_directional_obs(model, data, meta, CHASER_ID, theta_chaser)
        obs_runner = _agent_directional_obs(model, data, meta, RUNNER_ID, theta_runner)
        actions = {
            CHASER_ID: np.clip(policy.act(obs_chaser), 0.0, 1.0),
            RUNNER_ID: np.clip(policy.act(obs_runner), 0.0, 1.0),
        }
        _, _, terminated, truncated, info = env.step(actions)
        tagged = bool(info.get("tagged", False))
        if any(terminated.values()) or any(truncated.values()):
            break
    return {"steps": step, "tagged": tagged}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import myosuite.envs.myo.tasks.challenge  # noqa: F401 -- registers FBVs
    from myosuite.envs.myo.tasks.mimic.policy import ActorCritic
    from myosuite.utils import gym

    policy = ActorCritic.load(args.ckpt, obs_dim=OBS_DIM, act_dim=ACT_DIM)
    policy.eval()
    env = gym.make(ENV_ID)

    episodes = [
        run_episode(env, policy, seed=seed, max_steps=args.max_steps)
        for seed in range(args.n_episodes)
    ]
    env.close()

    steps = np.array([e["steps"] for e in episodes], dtype=np.float64)
    tag_rate = float(np.mean([e["tagged"] for e in episodes]))
    result = {
        "env_id": ENV_ID,
        "method": "bc_directional_v2 heading-reuse (both agents), no training",
        "checkpoint": str(args.ckpt),
        "n_episodes": args.n_episodes,
        "max_steps": args.max_steps,
        "mean_steps_alive": float(steps.mean()),
        "std_steps_alive": float(steps.std()),
        "chaser_tag_rate_pct": 100.0 * tag_rate,
    }
    print(json.dumps(result, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
