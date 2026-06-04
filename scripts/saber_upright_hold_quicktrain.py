#!/usr/bin/env python3
# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Quick IK/ID posture tuning + short PPO for upright saber holding."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import gymnasium as gym
import mujoco
import numpy as np
from stable_baselines3 import PPO

from myosuite import register_all_envs
from scripts.saber_pose_utils import (
    build_finger_flexor_mask,
    build_non_finger_qpos_mask,
    reset_non_arm_joints_to_upright,
)


def _safe_joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> float:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        return 0.0
    return float(data.qpos[int(model.jnt_qposadr[jid])])


def _hold_metrics(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, float]:
    lsid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "S_grasp_left")
    rsid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "S_grasp")
    lbid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_lightsaber")
    rbid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_lightsaber")
    d_left = float(np.linalg.norm(data.xpos[lbid] - data.site_xpos[lsid]))
    d_right = float(np.linalg.norm(data.xpos[rbid] - data.site_xpos[rsid]))
    torso = (
        abs(_safe_joint_qpos(model, data, "flex_extension"))
        + abs(_safe_joint_qpos(model, data, "axial_rotation"))
        + abs(_safe_joint_qpos(model, data, "lateral_bending"))
    )
    return d_left + d_right, torso


@dataclass
class Q0Candidate:
    qpos: np.ndarray
    qvel: np.ndarray
    score: float
    warmup_steps: int
    action_level: float


def _pick_q0_with_id(env: gym.Env, trials: int, seed: int) -> Q0Candidate:
    """Quick search for grasp posture with low torso bend and low ID effort."""
    rng = np.random.default_rng(seed)
    unwrapped = env.unwrapped
    model = unwrapped.model
    data = unwrapped.data
    best: Q0Candidate | None = None

    for _ in range(trials):
        warmup_steps = int(rng.integers(40, 180))
        action_level = float(rng.uniform(0.55, 0.95))
        env.reset(seed=seed)
        reset_qpos = data.qpos.copy()

        action = np.full(env.action_space.shape, action_level, dtype=np.float32)
        for _ in range(warmup_steps):
            env.step(action)

        reset_non_arm_joints_to_upright(model, data, reset_qpos)
        snap = getattr(unwrapped, "_snap_free_sabers_to_grasp_sites", None)
        if callable(snap):
            snap()

        # ID: static-support proxy (lower inverse generalized force is better).
        data.qacc[:] = 0.0
        mujoco.mj_inverse(model, data)
        id_norm = float(np.linalg.norm(data.qfrc_inverse))
        hold_err, torso = _hold_metrics(model, data)
        score = 8.0 * hold_err + 3.0 * torso + 0.001 * id_norm
        cand = Q0Candidate(
            qpos=data.qpos.copy(),
            qvel=data.qvel.copy(),
            score=score,
            warmup_steps=warmup_steps,
            action_level=action_level,
        )
        if best is None or cand.score < best.score:
            best = cand

    assert best is not None
    return best


class SaberHoldWrapper(gym.Wrapper):
    """Reward wrapper for holding free sabers while keeping torso upright."""

    def __init__(
        self,
        env: gym.Env,
        q0: np.ndarray,
        qd0: np.ndarray,
        clamp_alpha: float = 0.0,
        prior_mix: float = 0.0,
        prior_level: float = 0.6,
        fail_hold_err: float = 1.0,
    ):
        super().__init__(env)
        self._q0 = q0.copy()
        self._qd0 = qd0.copy()
        self._clamp_alpha = float(clamp_alpha)
        self._prior_mix = float(prior_mix)
        self._prior_level = float(prior_level)
        self._fail_hold_err = float(fail_hold_err)
        uw = self.env.unwrapped
        self._finger_mask = build_finger_flexor_mask(uw.model)
        self._non_finger_qpos_mask = build_non_finger_qpos_mask(uw.model)

    def set_curriculum(self, clamp_alpha: float, prior_mix: float) -> None:
        """Update deterministic assist settings between PPO stages."""
        self._clamp_alpha = float(clamp_alpha)
        self._prior_mix = float(prior_mix)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        uw = self.env.unwrapped
        uw.data.qpos[:] = self._q0
        uw.data.qvel[:] = self._qd0
        snap = getattr(uw, "_snap_free_sabers_to_grasp_sites", None)
        if callable(snap):
            snap()
        mujoco.mj_forward(uw.model, uw.data)
        obs = uw._obs_dict_to_vec(uw._get_obs_dict(uw._accessor))  # pylint: disable=protected-access
        return obs.astype(np.float32), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        prior = self._prior_level * self._finger_mask
        action_eff = (1.0 - self._prior_mix) * action + self._prior_mix * prior
        action_eff = np.clip(
            action_eff, self.env.action_space.low, self.env.action_space.high
        )
        obs, _, terminated, truncated, info = self.env.step(action_eff)
        uw = self.env.unwrapped
        if self._clamp_alpha > 0.0:
            a = float(np.clip(self._clamp_alpha, 0.0, 1.0))
            mask = self._non_finger_qpos_mask
            uw.data.qpos[mask] = (1.0 - a) * uw.data.qpos[mask] + a * self._q0[mask]
            uw.data.qvel[:] *= 1.0 - a
            mujoco.mj_forward(uw.model, uw.data)
            # Refresh obs after deterministic correction.
            obs = uw._obs_dict_to_vec(uw._get_obs_dict(uw._accessor)).astype(np.float32)  # pylint: disable=protected-access
        hold_err, torso = _hold_metrics(uw.model, uw.data)
        act_reg = float(np.mean(np.square(action_eff)))
        reward = -6.0 * hold_err - 2.0 * torso - 0.01 * act_reg
        # Strongly penalize loss of grasp and cut episode early.
        if hold_err > self._fail_hold_err:
            reward -= 50.0
            terminated = True
        if hold_err < 0.12:
            reward += 1.0
        info["quick/hold_err"] = hold_err
        info["quick/torso"] = torso
        return obs, reward, terminated, truncated, info


def _evaluate(env: gym.Env, model: PPO, episodes: int) -> dict[str, float]:
    hold_vals: list[float] = []
    torso_vals: list[float] = []
    returns: list[float] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        ep_ret = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += float(reward)
            done = bool(terminated or truncated)
            hold_vals.append(float(info.get("quick/hold_err", 0.0)))
            torso_vals.append(float(info.get("quick/torso", 0.0)))
        returns.append(ep_ret)
    return {
        "return_mean": float(np.mean(returns)),
        "hold_err_mean_m": float(np.mean(hold_vals)),
        "torso_abs_sum_mean_rad": float(np.mean(torso_vals)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument(
        "--stage1-frac",
        type=float,
        default=0.4,
        help="Fraction of timesteps in assisted stage-1 (0..1).",
    )
    parser.add_argument(
        "--stage1-clamp-alpha",
        type=float,
        default=0.9,
        help="Non-finger posture clamp strength in stage-1.",
    )
    parser.add_argument(
        "--stage1-prior-mix",
        type=float,
        default=0.7,
        help="Finger-flexor prior action mix in stage-1.",
    )
    parser.add_argument(
        "--stage2-clamp-alpha",
        type=float,
        default=0.25,
        help="Non-finger posture clamp strength in stage-2.",
    )
    parser.add_argument(
        "--stage2-prior-mix",
        type=float,
        default=0.15,
        help="Finger-flexor prior action mix in stage-2.",
    )
    parser.add_argument(
        "--fail-hold-err",
        type=float,
        default=1.0,
        help="Terminate episode when hold error exceeds this threshold (m).",
    )
    parser.add_argument("--q0-trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-out", default="outputs/saber_upright_hold_ppo.zip")
    parser.add_argument(
        "--stage1-checkpoint-out",
        default="outputs/saber_upright_hold_stage1.zip",
    )
    args = parser.parse_args()

    register_all_envs()
    base_env = gym.make("myoChallengeSaberP0-v0")
    q0 = _pick_q0_with_id(base_env, trials=args.q0_trials, seed=args.seed)
    print(
        f"[q0] score={q0.score:.4f} warmup_steps={q0.warmup_steps} "
        f"action={q0.action_level:.3f}"
    )

    train_env = SaberHoldWrapper(
        gym.make("myoChallengeSaberP0-v0"),
        q0.qpos,
        q0.qvel,
        clamp_alpha=args.stage1_clamp_alpha,
        prior_mix=args.stage1_prior_mix,
        fail_hold_err=args.fail_hold_err,
    )
    policy = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        gamma=0.99,
        seed=args.seed,
    )
    total_steps = int(args.timesteps)
    stage1_steps = int(max(1, min(total_steps - 1, total_steps * args.stage1_frac)))
    stage2_steps = total_steps - stage1_steps
    print(
        f"[curriculum] stage1_steps={stage1_steps} stage2_steps={stage2_steps} "
        f"stage1(clamp={args.stage1_clamp_alpha},prior={args.stage1_prior_mix}) "
        f"stage2(clamp={args.stage2_clamp_alpha},prior={args.stage2_prior_mix})"
    )
    policy.learn(total_timesteps=stage1_steps)
    policy.save(args.stage1_checkpoint_out)
    train_env.set_curriculum(
        clamp_alpha=args.stage2_clamp_alpha,
        prior_mix=args.stage2_prior_mix,
    )
    if stage2_steps > 0:
        policy.learn(total_timesteps=stage2_steps, reset_num_timesteps=False)
    policy.save(args.model_out)
    metrics = _evaluate(train_env, policy, episodes=5)
    print("[eval]", metrics)
    train_env.close()
    base_env.close()


if __name__ == "__main__":
    main()
