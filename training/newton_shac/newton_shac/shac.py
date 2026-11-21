# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Short-Horizon Actor-Critic (SHAC) trainer for differentiable musculoskeletal envs.

Algorithm reference:
    Xu et al., "Accelerated Policy Learning with Parallel Differentiable Simulation",
    ICLR 2022 (https://arxiv.org/abs/2204.07137). Clean-room PyTorch + MuJoCo Warp
    implementation for MyoSuite musculoskeletal control.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from myosuite import make_env
from newton_shac.diff_env import DifferentiablePoseEnv
from newton_shac.networks import ActorCritic


@dataclass(frozen=True)
class ShacConfig:
    """Hyperparameters for Short-Horizon Actor-Critic."""

    num_envs: int = 256
    horizon: int = 16
    total_env_steps: int = 102_400
    hidden_dims: tuple[int, int] = (128, 128)
    actor_lr: float = 2e-3
    critic_lr: float = 2e-3
    gamma: float = 0.97
    lam: float = 0.95
    critic_tau: float = 0.2
    critic_epochs: int = 8
    critic_minibatches: int = 4
    init_std: float = 0.35
    max_grad_norm: float = 1.0
    eval_interval_steps: int = 25_600
    eval_episodes: int = 64


@dataclass
class TrainRunMetrics:
    """Recorded learning curve and resource usage for a single seed."""

    method: str
    env_id: str
    seed: int
    eval_steps: list[int]
    eval_wall_time_s: list[float]
    eval_return_mean: list[float]
    eval_solved_rate: list[float]
    cpu_eval_return: float
    cpu_eval_solved_rate: float
    total_wall_time_s: float
    peak_gpu_mem_mb: float


@torch.no_grad()
def evaluate_on_gpu(
    eval_env: DifferentiablePoseEnv,
    model: ActorCritic,
    *,
    seed: int = 10_000,
) -> tuple[float, float]:
    """Evaluate deterministic policy over one full episode batch on unmodified reward."""
    obs = eval_env.reset(seed=seed)
    ep_returns = torch.zeros(
        eval_env.num_envs, dtype=torch.float32, device=eval_env.device
    )
    ever_solved = torch.zeros(
        eval_env.num_envs, dtype=torch.bool, device=eval_env.device
    )
    active = torch.ones(eval_env.num_envs, dtype=torch.bool, device=eval_env.device)

    for _ in range(eval_env.max_episode_steps):
        acts = torch.clamp(model.act_deterministic(obs), -1.0, 1.0)
        ctrl = eval_env.actions_to_ctrl(acts)
        q_next, v_next, a_next = eval_env._forward_substeps(
            eval_env.qpos, eval_env.qvel, eval_env.act, ctrl
        )
        eval_env.qpos = q_next
        eval_env.qvel = v_next
        eval_env.act = a_next
        eval_env.step_count = eval_env.step_count + 1
        rwd_true, _, solved, terminated = eval_env.compute_rewards(
            q_next, a_next, eval_env.target_angles
        )
        ep_returns = ep_returns + rwd_true * active.to(dtype=torch.float32)
        ever_solved = ever_solved | (solved & active)
        active = active & ~terminated
        if not torch.any(active):
            break
        obs = eval_env.build_obs(
            eval_env.qpos,
            eval_env.qvel,
            eval_env.act,
            eval_env.target_angles,
            eval_env.step_count,
        )

    return float(ep_returns.mean().cpu().item()), float(
        ever_solved.to(dtype=torch.float32).mean().cpu().item()
    )


@torch.no_grad()
def evaluate_on_cpu_env(
    env_id: str,
    model: ActorCritic,
    *,
    episodes: int = 16,
    seed: int = 20_000,
) -> tuple[float, float]:
    """Verify policy portability on unmodified CPU ``make_env(env_id)``."""
    cpu_env = make_env(env_id)
    dev = next(model.parameters()).device
    returns: list[float] = []
    solved_flags: list[float] = []
    try:
        for ep in range(episodes):
            obs_np, _ = cpu_env.reset(seed=seed + ep)
            ep_ret = 0.0
            ep_solved = False
            while True:
                obs_t = torch.as_tensor(
                    obs_np, dtype=torch.float32, device=dev
                ).unsqueeze(0)
                act_np = (
                    torch.clamp(model.act_deterministic(obs_t), -1.0, 1.0)
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )
                obs_np, rwd, term, trunc, info = cpu_env.step(act_np)
                ep_ret += float(rwd)
                ep_solved = ep_solved or bool(info.get("solved", False))
                if term or trunc:
                    break
            returns.append(ep_ret)
            solved_flags.append(1.0 if ep_solved else 0.0)
    finally:
        cpu_env.close()
    return float(np.mean(returns)), float(np.mean(solved_flags))


@torch.no_grad()
def _compute_td_lambda_targets(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    terminated: torch.Tensor,
    next_values: torch.Tensor,
    *,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    horizon, num_envs = rewards.shape
    targets = torch.zeros_like(rewards)
    Ai = torch.zeros(num_envs, dtype=torch.float32, device=rewards.device)
    Bi = torch.zeros(num_envs, dtype=torch.float32, device=rewards.device)
    lam_carry = torch.ones(num_envs, dtype=torch.float32, device=rewards.device)

    for t in range(horizon - 1, -1, -1):
        not_done = (~dones[t]).to(dtype=torch.float32)
        not_term = (~terminated[t]).to(dtype=torch.float32)
        lam_carry = lam_carry * lam * not_done + (1.0 - not_done)
        Ai = not_done * (
            (1.0 - lam) * gamma * next_values[t]
            + gamma * (Ai - lam_carry * next_values[t])
        )
        Ai = Ai + (1.0 - not_done) * not_term * gamma * next_values[t]
        Bi = gamma * (not_done * Bi) + rewards[t]
        targets[t] = (1.0 - lam) * Bi + Ai + lam_carry * (not_done * next_values[t])
    return targets


def train_shac(
    env_id: str,
    cfg: ShacConfig = ShacConfig(),
    *,
    seed: int = 0,
    device: str = "cuda:0",
    eval_env: DifferentiablePoseEnv | None = None,
) -> tuple[ActorCritic, TrainRunMetrics]:
    """Train a policy with Short-Horizon Actor-Critic and return ``(model, metrics)``."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = DifferentiablePoseEnv(env_id, num_envs=cfg.num_envs, device=device, seed=seed)
    own_eval = eval_env is None
    if eval_env is None:
        eval_env = DifferentiablePoseEnv(
            env_id, num_envs=cfg.eval_episodes, device=device, seed=seed + 5000
        )

    model = ActorCritic(
        env.num_obs,
        env.num_actions,
        hidden_dims=cfg.hidden_dims,
        init_std=cfg.init_std,
    ).to(env.device)
    target_model = copy.deepcopy(model).to(env.device)
    for p in target_model.parameters():
        p.requires_grad_(False)

    actor_params = [*model.actor_net.parameters(), model.std_param]
    critic_params = list(model.critic_net.parameters())
    actor_opt = torch.optim.Adam(actor_params, lr=cfg.actor_lr)
    critic_opt = torch.optim.Adam(critic_params, lr=cfg.critic_lr)

    steps_per_iter = cfg.num_envs * cfg.horizon
    num_iters = max(1, cfg.total_env_steps // steps_per_iter)

    if env.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(env.device)
        torch.cuda.synchronize(env.device)

    t_start = time.perf_counter()
    eval_steps: list[int] = [0]
    eval_times: list[float] = [0.0]
    r0, s0 = evaluate_on_gpu(eval_env, model)
    eval_returns: list[float] = [r0]
    eval_solved: list[float] = [s0]
    next_eval_at = cfg.eval_interval_steps

    env.reset(seed=seed)

    for it in range(1, num_iters + 1):
        frac = 1.0 - (it - 1) / max(num_iters, 1)
        cur_actor_lr = cfg.actor_lr * (0.1 + 0.9 * frac)
        for pg in actor_opt.param_groups:
            pg["lr"] = cur_actor_lr

        env.detach_state()
        target_model.obs_norm.load_state_dict(model.obs_norm.state_dict())
        obs = env.build_obs(
            env.qpos, env.qvel, env.act, env.target_angles, env.step_count
        )

        obs_buf = torch.empty(
            (cfg.horizon, cfg.num_envs, env.num_obs),
            dtype=torch.float32,
            device=env.device,
        )
        rwd_buf = torch.empty(
            (cfg.horizon, cfg.num_envs), dtype=torch.float32, device=env.device
        )
        dones_buf = torch.empty(
            (cfg.horizon, cfg.num_envs), dtype=torch.bool, device=env.device
        )
        term_buf = torch.empty(
            (cfg.horizon, cfg.num_envs), dtype=torch.bool, device=env.device
        )
        next_val_buf = torch.empty(
            (cfg.horizon, cfg.num_envs), dtype=torch.float32, device=env.device
        )

        actor_return = torch.zeros(cfg.num_envs, dtype=torch.float32, device=env.device)
        gamma_pow = torch.ones(cfg.num_envs, dtype=torch.float32, device=env.device)

        for t in range(cfg.horizon):
            obs_buf[t] = obs.detach()
            actions = model.act_reparam(obs, update_norm=True)
            next_obs, rwd_smooth, dones, info = env.step_diff(actions)

            actor_return = actor_return + gamma_pow * rwd_smooth
            gamma_next = gamma_pow * cfg.gamma
            if torch.any(dones):
                not_term_f = (~info["terminated"]).to(dtype=torch.float32)
                v_term = target_model.value(info["final_obs"]) * not_term_f
                actor_return = (
                    actor_return + gamma_next * dones.to(dtype=torch.float32) * v_term
                )
            gamma_pow = torch.where(dones, torch.ones_like(gamma_pow), gamma_next)

            with torch.no_grad():
                v_next = torch.where(
                    dones,
                    target_model.value(info["final_obs"].detach()),
                    target_model.value(next_obs.detach()),
                )
            rwd_buf[t] = rwd_smooth.detach()
            dones_buf[t] = dones.detach()
            term_buf[t] = info["terminated"].detach()
            next_val_buf[t] = v_next
            obs = next_obs

        not_just_reset = (~dones_buf[-1]).to(dtype=torch.float32)
        actor_return = actor_return + gamma_pow * not_just_reset * target_model.value(
            obs
        )
        actor_loss = -actor_return.mean()

        actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor_params, cfg.max_grad_norm)
        actor_opt.step()
        with torch.no_grad():
            model.std_param.clamp_(model.std_range[0], model.std_range[1])

        td_targets = _compute_td_lambda_targets(
            rwd_buf,
            dones_buf,
            term_buf,
            next_val_buf,
            gamma=cfg.gamma,
            lam=cfg.lam,
        )
        flat_obs = obs_buf.reshape(-1, env.num_obs)
        flat_targets = td_targets.reshape(-1)
        batch_n = flat_obs.shape[0]
        mb_size = max(1, batch_n // cfg.critic_minibatches)

        for _ in range(cfg.critic_epochs):
            perm = torch.randperm(batch_n, device=env.device)
            for start in range(0, batch_n, mb_size):
                idx = perm[start : start + mb_size]
                pred_v = model.value(flat_obs[idx])
                c_loss = nn.functional.mse_loss(pred_v, flat_targets[idx])
                critic_opt.zero_grad(set_to_none=True)
                c_loss.backward()
                nn.utils.clip_grad_norm_(critic_params, cfg.max_grad_norm)
                critic_opt.step()

        with torch.no_grad():
            for p, tp in zip(
                model.critic_net.parameters(), target_model.critic_net.parameters()
            ):
                tp.data.mul_(1.0 - cfg.critic_tau).add_(p.data, alpha=cfg.critic_tau)
            target_model.obs_norm.load_state_dict(model.obs_norm.state_dict())

        done_steps = it * steps_per_iter
        if done_steps >= next_eval_at or it == num_iters:
            if env.device.type == "cuda":
                torch.cuda.synchronize(env.device)
            elapsed = time.perf_counter() - t_start
            ret_m, sol_m = evaluate_on_gpu(eval_env, model)
            eval_steps.append(done_steps)
            eval_times.append(elapsed)
            eval_returns.append(ret_m)
            eval_solved.append(sol_m)
            next_eval_at += cfg.eval_interval_steps

    if env.device.type == "cuda":
        torch.cuda.synchronize(env.device)
        peak_mb = float(torch.cuda.max_memory_allocated(env.device)) / (1024.0 * 1024.0)
    else:
        peak_mb = 0.0
    total_time = time.perf_counter() - t_start
    cpu_ret, cpu_sol = evaluate_on_cpu_env(env_id, model)
    if own_eval:
        del eval_env

    return model, TrainRunMetrics(
        method="newton_shac",
        env_id=env_id,
        seed=seed,
        eval_steps=eval_steps,
        eval_wall_time_s=eval_times,
        eval_return_mean=eval_returns,
        eval_solved_rate=eval_solved,
        cpu_eval_return=cpu_ret,
        cpu_eval_solved_rate=cpu_sol,
        total_wall_time_s=total_time,
        peak_gpu_mem_mb=peak_mb,
    )
