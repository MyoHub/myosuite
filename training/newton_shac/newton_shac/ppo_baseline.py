# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Matched PPO baseline using identical environment, network, normalizer, and budget.

Algorithm reference:
    Schulman et al., "Proximal Policy Optimization Algorithms", 2017
    (https://arxiv.org/abs/1707.06347). Hyperparameters match MyoSuite's
    registered pose/reach PPO defaults (gamma=0.97, lam=0.95, bounded Gaussian std).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from newton_shac.diff_env import DifferentiablePoseEnv
from newton_shac.networks import ActorCritic
from newton_shac.shac import TrainRunMetrics, evaluate_on_cpu_env, evaluate_on_gpu


@dataclass(frozen=True)
class PpoConfig:
    """Hyperparameters for the matched PPO baseline."""

    num_envs: int = 256
    horizon: int = 16
    total_env_steps: int = 102_400
    hidden_dims: tuple[int, int] = (128, 128)
    lr: float = 1e-3
    gamma: float = 0.97
    lam: float = 0.95
    clip_param: float = 0.2
    entropy_coef: float = 0.001
    value_loss_coef: float = 1.0
    num_epochs: int = 8
    num_minibatches: int = 4
    init_std: float = 0.4
    max_grad_norm: float = 1.0
    eval_interval_steps: int = 25_600
    eval_episodes: int = 64


def train_ppo(
    env_id: str,
    cfg: PpoConfig = PpoConfig(),
    *,
    seed: int = 0,
    device: str = "cuda:0",
    eval_env: DifferentiablePoseEnv | None = None,
) -> tuple[ActorCritic, TrainRunMetrics]:
    """Train a matched PPO policy on the identical GPU env and return ``(model, metrics)``."""
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
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

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

    obs = env.reset(seed=seed)

    for it in range(1, num_iters + 1):
        obs_buf = torch.empty(
            (cfg.horizon, cfg.num_envs, env.num_obs),
            dtype=torch.float32,
            device=env.device,
        )
        act_buf = torch.empty(
            (cfg.horizon, cfg.num_envs, env.num_actions),
            dtype=torch.float32,
            device=env.device,
        )
        logp_buf = torch.empty(
            (cfg.horizon, cfg.num_envs), dtype=torch.float32, device=env.device
        )
        val_buf = torch.empty(
            (cfg.horizon, cfg.num_envs), dtype=torch.float32, device=env.device
        )
        rwd_buf = torch.empty(
            (cfg.horizon, cfg.num_envs), dtype=torch.float32, device=env.device
        )
        done_buf = torch.empty(
            (cfg.horizon, cfg.num_envs), dtype=torch.bool, device=env.device
        )

        with torch.no_grad():
            for t in range(cfg.horizon):
                obs_buf[t] = obs
                norm_obs = model.obs_norm(obs, update=True)
                dist = model.distribution(norm_obs)
                acts = dist.sample()
                logp = dist.log_prob(acts).sum(dim=-1)
                val = model.critic_net(norm_obs).squeeze(-1)

                next_obs, _, dones, info = env.step_diff(acts)
                rwd = info["rwd_true"]
                if torch.any(dones):
                    not_term = (~info["terminated"]).to(dtype=torch.float32)
                    v_final = model.value(info["final_obs"]) * not_term
                    rwd = rwd + cfg.gamma * dones.to(dtype=torch.float32) * v_final

                act_buf[t] = acts
                logp_buf[t] = logp
                val_buf[t] = val
                rwd_buf[t] = rwd
                done_buf[t] = dones
                obs = next_obs

            last_val = model.value(obs)
            adv_buf = torch.zeros_like(rwd_buf)
            gae = torch.zeros(cfg.num_envs, dtype=torch.float32, device=env.device)
            for t in range(cfg.horizon - 1, -1, -1):
                not_done = (~done_buf[t]).to(dtype=torch.float32)
                next_v = last_val if t == cfg.horizon - 1 else val_buf[t + 1]
                delta = rwd_buf[t] + cfg.gamma * not_done * next_v - val_buf[t]
                gae = delta + cfg.gamma * cfg.lam * not_done * gae
                adv_buf[t] = gae
            ret_buf = adv_buf + val_buf

        flat_obs = obs_buf.reshape(-1, env.num_obs)
        flat_act = act_buf.reshape(-1, env.num_actions)
        flat_logp = logp_buf.reshape(-1)
        flat_adv = adv_buf.reshape(-1)
        flat_ret = ret_buf.reshape(-1)
        flat_val = val_buf.reshape(-1)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        batch_n = flat_obs.shape[0]
        mb_size = max(1, batch_n // cfg.num_minibatches)
        for _ in range(cfg.num_epochs):
            perm = torch.randperm(batch_n, device=env.device)
            for start in range(0, batch_n, mb_size):
                idx = perm[start : start + mb_size]
                norm_mb = model.obs_norm(flat_obs[idx], update=False)
                dist_mb = model.distribution(norm_mb)
                new_logp = dist_mb.log_prob(flat_act[idx]).sum(dim=-1)
                entropy = dist_mb.entropy().sum(dim=-1).mean()
                ratio = torch.exp(new_logp - flat_logp[idx])

                surr1 = ratio * flat_adv[idx]
                surr2 = (
                    torch.clamp(ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param)
                    * flat_adv[idx]
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                new_v = model.critic_net(norm_mb).squeeze(-1)
                v_clipped = flat_val[idx] + torch.clamp(
                    new_v - flat_val[idx], -cfg.clip_param, cfg.clip_param
                )
                v_loss = (
                    0.5
                    * torch.max(
                        (new_v - flat_ret[idx]).pow(2),
                        (v_clipped - flat_ret[idx]).pow(2),
                    ).mean()
                )

                loss = (
                    policy_loss
                    + cfg.value_loss_coef * v_loss
                    - cfg.entropy_coef * entropy
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                with torch.no_grad():
                    model.std_param.clamp_(model.std_range[0], model.std_range[1])

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
        method="ppo",
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
