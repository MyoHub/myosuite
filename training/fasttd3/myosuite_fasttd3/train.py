#!/usr/bin/env python3
# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Train official FastTD3 on a MyoSuite env (CPU gym or mjlab twin).

Update loop adapted from https://github.com/younggyoseo/FastTD3 (MIT).
MyoSuite env wrappers live in ``myosuite_fasttd3.env``.

Install::

    pip install -e .[musclemimic]
    pip install -e training/fasttd3

Examples::

    python -m myosuite_fasttd3 myoMimicFullbody-v0 --smoke --backend cpu
    python -m myosuite_fasttd3 myoMimicFullbody-v0 --backend mjlab --num-envs 1024
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from tensordict import TensorDict
from torch.amp import GradScaler, autocast

_PKG = Path(__file__).resolve().parent.parent
_REPO = _PKG.parent.parent
for _path in (_REPO, _PKG):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _require_fast_td3():
    try:
        from fast_td3.fast_td3 import Actor, Critic
        from fast_td3.fast_td3_utils import EmpiricalNormalization, SimpleReplayBuffer
    except ImportError as exc:
        raise ImportError(
            "FastTD3 is not installed. pip install -e training/fasttd3 "
            "(git+https://github.com/younggyoseo/FastTD3.git)."
        ) from exc
    return Actor, Critic, EmpiricalNormalization, SimpleReplayBuffer


@dataclass
class TrainArgs:
    env_id: str
    seed: int = 1
    num_envs: int = 128
    total_timesteps: int = 50_000
    batch_size: int = 2048
    buffer_size: int = 2048
    learning_starts: int = 10
    num_updates: int = 2
    policy_frequency: int = 2
    gamma: float = 0.99
    tau: float = 0.1
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    actor_hidden: int = 512
    critic_hidden: int = 1024
    init_scale: float = 0.01
    num_atoms: int = 101
    # HumanoidBench FastTD3 support; mimic returns sit well inside this.
    v_min: float = -250.0
    v_max: float = 250.0
    policy_noise: float = 0.001
    noise_clip: float = 0.5
    std_min: float = 0.001
    std_max: float = 0.4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    sim_type: str = ""
    sim_dimension: int = 64
    actor_seq_len: int = 8
    critic_seq_len: int = 8
    use_cdq: bool = True
    obs_normalization: bool = True
    compile: bool = False
    amp: bool = False
    unit_interval_actions: bool = True
    out: Path = Path("runs/fasttd3")
    smoke: bool = False
    device: str = ""
    clip: Path | None = None
    backend: str = "auto"


def parse_args(argv: list[str] | None = None) -> TrainArgs:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("env_id", help="mjlab twin id, e.g. myoMimicFullbody-v0")
    p.add_argument("--smoke", action="store_true", help="tiny CPU/GPU plumbing run")
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--total-timesteps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default="")
    p.add_argument(
        "--backend",
        choices=("auto", "cpu", "mjlab"),
        default="auto",
        help="cpu = Gymnasium (Colab smoke); mjlab = Warp twin; auto = CUDA→mjlab else cpu",
    )
    p.add_argument("--out", type=Path, default=Path("runs/fasttd3"))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--learning-starts", type=int, default=None)
    p.add_argument("--v-min", type=float, default=None)
    p.add_argument("--v-max", type=float, default=None)
    p.add_argument(
        "--amp",
        action="store_true",
        help="bf16 AMP only (ignored on T4 / GPUs without bf16; fp16 overflows)",
    )
    p.add_argument("--no-unit-interval-actions", action="store_true")
    p.add_argument(
        "--clip",
        type=Path,
        default=None,
        help="Motion clip for MuscleMimic trajectory mode (else random targets)",
    )
    ns = p.parse_args(argv)
    args = TrainArgs(
        env_id=ns.env_id,
        seed=ns.seed,
        out=ns.out,
        device=ns.device,
        clip=ns.clip,
        backend=ns.backend,
    )
    if ns.smoke:
        args.smoke = True
    if ns.num_envs is not None:
        args.num_envs = ns.num_envs
    if ns.total_timesteps is not None:
        args.total_timesteps = ns.total_timesteps
    if ns.batch_size is not None:
        args.batch_size = ns.batch_size
    if ns.no_unit_interval_actions:
        args.unit_interval_actions = False
    if ns.v_min is not None:
        args.v_min = ns.v_min
    if ns.v_max is not None:
        args.v_max = ns.v_max
    if ns.amp:
        args.amp = True
    if ns.learning_starts is not None:
        args.learning_starts = ns.learning_starts
    elif not args.smoke:
        # Official FastTD3 uses 10 steps at 128+ envs. Few Colab envs need more.
        args.learning_starts = max(10, 4096 // max(args.num_envs, 1))
    return args


def pick_device(requested: str) -> torch.device:
    """mjlab Warp talks CUDA or CPU; MPS is not a sim backend."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def apply_run_preset(args: TrainArgs) -> TrainArgs:
    """Fill smoke sizes so notebooks can pass ``TrainArgs(env_id=..., smoke=True)``."""
    if not args.smoke:
        return args
    args.num_envs = 2
    args.total_timesteps = 8
    args.batch_size = 64
    args.buffer_size = 64
    args.learning_starts = 2
    args.num_updates = 1
    args.actor_hidden = 64
    args.critic_hidden = 64
    args.num_atoms = 11
    args.compile = False
    args.amp = False
    if not args.device:
        args.device = "cpu"
    if args.backend == "auto":
        args.backend = "cpu"
    return args


def train(args: TrainArgs) -> Path:
    apply_run_preset(args)
    Actor, Critic, EmpiricalNormalization, SimpleReplayBuffer = _require_fast_td3()
    from myosuite_fasttd3.env import make_fast_td3_env

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    bf16 = device.type == "cuda" and bool(
        getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    )
    amp_enabled = bool(args.amp and bf16)
    if args.amp and device.type == "cuda" and not bf16:
        print(
            "AMP ignored: this GPU has no bf16 (T4 fp16 overflows on muscle obs)",
            flush=True,
        )
    amp_device_type = "cuda" if device.type == "cuda" else "cpu"
    amp_dtype = torch.bfloat16
    scaler = GradScaler(enabled=False)

    envs = make_fast_td3_env(
        args.env_id,
        backend=args.backend,
        num_envs=args.num_envs,
        device=device,
        seed=args.seed,
        unit_interval_actions=args.unit_interval_actions,
        clip_path=str(args.clip) if args.clip is not None else None,
    )
    n_obs, n_act = envs.num_obs, envs.num_actions
    n_critic = n_obs if not envs.asymmetric_obs else envs.num_privileged_obs
    actor_kw = dict(
        n_obs=n_obs,
        n_act=n_act,
        num_envs=envs.num_envs,
        init_scale=args.init_scale,
        hidden_dim=args.actor_hidden,
        std_min=args.std_min,
        std_max=args.std_max,
        sim_type=args.sim_type,
        sim_dimension=args.sim_dimension,
        seq_len=args.actor_seq_len,
        device=device,
    )
    critic_kw = dict(
        n_obs=n_critic,
        n_act=n_act,
        num_atoms=args.num_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
        hidden_dim=args.critic_hidden,
        sim_type=args.sim_type,
        sim_dimension=args.sim_dimension,
        seq_len=args.critic_seq_len,
        device=device,
    )
    actor = Actor(**actor_kw)
    try:
        from tensordict import from_module

        actor_detach = Actor(**actor_kw)
        from_module(actor).data.to_module(actor_detach)
        policy = actor_detach.explore
    except Exception:
        policy = actor.explore
    qnet = Critic(**critic_kw)
    qnet_target = Critic(**critic_kw)
    qnet_target.load_state_dict(qnet.state_dict())

    actor_opt = optim.AdamW(
        list(actor.parameters()), lr=args.actor_lr, weight_decay=args.weight_decay
    )
    q_opt = optim.AdamW(
        list(qnet.parameters()), lr=args.critic_lr, weight_decay=args.weight_decay
    )
    if args.obs_normalization:
        obs_norm = EmpiricalNormalization(shape=n_obs, device=device)
        critic_obs_norm = EmpiricalNormalization(shape=n_critic, device=device)
    else:
        obs_norm = torch.nn.Identity()
        critic_obs_norm = torch.nn.Identity()
    rb = SimpleReplayBuffer(
        n_env=envs.num_envs,
        buffer_size=args.buffer_size,
        n_obs=n_obs,
        n_act=n_act,
        n_critic_obs=n_critic,
        asymmetric_obs=envs.asymmetric_obs,
        n_steps=1,
        gamma=args.gamma,
        device=device,
    )
    action_low, action_high = -1.0, 1.0

    def update_critic(data: TensorDict) -> dict[str, torch.Tensor]:
        with autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            observations = data["observations"]
            next_observations = data["next"]["observations"]
            critic_observations = (
                data["critic_observations"] if envs.asymmetric_obs else observations
            )
            next_critic_observations = (
                data["next"]["critic_observations"]
                if envs.asymmetric_obs
                else next_observations
            )
            actions = data["actions"]
            rewards = data["next"]["rewards"]
            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()
            bootstrap = (truncations | ~dones).float()
            noise = (
                torch.randn_like(actions)
                .mul(args.policy_noise)
                .clamp(-args.noise_clip, args.noise_clip)
            )
            next_actions = (actor(next_observations) + noise).clamp(
                action_low, action_high
            )
            discount = args.gamma ** data["next"]["effective_n_steps"]
            with torch.no_grad():
                q1p, q2p = qnet_target.projection(
                    next_critic_observations,
                    next_actions,
                    rewards,
                    bootstrap,
                    discount,
                )
                q1v = qnet_target.get_value(q1p)
                q2v = qnet_target.get_value(q2p)
                if args.use_cdq:
                    pick = torch.where(q1v.unsqueeze(1) < q2v.unsqueeze(1), q1p, q2p)
                    q1p = q2p = pick
            q1, q2 = qnet(critic_observations, actions)
            q_loss = (
                -torch.sum(q1p * F.log_softmax(q1, dim=1), dim=1).mean()
                - torch.sum(q2p * F.log_softmax(q2, dim=1), dim=1).mean()
            )
        q_opt.zero_grad(set_to_none=True)
        scaler.scale(q_loss).backward()
        scaler.unscale_(q_opt)
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(qnet.parameters(), args.max_grad_norm)
        scaler.step(q_opt)
        scaler.update()
        return {"qf_loss": q_loss.detach(), "qf": q1v.detach().mean()}

    def update_actor(data: TensorDict) -> dict[str, torch.Tensor]:
        with autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            critic_observations = (
                data["critic_observations"]
                if envs.asymmetric_obs
                else data["observations"]
            )
            q1, q2 = qnet(critic_observations, actor(data["observations"]))
            q1v = qnet.get_value(F.softmax(q1, dim=1))
            q2v = qnet.get_value(F.softmax(q2, dim=1))
            qv = torch.minimum(q1v, q2v) if args.use_cdq else (q1v + q2v) / 2.0
            a_loss = -qv.mean()
        actor_opt.zero_grad(set_to_none=True)
        scaler.scale(a_loss).backward()
        scaler.unscale_(actor_opt)
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
        scaler.step(actor_opt)
        scaler.update()
        return {"actor_loss": a_loss.detach()}

    @torch.no_grad()
    def soft_update() -> None:
        src_ps = [p.data for p in qnet.parameters()]
        tgt_ps = [p.data for p in qnet_target.parameters()]
        torch._foreach_mul_(tgt_ps, 1.0 - args.tau)
        torch._foreach_add_(tgt_ps, src_ps, alpha=args.tau)

    if envs.asymmetric_obs:
        obs, critic_obs = envs.reset_with_critic_obs()
    else:
        obs = envs.reset()
        critic_obs = None
    dones = None
    t0 = time.time()
    last_q = last_pi = last_qv = 0.0
    print(
        f"FastTD3 env={args.env_id} backend={args.backend} device={device} "
        f"n_obs={n_obs} n_act={n_act} asymmetric={envs.asymmetric_obs} "
        f"envs={envs.num_envs} steps={args.total_timesteps} smoke={args.smoke} "
        f"amp={amp_enabled} v=[{args.v_min:g},{args.v_max:g}] "
        f"learn_start={args.learning_starts}",
        flush=True,
    )
    try:
        for step in range(args.total_timesteps):
            with (
                torch.no_grad(),
                autocast(
                    device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
                ),
            ):
                actions = policy(obs=obs_norm(obs), dones=dones)
            next_obs, rewards, dones, infos = envs.step(actions.float())
            truncations = infos["time_outs"]
            true_next = torch.where(
                dones[:, None] > 0, infos["observations"]["raw"]["obs"], next_obs
            )
            transition = TensorDict(
                {
                    "observations": obs,
                    "actions": torch.as_tensor(
                        actions, device=device, dtype=torch.float32
                    ),
                    "next": {
                        "observations": true_next,
                        "rewards": rewards.float(),
                        "truncations": truncations.long(),
                        "dones": dones.long(),
                    },
                },
                batch_size=(envs.num_envs,),
                device=device,
            )
            if envs.asymmetric_obs:
                next_critic = infos["observations"]["critic"]
                true_next_c = torch.where(
                    dones[:, None] > 0,
                    infos["observations"]["raw"]["critic_obs"],
                    next_critic,
                )
                transition["critic_observations"] = critic_obs
                transition["next"]["critic_observations"] = true_next_c
                critic_obs = next_critic
            rb.extend(transition)
            obs = next_obs
            if step >= args.learning_starts:
                for u in range(args.num_updates):
                    data = rb.sample(max(1, args.batch_size // envs.num_envs))
                    data["observations"] = obs_norm(data["observations"])
                    data["next"]["observations"] = obs_norm(
                        data["next"]["observations"]
                    )
                    if envs.asymmetric_obs:
                        data["critic_observations"] = critic_obs_norm(
                            data["critic_observations"]
                        )
                        data["next"]["critic_observations"] = critic_obs_norm(
                            data["next"]["critic_observations"]
                        )
                    logs = update_critic(data)
                    last_q = float(logs["qf_loss"])
                    last_qv = float(logs["qf"])
                    if args.num_updates > 1:
                        if u % args.policy_frequency == 1:
                            last_pi = float(update_actor(data)["actor_loss"])
                    elif step % args.policy_frequency == 0:
                        last_pi = float(update_actor(data)["actor_loss"])
                    soft_update()
            if step == args.total_timesteps - 1 or (step + 1) % 10 == 0:
                fps = (step + 1) * envs.num_envs / max(time.time() - t0, 1e-6)
                print(
                    f"t={step + 1}/{args.total_timesteps} fps={fps:.0f} "
                    f"qloss={last_q:.3f} q={last_qv:.3f} pi={last_pi:.3f} "
                    f"r={float(rewards.mean()):.3f}",
                    flush=True,
                )
                if step >= args.learning_starts and not math.isfinite(last_q):
                    raise RuntimeError(
                        f"non-finite critic loss {last_q} at step {step + 1}"
                    )

        args.out.mkdir(parents=True, exist_ok=True)
        ckpt = args.out / f"{args.env_id}_{'smoke' if args.smoke else 'final'}.pt"
        torch.save(
            {
                "actor": actor.state_dict(),
                "qnet": qnet.state_dict(),
                "obs_norm": obs_norm.state_dict(),
                "env_id": args.env_id,
                "n_obs": n_obs,
                "n_act": n_act,
            },
            ckpt,
        )
        print("saved", ckpt, flush=True)
        if step >= args.learning_starts and not math.isfinite(last_q):
            raise RuntimeError(f"non-finite critic loss {last_q}")
        return ckpt
    finally:
        envs.close()


def main(argv: list[str] | None = None) -> None:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    train(parse_args(argv))


if __name__ == "__main__":
    main()
