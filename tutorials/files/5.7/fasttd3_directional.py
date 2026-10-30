#!/usr/bin/env python3
"""Deprecated homemade TD3 loop.

Official FastTD3 lives in ``training/fasttd3``
(``python -m myosuite_fasttd3 myoMimicFullbody-v0 --smoke --backend cpu``). Use that instead.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OBS_DIM, ACT_DIM = 528, 354
LABELS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
DEGS = list(range(0, 360, 45))


def get_device(smoke: bool) -> torch.device:
    if smoke:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_env(seed: int = 0, angle: float | None = None):
    from myosuite.envs.myo.tasks.mimic.cpu import MuscleMimicFullbodyDirectionalEnv
    from myosuite.terms.mimic_reward import mimic_root_vel_reward

    class ScratchDirectionalEnv(MuscleMimicFullbodyDirectionalEnv):
        """Same 528-d physics/obs; no gated KIT clips (from-scratch RL)."""

        def _load_gait_clip(self) -> None:
            self._gait_qpos_local = None
            self._gait_qvel_local = None
            self._gait_T = 1

        def _load_circular_clips(self) -> None:
            self._circ_clips = None

        def get_reward_dict(self, obs_dict):
            cur_vel3 = np.array(
                [float(self.data.qvel[0]), float(self.data.qvel[1]), 0.0],
                dtype=np.float64,
            )
            target_vel3 = np.array(
                [
                    self._target_speed * float(self._heading_dir[0]),
                    self._target_speed * float(self._heading_dir[1]),
                    0.0,
                ],
                dtype=np.float64,
            )
            heading_r = float(mimic_root_vel_reward(np, cur_vel3, target_vel3, scale=1.0))
            cur_speed = float(np.linalg.norm(cur_vel3))
            excess = max(0.0, cur_speed - self._target_speed * 1.05)
            act = obs_dict["act"]
            act_reg = -float(np.square(act).mean()) if act.size > 0 else 0.0
            pelvis_z = float(self.data.site_xpos[self._pelvis_site_id, 2])
            fell = pelvis_z < self._fall_pelvis_z
            upright = pelvis_z >= self._upright_pelvis_z
            heading_r_gated = heading_r if upright else 0.0
            survival_r = 1.0 if upright else 0.0
            dense = (
                self._w_heading * heading_r_gated
                + self._w_act_reg * act_reg
                + self._w_survival * survival_r
                - 5.0 * excess * excess
            )
            return {
                "gait_jpos": 0.0,
                "gait_jvel": 0.0,
                "heading": heading_r_gated,
                "survival": survival_r,
                "dense": dense,
                "sparse": heading_r_gated,
                "solved": heading_r_gated > 0.8,
                "done": fell,
            }

    env = ScratchDirectionalEnv(
        seed=seed, target_speed=0.65, w_jpos=0.0, w_jvel=0.0
    )
    if angle is not None:
        env._angle_override = float(angle)
    return env


class Actor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, ACT_DIM),
        )
        nn.init.zeros_(self.net[-1].bias)
        nn.init.uniform_(self.net[-1].weight, -3e-3, 3e-3)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(obs))


class Critic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        hid = 512
        self.q1 = nn.Sequential(
            nn.Linear(OBS_DIM + ACT_DIM, hid), nn.SiLU(),
            nn.Linear(hid, hid), nn.SiLU(), nn.Linear(hid, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(OBS_DIM + ACT_DIM, hid), nn.SiLU(),
            nn.Linear(hid, hid), nn.SiLU(), nn.Linear(hid, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class Replay:
    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.o = np.zeros((cap, OBS_DIM), np.float32)
        self.a = np.zeros((cap, ACT_DIM), np.float32)
        self.r = np.zeros((cap,), np.float32)
        self.n = np.zeros((cap, OBS_DIM), np.float32)
        self.d = np.zeros((cap,), np.float32)
        self.i = 0
        self.size = 0

    def add(self, o, a, r, n, d) -> None:
        for j in range(o.shape[0]):
            self.o[self.i] = o[j]
            self.a[self.i] = a[j]
            self.r[self.i] = r[j]
            self.n[self.i] = n[j]
            self.d[self.i] = d[j]
            self.i = (self.i + 1) % self.cap
            self.size = min(self.size + 1, self.cap)

    def sample(self, batch: int, device: torch.device):
        idx = np.random.randint(0, self.size, size=batch)
        t = lambda x: torch.as_tensor(x[idx], device=device)  # noqa: E731
        return t(self.o), t(self.a), t(self.r), t(self.n), t(self.d)


@torch.no_grad()
def act_np(actor: nn.Module, obs: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.as_tensor(obs, device=device, dtype=torch.float32)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    return actor(x).clamp(0, 1).cpu().numpy()


@torch.no_grad()
def eval_dirs(
    actor: nn.Module,
    device: torch.device,
    labels: list[str] | None = None,
    steps: int = 80,
    seed: int = 0,
) -> dict:
    labels = labels or LABELS
    env = make_env(seed)
    actor.eval()
    rows = []
    for lab in labels:
        theta = math.radians(DEGS[LABELS.index(lab)])
        env._angle_override = theta
        obs, _ = env.reset(seed=seed)
        start = env.data.qpos[:2].copy()
        n, ret, alive = 0, 0.0, True
        for _ in range(steps):
            obs, r, term, trunc, _ = env.step(act_np(actor, obs, device)[0])
            ret += float(r)
            n += 1
            if term or trunc:
                alive = False
                break
        disp = env.data.qpos[:2] - start
        dist = float(np.linalg.norm(disp))
        h = np.array([math.cos(theta), math.sin(theta)])
        align = float((disp / dist) @ h) if dist > 1e-4 else 0.0
        rows.append((lab, float(alive), n, align, dist, ret))
    env.close()
    print(f"{'Dir':>4}  {'Surv':>5}  {'Steps':>5}  {'Align':>7}  {'Dist':>6}")
    for lab, surv, n, align, dist, _ret in rows:
        print(f"{lab:>4}  {surv:>5.0%}  {n:>5d}  {align:>+7.3f}  {dist:>5.2f}m")
    return {
        "surv": float(np.mean([r[1] for r in rows])),
        "steps": float(np.mean([r[2] for r in rows])),
        "align": float(np.mean([r[3] for r in rows])),
        "dist": float(np.mean([r[4] for r in rows])),
        "return": float(np.mean([r[5] for r in rows])),
    }


def train(
    *,
    steps: int,
    n_envs: int,
    batch: int,
    device: torch.device,
    out: Path,
    smoke: bool = False,
) -> nn.Module:
    actor = Actor().to(device)
    critic = Critic().to(device)
    ta, tc = deepcopy(actor), deepcopy(critic)
    for p in list(ta.parameters()) + list(tc.parameters()):
        p.requires_grad_(False)
    a_opt = torch.optim.Adam(actor.parameters(), lr=1e-4)
    c_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda" if use_amp else "cpu", enabled=use_amp)

    explore, tgt_noise, gamma, tau, pfreq = 0.08, 0.02, 0.99, 0.01, 4
    if smoke:
        start_after, actor_after, utd = 64, 128, 1
    else:
        start_after = min(4_000, max(256, steps // 4))
        actor_after = min(8_000, max(512, steps // 2))
        utd = 2
    buf = Replay(cap=min(100_000, max(steps * n_envs, 2_000)))

    envs = [make_env(1000 + i) for i in range(n_envs)]
    obs = np.stack([e.reset(seed=1000 + i)[0] for i, e in enumerate(envs)])
    collected = updates = 0
    last_q = last_pi = 0.0
    t0 = last = time.time()
    print(f"train smoke={smoke} steps={steps} envs={n_envs} device={device} amp={use_amp}")

    while collected < steps:
        raw = act_np(actor, obs, device)
        act = np.clip(raw + explore * np.random.randn(*raw.shape).astype(np.float32), 0, 1)
        nxt = np.zeros_like(obs)
        rew = np.zeros(n_envs, np.float32)
        done = np.zeros(n_envs, np.float32)
        for i, env in enumerate(envs):
            nobs, r, term, trunc, _ = env.step(act[i])
            d = float(term or trunc)
            nxt[i], rew[i], done[i] = nobs, r, d
            if d:
                nobs, _ = env.reset()
                nxt[i] = nobs
        buf.add(obs, act, rew, nxt, done)
        obs = nxt
        collected += n_envs

        if buf.size >= start_after:
            for _ in range(utd):
                o, a, r, n, d = buf.sample(min(batch, buf.size), device)
                with torch.autocast("cuda", enabled=use_amp):
                    with torch.no_grad():
                        na = (ta(n) + tgt_noise * torch.randn_like(a).clamp(-0.06, 0.06)).clamp(0, 1)
                        q1t, q2t = tc(n, na)
                        target = r + (1.0 - d) * gamma * torch.min(q1t, q2t)
                    q1, q2 = critic(o, a)
                    c_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
                c_opt.zero_grad(set_to_none=True)
                scaler.scale(c_loss).backward()
                scaler.unscale_(c_opt)
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                scaler.step(c_opt)
                updates += 1
                last_q = float(c_loss.detach().cpu())
                if collected >= actor_after and updates % pfreq == 0:
                    with torch.autocast("cuda", enabled=use_amp):
                        q1a, _ = critic(o, actor(o))
                        a_loss = -q1a.mean()
                    a_opt.zero_grad(set_to_none=True)
                    scaler.scale(a_loss).backward()
                    scaler.unscale_(a_opt)
                    nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                    scaler.step(a_opt)
                    last_pi = float(a_loss.detach().cpu())
                    with torch.no_grad():
                        for p, tp in zip(actor.parameters(), ta.parameters()):
                            tp.data.mul_(1 - tau).add_(p.data, alpha=tau)
                scaler.update()
                with torch.no_grad():
                    for p, tp in zip(critic.parameters(), tc.parameters()):
                        tp.data.mul_(1 - tau).add_(p.data, alpha=tau)

        if time.time() - last > 15 or collected >= steps:
            fps = collected / max(time.time() - t0, 1e-6)
            print(
                f"t={collected}/{steps} fps={fps:.0f} buf={buf.size} "
                f"upd={updates} q={last_q:.3f} pi={last_pi:.3f}",
                flush=True,
            )
            last = time.time()

    for env in envs:
        env.close()
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / ("smoke.pt" if smoke else "fasttd3_scratch.pt")
    torch.save({"actor": actor.state_dict()}, ckpt)
    print("saved", ckpt)
    if not math.isfinite(last_q):
        raise RuntimeError(f"non-finite critic loss: {last_q}")
    if not smoke and last_pi == 0.0:
        raise RuntimeError("actor never updated — increase --steps")
    return actor


def render_east(actor: nn.Module, device: torch.device, path: Path, frames: int = 80) -> Path:
    import imageio
    import mujoco

    env = make_env(0, angle=0.0)
    renderer = mujoco.Renderer(env.model, height=320, width=420)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0, 0, 1]
    cam.distance, cam.azimuth, cam.elevation = 4.0, 135, -20
    obs, _ = env.reset(seed=0)
    imgs = []
    actor.eval()
    for _ in range(frames):
        obs, _, term, trunc, _ = env.step(act_np(actor, obs, device)[0])
        cam.lookat[:2] = env.data.qpos[:2]
        renderer.update_scene(env.data, camera=cam)
        imgs.append(np.asarray(renderer.render(), dtype=np.uint8))
        if term or trunc:
            break
    renderer.close()
    env.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=20, quality=6, macro_block_size=1) as w:
        for im in imgs:
            w.append_data(im)
    print("wrote", path, "frames", len(imgs))
    return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="CPU plumbing test (default if no --train)")
    p.add_argument("--train", action="store_true", help="Full run; requires GPU unless --force-cpu")
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=0)
    p.add_argument("--batch", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("runs/fasttd3_scratch"))
    args = p.parse_args()
    smoke = not args.train
    if smoke:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    device = get_device(smoke=smoke or args.force_cpu)
    if args.train and device.type == "cpu" and not args.force_cpu:
        raise SystemExit("No GPU. Use a CUDA/MPS runtime, or pass --force-cpu.")
    steps = args.steps or (400 if smoke else 200_000)
    n_envs = args.n_envs or (1 if smoke else 8)
    batch = args.batch or (64 if smoke else 1024)
    actor = train(
        steps=steps, n_envs=n_envs, batch=batch, device=device, out=args.out, smoke=smoke
    )
    labels = ["E", "N"] if smoke else LABELS
    eval_dirs(actor, device, labels=labels, steps=40 if smoke else 120)
    if not smoke:
        render_east(actor, device, args.out / "east.mp4")
    print("ok" if smoke else "train done")


if __name__ == "__main__":
    main()
