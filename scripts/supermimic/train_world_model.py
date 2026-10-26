# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Fit the SuperMimic world model offline on recorded Mimic rollouts and evaluate it.

Reports the k-step open-loop normalised state error on held-out rollouts and a
closed-loop check: the recording policy driven inside the world model (analytic
observations and rewards) against the simulator from the same start states.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from myosuite.envs.myo.backends.mjlab.tasks.cpu_reference import _musclemimic_cpu_model
from myosuite.learning.supermimic.env_io import Rollout, make_mimic_env
from myosuite.learning.supermimic.heads import MimicHeads
from myosuite.learning.supermimic.metrics import gradient_agreement
from myosuite.learning.supermimic.policy import Actor
from myosuite.learning.supermimic.state import StateSpec, forward_axis_from_quat
from myosuite.learning.supermimic.world_model import (
    WindowSampler,
    WorldModel,
    WorldModelCfg,
)

GROUPS = (
    "root_pos",
    "root_rot",
    "root_linvel",
    "root_angvel",
    "joint_pos",
    "joint_vel",
)


def group_errors(spec: StateSpec, e: torch.Tensor) -> dict[str, float]:
    """Mean squared normalised error per quantity group."""
    nj = spec.nj
    sizes = (3, 3, 3, 3, nj, nj)
    out, i = {}, 0
    for g, n in zip(GROUPS, sizes):
        out[g] = float((e[..., i : i + n] ** 2).mean())
        i += n
    return out


@torch.no_grad()
def open_loop_eval(
    wm: WorldModel, sampler: WindowSampler, n: int = 4096
) -> dict[str, float]:
    """1-step and k-step open-loop errors on held-out windows."""
    g = torch.Generator(device=sampler.starts.device).manual_seed(0)
    s0, u, tgt = sampler.sample(n, g)
    e = wm.spec.error(wm.rollout(s0, u), tgt)
    per_step = (e * e).mean(-1).mean(-1)
    return {
        "err_1": float(per_step[0]),
        f"err_{sampler.k}": float(per_step[-1]),
        "err_mean": float(per_step.mean()),
        **{f"k_{k}": v for k, v in group_errors(wm.spec, e[-1]).items()},
    }


@torch.no_grad()
def closed_loop_eval(
    wm: WorldModel,
    heads: MimicHeads,
    actor: Actor,
    ro: Rollout,
    noise: torch.Tensor,
    horizon: int,
) -> dict[str, float]:
    """Recording policy in the world model vs the (deterministic) recorded sim rollout."""
    sp = wm.spec
    envs = torch.nonzero(noise == 0).flatten()
    T = ro.ctrl.shape[0]
    d = ro.done[:, envs].float()
    csum = torch.cat([torch.zeros_like(d[:1]), d.cumsum(0)], 0)
    starts = [t for t in range(0, T - horizon, horizon)]
    sim_r, wm_r, site_div = [], [], []
    for t0 in starts:
        ok = (csum[t0 + horizon, :] - csum[t0, :]) == 0
        e = envs[ok]
        if len(e) == 0:
            continue
        s = ro.state[t0, e]
        f = ro.frame[t0, e]
        r_wm, div = [], []
        for t in range(horizon):
            qpos, qvel, act = sp.split(s)
            ctrl = torch.sigmoid(
                5.0 * (actor(heads.observation(qpos, qvel, act, f)) - 0.5)
            )
            s = wm.step(s, ctrl)
            f = f + 1
            r_wm.append(heads.reward(s[:, : sp.nq], s[:, sp.nq : sp.nq + sp.nv], f))
            s_sim = ro.state[t0 + t + 1, e]
            div.append(
                (
                    heads.kin.site_pos(s[:, : sp.nq])
                    - heads.kin.site_pos(s_sim[:, : sp.nq])
                )
                .norm(dim=-1)
                .mean(-1)
            )
        sim_r.append(ro.reward[t0 : t0 + horizon, e] / 5.0)
        wm_r.append(torch.stack(r_wm))
        site_div.append(torch.stack(div))
    sim_r, wm_r, site_div = (torch.cat(x, 1) for x in (sim_r, wm_r, site_div))
    return {
        "closed_sim_reward": float(sim_r.mean()),
        "closed_wm_reward": float(wm_r.mean()),
        "closed_reward_abs_err": float((sim_r - wm_r).abs().mean()),
        "closed_site_div_mid_m": float(site_div[horizon // 2 - 1].mean()),
        "closed_site_div_end_m": float(site_div[-1].mean()),
        "closed_windows": int(sim_r.shape[1]),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--eval", type=Path, required=True)
    p.add_argument("--clip", required=True)
    p.add_argument(
        "--checkpoint", required=True, help="recording policy (closed-loop check)"
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--hidden", type=int, nargs="+", default=[2048, 2048, 2048])
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--no-couplings", action="store_true")
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--init",
        type=Path,
        help="start from this world_model.pt (with --steps 0: evaluate only)",
    )
    p.add_argument(
        "--grad-check-starts",
        type=int,
        default=16,
        help="starts of the final gradient check (0: skip)",
    )
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    a = p.parse_args()
    dev = a.device
    torch.manual_seed(0)
    a.out.mkdir(parents=True, exist_ok=True)

    tr = Rollout.load(a.train, dev)
    ev = Rollout.load(a.eval, dev)
    ev_noise = torch.load(a.eval.with_suffix(".meta.pt"))["noise"].to(dev)
    env = make_mimic_env(a.clip, 1, dev)  # heads only: clip tensors and site ids
    heads = MimicHeads.from_env(env, "mimic_fullbody_robot")
    model = _musclemimic_cpu_model("fullbody")
    spec = StateSpec(
        model.nq,
        model.nv,
        model.na,
        heads.ctrl_dt,
        forward_axis_from_quat(heads.clip.qpos[0, 3:7].cpu()),
    ).to(dev)
    cfg = WorldModelCfg(
        hidden=tuple(a.hidden),
        window=a.window,
        lr=a.lr,
        batch=a.batch,
        project_couplings=not a.no_couplings,
    )
    wm = WorldModel(model, spec, env.cfg.decimation, env.physics_dt, cfg).to(dev)
    if a.init:
        wm.load_state_dict(torch.load(a.init, map_location=dev)["wm"])
    actor = Actor.from_rsl_rl(a.checkpoint, dev)
    train_s = WindowSampler(tr.state, tr.ctrl, tr.done, a.window)
    eval_s = WindowSampler(ev.state, ev.ctrl, ev.done, a.window)
    print(
        f"train windows {len(train_s)}, eval windows {len(eval_s)}, feat {spec.feat_dim}, delta {spec.delta_dim}",
        flush=True,
    )
    opt = torch.optim.Adam(wm.parameters(), lr=a.lr)
    torch.backends.cuda.matmul.allow_tf32 = a.tf32
    log = []
    t0 = time.time()
    rec = {"step": 0, **open_loop_eval(wm, eval_s)}
    print(json.dumps(rec), flush=True)
    for step in range(1, a.steps + 1):
        s0, u, tgt = train_s.sample(a.batch)
        loss, _ = wm.window_loss(s0, u, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), cfg.grad_clip)
        opt.step()
        if step % a.eval_every == 0 or step == a.steps:
            rec = {
                "step": step,
                "time_s": round(time.time() - t0),
                "train_loss": float(loss),
                **open_loop_eval(wm, eval_s),
            }
            if step % (5 * a.eval_every) == 0 or step == a.steps:
                rec.update(closed_loop_eval(wm, heads, actor, ev, ev_noise, a.horizon))
            log.append(rec)
            print(json.dumps(rec), flush=True)
            torch.save(
                {
                    "wm": wm.state_dict(),
                    "cfg": cfg.__dict__,
                    "forward_axis": spec.forward_axis.cpu(),
                },
                a.out / "world_model.pt",
            )
            (a.out / "log.json").write_text(json.dumps(log, indent=1))

    if a.grad_check_starts > 0:
        # World-model action gradients vs MuJoCo finite differences on held-out windows.
        g = torch.Generator(device=eval_s.starts.device).manual_seed(1)
        s0, u, _ = eval_s.sample(a.grad_check_starts, g)
        model.opt.timestep = env.physics_dt
        rec = {
            "step": a.steps,
            **gradient_agreement(wm, model, s0, u, env.cfg.decimation),
        }
        log.append(rec)
        print(json.dumps(rec), flush=True)
        (a.out / "log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
