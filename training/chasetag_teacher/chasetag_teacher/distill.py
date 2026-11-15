# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Distil the SuperMimic player (tracker + steering) into one 546 -> 354 policy on the FBVs observation (DAgger).

Collection: bouts in which each SuperMimic-labelled player executes either the teacher's or the student's action
(per bout), the other player being a teacher, the shipped baseline or idle; every step of a labelled player stores
(observation, teacher excitation). Training: an :class:`ActorCritic` (546 -> 354, the FBVs player network of
:func:`~myosuite.integrations.musclemimic.fbvs_player.load_fbvs_player`) regressing the excitations (MSE). The
input normalisation used in training is folded into the first layer, so the saved state dict is a plain
``ActorCritic`` that ``export_fbvs_onnx`` and the FBVsTrain self-play take as is.

The yaw-canonicalised student (:func:`canonicalise`) reads the observation rotated about z by minus the player's own
root yaw, so it is invariant to rotations of the scene by construction; :func:`export_canon_onnx` puts that rotation
in front of the network, so its ONNX still takes the raw 546-d observation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from myosuite.envs.myo.tasks.mimic.policy import ActorCritic

OBS_DIM, ACT_DIM = 546, 354


class StudentPlayer:
    """An ``ActorCritic`` state dict (546 -> 354) as a bout player; actions clamped to [0, 1].

    Args:
        path: The state dict (``student_r<k>.pt``).
        canon: The student reads canonicalised observations (:func:`canonicalise`).
    """

    def __init__(self, path: str | Path, canon: bool = False):
        net = ActorCritic.load(path, obs_dim=OBS_DIM, act_dim=ACT_DIM)
        self.net = CanonPlayerNet(net).eval() if canon else net
        self.canon = canon

    def reset(self, env: Any, agent: str) -> None:
        pass

    @torch.no_grad()
    def act(self, env: Any, agent: str, obs: np.ndarray) -> np.ndarray:
        if self.canon:
            return self.net(torch.as_tensor(np.asarray(obs, np.float32))[None])[
                0
            ].numpy()
        return np.clip(self.net.act(np.asarray(obs, np.float32)), 0.0, 1.0)


def collect_bout(env: Any, teacher_a: Any, teacher_b: Any, opp_b: Any, student: StudentPlayer | None, beta: float,
                 seed: int, task: str, noise: float, max_steps: int = 1000, every: int = 1, grace: int = 20,
                 takeover_z: float = 0.0, swap: bool = False) -> dict:  # fmt: skip
    """One bout. agent_0 is a teacher; agent_1 is ``teacher_b`` (labelled) or ``opp_b`` (unlabelled). Each labelled
    player executes the student's action with probability 1 - beta (chosen per bout; the teacher then reads the
    executed excitation as its previous action). Every ``every``-th step of a
    labelled player is stored until ``grace`` steps after it fell; the bout stops once every labelled player is past
    that. A student whose pelvis drops below ``takeover_z`` (m) hands over to the teacher for the rest of the bout.
    ``swap`` exchanges the two sides (``teacher_a`` plays agent_1)."""
    from chasetag_teacher.match import AGENTS  # noqa: PLC0415

    u = env.unwrapped
    u._config.task_choice = task
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed + 7)
    players = {
        "agent_0": teacher_a,
        "agent_1": teacher_b if teacher_b is not None else opp_b,
    }
    labelled = {"agent_0": True, "agent_1": teacher_b is not None}
    if swap:
        players = {"agent_0": players["agent_1"], "agent_1": players["agent_0"]}
        labelled = {"agent_0": labelled["agent_1"], "agent_1": labelled["agent_0"]}
    use_student = {
        a: student is not None and labelled[a] and rng.random() > beta for a in AGENTS
    }
    for a in AGENTS:
        players[a].reset(env, a)
    X, Y = [], []
    fell = {a: None for a in AGENTS}
    down = {a: 0 for a in AGENTS}  # steps since the fall
    took = {a: None for a in AGENTS}  # takeover time
    sid = u._meta.site_ids
    for step in range(max_steps):
        acts = {}
        for a in AGENTS:
            x = np.asarray(players[a].act(env, a, obs[a]), np.float32)
            if labelled[a]:
                if step % every == 0 and down[a] <= grace:
                    X.append(np.asarray(obs[a], np.float32))
                    Y.append(x)
                if use_student[a] and took[a] is None:
                    x = student.act(env, a, obs[a])
            x = x + rng.normal(0.0, noise, x.shape)
            acts[a] = np.clip(x, 0.0, 1.0)
            if (
                use_student[a]
                and took[a] is None
                and u.data.site_xpos[sid[a]["pelvis_site"]][2] < takeover_z
            ):
                took[a] = float(info.get("elapsed_s", 0.0))
            if (
                use_student[a] and took[a] is None
            ):  # the teacher's excitation observation: what was executed
                players[a].p.tracker.prev_action = acts[a].astype(np.float32)
        obs, _, term, trunc, info = env.step(acts)
        for a in AGENTS:
            if fell[a] is None and info["fell"].get(a):
                fell[a] = info["elapsed_s"]
            down[a] += fell[a] is not None
        stopped = all(down[a] > grace for a in AGENTS if labelled[a])
        if term["agent_0"] or trunc["agent_0"] or stopped:
            break
    end = (
        "tag"
        if info["tagged"]
        else (
            "chaser_fall"
            if info["chaser_fell"]
            else "stopped"
            if stopped
            else "time_up"
        )
    )
    return {"obs": np.array(X), "act": np.array(Y), "end": end, "t": float(info["elapsed_s"]),
            "student": [a for a in AGENTS if use_student[a]], "fell": fell,
            "took": took}  # fmt: skip


def fold_normalisation(net: ActorCritic, mean: torch.Tensor, std: torch.Tensor) -> None:
    """Make ``net(x)`` compute what it computed on ``(x - mean) / std``."""
    lin = net.trunk[0]
    with torch.no_grad():
        W = lin.weight / std
        lin.bias -= W @ mean
        lin.weight.copy_(W)


_XY_BLOCKS = (0, 89, 531, 534, 537, 540)
"""Starts of the world-frame xy pairs of the FBVs observation: root position, root linear velocity, own pelvis
position and velocity, opponent-relative position and velocity (the root angular velocity is body-frame)."""


def yaw_rotate(obs: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """The FBVs observation of the scene rotated by ``theta`` about the world z axis (an exact symmetry:
    checked against the simulator); muscle targets are unchanged."""
    out = obs.clone()
    c, s = theta.cos(), theta.sin()
    for i in _XY_BLOCKS:
        x, y = obs[:, i], obs[:, i + 1]
        out[:, i], out[:, i + 1] = c * x - s * y, s * x + c * y
    h = theta / 2  # root quaternion (w, x, y, z): q' = (cos h, 0, 0, sin h) * q
    ch, sh = h.cos(), h.sin()
    w, x, y, z = obs[:, 3], obs[:, 4], obs[:, 5], obs[:, 6]
    out[:, 3], out[:, 4], out[:, 5], out[:, 6] = (
        ch * w - sh * z,
        ch * x - sh * y,
        ch * y + sh * x,
        ch * z + sh * w,
    )
    return out


def train(obs: np.ndarray, act: np.ndarray, epochs: int = 20, batch: int = 4096, lr: float = 1e-3,
          device: str = "cpu", init: dict | None = None, log: Callable[[str], None] = print,
          yaw_aug: bool = False, obs_dim: int = OBS_DIM) -> dict:  # fmt: skip
    """Fit an ``ActorCritic`` student; returns ``{"state": plain state dict, "val_mse": ...}``.

    Args:
        obs: (N, obs_dim) observations (float16/32).
        act: (N, 354) teacher excitations.
        init: A previous result to start from (warm start; its normalisation is kept).
        yaw_aug: Rotate every training sample about the world z axis by a random angle (:func:`yaw_rotate`); the
            normalisation then uses rotated samples too.
        obs_dim: Observation size (546 for FBVs players, 528 for the directional walker).
    """
    obs_arr = np.asarray(obs)
    act_arr = np.asarray(act)
    if obs_arr.ndim != 2 or obs_arr.shape[1] != obs_dim:
        raise ValueError(f"obs must have shape (N, {obs_dim}), got {obs_arr.shape}")
    if act_arr.ndim != 2 or act_arr.shape != (len(obs_arr), ACT_DIM):
        raise ValueError(
            f"act must have shape ({len(obs_arr)}, {ACT_DIM}), got {act_arr.shape}"
        )
    n = len(obs_arr)
    if n == 0:
        raise ValueError("obs must be non-empty")
    # a large dataset stays in host memory; batches move to the device
    host = obs_arr.nbytes + act_arr.nbytes > 4 * 2**30
    X = torch.as_tensor(obs_arr) if host else torch.as_tensor(obs_arr).to(device)
    Y = torch.as_tensor(act_arr) if host else torch.as_tensor(act_arr).to(device)
    if init is not None:
        mean, std = init["mean"].to(device), init["std"].to(device)
    else:
        sub = X[::17].to(device).float()
        if yaw_aug:
            sub = yaw_rotate(sub, torch.rand(len(sub), device=device) * 2 * np.pi)
        mean = sub.mean(0)
        std = sub.std(0, correction=0).clamp(min=1e-2)
    net = ActorCritic(obs_dim, ACT_DIM).to(device)
    if init is not None:
        net.load_state_dict(init["raw_state"])
    perm = torch.randperm(n, device=X.device)
    n_val = min(max(1, n // 50), n - 1) if n > 1 else 1
    val = perm[:n_val]
    tr = perm[n_val:] if n > 1 else perm
    params = list(net.trunk.parameters()) + list(net.actor_mean.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    step_batch = min(batch, len(tr))
    nb = len(tr) // step_batch
    sched = (
        torch.optim.lr_scheduler.OneCycleLR(
            opt, lr, total_steps=epochs * nb, pct_start=0.05
        )
        if epochs * nb > 0
        else None
    )
    f = lambda o: net.actor_mean(net.trunk((o.float() - mean) / std))  # noqa: E731  (o on the device)

    def _val_mse() -> float:
        net.eval()
        with torch.no_grad():
            return float(np.mean([((f(X[val[j : j + 8192]].to(device)).clamp(0, 1) - Y[val[j : j + 8192]].to(device).float()) ** 2).mean().item()
                                  for j in range(0, len(val), 8192)]))  # fmt: skip

    v = float("nan")
    for ep in range(epochs):
        net.train()
        idx = tr[torch.randperm(len(tr), device=X.device)]
        tot = 0.0
        for i in range(nb):
            b = idx[i * step_batch : (i + 1) * step_batch]
            x = X[b].to(device, non_blocking=True).float()
            if yaw_aug:
                x = yaw_rotate(x, torch.rand(len(x), device=device) * 2 * np.pi)
            loss = ((f(x) - Y[b].to(device, non_blocking=True).float()) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            tot += loss.item()
        v = _val_mse()
        log(f"epoch {ep} train {tot / max(nb, 1):.5f} val {v:.5f}")
    if epochs == 0:
        v = _val_mse()
    raw = {k: x.detach().cpu().clone() for k, x in net.state_dict().items()}
    net = net.cpu()
    fold_normalisation(net, mean.cpu(), std.cpu())
    return {"state": {k: x.detach().clone() for k, x in net.state_dict().items()}, "raw_state": raw,
            "mean": mean.cpu(), "std": std.cpu(), "val_mse": v}  # fmt: skip


def root_yaw(obs: torch.Tensor) -> torch.Tensor:
    """Yaw of the player's root quaternion (w, x, y, z at 3:7 of the FBVs observation)."""
    w, x, y, z = obs[:, 3], obs[:, 4], obs[:, 5], obs[:, 6]
    return torch.atan2(
        2 * (w * z + x * y), w * w + x * x - y * y - z * z
    )  # exact for unnormalised quaternions too


def canonicalise(obs: torch.Tensor) -> torch.Tensor:
    """The observation rotated about z by minus the player's own root yaw: yaw-invariant by construction.

    Functional (no in-place writes) so it exports to ONNX as fixed ops in front of the network.
    """
    th = -root_yaw(obs)
    c, s = th.cos().unsqueeze(-1), th.sin().unsqueeze(-1)
    hc, hs = (th / 2).cos(), (th / 2).sin()
    w, x, y, z = obs[:, 3], obs[:, 4], obs[:, 5], obs[:, 6]
    quat = torch.stack(
        (hc * w - hs * z, hc * x - hs * y, hc * y + hs * x, hc * z + hs * w), -1
    )
    quat = quat * torch.where(
        quat[:, :1] < 0, -1.0, 1.0
    )  # q and -q are one rotation: pick w >= 0

    def rot(i: int) -> torch.Tensor:
        a, b = obs[:, i : i + 1], obs[:, i + 1 : i + 2]
        return torch.cat((c * a - s * b, s * a + c * b), -1)

    return torch.cat((rot(0), obs[:, 2:3], quat, obs[:, 7:89], rot(89), obs[:, 91:531], rot(531), obs[:, 533:534],
                      rot(534), obs[:, 536:537], rot(537), obs[:, 539:540], rot(540), obs[:, 542:]), -1)  # fmt: skip


class CanonPlayerNet(torch.nn.Module):
    """Raw 546-d FBVs observation -> excitations in [0, 1]: :func:`canonicalise`, then an ``ActorCritic`` mean.

    Args:
        net: The student, trained on canonicalised observations.
    """

    def __init__(self, net: ActorCritic):
        super().__init__()
        self.trunk, self.actor_mean = net.trunk, net.actor_mean

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Excitations (B, 354) for raw observations (B, 546)."""
        return torch.clamp(self.actor_mean(self.trunk(canonicalise(obs))), 0.0, 1.0)


def export_canon_onnx(state: dict, path: str | Path) -> None:
    """ONNX ``obs [batch, 546] -> action [batch, 354]`` of a student trained on canonicalised observations."""
    net = ActorCritic(OBS_DIM, ACT_DIM)
    net.load_state_dict(state)
    torch.onnx.export(CanonPlayerNet(net).eval(), torch.zeros(1, OBS_DIM), str(path), input_names=["obs"],
                      output_names=["action"], dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
                      opset_version=17, do_constant_folding=True, dynamo=False)  # fmt: skip


def canonicalise_inplace(obs: np.ndarray, chunk: int = 262144) -> np.ndarray:
    """:func:`canonicalise` a large (N, 546) array in float32 chunks, written back in its own dtype."""
    for j in range(0, len(obs), chunk):
        obs[j : j + chunk] = canonicalise(
            torch.as_tensor(obs[j : j + chunk]).float()
        ).numpy()
    return obs


def load_data(
    dirs: Sequence[str | Path], obs_dim: int = OBS_DIM
) -> tuple[np.ndarray, np.ndarray]:
    """The (N, 546) observations and (N, 354) teacher excitations (float16) of the bout files (``b*.npz``) in
    ``dirs``, as :mod:`scripts.dagger` writes them."""
    files = sorted(f for d in dirs for f in Path(d).glob("b*.npz"))
    sizes: list[int] = []
    for f in files:
        with np.load(f) as z:
            sizes.append(int(z["act"].shape[0]))
    X = np.empty((sum(sizes), obs_dim), np.float16)
    Y = np.empty((sum(sizes), ACT_DIM), np.float16)
    k = 0
    for f, n in zip(files, sizes):
        with np.load(f) as z:
            X[k : k + n], Y[k : k + n] = z["obs"], z["act"]
        k += n
    return X, Y
