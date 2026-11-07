# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Side-by-side video of Mimic policies (rsl_rl-layout checkpoints) and the reference clip.

Every policy starts at the same clip frame in its own single-env mjlab rollout; the
recorded states are rendered with CPU MuJoCo, muscles coloured by activation
(grey = 0, red = 1). A still contact sheet of a few frames is written next to the video.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from myosuite.envs.myo.backends.mjlab.tasks.cpu_reference import _musclemimic_cpu_model
from myosuite.learning.supermimic.env_io import MimicEnvIO, make_mimic_env
from myosuite.learning.supermimic.policy import Actor

_ENTITY = "mimic_fullbody_robot"


def _start_at_frame(env, io: MimicEnvIO, frame: int) -> None:
    """Reset to clip *frame* (the RSI state of that frame, zero activation)."""
    from mjlab.utils.lab_api.math import quat_apply

    env.reset()
    src = io.source
    ids = torch.zeros(1, dtype=torch.long, device=env.device)
    src._start_offsets[:] = frame
    src._last_step[:] = 0
    f = torch.full((1,), frame, dtype=torch.long, device=env.device)
    q, v = src.ref_qpos_at_frames(f).float(), src.ref_qvel_at_frames(f).float()
    root = torch.cat(
        [q[:, :3], q[:, 3:7], v[:, :3], quat_apply(q[:, 3:7], v[:, 3:6])], -1
    )
    ent = env.scene[_ENTITY]
    ent.write_root_state_to_sim(root, env_ids=ids)
    ent.write_joint_state_to_sim(q[:, 7:], v[:, 6:], env_ids=ids)
    env.sim.forward()
    env.obs_buf = env.observation_manager.compute()


@torch.no_grad()
def rollout(
    env, io: MimicEnvIO, actor: Actor | None, frame: int, steps: int
) -> dict[str, np.ndarray]:
    """States, rewards and the step of the first early termination of one policy."""
    _start_at_frame(env, io, frame)
    states, rewards, fell = [io.state()[0].cpu().numpy()], [], -1
    obs = env.obs_buf["actor"]
    for t in range(steps):
        out, rew, term, trunc, _ = env.step(actor(obs))
        obs = out["actor"]
        if bool(term[0]) and fell < 0:
            fell = t + 1
        if bool(term[0] | trunc[0]):
            break
        states.append(io.state()[0].cpu().numpy())
        rewards.append(float(rew[0]))
    return {"state": np.stack(states), "reward": np.asarray(rewards), "fell": fell}


class PanelRenderer:
    """Tracking-camera render of a ``[qpos | qvel | act]`` state, muscles coloured by activation."""

    def __init__(self, width: int, height: int, azimuth: float) -> None:
        self.model = _musclemimic_cpu_model("fullbody")
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, width)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, height)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height, width)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 3.4, -10.0, azimuth
        m = self.model
        self.muscle_tendons = m.actuator_trnid[:, 0]
        self.base_rgba = m.tendon_rgba.copy()

    def render(self, state: np.ndarray) -> np.ndarray:
        m, d = self.model, self.data
        nq, nv = m.nq, m.nv
        d.qpos[:] = state[:nq]
        d.qvel[:] = state[nq : nq + nv]
        act = np.clip(state[nq + nv :], 0.0, 1.0)
        rgba = self.base_rgba.copy()
        rgba[self.muscle_tendons] = np.stack(
            [0.55 + 0.45 * act, 0.55 * (1 - act), 0.55 * (1 - act), np.ones_like(act)],
            -1,
        )
        m.tendon_rgba[:] = rgba
        mujoco.mj_forward(m, d)
        self.cam.lookat[:] = d.qpos[:3] + np.array([0.0, 0.0, -0.1])
        self.renderer.update_scene(d, self.cam)
        return self.renderer.render()


def _label(
    img: np.ndarray, lines: list[str], font: ImageFont.ImageFont, header: int = 96
) -> np.ndarray:
    """Panel with a white text header above the render."""
    im = Image.new("RGB", (img.shape[1], img.shape[0] + header), (255, 255, 255))
    im.paste(Image.fromarray(img), (0, header))
    draw = ImageDraw.Draw(im)
    y = 6
    for i, text in enumerate(lines):
        draw.text((10, y), text, fill=(0, 0, 0) if i == 0 else (70, 70, 70), font=font)
        y += 22
    draw.line(
        [(img.shape[1] - 1, 0), (img.shape[1] - 1, im.height)],
        fill=(200, 200, 200),
        width=2,
    )
    return np.asarray(im)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clip", required=True)
    p.add_argument(
        "--policy",
        nargs=2,
        action="append",
        metavar=("LABEL", "CHECKPOINT"),
        required=True,
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--steps", type=int, default=480)
    p.add_argument(
        "--fps", type=int, default=50, help="video rate (the control rate is 100 Hz)"
    )
    p.add_argument("--size", type=int, nargs=2, default=(420, 480), metavar=("W", "H"))
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    env = make_mimic_env(a.clip, 1, a.device, seed=0)
    io = MimicEnvIO(env, _ENTITY)
    runs = []
    for label, ck in a.policy:
        r = rollout(env, io, Actor.from_rsl_rl(ck, a.device), a.start_frame, a.steps)
        print(
            f"{label}: {len(r['reward'])} steps, mean reward {r['reward'].mean():.2f}/step, "
            f"mean activation {r['state'][:, -354:].mean():.2f}, fell at {r['fell']}",
            flush=True,
        )
        runs.append((label, r))
    # Reference: the clip's own states (activation unknown -> drawn at zero).
    src = io.source
    n = min(a.steps + 1, src.n_frames - a.start_frame)
    f = torch.arange(a.start_frame, a.start_frame + n)
    ref = (
        torch.cat(
            [src.ref_qpos_at_frames(f), src.ref_qvel_at_frames(f), torch.zeros(n, 354)],
            -1,
        )
        .cpu()
        .numpy()
    )

    # Side view: camera azimuth perpendicular to the clip's walking direction.
    d = ref[-1, :2] - ref[0, :2]
    rend = PanelRenderer(
        *a.size, azimuth=float(np.degrees(np.arctan2(d[1], d[0]))) + 90.0
    )
    try:
        font = ImageFont.load_default(size=17)
    except TypeError:
        font = ImageFont.load_default()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        a.out, fps=a.fps, codec="libx264", quality=8, macro_block_size=8
    )
    stride = max(1, round(100 / a.fps))
    stills = []
    for t in range(0, n, stride):
        panels = [
            _label(
                rend.render(ref[t]), ["Reference clip", f"t = {t / 100:.2f} s"], font
            )
        ]
        for label, r in runs:
            k = min(t, len(r["state"]) - 1)
            name, *info = label.split("|")
            lines = [name, *info]
            rew = f"{r['reward'][:k].mean():.2f}" if k else "-"
            lines.append(
                f"reward {rew} / 5 per step, activation {r['state'][k, -354:].mean():.2f}"
            )
            if 0 <= r["fell"] <= t:
                lines.append(
                    f"terminated (off track > 0.3 m) at {r['fell'] / 100:.2f} s"
                )
            panels.append(_label(rend.render(r["state"][k]), lines, font))
        frame = np.concatenate(panels, axis=1)
        writer.append_data(frame)
        if t in {round(x) for x in np.linspace(0, n - 1, 4) // stride * stride}:
            stills.append(frame)
    writer.close()
    if stills:
        imageio.imwrite(a.out.with_suffix(".png"), np.concatenate(stills, axis=0))
    print(f"wrote {a.out} and {a.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
