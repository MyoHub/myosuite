"""Render a FastTD3 checkpoint to MP4. Safe under xvfb-run (Colab has no DISPLAY)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--env", default="myoMimicFullbody-v0")
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    if not os.environ.get("MUJOCO_GL"):
        os.environ["MUJOCO_GL"] = (
            "glfw" if sys.platform == "darwin" or os.environ.get("DISPLAY") else "egl"
        )

    import imageio
    import mujoco
    import numpy as np
    import torch
    from fast_td3.fast_td3 import Actor
    from fast_td3.fast_td3_utils import EmpiricalNormalization
    from myosuite.core.registry import make_env

    ckpt_path = Path(args.ckpt)
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    n_obs, n_act = int(blob["n_obs"]), int(blob["n_act"])
    hidden = int(blob["actor"]["net.0.weight"].shape[0])
    n_train_envs = int(blob["actor"]["noise_scales"].shape[0])
    actor = Actor(
        n_obs=n_obs,
        n_act=n_act,
        num_envs=n_train_envs,
        init_scale=0.01,
        hidden_dim=hidden,
        device=torch.device("cpu"),
    )
    actor.load_state_dict(blob["actor"])
    actor.eval()
    obs_norm = EmpiricalNormalization(shape=n_obs, device=torch.device("cpu"))
    norm_state = blob.get("obs_norm")
    if isinstance(norm_state, dict) and norm_state:
        obs_norm.load_state_dict(norm_state)
    obs_norm.eval()

    env = make_env(args.env)
    obs, _ = env.reset(seed=0)
    raw = env.unwrapped
    fps = round(1.0 / float(getattr(raw, "_ctrl_dt", 0.01)))
    renderer = mujoco.Renderer(raw.model, height=320, width=420)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 1.0]
    cam.distance, cam.azimuth, cam.elevation = 3.5, 135.0, -20.0

    frames = []
    for _ in range(args.steps):
        x = torch.as_tensor(np.asarray(obs, np.float32)).reshape(1, -1)
        if int(x.shape[-1]) != n_obs:
            raise ValueError(f"obs dim {x.shape[-1]} != checkpoint n_obs {n_obs}")
        with torch.no_grad():
            tanh_act = actor(obs_norm(x, update=False)).cpu().numpy()[0]
        muscle = (np.clip(tanh_act, -1.0, 1.0) + 1.0) * 0.5
        obs, _, terminated, truncated, _ = env.step(muscle)
        if raw.data.qpos.size > 2:
            cam.lookat[:2] = raw.data.qpos[:2]
        renderer.update_scene(raw.data, camera=cam)
        frames.append(np.asarray(renderer.render(), dtype=np.uint8).copy())
        if terminated or truncated:
            break
    renderer.close()
    env.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(out), fps=fps, quality=8, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"{ckpt_path.name}: {len(frames)} frames → {out}", flush=True)


if __name__ == "__main__":
    main()
