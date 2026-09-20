# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Render the Phase 6 Bolt-TD3-transfer rollout on myoSprint-v0 using mjlab's own OffscreenRenderer (not a
hand-built mujoco.Renderer script) -- with a full bone skeleton, including head/neck/spine (see
env_cfgs.py's _sprint_spec_fn_visual / _TORSO_EXTRA_MESHES: Bolt's own sprinter model has no separate
head/neck/spine bodies -- a single rigid "torso" represents the whole HAT segment -- so these are visual-only
meshes attached to the torso body, not new physics).

CPU only (no GPU involved anywhere in this script).

Run:
    KMP_DUPLICATE_LIB_OK=TRUE uv run python myosuite/tests/render_myosprint_mjlab.py
"""
from __future__ import annotations

import os
import sys

import imageio.v2 as imageio
import numpy as np
import torch

os.environ.setdefault("VV_WARP_DEVICE", "cpu")

BOLT_VV_ROOT = "/Users/caggiano/Desktop/myo_bolt_vv"
MSK_REPRO = f"{BOLT_VV_ROOT}/scripts/msk_envs_repro"
MSK_ENVS_ROOT = "/tmp/msk_envs_check"
sys.path[:0] = [MSK_REPRO, MSK_ENVS_ROOT]

CKPT = f"{MSK_REPRO}/artifacts/repro_sprint_19000.pt"
HORIZON = 90
FALL_Y = 0.6
OUT_MP4 = f"{MSK_REPRO}/artifacts/renders/myosprint_mjlab_native_render.mp4"


def main():
    import yaml
    from bolt_sprint_parity import bolt_action_to_mj_ctrl, load_orders, make_bolt_sprint_obs, muscle_obs_norm_fiber
    from msk_envs.train.nets.deterministic_policy import DeterministicPolicy
    from msk_envs.train.nets.normalizers import EmpiricalNormalization

    import myo_sim
    from myo_sim.sprinter_assets import load_map, load_muscles
    from mjlab.entity import EntityCfg
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.offscreen_renderer import OffscreenRenderer
    from mjlab.viewer.viewer_config import ViewerConfig

    from myosuite.envs.myo.backends.mjlab.tasks.myosprint.env_cfgs import make_sprint_env_cfg, SPRINT_ENTITY

    orders = load_orders()
    muscles = load_muscles()
    mapf = load_map()
    qadr, dadr = mapf["qposadr"], mapf["dofadr"]
    mjm = myo_sim.load_spec("myosprinter").compile()

    pose = yaml.safe_load(open(f"{MSK_REPRO}/starting_pose_run.yaml"))
    joint_pos, joint_vel = {}, {}
    lin_vel = (0.0, 0.0, 0.0)
    for name, spec in pose.items():
        q, v = float(spec.get("q", 0.0)), float(spec.get("v", 0.0))
        if name == "pelvis_tx":
            lin_vel = (v, lin_vel[1], lin_vel[2])
        elif name == "pelvis_ty":
            lin_vel = (lin_vel[0], v, lin_vel[2])
        elif name == "pelvis_tz":
            lin_vel = (lin_vel[0], lin_vel[1], v)
        elif name in ("pelvis_quat_w", "pelvis_tilt", "pelvis_list", "pelvis_rotation"):
            continue
        else:
            joint_pos[f"^{name}$"] = q
            joint_vel[f"^{name}$"] = v
    run_init_state = EntityCfg.InitialStateCfg(
        pos=(0.0, 1.05, 0.0), lin_vel=lin_vel, joint_pos=joint_pos, joint_vel=joint_vel
    )

    # visual=True: attaches real bone meshes (head/neck/spine included) for this render only -- the same
    # physics (BoltMuscleAction / myo_sim.warp) drives it either way.
    cfg = make_sprint_env_cfg(num_envs=1, init_state=run_init_state, default_activation=0.5, visual=True)
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset()
    action_term = env.action_manager.get_term("bolt_muscles")

    # mjlab's own offscreen renderer, tracking the sprinter's root body -- native mjlab camera/scene, not a
    # hand-rolled one.
    viewer_cfg = ViewerConfig(
        origin_type=ViewerConfig.OriginType.ASSET_ROOT,
        entity_name=SPRINT_ENTITY,
        distance=3.6,
        elevation=-12.0,
        azimuth=90.0,
        height=480,
        width=640,
    )
    renderer = OffscreenRenderer(env.sim.mj_model, viewer_cfg, env.scene)
    renderer.initialize()

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    aargs = ck.get("args", {})
    ons = ck["obs_normalizer_state"]
    mean = ons.get("_mean", ons.get("running_mean"))
    n_obs = int(mean.shape[-1])
    n_act = len(muscles) + 2
    actor = DeterministicPolicy(
        n_obs=n_obs, n_act=n_act, num_envs=1,
        hidden_dim=int(aargs.get("actor_hidden_dim", 512)),
        std_min=float(aargs.get("std_min", 0.001)),
        std_max=float(aargs.get("std_max", 0.4)),
        sim_type=str(aargs.get("sim_type", "")),
        sim_dimension=int(aargs.get("sim_dimension", 64)),
        seq_len=int(aargs.get("actor_seq_len", 8)),
        use_layer_norm=bool(aargs.get("use_layer_norm", False)),
        device=torch.device("cpu"),
    )
    actor_sd = {k: v for k, v in ck["actor_state_dict"].items() if k not in ("noise", "noise_scales")}
    actor.load_state_dict(actor_sd, strict=False)
    actor.eval()
    obs_norm = EmpiricalNormalization(shape=n_obs, device=torch.device("cpu"))
    obs_norm.load_state_dict(ck["obs_normalizer_state"])
    obs_norm.eval()
    print(f"loaded actor obs={n_obs} act={n_act}")

    # OffscreenRenderer.update() also loops over every world (up to 32) and re-adds each one's geoms a
    # second time via mjv_addGeoms -- meant for side-by-side multi-env visualization, but with num_envs=1
    # it just redundantly overlays env 0's own geoms a second time at an offset, producing a duplicate
    # skeleton. Drive its own mujoco.Renderer/camera/model directly instead (still mjlab's own renderer,
    # just without that loop) for the single-env case this render actually is.
    import mujoco

    dat = mujoco.MjData(env.sim.mj_model)

    def render_frame():
        dat.qpos[:] = env.sim.wp_data.qpos.numpy()[0]
        dat.qvel[:] = env.sim.wp_data.qvel.numpy()[0]
        mujoco.mj_forward(env.sim.mj_model, dat)
        renderer.renderer.update_scene(dat, camera=renderer._cam)  # noqa: SLF001
        return renderer.render().copy()

    frames = [render_frame()]

    fell = False
    for t in range(HORIZON):
        st = action_term.muscles.state.numpy()[0].copy()
        qpos = env.sim.wp_data.qpos.numpy()[0]
        qvel = env.sim.wp_data.qvel.numpy()[0]
        a_act = np.asarray(env.sim.wp_data.act.numpy()[0], np.float32).ravel()[:2]
        if a_act.size < 2:
            a_act = np.resize(a_act, 2)
        obs_np = make_bolt_sprint_obs(st[:, 0], muscle_obs_norm_fiber(st), a_act, qpos, qvel, qadr, dadr, orders)
        with torch.no_grad():
            nobs = obs_norm(torch.as_tensor(obs_np[None], dtype=torch.float32), update=False)
            act = actor(nobs).clamp(-1, 1).numpy()[0]
        ctrl = bolt_action_to_mj_ctrl(act, muscles, mapf, mjm.nu, orders)
        env.step(torch.as_tensor(ctrl[None], dtype=torch.float32))

        frames.append(render_frame())

        qpos = env.sim.wp_data.qpos.numpy()[0]
        if (t + 1) % 15 == 0:
            print(f"  t={t + 1} x={qpos[0]:.2f} y={qpos[1]:.3f}")
        if qpos[1] < FALL_Y:
            print(f"fell at step {t}")
            fell = True
            break

    renderer.close()
    os.makedirs(os.path.dirname(OUT_MP4), exist_ok=True)
    imageio.mimsave(OUT_MP4, frames, fps=30)
    print(f"wrote {OUT_MP4} ({len(frames)} frames), fell={fell}")


if __name__ == "__main__":
    main()
