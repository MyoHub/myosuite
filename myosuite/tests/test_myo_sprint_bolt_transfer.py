# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Phase 6 gate (myo_bolt_vv porting plan): the Bolt-trained TD3 checkpoint already verified in
myo_bolt_vv/scripts/msk_envs_repro/cpu_td3_side_by_side.py (closed-loop, 336-D Bolt-convention obs,
xT=18.08m over 90 steps vs Bolt's own 18.13m, no fall) must transfer through myoSprint-v0 on mjlab to
within the same band.

Run:
    KMP_DUPLICATE_LIB_OK=TRUE uv run python myosuite/tests/test_myo_sprint_bolt_transfer.py
"""
from __future__ import annotations

import os
import sys

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
XT_BOLT = 18.13
XT_WARP_REF = 18.08  # cpu_td3_side_by_side.py's own verified number for this checkpoint
XT_TOL = 2.0  # metres -- generous band; the point is "did it transfer", not exact reproduction of a
# different physics/obs pipeline (mjlab's decimation/actuator wiring differs in ways not yet proven
# bit-identical to WarpSprintEnv's own hand-rolled loop, unlike Phase 5's myo_sim.warp-vs-mjlab check)


def main():
    import yaml
    from bolt_sprint_parity import bolt_action_to_mj_ctrl, load_orders, make_bolt_sprint_obs, muscle_obs_norm_fiber
    from msk_envs.train.nets.deterministic_policy import DeterministicPolicy
    from msk_envs.train.nets.normalizers import EmpiricalNormalization

    import myo_sim
    from myo_sim.sprinter_assets import load_map, load_muscles
    from mjlab.entity import EntityCfg
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

    from myosuite.envs.myo.backends.mjlab.tasks.myosprint.env_cfgs import make_sprint_env_cfg

    orders = load_orders()
    muscles = load_muscles()
    mapf = load_map()
    qadr, dadr = mapf["qposadr"], mapf["dofadr"]
    mjm = myo_sim.load_spec("myosprinter").compile()

    # Bolt's own "running start" pose (nonzero forward/hip/knee velocity) -- the TD3 checkpoint was trained
    # to begin from this, not a neutral zero-velocity stand; see make_sprint_env_cfg()'s init_state docstring.
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
            continue  # root orientation stays neutral; their velocities are 0 in this pose anyway
        else:
            joint_pos[f"^{name}$"] = q
            joint_vel[f"^{name}$"] = v
    run_init_state = EntityCfg.InitialStateCfg(
        pos=(0.0, 1.05, 0.0), lin_vel=lin_vel, joint_pos=joint_pos, joint_vel=joint_vel
    )

    cfg = make_sprint_env_cfg(num_envs=1, init_state=run_init_state, default_activation=0.5)
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset()

    # Reach into the action term for its live MuscleSystem state (state[:,9] = geometric norm fiber length
    # -- the slot-9 fix from this session's earlier TD3-transfer work, mandatory: slot 1 stays 0 for rigid
    # tendons and drove the original closed-loop failure this whole Phase 6 gate is re-verifying didn't
    # regress).
    action_term = env.action_manager.get_term("bolt_muscles")

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

    xs, ys = [], []
    qpos_traj = [env.sim.wp_data.qpos.numpy()[0].copy()]
    fell = False
    for t in range(HORIZON):
        st = action_term.muscles.state.numpy()[0].copy()
        qpos = env.sim.wp_data.qpos.numpy()[0]
        qvel = env.sim.wp_data.qvel.numpy()[0]
        a_act = np.asarray(env.sim.wp_data.act.numpy()[0], np.float32).ravel()[:2]
        if a_act.size < 2:
            a_act = np.resize(a_act, 2)
        obs_np = make_bolt_sprint_obs(st[:, 0], muscle_obs_norm_fiber(st), a_act, qpos, qvel, qadr, dadr, orders)
        if t == 0:
            assert obs_np.shape[0] == n_obs, (obs_np.shape[0], n_obs)
        with torch.no_grad():
            nobs = obs_norm(torch.as_tensor(obs_np[None], dtype=torch.float32), update=False)
            act = actor(nobs).clamp(-1, 1).numpy()[0]
        ctrl = bolt_action_to_mj_ctrl(act, muscles, mapf, mjm.nu, orders)
        env.step(torch.as_tensor(ctrl[None], dtype=torch.float32))
        qpos = env.sim.wp_data.qpos.numpy()[0]
        xs.append(float(qpos[0]))
        ys.append(float(qpos[1]))
        qpos_traj.append(qpos.copy())
        if (t + 1) % 15 == 0 or qpos[1] < FALL_Y:
            print(f"  t={t + 1} x={qpos[0]:.2f} y={qpos[1]:.3f}")
        if qpos[1] < FALL_Y:
            print(f"fell at step {t}")
            fell = True
            break

    xt = xs[-1]
    y_min = min(ys)
    print(f"myoSprint-v0 (mjlab) closed-loop: {len(xs)} steps xT={xt:.2f} y_min={y_min:.3f} fell={fell}")
    print(f"reference: raw WarpSprintEnv xT={XT_WARP_REF}, Bolt xT={XT_BOLT}")

    out_dir = f"{MSK_REPRO}/artifacts/renders"
    os.makedirs(out_dir, exist_ok=True)
    np.savez(f"{out_dir}/trajs_mjlab_myosprint.npz", td3_mjlab_q=np.stack(qpos_traj))
    print(f"wrote {out_dir}/trajs_mjlab_myosprint.npz ({len(qpos_traj)} frames)")

    assert not fell, "policy fell through mjlab -- did not transfer"
    assert abs(xt - XT_BOLT) < XT_TOL, f"xT={xt:.2f} outside +-{XT_TOL}m of Bolt's {XT_BOLT}"
    print("PASS: Bolt-trained TD3 policy transfers through myoSprint-v0 on mjlab")


if __name__ == "__main__":
    main()
