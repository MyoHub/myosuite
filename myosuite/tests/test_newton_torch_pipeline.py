# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for experimental PyTorch DLPack accessor and muscle-stage pipeline on Newton."""

from __future__ import annotations

import functools
import importlib
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from myosuite import make_env
from myosuite.core.protocols import PhysicsPath
from myosuite.envs.muscle_stages import LowPassStage, StageOrderWarning
from myosuite.envs.wrappers import FatigueWrapper
from myosuite.terms.base_action import MotorNoiseCfg
from myosuite.terms.base_obs import joint_pos_obs, joint_vel_obs, muscle_act_obs
from myosuite.terms.base_reward import pose_reward

pytestmark = pytest.mark.tier2


def _require_newton_and_torch():
    if os.environ.get("MYOSUITE_REQUIRE_NEWTON") == "1":
        importlib.import_module("newton")
        wp = importlib.import_module("warp")
        torch = importlib.import_module("torch")
    else:
        pytest.importorskip("newton")
        wp = pytest.importorskip("warp")
        torch = pytest.importorskip("torch")
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    return torch, device


def test_torch_newton_accessor_matches_numpy_accessor_and_terms() -> None:
    torch, device = _require_newton_and_torch()
    from myosuite.integrations.newton import NewtonBatchSim

    sim = NewtonBatchSim("myoHandPoseRandom-v0", num_envs=3, device=device, seed=7)
    sim.reset(seed=7)
    rng = np.random.default_rng(7)
    actions = rng.uniform(-1.0, 1.0, size=(3, sim.nu)).astype(np.float32)
    sim.step(actions)

    np_acc = sim.accessor
    t_acc = sim.torch_accessor
    assert t_acc.physics_path == PhysicsPath.MJLAB
    assert t_acc.array_module() is torch

    np.testing.assert_allclose(
        t_acc.joint_pos().cpu().numpy(), np_acc.joint_pos(), atol=1e-6
    )
    np.testing.assert_allclose(
        t_acc.joint_vel().cpu().numpy(), np_acc.joint_vel(), atol=1e-6
    )
    np.testing.assert_allclose(
        t_acc.muscle_act().cpu().numpy(), np_acc.muscle_act(), atol=1e-6
    )
    np.testing.assert_allclose(
        t_acc.muscle_length().cpu().numpy(), np_acc.muscle_length(), atol=1e-6
    )
    np.testing.assert_allclose(
        t_acc.muscle_force().cpu().numpy(), np_acc.muscle_force(), atol=1e-5
    )

    first_site = sim.reference_model.site(0).name
    sid = t_acc.site_id(first_site)
    np.testing.assert_allclose(
        t_acc.site_xpos(sid).cpu().numpy(), np_acc.site_xpos(sid), atol=1e-6
    )
    np.testing.assert_allclose(
        t_acc.site_xpos([sid]).cpu().numpy(), np_acc.site_xpos([sid]), atol=1e-6
    )

    # Backend-agnostic term functions produce identical values on both accessors.
    np.testing.assert_allclose(
        joint_pos_obs(t_acc).cpu().numpy(), joint_pos_obs(np_acc), atol=1e-6
    )
    np.testing.assert_allclose(
        joint_vel_obs(t_acc).cpu().numpy(), joint_vel_obs(np_acc), atol=1e-6
    )
    np.testing.assert_allclose(
        muscle_act_obs(t_acc).cpu().numpy(), muscle_act_obs(np_acc), atol=1e-6
    )
    target = torch.zeros((3, sim.nq), dtype=torch.float32, device=t_acc.device)
    r_torch = pose_reward(t_acc, {"target_angles": target})["dense"].cpu().numpy()
    r_np = pose_reward(
        np_acc, {"target_angles": np.zeros((3, sim.nq), dtype=np.float32)}
    )["dense"]
    np.testing.assert_allclose(r_torch, r_np, atol=1e-6)


def test_newton_action_pipeline_fatigue_and_reafferentation_match_cpu() -> None:
    torch, device = _require_newton_and_torch()
    from myosuite.integrations.newton import (
        NewtonActionCfg,
        NewtonActionPipeline,
        NewtonBatchSim,
    )

    # 1. 3CC-r fatigue parity against CPU FatigueWrapper over 15 control steps + partial reset.
    cpu_base = make_env("myoElbowPose1D6MRandom-v0")
    reset_vec = np.linspace(0.05, 0.35, cpu_base.unwrapped.model.nu)
    cpu_fati = FatigueWrapper(cpu_base, fatigue_reset_vec=reset_vec)
    cpu_fati.reset(seed=11)

    cfg = NewtonActionCfg(
        normalize_act=True,
        muscle_fatigue=True,
        fatigue_reset_vec=tuple(float(v) for v in reset_vec),
    )
    sim = NewtonBatchSim(
        "myoElbowPose1D6MRandom-v0",
        num_envs=2,
        device=device,
        action_cfg=cfg,
        seed=11,
    )
    sim.reset(
        seed=11, qpos=cpu_fati.unwrapped.data.qpos, qvel=cpu_fati.unwrapped.data.qvel
    )
    sim.action_pipeline.reset()

    rng = np.random.default_rng(19)
    for _ in range(12):
        act = rng.uniform(-1.0, 1.0, size=(sim.nu,)).astype(np.float32)
        cpu_fati.step(act)
        act_t = torch.as_tensor(
            np.broadcast_to(act, (2, sim.nu)).copy(), device=sim.torch_accessor.device
        )
        sim.step_torch(act_t)

    t_qpos = sim.torch_accessor.joint_pos().cpu().numpy()
    for w in range(2):
        np.testing.assert_allclose(
            t_qpos[w], cpu_fati.unwrapped.data.qpos, atol=1e-4, rtol=1e-4
        )
    cpu_fati.close()

    # Partial reset resets only env 0's fatigue compartment.
    mf_before = sim.action_pipeline.fatigue_state.MF.clone()
    sim.reset(env_ids=[0])
    mf_after = sim.action_pipeline.fatigue_state.MF
    np.testing.assert_allclose(mf_after[0].cpu().numpy(), reset_vec, atol=1e-6)
    np.testing.assert_allclose(
        mf_after[1].cpu().numpy(), mf_before[1].cpu().numpy(), atol=1e-6
    )

    # 2. Reafferentation parity (EIP -> EPL reroute, EIP silenced) + from_env_id.
    reaf_cfg = NewtonActionCfg.from_env_id("myoReafHandPoseRandom-v0")
    assert reaf_cfg.reroute == ("EIP_r", "EPL_r")
    cpu_hand = make_env("myoReafHandPoseRandom-v0")
    cpu_hand.reset(seed=5)
    hand_model = cpu_hand.unwrapped.model
    reaf_pipe = NewtonActionPipeline(
        hand_model,
        reaf_cfg,
        num_envs=1,
        ctrl_dt=float(hand_model.opt.timestep * 10),
        device="cpu",
    )
    act_hand = rng.uniform(-1.0, 1.0, size=(hand_model.nu,)).astype(np.float32)
    cpu_hand.step(act_hand)
    proc_hand = reaf_pipe.process(torch.from_numpy(act_hand)).numpy()[0]
    np.testing.assert_allclose(proc_hand, cpu_hand.unwrapped.data.ctrl, atol=1e-6)
    assert proc_hand[hand_model.actuator(reaf_cfg.reroute[0]).id] == 0.0
    cpu_hand.close()


def test_newton_stage_ordering_motor_noise_and_site_mocap_updates() -> None:
    torch, device = _require_newton_and_torch()
    from myosuite.integrations.newton import (
        NewtonActionCfg,
        NewtonActionPipeline,
        NewtonBatchSim,
    )

    sim = NewtonBatchSim("myoArmReachRandom-v0", num_envs=2, device=device, seed=3)
    sim.reset(seed=3)

    # Stage order clash emits StageOrderWarning, and motor noise is seeded deterministically.
    with pytest.warns(StageOrderWarning):
        NewtonActionPipeline(
            sim.reference_model,
            NewtonActionCfg(
                motor_noise=MotorNoiseCfg.van_beers_2004(),
                excitation_stages=(
                    functools.partial(LowPassStage, 0.5, "lp_clash", 20),
                ),
            ),
            num_envs=2,
            ctrl_dt=sim.ctrl_dt,
            device="cpu",
            seed=42,
        )
    pipe_early = NewtonActionPipeline(
        sim.reference_model,
        NewtonActionCfg(
            motor_noise=MotorNoiseCfg.van_beers_2004(),
            excitation_stages=(functools.partial(LowPassStage, 0.5, "lp_early", 15),),
        ),
        num_envs=2,
        ctrl_dt=sim.ctrl_dt,
        device="cpu",
        seed=42,
    )
    pipe_late = NewtonActionPipeline(
        sim.reference_model,
        NewtonActionCfg(
            motor_noise=MotorNoiseCfg.van_beers_2004(),
            excitation_stages=(functools.partial(LowPassStage, 0.5, "lp_late", 25),),
        ),
        num_envs=2,
        ctrl_dt=sim.ctrl_dt,
        device="cpu",
        seed=42,
    )
    assert pipe_early.stage_names == ("lp_early", "noise")
    assert pipe_late.stage_names == ("noise", "lp_late")
    cmd0 = torch.zeros((2, sim.nu), dtype=torch.float32)
    cmd1 = torch.ones((2, sim.nu), dtype=torch.float32)
    _ = pipe_late.process(cmd0)
    out1 = pipe_late.process(cmd1)
    pipe_late.seed(42)
    pipe_late.reset()
    _ = pipe_late.process(cmd0)
    out2 = pipe_late.process(cmd1)
    assert torch.allclose(out1, out2)
    _ = pipe_early.process(cmd0)
    assert not torch.allclose(out1, pipe_early.process(cmd1))

    # Per-world target site position update reflects immediately in site_xpos.
    tgt_site = next(
        sim.reference_model.site(i).name
        for i in range(sim.reference_model.nsite)
        if "target" in sim.reference_model.site(i).name
    )
    target_pos = np.array([[0.15, -0.25, 1.05], [-0.10, -0.30, 0.95]], dtype=np.float32)
    sim.set_site_pos(tgt_site, target_pos)
    sid = sim.accessor.site_id(tgt_site)
    if sim.solver.mjw_model.site_pos.shape[0] == 2:
        np.testing.assert_allclose(sim.accessor.site_xpos(sid), target_pos, atol=1e-5)

    # Mocap body pose update works on MJCF models with mocap bodies.
    mocap_xml = """<mujoco>
      <worldbody>
        <body name="target" mocap="true" pos="0 0 0.5">
          <site name="t_site" pos="0 0 0"/>
        </body>
        <body name="link" pos="0 0 0.2">
          <joint name="h" type="hinge" axis="0 1 0"/>
          <geom type="sphere" size="0.05" mass="0.5"/>
        </body>
      </worldbody>
    </mujoco>"""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "mocap.xml"
        p.write_text(mocap_xml, encoding="utf-8")
        msim = NewtonBatchSim(p, num_envs=2, device=device)
        msim.reset()
        assert msim.solver.mj_model.nmocap == 1
        new_mpos = np.array([[0.3, -0.1, 0.7], [-0.2, 0.4, 0.9]], dtype=np.float32)
        msim.set_mocap_pose(0, pos=new_mpos)
        tsid = msim.accessor.site_id("t_site")
        np.testing.assert_allclose(msim.accessor.site_xpos(tsid), new_mpos, atol=1e-5)
