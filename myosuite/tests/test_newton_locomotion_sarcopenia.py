# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for experimental Newton leg locomotion, body/contact accessors, and sarcopenia."""

from __future__ import annotations

import importlib
import os

import numpy as np
import pytest

from myosuite import make_env

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


def test_body_kinematics_and_masked_contact_counts() -> None:
    torch, device = _require_newton_and_torch()
    from myosuite.integrations.newton import NewtonBatchSim

    cpu_env = make_env("myoLegWalk-v0")
    sim = NewtonBatchSim("myoLegWalk-v0", num_envs=2, device=device, seed=5)
    try:
        cpu_env.reset(seed=5)
        u = cpu_env.unwrapped
        sim.reset(qpos=u.data.qpos.copy(), qvel=u.data.qvel.copy())

        acc = sim.accessor
        t_acc = sim.torch_accessor
        mass_mask = u.model.body_mass > 0.0

        np.testing.assert_allclose(acc.body_xpos()[0], u.data.xpos, atol=1e-5)
        np.testing.assert_allclose(acc.body_xquat()[0], u.data.xquat, atol=1e-5)
        np.testing.assert_allclose(
            acc.body_xipos()[0, mass_mask], u.data.xipos[mass_mask], atol=1e-5
        )
        np.testing.assert_allclose(acc.body_cvel()[0], u.data.cvel, atol=1e-4)
        np.testing.assert_allclose(acc.subtree_com()[0], u.data.subtree_com, atol=1e-5)

        np.testing.assert_allclose(
            t_acc.body_xpos()[0].detach().cpu().numpy(), u.data.xpos, atol=1e-5
        )
        np.testing.assert_allclose(
            t_acc.body_cvel()[0].detach().cpu().numpy(), u.data.cvel, atol=1e-4
        )

        # Poison stale contact rows >= nacon to verify both accessors mask by nacon.
        nacon = int(sim.solver.mjw_data.nacon.numpy()[0])
        worldid_np = sim.solver.mjw_data.contact.worldid.numpy().copy()
        if worldid_np.size > nacon + 2:
            worldid_np[nacon] = 0
            worldid_np[nacon + 1] = 1
            sim.solver.mjw_data.contact.worldid.assign(worldid_np)
        counts_np = acc.contact_counts()
        counts_t = t_acc.contact_counts().detach().cpu().numpy()
        np.testing.assert_array_equal(counts_np, counts_t)
        assert int(counts_np.sum()) == nacon

        # Step until foot-ground contacts engage and check counts stay non-negative.
        zero_act = torch.zeros((2, sim.nu), dtype=torch.float32, device=t_acc.device)
        for _ in range(8):
            sim.step_torch(zero_act)
        assert np.all(acc.contact_counts() >= 0)
        np.testing.assert_array_equal(
            acc.contact_counts(), t_acc.contact_counts().detach().cpu().numpy()
        )
    finally:
        cpu_env.close()


def test_sarcopenia_halves_passive_and_active_forces_and_matches_cpu() -> None:
    torch, device = _require_newton_and_torch()
    from myosuite.integrations.newton import (
        NewtonActionCfg,
        NewtonBatchSim,
        NewtonTaskEnv,
    )

    cfg = NewtonActionCfg.from_env_id("myoSarcElbowPose1D6MRandom-v0")
    assert cfg.sarcopenia_force_scale == pytest.approx(0.5)

    healthy = NewtonBatchSim("myoElbowPose1D6MRandom-v0", num_envs=1, device=device)
    sarc = NewtonBatchSim("myoSarcElbowPose1D6MRandom-v0", num_envs=1, device=device)

    with pytest.raises(ValueError, match="already applied"):
        sarc.apply_sarcopenia(0.5)

    q_stretch = np.array([[2.1]], dtype=np.float32)
    healthy.reset(qpos=q_stretch)
    sarc.reset(qpos=q_stretch)
    f_pass_h = healthy.accessor.muscle_force()[0]
    f_pass_s = sarc.accessor.muscle_force()[0]
    assert np.max(np.abs(f_pass_h)) > 1.0
    np.testing.assert_allclose(f_pass_s, 0.5 * f_pass_h, rtol=1e-4, atol=1e-3)

    act_one = np.ones((1, healthy.nu), dtype=np.float32)
    healthy.solver.mjw_data.act.assign(act_one)
    sarc.solver.mjw_data.act.assign(act_one)
    healthy._refresh_derived()
    sarc._refresh_derived()
    f_act_h = healthy.accessor.muscle_force()[0] - f_pass_h
    f_act_s = sarc.accessor.muscle_force()[0] - f_pass_s
    assert np.max(np.abs(f_act_h)) > 10.0
    np.testing.assert_allclose(f_act_s, 0.5 * f_act_h, rtol=1e-4, atol=1e-2)

    cpu_env = make_env("myoSarcElbowPose1D6MRandom-v0")
    n_env = NewtonTaskEnv(
        "myoSarcElbowPose1D6MRandom-v0",
        num_envs=1,
        device=device,
        seed=11,
        auto_reset=False,
    )
    try:
        obs_c, _ = cpu_env.reset(seed=11)
        obs_n, _ = n_env.reset_torch(seed=11)
        np.testing.assert_allclose(obs_n[0].cpu().numpy(), obs_c, atol=1e-4)
        rng = np.random.default_rng(19)
        for _ in range(8):
            a = rng.uniform(-1.0, 1.0, size=(n_env.num_actions,)).astype(np.float32)
            obs_c, r_c, _, _, _ = cpu_env.step(a)
            obs_n, r_n, _, _, _ = n_env.step_torch(
                torch.as_tensor(a[None, :], device=n_env.torch_device)
            )
            np.testing.assert_allclose(obs_n[0].cpu().numpy(), obs_c, atol=5e-4)
            np.testing.assert_allclose(
                float(r_n[0].cpu().item()), float(r_c), atol=1e-3
            )
    finally:
        cpu_env.close()


@pytest.mark.parametrize("env_id", ["myoLegWalk-v0", "myoSarcLegWalk-v0"])
def test_leg_walk_and_sarcopenic_walk_match_cpu(env_id: str) -> None:
    torch, device = _require_newton_and_torch()
    from myosuite.integrations.newton import (
        NewtonActionCfg,
        NewtonTaskEnv,
        rollout_closed_loop_parity,
    )

    cfg = NewtonActionCfg.from_env_id(env_id)
    assert cfg.action_range == (0.0, 1.0)
    assert not cfg.muscle_sigmoid
    expected_sarc = 0.5 if "Sarc" in env_id else None
    assert cfg.sarcopenia_force_scale == expected_sarc

    cpu_env = make_env(env_id)
    n_env = NewtonTaskEnv(env_id, num_envs=2, device=device, seed=23, auto_reset=False)
    try:
        obs_c, _ = cpu_env.reset(seed=23)
        obs_n, _ = n_env.reset_torch(seed=23)
        np.testing.assert_allclose(
            obs_n[0].detach().cpu().numpy(), obs_c, atol=1e-3, rtol=1e-3
        )

        rng = np.random.default_rng(31)
        for _ in range(8):
            act = rng.uniform(0.0, 1.0, size=(n_env.num_actions,)).astype(np.float32)
            obs_c, r_c, term_c, trunc_c, info_c = cpu_env.step(act)
            act_batch = np.stack([act, 0.5 * act], axis=0)
            obs_n, r_n, term_n, trunc_n, info_n = n_env.step_torch(
                torch.as_tensor(
                    act_batch, dtype=torch.float32, device=n_env.torch_device
                )
            )
            obs_n0 = obs_n[0].detach().cpu().numpy()
            # Released newton==1.8.0 zeroes 1 mm geom_margin (fixed upstream in newton#5378),
            # shifting stance-foot muscle velocity/force slightly at contact while keeping poses tight.
            np.testing.assert_allclose(obs_n0[:163], obs_c[:163], atol=2e-2, rtol=2e-2)
            np.testing.assert_allclose(
                obs_n0[163:243], obs_c[163:243], atol=5e-2, rtol=5e-2
            )
            np.testing.assert_allclose(
                obs_n0[243:323], obs_c[243:323], atol=0.15, rtol=0.15
            )
            np.testing.assert_allclose(obs_n0[323:], obs_c[323:], atol=1e-2, rtol=1e-2)
            np.testing.assert_allclose(
                float(r_n[0].detach().cpu().item()), float(r_c), atol=5e-2, rtol=1e-2
            )
            assert bool(term_n[0].detach().cpu().item()) == bool(term_c)
            assert bool(trunc_n[0].detach().cpu().item()) == bool(trunc_c)
            assert bool(info_n["solved"][0].detach().cpu().item()) == bool(
                info_c.get("solved", False)
            )
    finally:
        cpu_env.close()

    if env_id == "myoLegWalk-v0":
        report = rollout_closed_loop_parity(
            env_id,
            lambda obs: np.clip(0.25 + 0.01 * obs[:80], 0.0, 1.0).astype(np.float32),
            steps=6,
            seed=7,
            device=device,
        )
        assert report.termination_matches
        assert report.max_abs_obs_diff < 0.15


def test_leg_walk_rsl_rl_vecenv_contract() -> None:
    torch, device = _require_newton_and_torch()
    pytest.importorskip("tensordict")
    from myosuite.integrations.newton import NewtonRslRlVecEnv

    vec = NewtonRslRlVecEnv("myoLegWalk-v0", num_envs=3, device=device, seed=0)
    assert vec.num_obs == 403
    assert vec.num_actions == 80
    td0 = vec.get_observations()
    assert td0["policy"].shape == (3, 403)
    actions = torch.full((3, 80), 0.2, dtype=torch.float32, device=vec.device)
    td1, rwd, dones, extras = vec.step(actions)
    assert td1["policy"].shape == (3, 403)
    assert rwd.shape == (3,)
    assert dones.shape == (3,)
    assert "/solved" in extras["log"]
