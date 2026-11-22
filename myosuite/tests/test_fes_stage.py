# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the portable Functional Electrical Stimulation (FES) ExcitationStage."""

from __future__ import annotations

import pickle

import mujoco
import numpy as np
import pytest

from myosuite import make_env
from myosuite.core.registry import EnvConfig
from myosuite.envs.muscle_stages import FESStage, fes_stage_factory
from myosuite.envs.wrappers import ExcitationStageWrapper
from myosuite.tests.support.host_sync import HostSyncCounter

pytestmark = pytest.mark.tier1


def test_fes_recruitment_deadzone_frequency_scaling_and_fatigue_recovery() -> None:
    # Command u=0.15 -> PW = 75 us < threshold_us=100 us -> zero recruitment.
    s_lo = FESStage(
        stim_freq_hz=20.0, step_dt=0.1, tau_fatigue_s=5.0, tau_recovery_s=10.0
    )
    s_hi = FESStage(
        stim_freq_hz=50.0, step_dt=0.1, tau_fatigue_s=5.0, tau_recovery_s=10.0
    )

    assert s_lo(np.array([0.15, 0.20]), np)[0] == pytest.approx(0.0, abs=1e-12)
    assert s_lo(np.array([0.15, 0.20]), np)[1] == pytest.approx(0.0, abs=1e-12)

    s_lo.reset()
    s_hi.reset()
    u_tet = np.array([0.95, 0.95])
    out_20hz = s_lo(u_tet, np)
    out_50hz = s_hi(u_tet, np)
    assert np.all(out_50hz > out_20hz)
    assert out_50hz[0] == pytest.approx(s_hi.freq_gain, abs=1e-9)

    # Sustained tetanic stimulation fatigues fitness down toward min_fitness=0.20.
    prev = float(out_50hz[0])
    for _ in range(150):
        cur = float(s_hi(u_tet, np)[0])
        assert cur <= prev + 1e-12
        prev = cur
    fatigued_fit = float(s_hi.fitness[0])
    assert fatigued_fit < 0.32
    assert fatigued_fit >= s_hi.min_fitness

    # Rest (u = 0) recovers fitness back toward 1.0.
    for _ in range(120):
        _ = s_hi(np.zeros(2), np)
    assert float(s_hi.fitness[0]) > fatigued_fit + 0.35


def test_fes_selective_channels_pytorch_parity_partial_reset_and_zero_sync() -> None:
    torch = pytest.importorskip("torch")

    rng = np.random.default_rng(4)
    u_batch = rng.uniform(0.0, 1.0, size=(4, 5)).astype(np.float32)

    stage_np = FESStage(
        stim_freq_hz=35.0,
        step_dt=0.05,
        tau_fatigue_s=4.0,
        stimulated_indices=(0, 3),
        passthrough_unstimulated=False,
    )
    stage_pt = FESStage(
        stim_freq_hz=35.0,
        step_dt=0.05,
        tau_fatigue_s=4.0,
        stimulated_indices=(0, 3),
        passthrough_unstimulated=False,
    )

    u_t = torch.tensor(u_batch)
    for _ in range(5):
        y_np = stage_np(u_batch, np)
        y_pt = stage_pt(u_t, torch)
    np.testing.assert_allclose(y_np, y_pt.numpy(), atol=1e-6)
    # Unstimulated paralyzed channels (1, 2, 4) output exact zero.
    np.testing.assert_allclose(y_np[:, [1, 2, 4]], 0.0, atol=1e-12)

    # Partial reset of env_ids=[1, 3] restores fresh fitness on rows 1 & 3 only.
    stage_np.reset(np.array([1, 3]))
    stage_pt.reset(torch.tensor([1, 3], dtype=torch.long))

    with HostSyncCounter(package_only=True) as syncs:
        y_pt_after = stage_pt(u_t, torch)
    assert syncs.total == 0, syncs.report()

    y_np_after = stage_np(u_batch, np)
    np.testing.assert_allclose(y_np_after, y_pt_after.numpy(), atol=1e-6)
    assert float(stage_np.fitness[1, 0]) > float(stage_np.fitness[0, 0])


def test_fes_excitation_stage_wrapper_pickle_and_seeded_reset() -> None:
    factory = fes_stage_factory(stim_freq_hz=40.0, step_dt=0.02, tau_fatigue_s=6.0)
    env = make_env(
        EnvConfig(
            env_id="myoElbowPose1D6MRandom-v0",
            features=[(ExcitationStageWrapper, {"make_stage": factory})],
        )
    )
    try:
        assert "fes" in env.unwrapped.ctrl_stages
        clone = pickle.loads(pickle.dumps(env))
        try:
            obs1, _ = env.reset(seed=17)
            act = np.full(env.action_space.shape, 0.8, dtype=np.float32)
            for _ in range(5):
                step1 = env.step(act)[0]

            obs2, _ = env.reset(seed=17)
            for _ in range(5):
                step2 = env.step(act)[0]

            obs_c, _ = clone.reset(seed=17)
            for _ in range(5):
                step_c = clone.step(act)[0]

            np.testing.assert_array_equal(obs1, obs2)
            np.testing.assert_array_equal(step1, step2)
            np.testing.assert_array_equal(step1, step_c)
            np.testing.assert_array_equal(obs1, obs_c)
        finally:
            clone.close()
    finally:
        env.close()


def test_fes_analytical_riener_recruitment_frequency_and_fitness_ode() -> None:
    """Validate recruitment, force-frequency, and fitness ODE against closed-form literature equations."""
    stage = FESStage(
        stim_freq_hz=25.0,
        f50_hz=25.0,
        freq_exponent=2.5,
        pulse_width_range_us=(0.0, 500.0),
        threshold_us=100.0,
        saturation_us=450.0,
        recruitment_steepness=2.5,
        step_dt=0.01,
        tau_fatigue_s=12.0,
        tau_recovery_s=30.0,
        min_fitness=0.20,
    )
    # At f == f50_hz (25 Hz), Hill force-frequency gain is exactly 0.5 (Riener & Quintern 1997).
    assert stage.freq_gain == pytest.approx(0.5, abs=1e-12)

    # Midpoint pulse width PW = 275 us <=> u = 275 / 500 = 0.55 gives r_pw = 0.5 -> a_rec = 0.25.
    stage.reset()
    out_mid = float(stage(np.array([0.55]), np)[0])
    assert out_mid == pytest.approx(0.25, abs=1e-12)

    # Constant recruitment a_rec = 0.40: compare discrete fitness trajectory against exact ODE
    # mu(t) = mu_inf + (1 - mu_inf) * exp(-lam * t) (Riener, Quintern & Schmidt 1996).
    stage_ode = FESStage(
        stim_freq_hz=25.0,
        f50_hz=25.0,
        pulse_width_range_us=(0.0, 500.0),
        threshold_us=100.0,
        saturation_us=450.0,
        step_dt=0.01,
        tau_fatigue_s=10.0,
        tau_recovery_s=25.0,
        min_fitness=0.20,
    )
    a_rec = 0.5  # u = 1.0 -> r_pw = 1.0, freq_gain = 0.5 -> a_rec = 0.5
    rate_fat = a_rec / 10.0
    rate_rec = (1.0 - a_rec) / 25.0
    lam = rate_fat + rate_rec
    mu_inf = (rate_fat * 0.20 + rate_rec) / lam

    u_const = np.array([1.0])
    for step in range(1, 401):
        _ = stage_ode(u_const, np)
        if step in (50, 150, 400):
            t = step * 0.01
            expected_mu = mu_inf + (1.0 - mu_inf) * np.exp(-lam * t)
            assert float(stage_ode.fitness[0]) == pytest.approx(expected_mu, abs=2e-3)

    for _ in range(18000):
        _ = stage_ode(u_const, np)
    assert float(stage_ode.fitness[0]) == pytest.approx(mu_inf, abs=1e-5)


def test_fes_neuroprosthetic_elbow_flexion_with_paralyzed_antagonists() -> None:
    """Use case: 2-channel biceps FES restores anti-gravity reach on a paralyzed 6-muscle arm."""
    probe = make_env("myoElbowPose1D6MFixed-v0")
    try:
        model = probe.unwrapped.model
        dt = float(probe.unwrapped.dt)
        bic_ids = tuple(
            i
            for i in range(model.nu)
            if "BIC"
            in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "")
        )
        assert len(bic_ids) >= 2
    finally:
        probe.close()

    # 1. Completely paralyzed arm (no stimulated channels, passthrough=False) cannot reach elevated pose.
    paralyzed_env = make_env(
        EnvConfig(
            env_id="myoElbowPose1D6MFixed-v0",
            features=[
                (
                    ExcitationStageWrapper,
                    {
                        "make_stage": fes_stage_factory(
                            stimulated_indices=(),
                            passthrough_unstimulated=False,
                            step_dt=dt,
                        )
                    },
                )
            ],
        )
    )
    # 2. Neuroprosthetic 2-channel surface FES over biceps brachii only (4 other muscles paralyzed).
    fes_env = make_env(
        EnvConfig(
            env_id="myoElbowPose1D6MFixed-v0",
            features=[
                (
                    ExcitationStageWrapper,
                    {
                        "make_stage": fes_stage_factory(
                            stimulated_indices=bic_ids,
                            passthrough_unstimulated=False,
                            stim_freq_hz=40.0,
                            step_dt=dt,
                            tau_fatigue_s=8.0,
                        )
                    },
                )
            ],
        )
    )
    try:
        for e in (paralyzed_env, fes_env):
            e.reset(seed=0)
            e.unwrapped.data.qpos[0] = 0.10
            e.unwrapped.data.qvel[:] = 0.0
            mujoco.mj_forward(e.unwrapped.model, e.unwrapped.data)

        target_angle = float(fes_env.unwrapped.target_jnt_value[0])
        assert target_angle > 1.0

        for _ in range(80):
            paralyzed_env.step(
                np.ones(paralyzed_env.action_space.shape, dtype=np.float32)
            )
        paralyzed_err = abs(float(paralyzed_env.unwrapped.data.qpos[0]) - target_angle)
        assert paralyzed_err > 0.8

        # Closed-loop PI pulse-width controller on the 2 biceps FES electrodes solves the pose.
        integral = 0.0
        solved_steps = 0
        non_bic = [i for i in range(fes_env.action_space.shape[0]) if i not in bic_ids]
        for _ in range(120):
            q = float(fes_env.unwrapped.data.qpos[0])
            qd = float(fes_env.unwrapped.data.qvel[0])
            err = target_angle - q
            integral = float(np.clip(integral + err * dt, -0.5, 0.5))
            pw_cmd = float(
                np.clip(0.22 + 1.4 * err + 1.2 * integral - 0.08 * qd, 0.0, 1.0)
            )
            action = np.full(fes_env.action_space.shape, 0.9, dtype=np.float32)
            for idx in bic_ids:
                action[idx] = pw_cmd
            _, _, _, _, info = fes_env.step(action)
            np.testing.assert_allclose(
                fes_env.unwrapped.data.ctrl[non_bic], 0.0, atol=1e-12
            )
            if bool(info.get("solved", False)):
                solved_steps += 1

        fes_err = abs(float(fes_env.unwrapped.data.qpos[0]) - target_angle)
        assert fes_err < 0.08
        assert solved_steps > 25
    finally:
        paralyzed_env.close()
        fes_env.close()
