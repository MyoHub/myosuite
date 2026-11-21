# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the portable Functional Electrical Stimulation (FES) ExcitationStage."""

from __future__ import annotations

import pickle

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
