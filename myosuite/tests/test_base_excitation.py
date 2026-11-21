# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for non-inertial base-excitation profiles, wrenches, and wrapper."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from myosuite import make_env
from myosuite.biomechanics.external_loads import inertial_frame_qfrc
from myosuite.core.model_builder import build_from_recipe
from myosuite.envs.wrappers import BaseExcitationWrapper, PerturbationWrapper
from myosuite.physics.base_excitation import (
    CompositeBaseProfile,
    HarmonicBaseProfile,
    PulseBaseProfile,
    RecordedBaseProfile,
    StochasticBaseProfile,
    inertial_body_wrenches,
)

pytestmark = pytest.mark.tier1


def test_upward_base_acceleration_matches_gravity_generalized_load() -> None:
    model, _ = build_from_recipe("arm_reach")
    data = mujoco.MjData(model)
    qpos = model.qpos0.copy()
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    # Upward support-frame acceleration (-gravity) produces downward fictitious load (+m*g),
    # whose generalized projection must equal -qfrc_bias at zero velocity (minus passive joint forces).
    qfrc_inertial = inertial_frame_qfrc(model, qpos, -model.opt.gravity)
    qfrc_gravity = -(data.qfrc_bias - data.qfrc_passive)
    np.testing.assert_allclose(qfrc_inertial, qfrc_gravity, rtol=1e-10, atol=1e-10)


def test_centrifugal_coriolis_and_torch_parity() -> None:
    torch = pytest.importorskip("torch")

    model, _ = build_from_recipe("elbow_standard")
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.75
    mujoco.mj_forward(model, data)

    dyn_mask = model.body_weldid != 0
    lin_acc = np.array([1.2, -0.4, 0.8])
    ang_vel = np.array([0.0, 0.0, 2.0])
    ang_acc = np.array([0.3, -0.2, 0.5])
    pivot = np.array([0.1, -0.05, 0.2])

    w_np = inertial_body_wrenches(
        np,
        model.body_mass,
        model.body_inertia,
        data.xipos,
        data.ximat,
        data.cvel,
        data.subtree_com[model.body_rootid],
        lin_acc,
        ang_vel=ang_vel,
        ang_acc=ang_acc,
        pivot=pivot,
        dynamic_mask=dyn_mask,
    )
    assert np.all(w_np[~dyn_mask] == 0.0)
    assert np.any(np.abs(w_np[dyn_mask]) > 0.0)

    # Batch of 4 identical worlds in PyTorch must match NumPy to float32 precision.
    b = 4
    w_th = inertial_body_wrenches(
        torch,
        torch.as_tensor(model.body_mass, dtype=torch.float64),
        torch.as_tensor(model.body_inertia, dtype=torch.float64),
        torch.as_tensor(np.broadcast_to(data.xipos, (b, *data.xipos.shape)).copy()),
        torch.as_tensor(np.broadcast_to(data.ximat, (b, *data.ximat.shape)).copy()),
        torch.as_tensor(np.broadcast_to(data.cvel, (b, *data.cvel.shape)).copy()),
        torch.as_tensor(
            np.broadcast_to(
                data.subtree_com[model.body_rootid], (b, model.nbody, 3)
            ).copy()
        ),
        torch.as_tensor(np.broadcast_to(lin_acc, (b, 3)).copy()),
        ang_vel=torch.as_tensor(np.broadcast_to(ang_vel, (b, 3)).copy()),
        ang_acc=torch.as_tensor(np.broadcast_to(ang_acc, (b, 3)).copy()),
        pivot=torch.as_tensor(np.broadcast_to(pivot, (b, 3)).copy()),
        dynamic_mask=dyn_mask,
    )
    assert w_th.shape == (b, model.nbody, 6)
    np.testing.assert_allclose(w_th[0].numpy(), w_np, rtol=1e-11, atol=1e-11)

    # Analytic point-mass check: m=2 kg at r=(1, 0, 0), v=(0.5, 0, 0), omega=(0, 0, 3)
    # Centrifugal = +m * omega^2 * R x_hat = (18, 0, 0)
    # Coriolis = -2 m (omega x v) = -4 * (0, 1.5, 0) = (0, -6, 0)
    w_pt = inertial_body_wrenches(
        np,
        np.array([2.0]),
        np.zeros((1, 3)),
        np.array([[1.0, 0.0, 0.0]]),
        np.eye(3).reshape(1, 9),
        np.array([[0.0, 0.0, 0.0, 0.5, 0.0, 0.0]]),
        np.zeros((1, 3)),
        np.zeros(3),
        ang_vel=np.array([0.0, 0.0, 3.0]),
    )
    np.testing.assert_allclose(w_pt[0, :3], [18.0, -6.0, 0.0], atol=1e-12)


def test_profiles_and_wrapper_seeded_reproducibility() -> None:
    times = np.linspace(0.0, 1.0, 11)
    rec = RecordedBaseProfile(
        times=times,
        lin_acc=np.column_stack([np.sin(2.0 * np.pi * times), np.zeros((11, 2))]),
        loop=True,
    )
    comp = CompositeBaseProfile(
        profiles=(
            HarmonicBaseProfile(
                lin_amp=(1.5, 0.5, 0.8),
                freq_hz=2.0,
                ang_amp=(0.1, 0.2, 0.0),
                random_phase=True,
            ),
            StochasticBaseProfile(lin_std=(0.4, 0.4, 0.6), beta=1.0, buffer_steps=32),
            PulseBaseProfile(
                start_s=0.02, duration_s=0.08, peak_lin_acc=(3.0, 0.0, 0.0)
            ),
            rec,
        )
    )

    env = BaseExcitationWrapper(make_env("myoElbowPose1D6MRandom-v0"), comp)
    try:
        obs_a, _ = env.reset(seed=42)
        zero_act = np.zeros(env.action_space.shape, dtype=np.float32)
        traj_a = [obs_a]
        for _ in range(8):
            obs, _, _, _, info = env.step(zero_act)
            assert "base_excitation" in info
            assert info["base_excitation"]["lin_acc"].shape == (3,)
            traj_a.append(obs)

        obs_b, _ = env.reset(seed=42)
        traj_b = [obs_b]
        for _ in range(8):
            obs, _, _, _, _ = env.step(zero_act)
            traj_b.append(obs)

        np.testing.assert_array_equal(np.stack(traj_a), np.stack(traj_b))
    finally:
        env.close()


def test_wrapper_composes_with_perturbation_wrapper() -> None:
    base_env = make_env("myoElbowPose1D6MFixed-v0")
    pert_env = PerturbationWrapper(base_env)
    pert_env.add_perturbation(
        {"body": "r_ulna_radius_hand", "force": [0.0, 0.0, 5.0], "start": 0, "end": 5}
    )
    env = BaseExcitationWrapper(
        pert_env,
        PulseBaseProfile(start_s=0.0, duration_s=0.2, peak_lin_acc=(0.0, 0.0, 4.0)),
    )
    try:
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert info["base_excitation"]["lin_acc"][2] >= 0.0
    finally:
        env.close()
