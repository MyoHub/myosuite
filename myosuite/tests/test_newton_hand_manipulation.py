# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for experimental Newton dexterous hand manipulation tasks."""

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


@pytest.mark.parametrize(
    "env_id",
    [
        "myoHandObjHoldFixed-v0",
        "myoHandObjHoldRandom-v0",
        "myoHandPenTwirlFixed-v0",
        "myoHandPenTwirlRandom-v0",
        "myoHandKeyTurnFixed-v0",
        "myoHandKeyTurnRandom-v0",
        "myoFatiHandObjHoldRandom-v0",
    ],
)
def test_hand_manipulation_matches_cpu_step_by_step(env_id: str) -> None:
    torch, device = _require_newton_and_torch()
    from myosuite.integrations.newton import NewtonTaskEnv

    cpu_env = make_env(env_id)
    n_env = NewtonTaskEnv(env_id, num_envs=2, device=device, seed=19, auto_reset=False)
    try:
        obs_c, _ = cpu_env.reset(seed=19)
        obs_n, _ = n_env.reset_torch(seed=19)
        np.testing.assert_allclose(
            obs_n[0].detach().cpu().numpy(), obs_c, atol=1e-4, rtol=1e-4
        )
        # Re-seeding reproduces per-episode object size / goal / orientation / key pose.
        obs_rep, _ = n_env.reset_torch(seed=19)
        np.testing.assert_allclose(
            obs_rep.detach().cpu().numpy(), obs_n.detach().cpu().numpy(), atol=1e-6
        )

        rng = np.random.default_rng(23)
        for _ in range(5):
            act = rng.uniform(-1.0, 1.0, size=(n_env.num_actions,)).astype(np.float32)
            obs_c, r_c, term_c, trunc_c, info_c = cpu_env.step(act)
            act_batch = np.stack([act, -0.3 * act], axis=0)
            obs_n, r_n, term_n, trunc_n, info_n = n_env.step_torch(
                torch.as_tensor(
                    act_batch, dtype=torch.float32, device=n_env.torch_device
                )
            )
            obs_n0 = obs_n[0].detach().cpu().numpy()
            na = n_env.sim.na
            # Muscle activations and task site positions stay tight even when newton==1.8.0
            # zeroes 1 mm geom_margin (~3e-2 rad/s on contacted key/finger velocities).
            np.testing.assert_allclose(obs_n0[-na:], obs_c[-na:], atol=2e-3, rtol=2e-3)
            if "PenTwirl" in env_id:
                np.testing.assert_allclose(obs_n0, obs_c, atol=2e-2, rtol=2e-2)
            elif "KeyTurn" in env_id:
                np.testing.assert_allclose(
                    obs_n0[48 : 48 + 6], obs_c[48 : 48 + 6], atol=5e-3, rtol=5e-3
                )
                np.testing.assert_allclose(obs_n0, obs_c, atol=6e-2, rtol=6e-2)
            else:
                np.testing.assert_allclose(
                    obs_n0[46:], obs_c[46:], atol=2e-2, rtol=2e-2
                )
                np.testing.assert_allclose(obs_n0, obs_c, atol=6e-2, rtol=6e-2)
            np.testing.assert_allclose(
                float(r_n[0].detach().cpu().item()), float(r_c), atol=0.25, rtol=5e-2
            )
            assert bool(term_n[0].detach().cpu().item()) == bool(term_c)
            assert bool(trunc_n[0].detach().cpu().item()) == bool(trunc_c)
            assert bool(info_n["solved"][0].detach().cpu().item()) == bool(
                info_c.get("solved", False)
            )
    finally:
        cpu_env.close()


def test_hand_manipulation_vecenv_and_closed_loop_parity() -> None:
    torch, device = _require_newton_and_torch()
    pytest.importorskip("tensordict")
    from myosuite.integrations.newton import (
        NewtonRslRlVecEnv,
        rollout_closed_loop_parity,
    )

    vec = NewtonRslRlVecEnv(
        "myoHandPenTwirlRandom-v0", num_envs=4, device=device, seed=3
    )
    td0 = vec.get_observations()
    assert td0["policy"].shape == (4, vec.num_obs)
    actions = torch.zeros((4, vec.num_actions), dtype=torch.float32, device=vec.device)
    td1, rwd, dones, extras = vec.step(actions)
    assert td1["policy"].shape == (4, vec.num_obs)
    assert rwd.shape == (4,)
    assert dones.shape == (4,)
    assert "/solved" in extras["log"]

    report = rollout_closed_loop_parity(
        "myoHandKeyTurnRandom-v0",
        lambda obs: np.tanh(0.05 * obs[:39]).astype(np.float32),
        steps=6,
        seed=5,
        device=device,
    )
    assert report.termination_matches
    assert report.max_abs_obs_diff < 2e-2


@pytest.mark.parametrize(
    "env_id", ["myoHandObjHoldRandom-v0", "myoHandKeyTurnRandom-v0"]
)
def test_hand_manipulation_exact_parity_with_preserved_geom_margin(env_id: str) -> None:
    torch, device = _require_newton_and_torch()
    from myosuite.integrations.newton import NewtonTaskEnv

    cpu_env = make_env(env_id)
    n_env = NewtonTaskEnv(env_id, num_envs=2, device=device, seed=19, auto_reset=False)
    try:
        gm = n_env.sim.solver.mjw_model.geom_margin.numpy().copy()
        for sg, cg in enumerate(n_env.sim.mapping.src_to_conv_geom):
            if cg >= 0:
                gm[:, cg] = n_env.sim.reference_model.geom_margin[sg]
        n_env.sim.solver.mjw_model.geom_margin.assign(gm)

        obs_c, _ = cpu_env.reset(seed=19)
        obs_n, _ = n_env.reset_torch(seed=19)
        np.testing.assert_allclose(
            obs_n[0].detach().cpu().numpy(), obs_c, atol=1e-5, rtol=1e-5
        )
        rng = np.random.default_rng(23)
        for _ in range(5):
            act = rng.uniform(-1.0, 1.0, size=(n_env.num_actions,)).astype(np.float32)
            obs_c, r_c, _, _, _ = cpu_env.step(act)
            obs_n, r_n, _, _, _ = n_env.step_torch(
                torch.as_tensor(
                    np.stack([act, -0.3 * act], axis=0),
                    dtype=torch.float32,
                    device=n_env.torch_device,
                )
            )
            np.testing.assert_allclose(
                obs_n[0].detach().cpu().numpy(), obs_c, atol=5e-4, rtol=5e-4
            )
            np.testing.assert_allclose(
                float(r_n[0].detach().cpu().item()), float(r_c), atol=5e-4, rtol=5e-4
            )
    finally:
        cpu_env.close()
