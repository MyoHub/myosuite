# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""``com_vel`` of the walk envs is the velocity of the whole-body centre of mass.

The oracles are MuJoCo's ``mj_subtreeVel`` and a finite difference of the centre-of-mass
position, not another backend.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from myosuite import make_env

pytestmark = pytest.mark.tier1

WALK_IDS = (
    "myoLegWalk-v0",
    "myoLegRoughTerrainWalk-v0",
    "myoLegHillyTerrainWalk-v0",
    "myoLegStairTerrainWalk-v0",
    "myoFatiLegWalk-v0",
)


def _true_com_vel(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Planar COM velocity in the env's convention (the model faces -y)."""
    mujoco.mj_subtreeVel(model, data)
    return -data.subtree_linvel[0, :2]


@pytest.fixture(scope="module")
def walk_model() -> mujoco.MjModel:
    env = make_env("myoLegWalk-v0")
    model = env.unwrapped.model
    env.close()
    return model


def _tumbling_data(model: mujoco.MjModel, seed: int) -> mujoco.MjData:
    """The walk keyframe with every segment rotating."""
    data = mujoco.MjData(model)
    data.qpos[:] = model.key_qpos[2]
    data.qvel[:] = np.random.default_rng(seed).normal(0.0, 2.0, model.nv)
    mujoco.mj_forward(model, data)
    return data


def _com_linear_velocity(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    from myosuite.terms.base_obs import com_linear_velocity

    return com_linear_velocity(
        np,
        model.body_mass,
        data.cvel,
        data.xipos,
        data.subtree_com[model.body_rootid],
    )


@pytest.mark.parametrize("env_id", WALK_IDS)
def test_com_vel_is_the_centre_of_mass_velocity(env_id: str) -> None:
    """Observation, reward input and obs vector hold ``-subtree_linvel`` of the model."""
    env = make_env(env_id)
    base = env.unwrapped
    model, data = base.model, base.data
    keys = base.obs_keys
    rng = np.random.default_rng(0)

    env.reset(seed=0)
    np.testing.assert_allclose(
        base._get_obs_dict(base._accessor)["com_vel"][0],
        _true_com_vel(model, data),
        atol=1e-6,
        err_msg="after reset",
    )
    for step in range(60):
        action = rng.uniform(0.0, 1.0, env.action_space.shape).astype(np.float32)
        obs, _, terminated, truncated, info = env.step(action)
        expected = _true_com_vel(model, data)
        obs_dict, rwd_dict = info["obs_dict"], info["rwd_dict"]
        np.testing.assert_allclose(
            obs_dict["com_vel"][0], expected, atol=1e-6, err_msg=f"step {step}"
        )
        start = sum(np.size(obs_dict[k]) for k in keys[: keys.index("com_vel")])
        np.testing.assert_allclose(obs[start : start + 2], expected, atol=1e-6)
        vel_reward = np.exp(-np.square(base.target_y_vel - expected[1])) + np.exp(
            -np.square(base.target_x_vel - expected[0])
        )
        assert rwd_dict["vel_reward"] == pytest.approx(vel_reward, abs=1e-6)
        if terminated or truncated:
            env.reset()
    env.close()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_com_linear_velocity_matches_mujoco_and_finite_difference(
    walk_model: mujoco.MjModel, seed: int
) -> None:
    """The shared term equals ``mj_subtreeVel`` and d/dt of the COM position."""
    model, data = walk_model, _tumbling_data(walk_model, seed)
    velocity = _com_linear_velocity(model, data)
    mujoco.mj_subtreeVel(model, data)
    np.testing.assert_allclose(velocity, data.subtree_linvel[0], atol=1e-9)

    def com(dt: float) -> np.ndarray:
        moved = mujoco.MjData(model)
        moved.qpos[:] = data.qpos
        mujoco.mj_integratePos(model, moved.qpos, data.qvel, dt)
        mujoco.mj_kinematics(model, moved)
        mujoco.mj_comPos(model, moved)
        return model.body_mass @ moved.xipos / model.body_mass.sum()

    h = 1e-5
    np.testing.assert_allclose(velocity, (com(h) - com(-h)) / (2 * h), atol=1e-6)
    # The 2.x formula (mass-weighted cvel) is not this velocity on a tumbling body.
    legacy = model.body_mass @ data.cvel[:, 3:] / model.body_mass.sum()
    assert np.abs(legacy - velocity).max() > 0.05


def test_com_linear_velocity_is_batched_on_numpy_and_torch(
    walk_model: mujoco.MjModel,
) -> None:
    """A leading batch axis gives the per-state velocities, on both array modules."""
    torch = pytest.importorskip("torch")
    from myosuite.terms.base_obs import com_linear_velocity

    model = walk_model
    states = [_tumbling_data(model, seed) for seed in (0, 1)]
    single = np.stack([_com_linear_velocity(model, d) for d in states])
    cvel, xipos, root_com = (
        np.stack([getattr(d, name) for d in states])
        for name in ("cvel", "xipos", "subtree_com")
    )
    root_com = root_com[:, model.body_rootid]
    np.testing.assert_allclose(
        com_linear_velocity(np, model.body_mass, cvel, xipos, root_com),
        single,
        atol=1e-12,
    )
    batched = com_linear_velocity(
        torch,
        torch.as_tensor(model.body_mass),
        torch.as_tensor(cvel),
        torch.as_tensor(xipos),
        torch.as_tensor(root_com),
    )
    np.testing.assert_allclose(batched.numpy(), single, atol=1e-12)
