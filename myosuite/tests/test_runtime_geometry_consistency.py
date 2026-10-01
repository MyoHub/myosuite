# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Envs that resize or reshape objects per episode match a recompiled model.

After reset, the edited compiled model must equal a recompile of its spec
with the same geoms and body masses in the derived collision and inertia
fields, and give the same contacts in the same states.
"""

from __future__ import annotations

from collections.abc import Callable

import mujoco
import numpy as np
import pytest

import myosuite
from myosuite.utils import gym

myosuite.register_all_envs()

pytestmark = pytest.mark.tier1

_PRIMITIVES = {
    int(mujoco.mjtGeom.mjGEOM_SPHERE),
    int(mujoco.mjtGeom.mjGEOM_CAPSULE),
    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
    int(mujoco.mjtGeom.mjGEOM_CYLINDER),
    int(mujoco.mjtGeom.mjGEOM_BOX),
}

# env id -> ids of the bodies whose geoms or mass the env edits on reset.
_EDITED_BODIES: dict[str, Callable] = {
    "myoHandReorient8-v0": lambda env: (env.obj_bid, env.target_obj_bid),
    "myoHandReorient100-v0": lambda env: (env.obj_bid, env.target_obj_bid),
    "myoHandReorientID-v0": lambda env: (env.obj_bid, env.target_obj_bid),
    "myoHandReorientOOD-v0": lambda env: (env.obj_bid, env.target_obj_bid),
    "myoHandObjHoldRandom-v0": lambda env: (env.model.geom_bodyid[-1],),
    "myoChallengeBimanual-v0": lambda env: (env.obj_bid,),
}


def _primitive_geoms(model: mujoco.MjModel, body_id: int) -> list[int]:
    start = model.body_geomadr[body_id]
    return [
        g
        for g in range(start, start + model.body_geomnum[body_id])
        if model.geom_type[g] in _PRIMITIVES
    ]


def _recompiled(env, body_ids: tuple[int, ...]) -> mujoco.MjModel:
    """Recompile the env's spec with its current geoms and body masses."""
    model, spec = env.model, env._mj_spec.copy()
    bodies = spec.bodies
    for b in body_ids:
        start = model.body_geomadr[b]
        for gid, geom in enumerate(bodies[b].geoms, start):
            if model.geom_type[gid] not in _PRIMITIVES:
                continue
            geom.type = mujoco.mjtGeom(int(model.geom_type[gid]))
            geom.fromto = [np.nan] * 6
            geom.size, geom.pos = model.geom_size[gid], model.geom_pos[gid]
            geom.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT
            geom.quat = model.geom_quat[gid]
    compiled = spec.compile()
    for b in body_ids:
        scale = model.body_mass[b] / compiled.body_mass[b]
        for geom in bodies[b].geoms:
            geom.density *= scale
            geom.mass *= scale
    return spec.compile()


def _assert_derived_fields_match(
    model: mujoco.MjModel, ref: mujoco.MjModel, body_ids: tuple[int, ...]
) -> None:
    close = {"rtol": 1e-9, "atol": 1e-12}
    for b in body_ids:
        geoms = _primitive_geoms(model, b)
        np.testing.assert_allclose(
            model.geom_aabb[geoms], ref.geom_aabb[geoms], **close
        )
        np.testing.assert_allclose(
            model.geom_rbound[geoms], ref.geom_rbound[geoms], **close
        )
        if model.body_weldid[b] == 0:
            continue  # Static bodies keep their inertial frame (and BVH frame).
        nodes = slice(model.body_bvhadr[b], model.body_bvhadr[b] + model.body_bvhnum[b])
        np.testing.assert_allclose(model.bvh_aabb[nodes], ref.bvh_aabb[nodes], **close)
        np.testing.assert_array_equal(model.bvh_nodeid[nodes], ref.bvh_nodeid[nodes])
        for field in ("body_inertia", "body_ipos", "body_iquat", "body_invweight0"):
            np.testing.assert_allclose(
                getattr(model, field)[b], getattr(ref, field)[b], err_msg=field, **close
            )
        dofs = slice(model.body_dofadr[b], model.body_dofadr[b] + model.body_dofnum[b])
        np.testing.assert_allclose(
            model.dof_invweight0[dofs], ref.dof_invweight0[dofs], **close
        )


def _contacts(model: mujoco.MjModel, source: mujoco.MjData) -> list[tuple[int, ...]]:
    data = mujoco.MjData(model)
    data.qpos[:] = source.qpos
    data.mocap_pos[:] = source.mocap_pos
    data.mocap_quat[:] = source.mocap_quat
    mujoco.mj_forward(model, data)
    return sorted(tuple(sorted(pair)) for pair in data.contact.geom.tolist())


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("env_id", list(_EDITED_BODIES))
def test_episode_geometry_matches_recompiled_model(env_id: str, seed: int) -> None:
    env = gym.make(env_id).unwrapped
    env.reset()  # Bimanual skips the object rescale on its first reset.
    env.reset(seed=seed)
    body_ids = tuple(int(b) for b in _EDITED_BODIES[env_id](env))
    ref = _recompiled(env, body_ids)
    _assert_derived_fields_match(env.model, ref, body_ids)

    rng = np.random.default_rng(seed)
    for step in range(25):
        assert _contacts(env.model, env.data) == _contacts(ref, env.data), step
        env.step(rng.uniform(env.action_space.low, env.action_space.high))
