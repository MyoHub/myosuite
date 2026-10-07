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

import types
from collections.abc import Callable

import mujoco
import numpy as np
import pytest

import myosuite  # noqa: F401
from myosuite import make_env


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
    "myoChallengeRelocateP2-v0": lambda env: (env.object_bid,),
    "myoChallengeRelocateP2eval-v0": lambda env: (env.object_bid,),
    "myoChallengeDieReorientP1-v0": lambda env: (env.object_bid, env.goal_bid),
    "myoChallengeDieReorientP2-v0": lambda env: (env.object_bid, env.goal_bid),
    "myoChallengeBaodingP2-v1": lambda env: (env.object1_bid, env.object2_bid),
    "myoElbowPose1D6MExoRandom-v0": lambda env: (
        env.model.body(env.weight_bodyname).id,
    ),
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
        if bodies[b].explicitinertial:
            scale = model.body_mass[b] / bodies[b].mass
            bodies[b].mass = model.body_mass[b]
            if np.isfinite(bodies[b].fullinertia[0]):
                bodies[b].fullinertia *= scale
            else:
                bodies[b].inertia *= scale
            continue
        scale = model.body_mass[b] / compiled.body_mass[b]
        for geom in bodies[b].geoms:
            geom.density *= scale
            geom.mass *= scale
    ref = spec.compile()
    # Body poses the env sets on reset (goal, object start) are inputs, not
    # derived fields: take them from the env like the state.
    ref.body_pos[:] = model.body_pos
    ref.body_quat[:] = model.body_quat
    return ref


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
    env = make_env(env_id).unwrapped
    env.reset()  # Bimanual skips the object rescale on its first reset.
    env.reset(seed=seed)
    body_ids = tuple(int(b) for b in _EDITED_BODIES[env_id](env))
    ref = _recompiled(env, body_ids)
    _assert_derived_fields_match(env.model, ref, body_ids)

    rng = np.random.default_rng(seed)
    for step in range(25):
        assert _contacts(env.model, env.data) == _contacts(ref, env.data), step
        env.step(rng.uniform(env.action_space.low, env.action_space.high))


@pytest.mark.parametrize("seed", [0, 3])
def test_bimanual_first_reset_mass_draw_matches_recompiled_model(seed: int) -> None:
    """The first reset draws a new box mass but skips the rescale: inertia follows."""
    env = make_env("myoChallengeBimanual-v0").unwrapped
    env.reset(seed=seed)
    body_ids = (int(env.obj_bid),)
    _assert_derived_fields_match(env.model, _recompiled(env, body_ids), body_ids)


def _bounds_rule(verts: np.ndarray) -> tuple[np.ndarray, float]:
    """A mesh geom's aabb (center, half size) and rbound, from its (float32) vertices."""
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    center, half = (lo + hi) / 2, (hi - lo) / 2
    return np.concatenate([center, half]), float(np.linalg.norm(np.abs(center) + half))


def _world_mesh_model(model: mujoco.MjModel, data: mujoco.MjData, geom: int):
    """Standalone compile of one mesh geom's current world-space triangles."""
    mesh = int(model.geom_dataid[geom])
    adr, num = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
    verts = model.mesh_vert[adr : adr + num] @ data.geom_xmat[geom].reshape(3, 3).T
    fadr, fnum = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
    spec = mujoco.MjSpec()
    spec.add_mesh(
        name="m",
        uservert=(verts + data.geom_xpos[geom]).ravel(),
        userface=model.mesh_face[fadr : fadr + fnum].ravel(),
    )
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_MESH, meshname="m", contype=0, conaffinity=0
    )
    ref = spec.compile()
    return ref, mujoco.MjData(ref)


def test_bimanual_visual_mesh_follows_collision_box() -> None:
    """The visual box is rescaled with the collision box and stays ray-castable.

    Its vertices are scaled like the box size, its bounding box/sphere follow
    the compiler's rule and rays hit it where they hit a standalone compile of
    the same world-space triangles.
    """
    env = make_env("myoChallengeBimanual-v0").unwrapped
    fresh = env._mj_spec.compile()
    vis, box = env.obj_gid - 1, env.obj_gid
    mesh = int(fresh.geom_dataid[vis])
    adr, num = int(fresh.mesh_vertadr[mesh]), int(fresh.mesh_vertnum[mesh])
    aabb, rbound = _bounds_rule(fresh.mesh_vert[adr : adr + num])
    np.testing.assert_allclose(fresh.geom_aabb[vis], aabb, atol=1e-8)
    assert fresh.geom_rbound[vis] == pytest.approx(rbound, abs=1e-8)
    # Nominal visual vertices in the box frame (the box has the body's axes).
    q = fresh.geom_quat[vis]
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, q)
    nominal = (
        fresh.mesh_vert[adr : adr + num] @ rot.reshape(3, 3).T
        + fresh.geom_pos[vis]
        - fresh.geom_pos[box]
    )

    m, d = env.model, env.data
    for seed in (0, 1, 2):  # nominal size, then two rescaled boxes
        env.reset(seed=seed)
        scales = m.geom_size[box] / fresh.geom_size[box]
        assert (seed == 0) == np.allclose(scales, 1.0)
        verts = m.mesh_vert[adr : adr + num]
        np.testing.assert_allclose(
            np.ptp(verts, axis=0), np.ptp(nominal, axis=0) * scales
        )
        aabb, rbound = _bounds_rule(verts)
        np.testing.assert_allclose(m.geom_aabb[vis], aabb, atol=1e-8)
        assert m.geom_rbound[vis] == pytest.approx(rbound, abs=1e-8)

        ref, ref_data = _world_mesh_model(m, d, vis)
        mujoco.mj_forward(ref, ref_data)
        group = np.zeros(6, np.uint8)
        group[m.geom_group[vis]] = 1
        geom_id = np.zeros(1, np.int32)
        center = d.geom_xpos[vis]
        for axis in np.vstack([np.eye(3), -np.eye(3)]):
            start = center - 0.08 * axis
            expected = mujoco.mj_ray(ref, ref_data, start, axis, None, 1, -1, geom_id)
            assert expected > 0
            got = mujoco.mj_rayMesh(m, d, vis, start, axis)
            assert got == pytest.approx(expected, abs=1e-6)
            if axis[2] == 0:  # sideways nothing else is in the way
                got = mujoco.mj_ray(m, d, start, axis, group, 1, -1, geom_id)
                assert geom_id[0] == vis and got == pytest.approx(expected, abs=1e-6)


def test_bimanual_rescale_uploads_visual_mesh(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rescaled visual box is pushed to an existing render context."""
    env = make_env("myoChallengeBimanual-v0").unwrapped
    uploads: list[tuple[object, int]] = []
    monkeypatch.setattr(
        mujoco, "mjr_uploadMesh", lambda model, con, mesh: uploads.append((con, mesh))
    )
    viewer = types.SimpleNamespace(con="context", make_context_current=lambda: None)
    env._mujoco_renderer = types.SimpleNamespace(
        _viewers={"rgb_array": viewer}, close=lambda: None
    )
    env.reset(seed=0)  # the first reset keeps the nominal size
    assert uploads == []
    env.reset(seed=1)
    assert uploads == [("context", env.obj_mid)]
