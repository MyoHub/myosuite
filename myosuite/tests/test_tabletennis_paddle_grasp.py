# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Table-tennis reset state matches the public ``myoarm_tabletennis.xml``.

Reference values are the public keyframe / compiled model, measured against
myosuite 2.12.2's ``myoChallengeTableTennisP0-v0`` under the same MuJoCo build.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.tier2

# Public ``myoarm_tabletennis.xml`` keyframe "default", paddle freejoint.
PUBLIC_PADDLE_POS = np.array([1.91, 0.65, 1.18])
PUBLIC_PADDLE_QUAT = np.array([0.699445, -0.105711, 0.698888, -0.105627])


@pytest.fixture(scope="module")
def tabletennis_env():
    from myosuite import register_all_envs

    register_all_envs()
    import gymnasium as gym

    env = gym.make("myoChallengeTableTennisP0-v0")
    yield env
    env.close()


def test_paddle_world_pose_matches_public_keyframe(tabletennis_env) -> None:
    """Reset leaves the paddle exactly at the public spawn — no snapping."""
    tabletennis_env.reset(seed=0)
    model = tabletennis_env.unwrapped.model
    data = tabletennis_env.unwrapped.data
    bid = model.body("paddle").id
    assert data.xpos[bid] == pytest.approx(PUBLIC_PADDLE_POS, abs=1e-9)
    quat = np.asarray(data.xquat[bid])
    assert abs(float(np.dot(quat, PUBLIC_PADDLE_QUAT))) == pytest.approx(1.0, abs=1e-6)


def test_pelvis_slide_axes_track_world_axes(tabletennis_env) -> None:
    """``pelvis_x``/``pelvis_y`` drive the actor along the legacy world axes.

    The axes live on the calibrated root body, whose yaw differs from legacy's
    ``euler="0 0 3.14"``; uncompensated axes sent the actor 0.59 m sideways and
    left the paddle out of reach.
    """
    tabletennis_env.reset(seed=0)
    model = tabletennis_env.unwrapped.model
    data = tabletennis_env.unwrapped.data
    root = data.xmat[model.body("Full Body").id].reshape(3, 3)
    for name, world in (("pelvis_x", [-1.0, 0.0, 0.0]), ("pelvis_y", [0.0, -1.0, 0.0])):
        axis = root @ np.asarray(model.jnt_axis[model.joint(name).id])
        assert float(np.dot(axis, world)) > 0.999


def test_paddle_handle_sits_in_grasp(tabletennis_env) -> None:
    """The legacy arm pose already reaches the paddle: handle inside S_grasp."""
    tabletennis_env.reset(seed=0)
    model = tabletennis_env.unwrapped.model
    data = tabletennis_env.unwrapped.data
    handle = data.geom_xpos[model.geom("handle").id]
    grasp = data.site_xpos[model.site("S_grasp").id]
    assert float(np.linalg.norm(handle - grasp)) < 0.025


def test_paddle_visual_aligns_with_handle_axis(tabletennis_env) -> None:
    """Textured mesh is long along the handle, thin along the face normal."""
    tabletennis_env.reset(seed=0)
    model = tabletennis_env.unwrapped.model
    data = tabletennis_env.unwrapped.data
    gid = model.geom("paddle").id
    mid = int(model.geom_dataid[gid])
    adr = int(model.mesh_vertadr[mid])
    nvert = int(model.mesh_vertnum[mid])
    verts = model.mesh_vert[adr : adr + nvert]
    world = verts @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
    body_rot = data.xmat[model.body("paddle").id].reshape(3, 3)
    local = (world - data.geom_xpos[gid]) @ body_rot
    extent = local.max(axis=0) - local.min(axis=0)
    # Handle axis = body +X; face normal = body +Z (mesh thin axis).
    assert float(extent[0]) > 0.18
    assert float(extent[2]) < 0.05


def test_paddle_collision_geoms_are_hidden(tabletennis_env) -> None:
    """Pad/handle match legacy ``class="collision"`` (group 4, not drawn)."""
    tabletennis_env.reset(seed=0)
    model = tabletennis_env.unwrapped.model
    assert int(model.geom_group[model.geom("pad").id]) == 4
    assert int(model.geom_group[model.geom("handle").id]) == 4
    assert int(model.geom_group[model.geom("paddle").id]) in (0, 1, 2)


def test_prop_masses_match_public(tabletennis_env) -> None:
    """Legacy ``<inertial>`` + zero-mass collision geoms, not geom density."""
    tabletennis_env.reset(seed=0)
    model = tabletennis_env.unwrapped.model
    assert float(model.body_mass[model.body("paddle").id]) == pytest.approx(0.15)
    assert float(model.body_mass[model.body("pingpong").id]) == pytest.approx(2.7e-3)
    assert float(model.body_mass[model.body("tabletennis_table").id]) == pytest.approx(
        1012.3920413955552, rel=1e-9
    )


def test_scene_rests_on_the_floor(tabletennis_env) -> None:
    """Ground plane is at z=0 like the legacy scene, under table and feet."""
    tabletennis_env.reset(seed=0)
    model = tabletennis_env.unwrapped.model
    data = tabletennis_env.unwrapped.data
    ground_z = float(data.geom_xpos[model.geom("ground").id][2])
    assert ground_z == pytest.approx(0.0, abs=1e-9)
    for body in ("calcn_r", "calcn_l", "toes_r", "toes_l"):
        assert float(data.xpos[model.body(body).id][2]) - ground_z < 0.1
        assert float(data.xpos[model.body(body).id][2]) - ground_z > -0.01
