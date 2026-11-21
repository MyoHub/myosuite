# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for parameterized workspace fixtures, boundaries, and MjSpec composition."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from myosuite.core.model_builder import ModelBuilder
from myosuite.scenes import (
    FixtureSpec,
    SceneSpec,
    SiteMarkerSpec,
    WorkspaceConfig,
    apply_scene_to_spec,
    get_scene,
    make_workspace_scene,
    quat_from_euler_xyz,
    scene_to_mjcf_xml,
)

pytestmark = pytest.mark.tier1


def test_workspace_scene_compiles_with_collision_and_visual_boundaries() -> None:
    cfg = WorkspaceConfig(
        include_seat=True,
        include_armrests=True,
        include_surface=True,
        include_panel=True,
        panel_mocap=True,
        target_grid=(2, 3),
        boundary_half_extents=(0.50, 0.40, 0.50),
        boundary_collide=False,
    )
    scene = make_workspace_scene(cfg)
    model = mujoco.MjModel.from_xml_string(scene_to_mjcf_xml(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Physical surface collides; visual boundary walls have contype=conaffinity=0.
    surf_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "work_surface_geom")
    wall_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "boundary_left_geom")
    assert surf_gid >= 0 and wall_gid >= 0
    assert model.geom_contype[surf_gid] == 1
    assert model.geom_conaffinity[surf_gid] == 1
    assert model.geom_contype[wall_gid] == 0
    assert model.geom_conaffinity[wall_gid] == 0

    # Mocap panel moves its 6 rigidly attached target sites in lockstep.
    panel_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "interaction_panel")
    mocap_id = int(model.body_mocapid[panel_bid])
    assert mocap_id >= 0
    t00 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "panel_target_0_0")
    t12 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "panel_target_1_2")
    assert t00 >= 0 and t12 >= 0
    before = data.site_xpos[t00].copy()

    delta = np.array([0.08, -0.05, 0.04])
    data.mocap_pos[mocap_id] += delta
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.site_xpos[t00] - before, delta, atol=1e-12)


def test_apply_scene_to_spec_attaches_workspace_without_adding_dofs() -> None:
    base_model, _ = ModelBuilder().attach_fragment("elbow").build()
    scene = get_scene("seated_workspace")

    composed_model, _ = (
        ModelBuilder()
        .attach_fragment("elbow")
        .apply_transform(lambda s: apply_scene_to_spec(s, scene))
        .build()
    )
    # Static fixtures add zero generalized coordinates; all 6 panel targets resolve.
    assert composed_model.nq == base_model.nq
    assert composed_model.nv == base_model.nv
    assert composed_model.nbody == base_model.nbody + len(scene.fixtures)
    for r in range(2):
        for c in range(3):
            sid = mujoco.mj_name2id(
                composed_model, mujoco.mjtObj.mjOBJ_SITE, f"panel_target_{r}_{c}"
            )
            assert sid >= 0


def test_quat_from_euler_and_fixture_validation() -> None:
    q = quat_from_euler_xyz(0.0, -0.5 * math.pi, 0.0)
    assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-12)
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(q))
    # -90 deg pitch about Y maps +X to +Z.
    np.testing.assert_allclose(
        mat.reshape(3, 3) @ [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], atol=1e-12
    )

    bad = SceneSpec(
        fixtures=[
            FixtureSpec(
                name="bad",
                shape="octahedron",
                sites=(SiteMarkerSpec(name="s0"),),
            )
        ]
    )
    with pytest.raises(ValueError, match="Unsupported shape"):
        scene_to_mjcf_xml(bad)
