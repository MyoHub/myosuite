# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for parameterized workspace fixtures, boundaries, and CPU/mjlab composition."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from myosuite.core.model_builder import ModelBuilder
from myosuite.physics.quat_math import euler2quat, quat_from_euler_xyz
from myosuite.scenes import (
    FixtureSpec,
    FloorSpec,
    ObjectSpec,
    SceneSpec,
    SiteMarkerSpec,
    WorkspaceConfig,
    apply_scene_to_spec,
    get_scene,
    make_workspace_scene,
    scene_to_mjcf_xml,
    scene_to_mjlab_cfg,
)

pytestmark = pytest.mark.tier1


def test_scene_spec_positional_compatibility() -> None:
    legacy = SceneSpec(FloorSpec(), [], (0.0, 0.0, -1.0))
    assert legacy.gravity == (0.0, 0.0, -1.0)
    assert legacy.fixtures == []
    model = mujoco.MjModel.from_xml_string(scene_to_mjcf_xml(legacy))
    np.testing.assert_allclose(model.opt.gravity, [0.0, 0.0, -1.0], atol=1e-12)


def test_apply_scene_to_spec_preserves_existing_floor() -> None:
    existing_xml = (
        "<mujoco><worldbody>"
        "<geom name='floor' type='plane' size='1 1 .1'/>"
        "</worldbody></mujoco>"
    )
    mj_spec = mujoco.MjSpec.from_string(existing_xml)
    scene = SceneSpec(
        floor=FloorSpec(kind="flat", size=12.0),
        fixtures=[FixtureSpec(name="shelf", size=(0.2, 0.3, 0.02))],
    )
    apply_scene_to_spec(mj_spec, scene, include_floor=True)
    model = mj_spec.compile()
    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert floor_gid >= 0
    np.testing.assert_allclose(model.geom_size[floor_gid], [1.0, 1.0, 0.1], atol=1e-12)

    empty_spec = mujoco.MjSpec()
    apply_scene_to_spec(empty_spec, scene, include_floor=True)
    model_added = empty_spec.compile()
    added_gid = mujoco.mj_name2id(model_added, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert added_gid >= 0
    np.testing.assert_allclose(
        model_added.geom_size[added_gid], [12.0, 12.0, 0.1], atol=1e-12
    )


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

    surf_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "work_surface_geom")
    wall_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "boundary_left_geom")
    assert surf_gid >= 0 and wall_gid >= 0
    assert model.geom_contype[surf_gid] == 1
    assert model.geom_conaffinity[surf_gid] == 1
    assert model.geom_contype[wall_gid] == 0
    assert model.geom_conaffinity[wall_gid] == 0

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
    assert composed_model.nq == base_model.nq
    assert composed_model.nv == base_model.nv
    assert composed_model.nbody == base_model.nbody + len(scene.fixtures)
    for r in range(2):
        for c in range(3):
            sid = mujoco.mj_name2id(
                composed_model, mujoco.mjtObj.mjOBJ_SITE, f"panel_target_{r}_{c}"
            )
            assert sid >= 0


def test_quat_from_euler_matches_euler2quat_and_validates_shapes() -> None:
    rng = np.random.default_rng(0)
    angles = rng.uniform(-math.pi, math.pi, size=(100, 3))
    for rx, ry, rz in angles:
        q = quat_from_euler_xyz(float(rx), float(ry), float(rz))
        np.testing.assert_allclose(q, euler2quat((rx, ry, rz)), atol=1e-15)

    q = quat_from_euler_xyz(0.0, -0.5 * math.pi, 0.0)
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(q))
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


def test_cpu_and_mjlab_scene_pose_quat_collision_and_site_parity() -> None:
    pytest.importorskip("mjlab")
    from mjlab.scene import Scene, SceneCfg
    from mjlab.sim import Simulation, SimulationCfg

    scene_spec = SceneSpec(
        floor=FloorSpec(kind="flat"),
        objects=[
            ObjectSpec(name="cube", shape="box", size=0.03, mass=0.12),
            ObjectSpec(name="peg", shape="capsule", size=0.02, count=2, mass=0.08),
        ],
        fixtures=[
            FixtureSpec(
                name="tilted_panel",
                shape="box",
                size=(0.015, 0.22, 0.14),
                pos=(0.52, -0.12, 1.04),
                quat=quat_from_euler_xyz(0.18, -0.25, 0.32),
                mocap=True,
                collide=True,
                sites=(
                    SiteMarkerSpec(name="anchor_a", pos=(-0.017, 0.08, 0.05)),
                    SiteMarkerSpec(name="anchor_b", pos=(-0.017, -0.06, -0.04)),
                ),
            ),
            FixtureSpec(
                name="visual_zone",
                shape="ellipsoid",
                size=(0.25, 0.20, 0.18),
                pos=(0.30, 0.15, 0.88),
                quat=quat_from_euler_xyz(-0.10, 0.12, -0.20),
                collide=False,
            ),
        ],
    )

    cpu_model = mujoco.MjModel.from_xml_string(scene_to_mjcf_xml(scene_spec))
    cpu_data = mujoco.MjData(cpu_model)
    mujoco.mj_forward(cpu_model, cpu_data)

    mj_scene = Scene(
        SceneCfg(num_envs=2, entities=scene_to_mjlab_cfg(scene_spec)),
        device="cpu",
    )
    sim = Simulation(
        num_envs=2,
        cfg=SimulationCfg(),
        model=mj_scene.compile(),
        device="cpu",
    )
    mj_scene.initialize(mj_model=sim.mj_model, model=sim.model, data=sim.data)
    mj_scene.reset()
    sim.forward()

    mj_model = sim.mj_model
    xpos = sim.data.xpos.cpu().numpy()
    xquat = sim.data.xquat.cpu().numpy()
    site_xpos = sim.data.site_xpos.cpu().numpy()

    def _find_suffix_id(obj_type: mujoco.mjtObj, count: int, suffix: str) -> int:
        matches = [
            idx
            for idx in range(count)
            if (mujoco.mj_id2name(mj_model, obj_type, idx) or "") in (suffix,)
            or (mujoco.mj_id2name(mj_model, obj_type, idx) or "").endswith(f"/{suffix}")
        ]
        assert len(matches) == 1, (suffix, matches)
        return matches[0]

    for body_name in ("tilted_panel", "visual_zone", "cube", "peg_0", "peg_1"):
        cpu_bid = mujoco.mj_name2id(cpu_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        mj_bid = _find_suffix_id(mujoco.mjtObj.mjOBJ_BODY, mj_model.nbody, body_name)
        for env_i in range(2):
            np.testing.assert_allclose(
                xpos[env_i, mj_bid], cpu_data.xpos[cpu_bid], atol=1e-6
            )
            dot = abs(float(np.dot(xquat[env_i, mj_bid], cpu_data.xquat[cpu_bid])))
            assert dot == pytest.approx(1.0, abs=1e-6)

    for geom_name in ("tilted_panel_geom", "visual_zone_geom"):
        cpu_gid = mujoco.mj_name2id(cpu_model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        mj_gid = _find_suffix_id(mujoco.mjtObj.mjOBJ_GEOM, mj_model.ngeom, geom_name)
        assert int(mj_model.geom_contype[mj_gid]) == int(
            cpu_model.geom_contype[cpu_gid]
        )
        assert int(mj_model.geom_conaffinity[mj_gid]) == int(
            cpu_model.geom_conaffinity[cpu_gid]
        )

    for site_name in ("anchor_a", "anchor_b"):
        cpu_sid = mujoco.mj_name2id(cpu_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        mj_sid = _find_suffix_id(mujoco.mjtObj.mjOBJ_SITE, mj_model.nsite, site_name)
        for env_i in range(2):
            np.testing.assert_allclose(
                site_xpos[env_i, mj_sid], cpu_data.site_xpos[cpu_sid], atol=1e-6
            )


def test_scene_object_shares_model_builder_free_body_contract_and_acyclic_imports() -> (
    None
):
    import ast
    from pathlib import Path

    import myosuite.scenes as scenes_pkg
    import myosuite.scenes.library as library_mod
    import myosuite.scenes.workspaces as workspaces_mod

    assert not hasattr(scenes_pkg, "quat_from_euler_xyz")
    assert not hasattr(library_mod, "quat_from_euler_xyz")

    ws_tree = ast.parse(Path(workspaces_mod.__file__).read_text(encoding="utf-8"))
    ws_imports = {
        node.module
        for node in ast.walk(ws_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "myosuite.scenes.library" not in ws_imports

    scene_model = mujoco.MjModel.from_xml_string(
        scene_to_mjcf_xml(
            SceneSpec(
                floor=FloorSpec(kind="none"),
                objects=[
                    ObjectSpec(
                        name="prop",
                        shape="box",
                        size=0.04,
                        mass=0.25,
                        rgba=(0.2, 0.4, 0.8, 1.0),
                    )
                ],
            )
        )
    )
    builder_model, _ = (
        ModelBuilder()
        .add_free_body(
            "prop",
            pos=(0.0, 0.0, 0.04),
            geom_type=mujoco.mjtGeom.mjGEOM_BOX,
            geom_size=(0.04, 0.04, 0.04),
            rgba=(0.2, 0.4, 0.8, 1.0),
            mass=0.25,
        )
        .build()
    )
    assert mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_JOINT, "prop_free") >= 0
    assert mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_GEOM, "prop") >= 0
    np.testing.assert_allclose(
        scene_model.body_mass, builder_model.body_mass, atol=1e-12
    )
    np.testing.assert_allclose(
        scene_model.geom_size, builder_model.geom_size, atol=1e-12
    )
    np.testing.assert_allclose(scene_model.qpos0, builder_model.qpos0, atol=1e-12)
