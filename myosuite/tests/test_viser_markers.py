# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for ``myosuite.viz.viser_markers``."""

from __future__ import annotations

import ast
from collections.abc import Iterator
import json
from pathlib import Path
import socket
import subprocess
import sys

from mjviser import ViserMujocoScene
import mujoco
import numpy as np
import pytest
import viser

from myosuite import make_env
from myosuite.core.trajectory_io import MotionClip
from myosuite.viz.ghost_body_viz import GhostBodyViz
from myosuite.viz.site_marker_viz import SiteMarkerViz
import myosuite.viz.viser_markers as vm_mod
from myosuite.viz.viser_markers import (
    ViserMarkers,
    ViserMultiEnvScene,
    square_grid_offsets,
)

pytestmark = pytest.mark.tier1

_REPO = Path(__file__).resolve().parents[2]

_FREEJOINT_XML = """
<mujoco>
  <asset>
    <mesh name="tet" vertex="0 0 0  0.1 0 0  0 0.1 0  0 0 0.1"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1"/>
    <body name="root" pos="0 0 1">
      <freejoint/>
      <geom type="sphere" size="0.08"/>
      <geom type="mesh" mesh="tet" pos="0 0 0"/>
      <site name="tip" pos="0 0 0.1" size="0.02"/>
    </body>
    <body name="target" mocap="true" pos="0.2 -0.1 0.5">
      <geom type="sphere" size="0.04" rgba="1 0 0 0.8"/>
    </body>
  </worldbody>
</mujoco>
"""


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def viser_server() -> Iterator[viser.ViserServer]:
    server = viser.ViserServer(host="127.0.0.1", port=_free_port(), verbose=False)
    try:
        yield server
    finally:
        server.stop()


def _init_geom(
    scn: mujoco.MjvScene,
    geom_type: mujoco.mjtGeom,
    size: tuple[float, float, float],
    pos: tuple[float, float, float],
    mat: np.ndarray | None = None,
    rgba: tuple[float, float, float, float] = (1.0, 0.2, 0.1, 0.8),
    dataid: int = -1,
) -> None:
    idx = int(scn.ngeom)
    mat_flat = (
        np.eye(3, dtype=np.float64).flatten()
        if mat is None
        else np.asarray(mat, dtype=np.float64).reshape(9)
    )
    mujoco.mjv_initGeom(
        scn.geoms[idx],
        geom_type,
        np.array(size, dtype=np.float64),
        np.array(pos, dtype=np.float64),
        mat_flat,
        np.array(rgba, dtype=np.float64),
    )
    if dataid >= 0:
        scn.geoms[idx].dataid = dataid
    scn.ngeom = idx + 1


def test_viser_markers_module_keeps_optional_deps_lazy_and_uses_public_apis() -> None:
    code = (
        "import json, sys\n"
        "import myosuite.viz.viser_markers\n"
        "heavy = ['viser', 'mjviser', 'trimesh', 'torch']\n"
        "print(json.dumps([m for m in heavy if m in sys.modules]))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == []

    tree = ast.parse(Path(vm_mod.__file__).read_text(encoding="utf-8"))
    forbidden = {
        "_handle_from_node_name",
        "_impl",
        "_scene_offset",
        "_update_decor_from_mjvscene",
        "_get_unit_mesh",
    }
    attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert forbidden.isdisjoint(attrs)


def test_viser_markers_primitives_arrows_and_meshes(
    viser_server: viser.ViserServer,
) -> None:
    model = mujoco.MjModel.from_xml_string(_FREEJOINT_XML)
    scene = ViserMujocoScene(viser_server, model, num_envs=1)
    markers = ViserMarkers(scene, model)
    assert markers.root_path == "/fixed_bodies/myosuite_markers"

    user_scn = mujoco.MjvScene(model, maxgeom=32)
    _init_geom(
        user_scn,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        (0.05, 0.0, 0.0),
        (0.1, 0.2, 0.3),
        rgba=(1.0, 0.0, 0.0, 0.7),
    )
    _init_geom(
        user_scn,
        mujoco.mjtGeom.mjGEOM_BOX,
        (0.1, 0.2, 0.3),
        (-0.2, 0.0, 0.4),
        rgba=(0.0, 1.0, 0.0, 0.9),
    )
    _init_geom(
        user_scn,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        (0.03, 0.15, 0.0),
        (0.0, 0.4, 0.5),
        rgba=(0.0, 0.4, 1.0, 0.6),
    )
    _init_geom(
        user_scn,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        (0.04, 0.12, 0.0),
        (0.3, -0.2, 0.2),
    )
    _init_geom(
        user_scn,
        mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        (0.03, 0.05, 0.07),
        (0.2, 0.2, 0.6),
    )
    _init_geom(
        user_scn,
        mujoco.mjtGeom.mjGEOM_MESH,
        (1.0, 1.0, 1.0),
        (0.5, 0.5, 0.5),
        dataid=0,
    )
    rot_x_from_z = np.array(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64
    )
    _init_geom(
        user_scn,
        mujoco.mjtGeom.mjGEOM_ARROW,
        (0.02, 0.05, 0.5),
        (1.0, 2.0, 3.0),
        mat=rot_x_from_z,
        rgba=(1.0, 0.5, 0.0, 0.85),
    )
    _init_geom(
        user_scn,
        mujoco.mjtGeom.mjGEOM_PLANE,
        (5.0, 5.0, 0.1),
        (0.0, 0.0, 0.0),
    )
    _init_geom(
        user_scn,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        (0.1, 0.1, 0.1),
        (9.0, 9.0, 9.0),
        rgba=(1.0, 1.0, 0.0, 0.0),
    )

    markers.update(user_scn)

    expected_keys = {
        "sphere",
        "box",
        "capsule",
        "cylinder",
        "ellipsoid",
        "mesh_0",
        "arrows",
    }
    assert set(markers.handles.keys()) == expected_keys
    for handle in markers.handles.values():
        assert handle.visible is True

    np.testing.assert_allclose(
        markers.handles["sphere"].batched_positions,
        [[0.1, 0.2, 0.3]],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        markers.handles["sphere"].batched_scales,
        [[0.05, 0.05, 0.05]],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        markers.handles["box"].batched_scales,
        [[0.1, 0.2, 0.3]],
        atol=1e-6,
    )

    markers.clear()
    assert markers.handles == {}


def test_viser_markers_handle_reuse_recreation_and_visibility(
    viser_server: viser.ViserServer,
) -> None:
    model = mujoco.MjModel.from_xml_string(_FREEJOINT_XML)
    scene = ViserMujocoScene(viser_server, model, num_envs=1)
    markers = ViserMarkers(scene, model)
    scn = mujoco.MjvScene(model, maxgeom=16)

    _init_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, (0.04, 0, 0), (0.0, 0.0, 0.1))
    _init_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, (0.04, 0, 0), (0.2, 0.0, 0.1))
    _init_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, (0.1, 0.1, 0.1), (0.0, 0.3, 0.2))
    markers.update(scn)
    sphere_h0 = markers.handles["sphere"]
    box_h0 = markers.handles["box"]

    scn.ngeom = 0
    _init_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, (0.06, 0, 0), (1.0, 2.0, 3.0))
    _init_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, (0.06, 0, 0), (4.0, 5.0, 6.0))
    markers.update(scn)
    assert markers.handles["sphere"] is sphere_h0
    assert markers.handles["sphere"].visible is True
    assert box_h0.visible is False
    np.testing.assert_allclose(
        sphere_h0.batched_positions,
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        atol=1e-6,
    )

    scn.ngeom = 0
    for k in range(3):
        _init_geom(
            scn, mujoco.mjtGeom.mjGEOM_SPHERE, (0.05, 0, 0), (float(k), 0.0, 0.0)
        )
    markers.update(scn)
    sphere_h2 = markers.handles["sphere"]
    assert sphere_h2 is not sphere_h0
    assert sphere_h2.batched_positions.shape == (3, 3)

    for k in range(3):
        scn.geoms[k].rgba[3] = 0.0
    markers.update(scn)
    assert sphere_h2.visible is False


def test_camera_tracking_aligns_fixed_frame_and_motion_clip_markers(
    viser_server: viser.ViserServer,
) -> None:
    model = mujoco.MjModel.from_xml_string(_FREEJOINT_XML)
    data = mujoco.MjData(model)
    data.qpos[:3] = [2.0, -1.5, 1.0]
    mujoco.mj_forward(model, data)

    scene = ViserMujocoScene(viser_server, model, num_envs=1)
    scene.camera_tracking_enabled = True
    scene.update_from_mjdata(data)

    # Public fixed_bodies_frame shifts by -tracked_pos while markers under it stay in world coords.
    np.testing.assert_allclose(
        scene.fixed_bodies_frame.position, [-2.0, 1.5, -1.0], atol=1e-6
    )

    markers = ViserMarkers(scene, model)
    site_xpos = np.array(
        [
            [[2.0, -1.5, 1.1], [2.2, -1.3, 1.0]],
            [[2.4, -1.2, 0.9], [2.5, -1.0, 1.2]],
        ],
        dtype=np.float64,
    )
    clip = MotionClip(
        qpos=np.tile(data.qpos, (2, 1)),
        qvel=None,
        site_xpos=site_xpos,
        site_names=["s0", "s1"],
    )
    scn0 = markers.update_from_clip(
        clip, 0, rgba=(1.0, 0.4, 0.0, 0.65), sphere_radius=0.035
    )
    assert scn0.ngeom == 2
    h0 = markers.handles["sphere"]
    assert h0.name.startswith(scene.fixed_bodies_frame.name + "/")
    np.testing.assert_allclose(h0.batched_positions, site_xpos[0], atol=1e-6)
    np.testing.assert_allclose(markers.opacities["sphere"], [0.65, 0.65], atol=1e-5)
    if h0.batched_opacities is not None:
        np.testing.assert_allclose(h0.batched_opacities, [0.65, 0.65], atol=1e-5)

    viz_fn = SiteMarkerViz(
        clip=clip,
        site_names=("s0", "s1"),
        model=model,
        rgba=(1.0, 0.4, 0.0, 0.65),
        sphere_radius=0.035,
    ).as_viz_fn()
    scn1 = markers.update_from_callback(1, data, viz_fn)
    assert scn1 is scn0
    assert markers.handles["sphere"] is h0
    np.testing.assert_allclose(h0.batched_positions, site_xpos[1], atol=1e-6)

    ghost_fn = GhostBodyViz.from_motion_clip(clip, model, bone_meshes=True).as_viz_fn()
    markers.update_from_callback(0, data, ghost_fn)
    assert "mesh_0" in markers.handles


def test_viser_multi_env_scene_freejoint_and_fixed_base(
    viser_server: viser.ViserServer,
) -> None:
    model = mujoco.MjModel.from_xml_string(_FREEJOINT_XML)
    multi = ViserMultiEnvScene(viser_server, model, num_envs=4, spacing=3.0)
    assert multi.camera_tracking_enabled is False
    expected_offsets = np.array(
        [[-1.5, -1.5], [1.5, -1.5], [-1.5, 1.5], [1.5, 1.5]], dtype=np.float64
    )
    np.testing.assert_allclose(multi.grid_offsets, expected_offsets)
    np.testing.assert_allclose(
        square_grid_offsets(1, spacing=3.0), np.zeros((1, 2), dtype=np.float64)
    )

    qpos = np.tile(model.qpos0, (4, 1))
    qpos[:, 2] = [1.0, 1.1, 1.2, 1.3]
    qpos_orig = qpos.copy()
    mocap_pos = np.tile(np.array([[[0.2, -0.1, 0.5]]]), (4, 1, 1))
    mocap_orig = mocap_pos.copy()

    per_env_scenes = []
    for env_i in range(4):
        scn = mujoco.MjvScene(model, maxgeom=4)
        _init_geom(
            scn,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            (0.03, 0, 0),
            (0.1 * env_i, 0.0, 0.8),
        )
        per_env_scenes.append(scn)

    multi.update(qpos, mocap_pos=mocap_pos, user_scenes=per_env_scenes)
    np.testing.assert_array_equal(qpos, qpos_orig)
    np.testing.assert_array_equal(mocap_pos, mocap_orig)

    assert multi.last_body_xpos is not None
    np.testing.assert_allclose(
        multi.last_body_xpos[:, 1, :2], expected_offsets, atol=1e-6
    )
    np.testing.assert_allclose(
        multi.last_body_xpos[:, 1, 2], [1.0, 1.1, 1.2, 1.3], atol=1e-6
    )
    assert multi.last_mocap_pos is not None
    np.testing.assert_allclose(
        multi.last_mocap_pos[:, 0, :2],
        expected_offsets + np.array([0.2, -0.1]),
        atol=1e-6,
    )

    expected_marker_xy = expected_offsets + np.column_stack(
        [0.1 * np.arange(4), np.zeros(4)]
    )
    marker_pos = multi.markers.handles["sphere"].batched_positions
    np.testing.assert_allclose(marker_pos[:, :2], expected_marker_xy, atol=1e-6)
    np.testing.assert_allclose(marker_pos[:, 2], 0.8, atol=1e-6)

    env = make_env("myoElbowPose1D6MRandom-v0")
    try:
        elbow_model = env.unwrapped.model
        elbow_multi = ViserMultiEnvScene(
            viser_server, elbow_model, num_envs=4, spacing=2.0
        )
        elbow_qpos = np.tile(elbow_model.qpos0, (4, 1))
        elbow_qpos[:, 0] = [0.2, 0.5, 0.8, 1.1]
        elbow_multi.update(elbow_qpos)

        forearm_id = mujoco.mj_name2id(
            elbow_model, mujoco.mjtObj.mjOBJ_BODY, "r_ulna_radius_hand"
        )
        ref_data = mujoco.MjData(elbow_model)
        expected_forearm_xy = np.empty((4, 2), dtype=np.float64)
        for env_i in range(4):
            ref_data.qpos[:] = elbow_qpos[env_i]
            mujoco.mj_forward(elbow_model, ref_data)
            expected_forearm_xy[env_i] = (
                ref_data.xpos[forearm_id, :2] + elbow_multi.grid_offsets[env_i]
            )
        assert elbow_multi.last_body_xpos is not None
        np.testing.assert_allclose(elbow_multi.last_body_xpos[:, 0, :], 0.0, atol=1e-6)
        np.testing.assert_allclose(
            elbow_multi.last_body_xpos[:, forearm_id, :2],
            expected_forearm_xy,
            atol=1e-6,
        )
    finally:
        env.close()
