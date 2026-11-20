# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for ``myosuite.viz.viser_markers``."""

from __future__ import annotations

from collections.abc import Iterator
import json
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

from mjviser import ViserMujocoScene
import mujoco
import numpy as np
import pytest
import viser

from myosuite import make_env
from myosuite.core.trajectory_io import MotionClip
from myosuite.viz.site_marker_viz import SiteMarkerViz
from myosuite.viz.viser_markers import (
    ViserMarkers,
    ViserMultiEnvScene,
    apply_viser_floor,
    apply_viser_sky,
    square_grid_offsets,
)

pytestmark = pytest.mark.tier1

_REPO = Path(__file__).resolve().parents[2]

_STYLED_FREEJOINT_XML = """
<mujoco>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="0.4 0.6 0.8" rgb2="0.1 0.2 0.3" width="32" height="32"/>
    <texture name="grid" type="2d" builtin="checker"
             rgb1="0.2 0.3 0.4" rgb2="0.4 0.5 0.6" width="32" height="32"/>
    <material name="grid_mat" texture="grid"/>
    <mesh name="tet" vertex="0 0 0  0.1 0 0  0 0.1 0  0 0 0.1"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1" material="grid_mat"/>
    <body name="root" pos="0 0 1">
      <freejoint/>
      <geom type="sphere" size="0.08"/>
      <site name="tip" pos="0 0 0.1" size="0.02"/>
    </body>
    <body name="target" mocap="true" pos="0.2 -0.1 0.5">
      <geom type="sphere" size="0.04" rgba="1 0 0 0.8"/>
    </body>
  </worldbody>
</mujoco>
"""

_PLAIN_HINGE_XML = """
<mujoco>
  <worldbody>
    <body name="arm" pos="0 0 0.5">
      <joint name="elbow" type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.04 0.15"/>
    </body>
    <body name="goal" mocap="true" pos="0.3 0.1 0.5">
      <geom type="sphere" size="0.03"/>
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


def test_viser_markers_module_keeps_optional_deps_lazy() -> None:
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


def test_viser_markers_primitives_arrows_and_meshes(
    viser_server: viser.ViserServer,
) -> None:
    model = mujoco.MjModel.from_xml_string(_STYLED_FREEJOINT_XML)
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

    # Arrow #0 pointing along +X (90 deg about +Y): explicit head radius size[1]=0.05.
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
    # Arrow #1 along +Z with size[1]=0 -> fallback head radius = 2 * size[0] = 0.06.
    _init_geom(
        user_scn,
        mujoco.mjtGeom.mjGEOM_ARROW1,
        (0.03, 0.0, 1.0),
        (0.0, 0.0, 0.5),
        rgba=(0.2, 0.8, 0.4, 1.0),
    )
    # Plane and alpha=0 geoms must be skipped.
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
        rgba=(1.0, 1.0, 1.0, 0.0),
    )

    markers.update(user_scn)

    expected_keys = {
        "sphere",
        "box",
        "capsule",
        "cylinder",
        "ellipsoid",
        "mesh_0",
        "arrow_shaft",
        "arrow_head",
    }
    assert set(markers.handles.keys()) == expected_keys
    for handle in markers.handles.values():
        assert handle.visible is True

    # Verify sphere ignored the alpha=0 entry.
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

    # Verify arrow shaft (80%) and cone head (20% along mat[:, 2]) geometry.
    shaft_h = markers.handles["arrow_shaft"]
    head_h = markers.handles["arrow_head"]
    assert shaft_h.batched_positions.shape == (2, 3)
    assert head_h.batched_positions.shape == (2, 3)
    np.testing.assert_allclose(
        shaft_h.batched_scales,
        [[0.02, 0.02, 0.4], [0.03, 0.03, 0.8]],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        head_h.batched_positions,
        [[1.4, 2.0, 3.0], [0.0, 0.0, 1.3]],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        head_h.batched_scales,
        [[0.05, 0.05, 0.1], [0.06, 0.06, 0.2]],
        atol=1e-6,
    )

    markers.clear()
    assert markers.handles == {}


def test_viser_markers_handle_reuse_recreation_and_visibility(
    viser_server: viser.ViserServer,
) -> None:
    model = mujoco.MjModel.from_xml_string(_STYLED_FREEJOINT_XML)
    scene = ViserMujocoScene(viser_server, model, num_envs=1)
    markers = ViserMarkers(scene, model)
    scn = mujoco.MjvScene(model, maxgeom=16)

    # Frame 0: 2 spheres + 1 box.
    _init_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, (0.04, 0, 0), (0.0, 0.0, 0.1))
    _init_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, (0.04, 0, 0), (0.2, 0.0, 0.1))
    _init_geom(scn, mujoco.mjtGeom.mjGEOM_BOX, (0.1, 0.1, 0.1), (0.0, 0.3, 0.2))
    markers.update(scn)
    sphere_h0 = markers.handles["sphere"]
    box_h0 = markers.handles["box"]

    # Frame 1: still 2 spheres (moved), 0 boxes -> sphere handle reused, box hidden.
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

    # Frame 2: 3 spheres -> instance count changed, so handle must be recreated.
    scn.ngeom = 0
    for k in range(3):
        _init_geom(
            scn, mujoco.mjtGeom.mjGEOM_SPHERE, (0.05, 0, 0), (float(k), 0.0, 0.0)
        )
    markers.update(scn)
    sphere_h2 = markers.handles["sphere"]
    assert sphere_h2 is not sphere_h0
    assert sphere_h2.batched_positions.shape == (3, 3)

    # Frame 3: alpha=0 on all geoms -> hidden.
    for k in range(3):
        scn.geoms[k].rgba[3] = 0.0
    markers.update(scn)
    assert sphere_h2.visible is False


def test_viser_markers_scene_offset_fallback_without_fixed_frame(
    viser_server: viser.ViserServer,
) -> None:
    model = mujoco.MjModel.from_xml_string(_PLAIN_HINGE_XML)
    scene_stub = SimpleNamespace(
        server=viser_server,
        _scene_offset=np.array([-1.5, 2.5, -0.5], dtype=np.float64),
    )
    markers = ViserMarkers(scene_stub, model, root_name="/custom_markers")
    assert markers.root_path == "/custom_markers"

    scn = mujoco.MjvScene(model, maxgeom=4)
    _init_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, (0.05, 0, 0), (1.0, 0.5, 1.0))
    markers.update(scn)
    np.testing.assert_allclose(
        markers.handles["sphere"].batched_positions,
        [[-0.5, 3.0, 0.5]],
        atol=1e-6,
    )


def test_viser_markers_site_marker_viz_callback_integration(
    viser_server: viser.ViserServer,
) -> None:
    model = mujoco.MjModel.from_xml_string(_STYLED_FREEJOINT_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    scene = ViserMujocoScene(viser_server, model, num_envs=1)
    markers = ViserMarkers(scene, model)

    site_xpos = np.array(
        [
            [[0.1, 0.2, 0.9], [-0.1, 0.3, 1.1]],
            [[0.4, -0.2, 0.8], [0.2, 0.5, 1.2]],
        ],
        dtype=np.float64,
    )
    clip = MotionClip(
        qpos=np.tile(model.qpos0, (2, 1)),
        qvel=None,
        site_xpos=site_xpos,
        site_names=["s0", "s1"],
    )
    viz_fn = SiteMarkerViz(
        clip=clip,
        site_names=("s0", "s1"),
        model=model,
        rgba=(1.0, 0.4, 0.0, 0.65),
        sphere_radius=0.035,
    ).as_viz_fn()

    scn0 = markers.update_from_callback(0, data, viz_fn)
    assert scn0.ngeom == 2
    h0 = markers.handles["sphere"]
    np.testing.assert_allclose(h0.batched_positions, site_xpos[0], atol=1e-6)
    np.testing.assert_allclose(
        h0._impl.props.batched_opacities, [0.65, 0.65], atol=1e-5
    )

    scn1 = markers.update_from_callback(1, data, viz_fn)
    assert scn1 is scn0
    assert markers.handles["sphere"] is h0
    np.testing.assert_allclose(h0.batched_positions, site_xpos[1], atol=1e-6)


def test_apply_viser_sky_and_floor(viser_server: viser.ViserServer) -> None:
    styled_model = mujoco.MjModel.from_xml_string(_STYLED_FREEJOINT_XML)
    plain_model = mujoco.MjModel.from_xml_string(_PLAIN_HINGE_XML)
    ViserMujocoScene(viser_server, styled_model, num_envs=1)

    # Model-derived skybox and floor colors.
    sky_colors = apply_viser_sky(viser_server, styled_model, sky="model")
    assert sky_colors is not None
    top_rgb, bot_rgb = sky_colors
    assert top_rgb[2] > bot_rgb[2] and top_rgb[0] > bot_rgb[0]

    assert apply_viser_sky(viser_server, plain_model, sky="model") is None

    explicit_sky = ((12, 34, 56), (200, 210, 220))
    assert apply_viser_sky(viser_server, plain_model, sky=explicit_sky) == explicit_sky
    assert apply_viser_sky(viser_server, styled_model, sky=None) is None
    with pytest.raises(ValueError, match="Unsupported sky"):
        apply_viser_sky(viser_server, styled_model, sky="mars")

    # Model-derived checkerboard floor color vs explicit RGB vs plain model.
    floor_rgb = apply_viser_floor(viser_server, styled_model, floor_color="model")
    assert floor_rgb is not None
    grid_handle = viser_server.scene._handle_from_node_name["/fixed_bodies/world/floor"]
    assert grid_handle._impl.props.plane_color == floor_rgb
    assert grid_handle.visible is True

    assert apply_viser_floor(viser_server, plain_model, floor_color="model") is None

    custom_floor = (30, 60, 90)
    assert (
        apply_viser_floor(viser_server, styled_model, floor_color=custom_floor)
        == custom_floor
    )
    assert grid_handle._impl.props.plane_color == custom_floor

    assert apply_viser_floor(viser_server, styled_model, floor_color=None) is None
    assert grid_handle.visible is False
    with pytest.raises(ValueError, match="Unsupported floor_color"):
        apply_viser_floor(viser_server, styled_model, floor_color="lava")


def test_viser_multi_env_scene_freejoint_and_markers(
    viser_server: viser.ViserServer,
) -> None:
    model = mujoco.MjModel.from_xml_string(_STYLED_FREEJOINT_XML)
    multi = ViserMultiEnvScene(viser_server, model, num_envs=4, spacing=3.0)
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

    # Caller inputs must not be mutated in place.
    np.testing.assert_array_equal(qpos, qpos_orig)
    np.testing.assert_array_equal(mocap_pos, mocap_orig)

    # Root body (id 1) and mocap target (id 2) must sit on the 2x2 grid.
    last_xpos = multi.scene._last_body_xpos
    assert last_xpos is not None
    np.testing.assert_allclose(last_xpos[:, 1, :2], expected_offsets, atol=1e-6)
    np.testing.assert_allclose(last_xpos[:, 1, 2], [1.0, 1.1, 1.2, 1.3], atol=1e-6)
    assert multi.scene._last_mocap_pos is not None
    np.testing.assert_allclose(
        multi.scene._last_mocap_pos[:, 0, :2],
        expected_offsets + np.array([0.2, -0.1]),
        atol=1e-6,
    )

    # Per-env markers must be shifted by the exact same grid offsets.
    expected_marker_xy = expected_offsets + np.column_stack(
        [0.1 * np.arange(4), np.zeros(4)]
    )
    marker_pos = multi.markers.handles["sphere"].batched_positions
    np.testing.assert_allclose(marker_pos[:, :2], expected_marker_xy, atol=1e-6)
    np.testing.assert_allclose(marker_pos[:, 2], 0.8, atol=1e-6)


def test_viser_multi_env_scene_fixed_base_myosuite_env(
    viser_server: viser.ViserServer,
) -> None:
    env = make_env("myoElbowPose1D6MRandom-v0")
    try:
        model = env.unwrapped.model
        assert not any(
            int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)
            for j in range(model.njnt)
        )
        multi = ViserMultiEnvScene(viser_server, model, num_envs=4, spacing=2.0)

        qpos = np.tile(model.qpos0, (4, 1))
        qpos[:, 0] = [0.2, 0.5, 0.8, 1.1]
        per_env_scenes = []
        for env_i in range(4):
            scn = mujoco.MjvScene(model, maxgeom=4)
            _init_geom(
                scn,
                mujoco.mjtGeom.mjGEOM_BOX,
                (0.02, 0.02, 0.02),
                (0.0, 0.0, 0.1 * (env_i + 1)),
            )
            per_env_scenes.append(scn)

        multi.update(qpos, user_scenes=per_env_scenes)

        # Reference single-env FK for comparison.
        ref_data = mujoco.MjData(model)
        expected_forearm_xy = np.empty((4, 2), dtype=np.float64)
        forearm_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "r_ulna_radius_hand"
        )
        assert forearm_id > 0
        for env_i in range(4):
            ref_data.qpos[:] = qpos[env_i]
            mujoco.mj_forward(model, ref_data)
            expected_forearm_xy[env_i] = (
                ref_data.xpos[forearm_id, :2] + multi.grid_offsets[env_i]
            )

        assert multi.scene._last_body_xpos is not None
        # World body (id 0) stays unshifted at origin; non-world bodies shift per env.
        np.testing.assert_allclose(multi.scene._last_body_xpos[:, 0, :], 0.0, atol=1e-6)
        np.testing.assert_allclose(
            multi.scene._last_body_xpos[:, forearm_id, :2],
            expected_forearm_xy,
            atol=1e-6,
        )

        # Dynamic batched mesh handle in ViserMujocoScene reflects all 4 grid cells.
        assert len(multi.scene._mesh_groups) >= 1
        np.testing.assert_allclose(
            multi.scene._mesh_groups[0].handle.batched_positions[:, :2],
            expected_forearm_xy,
            atol=1e-6,
        )

        # Per-env box markers match the grid offsets too.
        box_pos = multi.markers.handles["box"].batched_positions
        np.testing.assert_allclose(box_pos[:, :2], multi.grid_offsets, atol=1e-6)
        np.testing.assert_allclose(box_pos[:, 2], [0.1, 0.2, 0.3, 0.4], atol=1e-6)
    finally:
        env.close()
