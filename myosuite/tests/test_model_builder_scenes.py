# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""
Scene-composition tests for ModelBuilder.

Each test mirrors the physical setup of a real MyoSuite challenge environment
(Baoding balls, soccer, table tennis, relocate) to verify that place_fragment,
add_free_body, and add_mesh_body produce the expected geometry and DoF counts.

Tests are split into two groups:
- myo_sim-dependent: require the myo_sim pip package (hand/arm/leg fragments).
- Asset-only: only require the mesh/texture files checked into the repo.

These tests require the myo_sim package to be installed.
Skip markers are applied globally so CI without the package stays green.

Parity notes
------------
ModelBuilder composes challenge scenes **partially**, not perfectly.  Each
recipe documents which features are natively replicated and which need
``apply_transform``.  The tests here verify the natively replicated subset.
"""

from __future__ import annotations

import pytest
import numpy as np

pytestmark = [pytest.mark.tier2]


def _hand_fragment_available() -> bool:
    try:
        import myo_sim  # type: ignore[import-untyped]

        if (myo_sim.MODELS_DIR / "hand" / "myohand_r.xml").exists():
            return True
    except (ImportError, AttributeError):
        pass
    return (
        __import__("pathlib").Path(__file__).parents[1]
        / "simhive"
        / "myo_sim"
        / "hand"
        / "myohand.xml"
    ).exists()


_SKIP = pytest.mark.skipif(
    not _hand_fragment_available(),
    reason="myo_sim pip package or local override checkout required",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _nq_for(fragment: str) -> int:
    """Return the expected nq of a bare fragment (no free bodies)."""
    from myosuite.core.model_builder import ModelBuilder

    model, _ = ModelBuilder().attach_fragment(fragment).build()
    return model.nq


# ---------------------------------------------------------------------------
# Baoding-ball style scene: hand + 2 free spheres
# ---------------------------------------------------------------------------


@_SKIP
def test_baoding_scene_body_count():
    """Hand + 2 free spheres: each sphere adds exactly 1 body."""
    import mujoco
    from myosuite.core.model_builder import ModelBuilder

    model_hand, _ = ModelBuilder().attach_fragment("hand").build()

    model_scene, _ = (
        ModelBuilder()
        .attach_fragment("hand")
        .add_free_body(
            "ball1",
            pos=[-0.227, -0.511, 1.452],
            geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
            geom_size=[0.022, 0, 0],
            rgba=[1.0, 0.8, 0.0, 1.0],
        )
        .add_free_body(
            "ball2",
            pos=[-0.256, -0.552, 1.442],
            geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
            geom_size=[0.022, 0, 0],
            rgba=[0.8, 0.6, 0.4, 1.0],
        )
        .build()
    )

    assert model_scene.nbody == model_hand.nbody + 2
    assert model_scene.nq == model_hand.nq + 14  # 7 dof per free sphere


@_SKIP
def test_baoding_scene_ball_positions():
    """Ball bodies are placed at the positions provided to add_free_body."""
    import mujoco
    from myosuite.core.model_builder import ModelBuilder

    pos1 = np.array([-0.227, -0.511, 1.452])
    pos2 = np.array([-0.256, -0.552, 1.442])

    model, _ = (
        ModelBuilder()
        .attach_fragment("hand")
        .add_free_body(
            "ball1",
            pos=pos1,
            geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
            geom_size=[0.022, 0, 0],
            rgba=[1.0, 0.8, 0.0, 1.0],
        )
        .add_free_body(
            "ball2",
            pos=pos2,
            geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
            geom_size=[0.022, 0, 0],
            rgba=[0.8, 0.6, 0.4, 1.0],
        )
        .build()
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    for name, expected in [("ball1", pos1), ("ball2", pos2)]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert bid >= 0, f"Body '{name}' not found"
        np.testing.assert_allclose(
            data.xpos[bid], expected, atol=1e-6, err_msg=f"{name} position mismatch"
        )


@_SKIP
def test_baoding_ball_size_range():
    """Ball radius randomisation (0.018–0.024 m) reflects in compiled geom size."""
    import mujoco
    from myosuite.core.model_builder import ModelBuilder

    for radius in [0.018, 0.022, 0.024]:
        model, _ = (
            ModelBuilder()
            .attach_fragment("hand")
            .add_free_body(
                "ball",
                pos=[0.0, 0.0, 1.4],
                geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
                geom_size=[radius, 0, 0],
                rgba=[1, 0.8, 0, 1],
            )
            .build()
        )
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
        assert gid >= 0
        assert model.geom_size[gid, 0] == pytest.approx(radius)


# ---------------------------------------------------------------------------
# Soccer-style scene: leg + free sphere
# ---------------------------------------------------------------------------


@_SKIP
def test_soccer_scene_body_count():
    """Leg fragment + soccer ball: adds exactly 1 body and 7 dof."""
    import mujoco
    from myosuite.core.model_builder import ModelBuilder

    model_leg, _ = ModelBuilder().attach_fragment("leg").build()

    model_scene, _ = (
        ModelBuilder()
        .attach_fragment("leg")
        .add_free_body(
            "soccer_ball",
            pos=[2.0, 0.0, 0.117],
            geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
            geom_size=[0.117, 0, 0],
            rgba=[0.9, 0.9, 0.9, 1.0],
        )
        .build()
    )

    assert model_scene.nbody == model_leg.nbody + 1
    assert model_scene.nq == model_leg.nq + 7


@_SKIP
def test_soccer_ball_size():
    """Soccer ball radius matches FIFA size-5 spec (0.117 m)."""
    import mujoco
    from myosuite.core.model_builder import ModelBuilder

    FIFA_RADIUS = 0.117
    model, _ = (
        ModelBuilder()
        .attach_fragment("leg")
        .add_free_body(
            "soccer_ball",
            pos=[2.0, 0.0, FIFA_RADIUS],
            geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
            geom_size=[FIFA_RADIUS, 0, 0],
            rgba=[0.9, 0.9, 0.9, 1.0],
        )
        .build()
    )
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "soccer_ball")
    assert model.geom_size[gid, 0] == pytest.approx(FIFA_RADIUS)


# ---------------------------------------------------------------------------
# Table-tennis-style scene: arm + pingpong ball + paddle
# ---------------------------------------------------------------------------


@_SKIP
def test_tabletennis_scene_body_count():
    """Arm + pingpong ball + paddle cylinder: adds 2 bodies, 14 dof."""
    import mujoco
    from myosuite.core.model_builder import ModelBuilder

    model_arm, _ = ModelBuilder().attach_fragment("arm").build()

    model_scene, _ = (
        ModelBuilder()
        .attach_fragment("arm")
        .add_free_body(
            "pingpong",
            pos=[0.95, 0.0, 1.252],
            geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
            geom_size=[0.02, 0, 0],
            rgba=[0.98, 0.70, 0.015, 1.0],
        )
        .add_free_body(
            "paddle",
            pos=[1.8, 0.5, 1.13],
            geom_type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            geom_size=[0.093, 0.020, 0],
            rgba=[0.7, 0.1, 0.1, 1.0],
        )
        .build()
    )

    assert model_scene.nbody == model_arm.nbody + 2
    assert model_scene.nq == model_arm.nq + 14


@_SKIP
def test_tabletennis_geom_types():
    """Pingpong is a sphere; paddle is a cylinder."""
    import mujoco
    from myosuite.core.model_builder import ModelBuilder

    model, _ = (
        ModelBuilder()
        .attach_fragment("arm")
        .add_free_body(
            "pingpong",
            pos=[0.95, 0.0, 1.252],
            geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
            geom_size=[0.02, 0, 0],
            rgba=[0.98, 0.70, 0.015, 1.0],
        )
        .add_free_body(
            "paddle",
            pos=[1.8, 0.5, 1.13],
            geom_type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            geom_size=[0.093, 0.020, 0],
            rgba=[0.7, 0.1, 0.1, 1.0],
        )
        .build()
    )

    ball_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pingpong")
    pad_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "paddle")

    assert model.geom_type[ball_gid] == mujoco.mjtGeom.mjGEOM_SPHERE
    assert model.geom_type[pad_gid] == mujoco.mjtGeom.mjGEOM_CYLINDER

    # Ball radius: 0.02 m (official 40 mm diameter)
    assert model.geom_size[ball_gid, 0] == pytest.approx(0.02)
    # Paddle: radius 0.093 m, half-height 0.020 m
    assert model.geom_size[pad_gid, 0] == pytest.approx(0.093)
    assert model.geom_size[pad_gid, 1] == pytest.approx(0.020)


@_SKIP
def test_tabletennis_object_positions():
    """Pingpong and paddle are placed at the positions used in the real env."""
    import mujoco
    from myosuite.core.model_builder import ModelBuilder

    ball_pos = np.array([0.95, 0.0, 1.252])
    paddle_pos = np.array([1.8, 0.5, 1.13])

    model, _ = (
        ModelBuilder()
        .attach_fragment("arm")
        .add_free_body(
            "pingpong",
            pos=ball_pos,
            geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
            geom_size=[0.02, 0, 0],
            rgba=[0.98, 0.70, 0.015, 1.0],
        )
        .add_free_body(
            "paddle",
            pos=paddle_pos,
            geom_type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            geom_size=[0.093, 0.020, 0],
            rgba=[0.7, 0.1, 0.1, 1.0],
        )
        .build()
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    for name, expected in [("pingpong", ball_pos), ("paddle", paddle_pos)]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert bid >= 0
        np.testing.assert_allclose(data.xpos[bid], expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Cross-cutting: place_fragment positioning
# ---------------------------------------------------------------------------


@_SKIP
def test_place_fragment_elbow_offset():
    """place_fragment sets a non-zero attachment offset visible after mj_forward."""
    from myosuite.core.model_builder import ModelBuilder

    offset = np.array([0.1, 0.0, 0.5])
    model, _ = ModelBuilder().place_fragment("elbow", pos=offset).build()
    assert model.nbody > 1  # worldbody + at least one elbow body


@_SKIP
def test_place_fragment_different_pos_different_model():
    """Two place_fragment calls with different pos produce different models."""
    from myosuite.core.model_builder import ModelBuilder

    m1, _ = ModelBuilder().place_fragment("elbow", pos=[0, 0, 0]).build()
    m2, _ = ModelBuilder().place_fragment("elbow", pos=[0.5, 0, 0]).build()

    # Different cache keys → different model objects
    assert m1 is not m2


# ---------------------------------------------------------------------------
# mass / condim on add_free_body (challenge-env physics params)
# ---------------------------------------------------------------------------


def test_free_body_mass_is_applied():
    """add_free_body mass= sets geom mass that survives compilation."""
    import mujoco
    from myosuite.core.model_builder import ModelBuilder

    MASS = 0.043  # baoding-ball mass
    model, _ = (
        ModelBuilder()
        .add_free_body(
            "ball",
            geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
            geom_size=[0.022, 0, 0],
            rgba=[1, 0.8, 0.31, 1],
            mass=MASS,
        )
        .build()
    )
    # body_mass index 1 is the non-worldbody body
    assert model.body_mass[1] == pytest.approx(MASS)


def test_free_body_condim_default_and_override():
    """condim default is 3; override to 4 is reflected in compiled geom_condim."""
    import mujoco
    from myosuite.core.model_builder import ModelBuilder

    model_default, _ = (
        ModelBuilder()
        .add_free_body(
            "b",
            geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
            geom_size=[0.02, 0, 0],
            rgba=[1, 0, 0, 1],
        )
        .build()
    )
    model_condim4, _ = (
        ModelBuilder()
        .add_free_body(
            "b",
            geom_type=mujoco.mjtGeom.mjGEOM_SPHERE,
            geom_size=[0.02, 0, 0],
            rgba=[1, 0, 0, 1],
            condim=4,
        )
        .build()
    )
    gid = mujoco.mj_name2id(model_default, mujoco.mjtObj.mjOBJ_GEOM, "b")
    assert model_default.geom_condim[gid] == 3
    gid4 = mujoco.mj_name2id(model_condim4, mujoco.mjtObj.mjOBJ_GEOM, "b")
    assert model_condim4.geom_condim[gid4] == 4


# ---------------------------------------------------------------------------
# Challenge recipe tests: what ModelBuilder replicates natively
# ---------------------------------------------------------------------------

_SKIP_SIMHIVE = _SKIP  # alias for clarity

_ASSETS = __import__("pathlib").Path(__file__).parents[1] / "envs" / "myo" / "assets"
_SKIP_ASSETS = pytest.mark.skipif(
    not (_ASSETS / "tabletennis_table.obj").exists(),
    reason="tabletennis mesh assets not present",
)


@_SKIP_SIMHIVE
def test_challenge_baoding_recipe_body_count():
    """challenge_baoding recipe: hand + 2 balls → correct body and DoF counts."""
    import myosuite.core.model_recipes  # noqa: F401 — registers recipes
    from myosuite.core.model_builder import build_from_recipe, ModelBuilder

    model_hand, _ = ModelBuilder().attach_fragment("hand").build()
    try:
        model, _ = build_from_recipe("challenge_baoding")
    except FileNotFoundError:
        pytest.skip("hand fragment not found; myo_sim not installed")

    assert model.nbody == model_hand.nbody + 2
    assert model.nq == model_hand.nq + 14  # 7 dof × 2 balls


@_SKIP_SIMHIVE
def test_challenge_baoding_recipe_ball_mass():
    """challenge_baoding recipe: balls have the correct 43 g mass."""
    import mujoco
    import myosuite.core.model_recipes  # noqa: F401
    from myosuite.core.model_builder import build_from_recipe

    try:
        model, _ = build_from_recipe("challenge_baoding")
    except FileNotFoundError:
        pytest.skip("hand fragment not found; myo_sim not installed")

    for name in ("ball1", "ball2"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert bid >= 0, f"Body '{name}' not found"
        assert model.body_mass[bid] == pytest.approx(0.043, abs=1e-4)


@_SKIP_SIMHIVE
def test_challenge_baoding_recipe_ball_condim():
    """challenge_baoding recipe: balls use condim=4 (torsional friction)."""
    import mujoco
    import myosuite.core.model_recipes  # noqa: F401
    from myosuite.core.model_builder import build_from_recipe

    try:
        model, _ = build_from_recipe("challenge_baoding")
    except FileNotFoundError:
        pytest.skip("hand fragment not found; myo_sim not installed")

    for name in ("ball1", "ball2"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert model.geom_condim[gid] == 4


@_SKIP_ASSETS
def test_challenge_tabletennis_mesh_bodies_no_arm():
    """Table, net, and paddle mesh bodies compile without the arm fragment."""
    import mujoco
    from myosuite.core.model_builder import ModelBuilder

    asset_dir = _ASSETS
    model, _ = (
        ModelBuilder()
        .add_mesh_body(
            "tabletennis_table",
            mesh_file=asset_dir / "tabletennis_table.obj",
            texture_file=asset_dir / "tabletennis.png",
        )
        .add_mesh_body(
            "tabletennis_net",
            mesh_file=asset_dir / "tabletennis_net.obj",
            texture_file=asset_dir / "tabletennis.png",
        )
        .add_mesh_body(
            "paddle",
            mesh_file=asset_dir / "paddle.obj",
            texture_file=asset_dir / "paddle_1k.png",
        )
        .build()
    )
    assert isinstance(model, mujoco.MjModel)
    assert model.nmesh == 3
    # Each add_mesh_body registers its own texture (no cross-body deduplication),
    # so 3 calls → 3 textures (table×2 + paddle×1) even though the table PNG is
    # referenced twice.
    assert model.ntex == 3
    assert model.nmat == 3


@_SKIP_SIMHIVE
@_SKIP_ASSETS
def test_challenge_tabletennis_recipe_body_and_mesh_count():
    """challenge_tabletennis recipe: arm + ball + 3 mesh props."""
    import myosuite.core.model_recipes  # noqa: F401
    from myosuite.core.model_builder import build_from_recipe, ModelBuilder

    model_arm, _ = ModelBuilder().attach_fragment("arm").build()
    try:
        model, _ = build_from_recipe("challenge_tabletennis")
    except FileNotFoundError:
        pytest.skip("arm fragment not found; myo_sim not installed")

    # 1 free ball + 3 mesh bodies
    assert model.nbody == model_arm.nbody + 4
    # 7 dof per freejoint × 4 free bodies
    assert model.nq == model_arm.nq + 28
    # arm fragment brings its own meshes; we added 3 more
    assert model.nmesh == model_arm.nmesh + 3


def _mesh_world_aabb(model, data, geom_name: str):
    """World-frame AABB of a mesh geom's vertices."""
    import mujoco

    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    mid = int(model.geom_dataid[gid])
    adr = int(model.mesh_vertadr[mid])
    nvert = int(model.mesh_vertnum[mid])
    verts = model.mesh_vert[adr : adr + nvert]
    rot = data.geom_xmat[gid].reshape(3, 3)
    world = verts @ rot.T + data.geom_xpos[gid]
    return world.min(axis=0), world.max(axis=0)


@_SKIP_SIMHIVE
@_SKIP_ASSETS
def test_challenge_tabletennis_table_is_horizontal():
    """Visual table/net sit in the collision table's XY plane, not on edge.

    ``challenge_tabletennis`` copies radian ``euler="1.57 0 0"`` from the
    legacy XML, but myo_sim composed specs compile in degrees. Passing the
    radian literals through unchanged stands the table visual on its side.
    """
    import mujoco
    import myosuite.core.model_recipes  # noqa: F401
    from myosuite.core.model_builder import build_from_recipe

    try:
        model, _ = build_from_recipe("challenge_tabletennis")
    except FileNotFoundError:
        pytest.skip("arm fragment not found; myo_sim not installed")

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    tmin, tmax = _mesh_world_aabb(model, data, "mesh_tabletennis_table")
    # Full table ~2.74 m long (X) × 1.52 m wide (Y) × 0.80 m high (Z).
    assert tmax[0] - tmin[0] > 2.5
    assert tmax[1] - tmin[1] > 1.4
    assert tmax[2] - tmin[2] < 1.0
    assert tmin[2] == pytest.approx(0.0, abs=0.05)
    assert tmax[2] == pytest.approx(0.795, abs=0.05)

    nmin, nmax = _mesh_world_aabb(model, data, "mesh_tabletennis_net")
    # Net spans table width (Y) and sits on the table top (Z ≈ 0.73–0.95).
    assert nmax[1] - nmin[1] > 1.5
    assert nmax[0] - nmin[0] < 0.1
    assert nmin[2] > 0.7
    assert nmax[2] < 1.05

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "paddle")
    # euler="-0.3 1.57 0" rad, not ~1.57°.
    assert model.body_quat[bid, 0] == pytest.approx(0.6994, abs=0.02)


@_SKIP_SIMHIVE
def test_challenge_relocate_recipe_object_mass():
    """challenge_relocate recipe: object has the correct 100 g mass."""
    import mujoco
    import myosuite.core.model_recipes  # noqa: F401
    from myosuite.core.model_builder import build_from_recipe

    try:
        model, _ = build_from_recipe("challenge_relocate")
    except FileNotFoundError:
        pytest.skip("arm fragment not found; myo_sim not installed")

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    assert bid >= 0
    assert model.body_mass[bid] == pytest.approx(0.1, abs=1e-4)
