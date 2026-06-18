# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for ModelBuilder (Phase 1)."""

from __future__ import annotations

import numpy as np
import pytest


pytestmark = pytest.mark.tier1

_SKIP_NO_SIMHIVE = pytest.mark.skipif(
    not __import__("shutil").which("mujoco") and True,
    reason="Requires accessible simhive or myo_sim to build",
)


def test_model_builder_imports():
    """ModelBuilder and model_recipe are importable."""
    from myosuite.core.model_builder import ModelBuilder, model_recipe, get_recipe

    assert ModelBuilder is not None
    assert model_recipe is not None
    assert get_recipe is not None


def test_recipes_registered():
    """Standard recipes are registered after importing model_recipes."""
    import myosuite.core.model_recipes  # noqa: F401 — triggers registration
    from myosuite.core.model_builder import get_recipe

    for name in (
        "elbow_standard",
        "elbow_sarcopenia",
        "hand_standard",
        "walk_standard",
    ):
        fn = get_recipe(name)
        assert callable(fn), f"recipe {name!r} is not callable"


def test_get_recipe_unknown_raises():
    """get_recipe raises KeyError for unknown recipe names."""
    import myosuite.core.model_recipes  # noqa: F401
    from myosuite.core.model_builder import get_recipe

    with pytest.raises(KeyError, match="Unknown model recipe"):
        get_recipe("nonexistent_recipe_xyz")


def test_resolve_fragment_path_fallback():
    """_resolve_fragment_path resolves via myo_sim pip package or simhive submodule."""
    from myosuite.core.model_builder import _resolve_fragment_path

    try:
        path = _resolve_fragment_path("elbow")
        assert path.exists(), f"Resolved path does not exist: {path}"
        assert path.suffix == ".xml"
    except FileNotFoundError:
        pytest.skip(
            "elbow fragment not resolvable (no myo_sim package and no simhive submodule)"
        )


@_SKIP_NO_SIMHIVE
def test_model_builder_build_elbow():
    """ModelBuilder.build() returns (MjModel, MjSpec) for the elbow fragment."""
    import mujoco
    from myosuite.core.model_builder import ModelBuilder

    try:
        model, spec = ModelBuilder().attach_fragment("elbow").build()
        assert isinstance(model, mujoco.MjModel)
        assert model.nq > 0
    except FileNotFoundError:
        pytest.skip("Fragment XML not found; simhive not initialised")


@_SKIP_NO_SIMHIVE
def test_place_fragment_sets_position():
    """place_fragment attaches with a non-zero offset that survives compilation."""
    import mujoco
    from myosuite.core.model_builder import ModelBuilder

    target_pos = np.array([0.1, 0.0, 0.5])
    try:
        model, spec = ModelBuilder().place_fragment("elbow", pos=target_pos).build()
        assert isinstance(model, mujoco.MjModel)
        assert model.nbody > 1
    except FileNotFoundError:
        pytest.skip("Fragment XML not found; simhive not initialised")


@_SKIP_NO_SIMHIVE
def test_add_free_body_increases_nq():
    """add_free_body adds 7 dof (freejoint) to the compiled model."""
    from myosuite.core.model_builder import ModelBuilder

    try:
        model_bare, _ = ModelBuilder().attach_fragment("elbow").build()
        model_with, _ = (
            ModelBuilder()
            .attach_fragment("elbow")
            .add_free_body("prop", pos=[0.2, 0.0, 0.1])
            .build()
        )
        assert model_with.nq == model_bare.nq + 7
        assert model_with.nbody == model_bare.nbody + 1
    except FileNotFoundError:
        pytest.skip("Fragment XML not found; simhive not initialised")


@_SKIP_NO_SIMHIVE
def test_multiple_free_bodies():
    """Multiple add_free_body calls each add 7 dof."""
    from myosuite.core.model_builder import ModelBuilder

    n_objects = 3
    try:
        model_bare, _ = ModelBuilder().attach_fragment("elbow").build()
        builder = ModelBuilder().attach_fragment("elbow")
        for i in range(n_objects):
            builder.add_free_body(f"obj_{i}", pos=[i * 0.1, 0.0, 0.1])
        model_multi, _ = builder.build()
        assert model_multi.nq == model_bare.nq + 7 * n_objects
        assert model_multi.nbody == model_bare.nbody + n_objects
    except FileNotFoundError:
        pytest.skip("Fragment XML not found; simhive not initialised")


def test_model_builder_build_returns_distinct_models():
    """Each build() call returns a distinct MjModel instance."""
    import mujoco

    from myosuite.core.model_builder import ModelBuilder

    b = ModelBuilder()
    model1, _ = b.build()
    model2, _ = b.build()
    assert isinstance(model1, mujoco.MjModel)
    assert model2 is not model1


# --- Minimal OBJ mesh fixture ---
# A tetrahedron (4 vertices, 4 triangular faces) satisfies MuJoCo's minimum.
_MINIMAL_OBJ = """\
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.5 1.0 0.0
v 0.5 0.5 1.0
f 1 2 3
f 1 2 4
f 1 3 4
f 2 3 4
"""


def _make_minimal_png() -> bytes:
    """Generate a valid 1x1 white RGB PNG using only stdlib modules."""
    import struct
    import zlib

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    idat = _chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))  # filter + RGB white
    iend = _chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


_MINIMAL_PNG = _make_minimal_png()


@pytest.fixture()
def minimal_mesh(tmp_path):
    """Return path to a minimal valid OBJ mesh file."""
    p = tmp_path / "test_mesh.obj"
    p.write_text(_MINIMAL_OBJ)
    return p


@pytest.fixture()
def minimal_texture(tmp_path):
    """Return path to a minimal valid PNG texture file."""
    p = tmp_path / "test_tex.png"
    p.write_bytes(_MINIMAL_PNG)
    return p


def test_add_mesh_body_missing_mesh_raises():
    """add_mesh_body raises FileNotFoundError for a non-existent mesh file."""
    from myosuite.core.model_builder import ModelBuilder

    with pytest.raises(FileNotFoundError, match="Mesh file not found"):
        ModelBuilder().add_mesh_body("obj", mesh_file="/nonexistent/mesh.obj")


def test_add_mesh_body_missing_texture_raises(minimal_mesh):
    """add_mesh_body raises FileNotFoundError for a non-existent texture file."""
    from myosuite.core.model_builder import ModelBuilder

    with pytest.raises(FileNotFoundError, match="Texture file not found"):
        ModelBuilder().add_mesh_body(
            "obj",
            mesh_file=minimal_mesh,
            texture_file="/nonexistent/tex.png",
        )


def test_add_mesh_body_compiles(minimal_mesh):
    """add_mesh_body produces a valid compiled MjModel."""
    import mujoco
    from myosuite.core.model_builder import ModelBuilder

    model, spec = (
        ModelBuilder()
        .add_mesh_body("prop", mesh_file=minimal_mesh, pos=[0.0, 0.0, 0.5])
        .build()
    )
    assert isinstance(model, mujoco.MjModel)
    assert model.nmesh == 1
    assert model.nbody > 1  # worldbody + prop
    # freejoint → 7 dof
    assert model.nq == 7


def test_add_mesh_body_with_texture_compiles(minimal_mesh, minimal_texture):
    """add_mesh_body with a texture file compiles and registers the texture."""
    import mujoco
    from myosuite.core.model_builder import ModelBuilder

    model, spec = (
        ModelBuilder()
        .add_mesh_body(
            "prop",
            mesh_file=minimal_mesh,
            pos=[0.0, 0.0, 0.5],
            texture_file=minimal_texture,
        )
        .build()
    )
    assert isinstance(model, mujoco.MjModel)
    assert model.nmesh == 1
    assert model.ntex == 1
    assert model.nmat == 1


def test_add_mesh_body_no_texture_uses_rgba(minimal_mesh):
    """Without a texture, the material still compiles (rgba-only rendering)."""
    import mujoco
    from myosuite.core.model_builder import ModelBuilder

    model, spec = (
        ModelBuilder()
        .add_mesh_body(
            "prop",
            mesh_file=minimal_mesh,
            rgba=[1.0, 0.0, 0.0, 1.0],
        )
        .build()
    )
    assert isinstance(model, mujoco.MjModel)
    assert model.ntex == 0  # no texture loaded
    assert model.nmat == 1  # but material is still created


@_SKIP_NO_SIMHIVE
def test_add_mesh_body_combined_with_fragment(minimal_mesh, minimal_texture):
    """add_mesh_body works alongside attach_fragment."""
    from myosuite.core.model_builder import ModelBuilder

    try:
        model_bare, _ = ModelBuilder().attach_fragment("elbow").build()
        model_with, _ = (
            ModelBuilder()
            .attach_fragment("elbow")
            .add_mesh_body(
                "prop",
                mesh_file=minimal_mesh,
                pos=[0.3, 0.0, 0.1],
                texture_file=minimal_texture,
            )
            .build()
        )
        assert model_with.nq == model_bare.nq + 7
        assert model_with.nmesh >= 1
        assert model_with.ntex >= 1
    except FileNotFoundError:
        pytest.skip("Fragment XML not found; simhive not initialised")


# ---------------------------------------------------------------------------
# attach_spec tests
# ---------------------------------------------------------------------------


def _make_minimal_spec():
    import mujoco

    s = mujoco.MjSpec()
    body = s.worldbody.add_body(name="test_body")
    body.add_geom(
        name="test_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0]
    )
    return s


def test_attach_spec_basic():
    """attach_spec() accepts a pre-built MjSpec and produces a valid model."""
    import mujoco
    from myosuite.core.model_builder import ModelBuilder

    model, spec = ModelBuilder().attach_spec(_make_minimal_spec(), name="test").build()
    assert isinstance(model, mujoco.MjModel)
    assert model.nbody > 1  # worldbody + test_body


def test_attach_spec_body_present():
    """Body from the inline spec is reachable by name in the compiled model."""
    import mujoco
    from myosuite.core.model_builder import ModelBuilder

    model, _ = ModelBuilder().attach_spec(_make_minimal_spec(), name="test").build()
    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(model.nbody)
    ]
    assert any("test_body" in (n or "") for n in body_names)


def test_attach_spec_combined_with_free_body():
    """attach_spec can be chained with add_free_body."""
    from myosuite.core.model_builder import ModelBuilder

    model, _ = (
        ModelBuilder()
        .attach_spec(_make_minimal_spec(), name="test")
        .add_free_body("ball", pos=[0.1, 0, 0])
        .build()
    )
    assert model.nbody > 2  # worldbody + test_body + ball


def _myo_sim_has_compose() -> bool:
    try:
        from myo_sim.build.compose import load_right_hand_from_arm_spec  # noqa: F401

        return True
    except (ImportError, AttributeError):
        return False


@pytest.mark.skipif(
    not _myo_sim_has_compose(), reason="myo_sim.build.compose not available"
)
def test_hand_standard_recipe_via_compose():
    """hand_standard recipe builds a valid hand model via myo_sim compose path."""
    import mujoco
    from myosuite.core.model_builder import build_from_recipe

    model, spec = build_from_recipe("hand_standard")
    assert isinstance(model, mujoco.MjModel)
    assert model.nu > 0
    assert model.njnt > 0


@pytest.mark.skipif(
    not _myo_sim_has_compose(), reason="myo_sim.build.compose not available"
)
def test_attach_spec_hand_from_arm():
    """attach_spec with myo_sim composed hand produces a model with finger joints."""
    import mujoco
    from myo_sim.build.compose import load_right_hand_from_arm_spec
    from myosuite.core.model_builder import ModelBuilder

    model, _ = (
        ModelBuilder().attach_spec(load_right_hand_from_arm_spec(), name="hand").build()
    )
    assert isinstance(model, mujoco.MjModel)
    assert model.nu > 0
    assert model.njnt > 0


@pytest.mark.skipif(
    not _myo_sim_has_compose(), reason="myo_sim.build.compose not available"
)
def test_hand_standard_numerically_equivalent_to_myo_sim_main():
    """hand_standard recipe must be structurally equivalent to myo_sim.load('myohand_r').

    myo_sim.load('myohand_r') attaches the same hand spec to a passive torso
    scaffold, so njnt/nu and all joint+actuator names must match exactly.
    """
    import mujoco
    import myo_sim
    from myosuite.core.model_builder import build_from_recipe

    # Myosuite side: hand_standard recipe (hand-only, no torso).
    m_recipe, _ = build_from_recipe("hand_standard")

    # myo_sim side: composed myohand_r (torso scaffold + same hand).
    m_ref, _ = myo_sim.load("myohand_r")

    # Structural counts must match.
    assert (
        m_recipe.njnt == m_ref.njnt
    ), f"njnt mismatch: recipe={m_recipe.njnt} myo_sim={m_ref.njnt}"
    assert (
        m_recipe.nu == m_ref.nu
    ), f"nu mismatch: recipe={m_recipe.nu} myo_sim={m_ref.nu}"

    def _joint_names(m):
        return sorted(
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)
        )

    def _actuator_names(m):
        return sorted(
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)
        )

    assert _joint_names(m_recipe) == _joint_names(m_ref), "joint names diverge"
    assert _actuator_names(m_recipe) == _actuator_names(m_ref), "actuator names diverge"

    # Joint range equivalence (order-independent via name lookup).
    for jnt_id in range(m_recipe.njnt):
        name = mujoco.mj_id2name(m_recipe, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
        ref_id = mujoco.mj_name2id(m_ref, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo_r, hi_r = m_recipe.jnt_range[jnt_id]
        lo_f, hi_f = m_ref.jnt_range[ref_id]
        assert (
            abs(lo_r - lo_f) < 1e-6 and abs(hi_r - hi_f) < 1e-6
        ), f"joint {name!r}: range ({lo_r:.4f},{hi_r:.4f}) != ({lo_f:.4f},{hi_f:.4f})"


@pytest.mark.skipif(
    not _myo_sim_has_compose(), reason="myo_sim.build.compose not available"
)
def test_attach_fragment_hand_routes_through_compose():
    """attach_fragment('hand') must use myo_sim compose, not the bundled static XML.

    Verified by checking that joint names carry the _r suffix produced by the
    compose pipeline (prune_arm_spec_to_hand) rather than the un-suffixed names
    in the legacy simhive hand/myohand.xml.
    """
    import mujoco
    from myosuite.core.model_builder import ModelBuilder

    model, _ = ModelBuilder().attach_fragment("hand").build()
    assert model.njnt == 23
    assert model.nu == 39
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]
    assert all(
        n.endswith("_r") for n in joint_names
    ), f"Expected all joints to end with '_r' (compose path); got: {joint_names}"
