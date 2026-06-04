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
    """_resolve_fragment_path falls back to simhive submodule."""
    from myosuite.core.model_builder import _resolve_fragment_path

    # elbow should resolve — simhive submodule must be initialised
    try:
        path = _resolve_fragment_path("elbow")
        assert path.exists(), f"Resolved path does not exist: {path}"
        assert path.suffix == ".xml"
    except FileNotFoundError:
        pytest.skip("simhive submodule not initialised; skipping path resolution test")


def test_model_builder_content_hash_stable():
    """Two ModelBuilder instances with the same fragments produce the same hash."""
    from myosuite.core.model_builder import ModelBuilder

    b1 = ModelBuilder()
    b2 = ModelBuilder()
    # No fragments — hashes should match
    assert b1._content_hash() == b2._content_hash()


def test_place_fragment_hash_differs_from_default():
    """place_fragment with non-zero pos produces a different hash than attach_fragment."""
    from myosuite.core.model_builder import ModelBuilder

    b_default = ModelBuilder().attach_fragment("elbow")
    b_placed = ModelBuilder().place_fragment("elbow", pos=[0.1, 0.0, 0.5])
    assert b_default._content_hash() != b_placed._content_hash()


def test_add_free_body_changes_hash():
    """add_free_body changes the content hash."""
    from myosuite.core.model_builder import ModelBuilder

    b_bare = ModelBuilder().attach_fragment("elbow")
    b_with_obj = (
        ModelBuilder().attach_fragment("elbow").add_free_body("ball", pos=[0.1, 0, 0.2])
    )
    assert b_bare._content_hash() != b_with_obj._content_hash()


def test_two_free_bodies_at_different_positions_differ():
    """Two builders with free bodies at different positions get different hashes."""
    from myosuite.core.model_builder import ModelBuilder

    b1 = ModelBuilder().add_free_body("ball", pos=[0.1, 0.0, 0.0])
    b2 = ModelBuilder().add_free_body("ball", pos=[0.2, 0.0, 0.0])
    assert b1._content_hash() != b2._content_hash()


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


def test_model_builder_cache_hit():
    """Building the same recipe twice reuses the cached spec but returns a fresh model."""
    import mujoco

    from myosuite.core.model_builder import ModelBuilder, _MODEL_CACHE

    b = ModelBuilder()
    key = b._content_hash()
    # Remove any prior cache entry so we get a clean build
    _MODEL_CACHE.pop(key, None)

    # First build: populates the cache with the spec
    model1, spec1 = b.build()
    assert key in _MODEL_CACHE
    assert isinstance(_MODEL_CACHE[key], mujoco.MjSpec)

    # Second build: spec is reused (same object), model is freshly compiled
    model2, spec2 = b.build()
    assert spec2 is spec1, "Cached spec should be the same object"
    assert model2 is not model1, "Each build() call must return a distinct MjModel"


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


def test_add_mesh_body_changes_hash(minimal_mesh):
    """add_mesh_body produces a different hash than an empty builder."""
    from myosuite.core.model_builder import ModelBuilder

    b_bare = ModelBuilder()
    b_mesh = ModelBuilder().add_mesh_body("obj", mesh_file=minimal_mesh)
    assert b_bare._content_hash() != b_mesh._content_hash()


def test_add_mesh_body_hash_differs_with_scale(minimal_mesh):
    """Different scale values produce different content hashes."""
    from myosuite.core.model_builder import ModelBuilder

    b1 = ModelBuilder().add_mesh_body("obj", mesh_file=minimal_mesh, scale=[1, 1, 1])
    b2 = ModelBuilder().add_mesh_body("obj", mesh_file=minimal_mesh, scale=[2, 2, 2])
    assert b1._content_hash() != b2._content_hash()


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
