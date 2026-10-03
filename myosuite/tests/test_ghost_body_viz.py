# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for GhostBodyViz (myosuite/viz/ghost_body_viz.py)."""

from __future__ import annotations

import io
from pathlib import Path

import mujoco
import numpy as np
import pytest

from myosuite.viz.ghost_body_viz import (
    CombinedViz,
    GhostBodyViz,
    _capsule_between,
    make_ghost_viz_fn,
)

# ── helpers ──────────────────────────────────────────────────────────────────

_T = 10
_NBODY = 5

RNG = np.random.default_rng(42)


def _rand_xpos(T: int = _T, nbody: int = _NBODY) -> np.ndarray:
    return RNG.standard_normal((T, nbody, 3)).astype(np.float64)


def _rand_xquat(T: int = _T, nbody: int = _NBODY) -> np.ndarray:
    q = RNG.standard_normal((T, nbody, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return q.astype(np.float64)


def _minimal_ghost() -> GhostBodyViz:
    parent_ids = np.array([0, 0, 1, 2, 3], dtype=np.int32)
    visible = np.array([1, 2, 3, 4], dtype=np.int32)
    return GhostBodyViz(
        xpos=_rand_xpos(),
        xquat=_rand_xquat(),
        parent_ids=parent_ids,
        visible_body_ids=visible,
    )


_SCENE_MODEL = mujoco.MjModel.from_xml_string(
    "<mujoco><worldbody><geom name='f' type='plane' size='1 1 0.01'/></worldbody></mujoco>"
)


def _fake_user_scn(maxgeom: int = 200) -> mujoco.MjvScene:
    scn = mujoco.MjvScene(_SCENE_MODEL, maxgeom=maxgeom)
    scn.ngeom = 0
    return scn


# ── _capsule_between ─────────────────────────────────────────────────────────


def test_capsule_between_centre():
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([0.0, 0.0, 1.0])
    centre, mat9, size = _capsule_between(p0, p1, 0.02)
    np.testing.assert_allclose(centre, [0.0, 0.0, 0.5], atol=1e-9)


def test_capsule_between_size():
    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([0.0, 0.0, 2.0])
    _, _, size = _capsule_between(p0, p1, 0.03)
    assert abs(size[2] - 1.0) < 1e-9, "half_len should be 1.0"
    assert abs(size[0] - 0.03) < 1e-9


def test_capsule_between_degenerate():
    """Degenerate (zero-length) segment should not crash."""
    p = np.array([1.0, 2.0, 3.0])
    centre, mat9, size = _capsule_between(p, p.copy(), 0.02)
    assert centre.shape == (3,)
    assert mat9.shape == (9,)


def test_capsule_between_mat9_orthonormal():
    p0 = np.array([0.1, 0.2, 0.3])
    p1 = np.array([0.5, -0.3, 0.9])
    _, mat9, _ = _capsule_between(p0, p1, 0.02)
    R = mat9.reshape(3, 3)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-7)


# ── GhostBodyViz init ─────────────────────────────────────────────────────────


def test_ghost_init_fields():
    viz = _minimal_ghost()
    assert viz._T == _T
    assert viz._skel_rgba.shape == (4,)
    assert viz._joint_rgba.shape == (4,)


def test_ghost_frame_count():
    viz = _minimal_ghost()
    assert viz._T == _T


# ── draw: geom count ─────────────────────────────────────────────────────────


def test_draw_adds_geoms():
    viz = _minimal_ghost()
    scn = _fake_user_scn()
    assert scn.ngeom == 0
    viz.draw(0, scn)
    assert scn.ngeom > 0


def test_draw_appends_geoms():
    """Ghost draw must append, not reset ngeom."""
    viz = _minimal_ghost()
    scn = _fake_user_scn()
    scn.ngeom = 3  # simulate existing site-marker geoms
    viz.draw(0, scn)
    assert scn.ngeom > 3


def test_draw_respects_maxgeom():
    """draw() must not exceed maxgeom."""
    viz = _minimal_ghost()
    scn = _fake_user_scn(maxgeom=4)
    viz.draw(0, scn)
    assert scn.ngeom <= 4


def test_draw_frame_wraps():
    """Frame index must wrap around clip length."""
    viz = _minimal_ghost()
    scn1 = _fake_user_scn()
    scn2 = _fake_user_scn()
    viz.draw(0, scn1)
    viz.draw(_T, scn2)  # same as frame 0 after modulo
    assert scn1.ngeom == scn2.ngeom


# ── as_viz_fn ─────────────────────────────────────────────────────────────────


def test_as_viz_fn_callable():
    viz = _minimal_ghost()
    fn = viz.as_viz_fn()
    scn = _fake_user_scn()
    # Provide dummy model/data
    fn(0, None, None, scn)
    assert scn.ngeom > 0


# ── CombinedViz ───────────────────────────────────────────────────────────────


def test_combined_viz_calls_both():
    calls = []

    def cb1(si, m, d, scn):
        calls.append("cb1")

    def cb2(si, m, d, scn):
        calls.append("cb2")

    combined = CombinedViz(callbacks=[cb1, cb2])
    combined(0, None, None, _fake_user_scn())
    assert calls == ["cb1", "cb2"]


def test_combined_viz_order():
    """CombinedViz should call callbacks in the given order."""
    order = []
    for i in range(3):
        idx = i

        def cb(si, m, d, scn, _i=idx):
            order.append(_i)

        if i == 0:
            cb0 = cb
        elif i == 1:
            cb1 = cb
        else:
            cb2 = cb
    CombinedViz(callbacks=[cb0, cb1, cb2])(0, None, None, _fake_user_scn())
    assert order == [0, 1, 2]


# ── from_clip_and_model ───────────────────────────────────────────────────────


def _make_npz(T: int = 8, nbody: int = 5) -> bytes:
    buf = io.BytesIO()
    xpos = RNG.standard_normal((T, nbody, 3)).astype(np.float64)
    xquat = RNG.standard_normal((T, nbody, 4))
    xquat /= np.linalg.norm(xquat, axis=-1, keepdims=True)
    np.savez(buf, xpos=xpos, xquat=xquat.astype(np.float64))
    buf.seek(0)
    return buf.read()


def _minimal_mjmodel(nbody: int = 5) -> mujoco.MjModel:
    """Build a tiny chain model with `nbody` bodies."""
    xml_bodies = "\n".join(
        f'<body name="b{i}" pos="0 0 {0.1 * i}"><geom size="0.02"/></body>'
        for i in range(1, nbody)
    )
    xml = f"""
    <mujoco>
      <worldbody>
        <geom name="floor" type="plane" size="1 1 0.01"/>
        {xml_bodies}
      </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


@pytest.fixture()
def clip_npz_path(tmp_path: Path) -> Path:
    p = tmp_path / "clip.npz"
    p.write_bytes(_make_npz(T=8, nbody=5))
    return p


def test_from_clip_and_model(clip_npz_path: Path):
    model = _minimal_mjmodel(nbody=5)
    viz = GhostBodyViz.from_clip_and_model(clip_npz_path, model)
    assert viz._T == 8
    assert viz.xpos.shape[1] <= 5


def test_from_clip_model_draw(clip_npz_path: Path):
    model = _minimal_mjmodel(nbody=5)
    viz = GhostBodyViz.from_clip_and_model(clip_npz_path, model)
    scn = _fake_user_scn()
    viz.draw(0, scn)
    assert scn.ngeom >= 0  # at least no crash


def test_make_ghost_viz_fn(clip_npz_path: Path):
    model = _minimal_mjmodel(nbody=5)
    fn = make_ghost_viz_fn(str(clip_npz_path), model)
    scn = _fake_user_scn()
    fn(0, None, None, scn)
    assert scn.ngeom >= 0
