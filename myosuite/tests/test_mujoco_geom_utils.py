# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""refresh_geom_derived_fields makes an edited model equal to a recompile."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from myosuite.utils.mujoco_geom_utils import refresh_geom_derived_fields

pytestmark = pytest.mark.tier1

# "obj": two colliding geoms (a multi-node BVH behind the floor's node) and a
# non-colliding marker that carries mass; inertia bounds and user statistics
# are set. "shelf" is a static body.
_XML = """
<mujoco>
  <compiler boundmass="0.05" boundinertia="2e-5" balanceinertia="true"/>
  <statistic extent="5" center="0 -1 1"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="obj" pos="0 0 0.2">
      <freejoint/>
      <geom name="main" type="ellipsoid" size="0.015 0.015 0.045" density="1500"/>
      <geom name="side" type="sphere" size="0.01" pos="0.03 0 0"/>
      <geom name="mark" type="cylinder" size="0.013 0.002" pos="0 0 0.035"
            contype="0" conaffinity="0"/>
    </body>
    <body name="probe" mocap="true" pos="0 0.03 0.2">
      <geom name="tip" type="sphere" size="0.005"/>
    </body>
    <body name="shelf" pos="0.5 0 0.3">
      <geom name="board" type="box" size="0.05 0.05 0.01"/>
    </body>
  </worldbody>
</mujoco>
"""

_EDITS = {
    "main": (mujoco.mjtGeom.mjGEOM_BOX, [0.035, 0.035, 0.035], [0, 0, 0], [1, 0, 0, 0]),
    "side": (
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        [0.008, 0.02, 0.0],
        [0.04, 0.01, 0],
        [np.cos(np.pi / 8), 0, np.sin(np.pi / 8), 0],
    ),
    "mark": (
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        [0.013, 0.002, 0],
        [0, 0, 0.05],
        [1, 0, 0, 0],
    ),
}
_MASS = 1.2


def _edit_compiled(model: mujoco.MjModel) -> None:
    for name, (gtype, size, pos, quat) in _EDITS.items():
        geom = model.geom(name)
        geom.type, geom.size, geom.pos, geom.quat = int(gtype), size, pos, quat
    model.body("obj").mass = _MASS


def _recompiled(spec: mujoco.MjSpec) -> mujoco.MjModel:
    spec = spec.copy()
    geoms = spec.body("obj").geoms
    for geom in geoms:
        geom.type, geom.size, geom.pos, geom.quat = _EDITS[geom.name]
    scale = _MASS / spec.compile().body("obj").mass[0]
    for geom in geoms:
        geom.density *= scale
    return spec.compile()


def _arrays(model: mujoco.MjModel) -> dict[str, np.ndarray]:
    fields = {
        k: v
        for k in dir(model)
        if not k.startswith("_") and isinstance(v := getattr(model, k), np.ndarray)
    }
    for k in ("meaninertia", "meanmass", "meansize", "extent", "center"):
        fields[f"stat.{k}"] = np.asarray(getattr(model.stat, k))
    return fields


def _contacts(model: mujoco.MjModel) -> list[tuple[int, ...]]:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return sorted(tuple(sorted(pair)) for pair in data.contact.geom.tolist())


def test_edited_model_matches_recompiled_model() -> None:
    spec = mujoco.MjSpec.from_string(_XML)
    model = spec.compile()
    _edit_compiled(model)
    refresh_geom_derived_fields(model, spec, [model.body("obj").id])

    expected = _arrays(_recompiled(spec))
    actual = _arrays(model)
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        np.testing.assert_allclose(
            actual[name], value, rtol=1e-12, atol=1e-12, err_msg=name
        )


def test_refresh_restores_culled_contacts() -> None:
    spec = mujoco.MjSpec.from_string(_XML)
    model = spec.compile()
    _edit_compiled(model)
    # The probe sits inside the grown box but outside its stale bounds.
    pair = (model.geom("main").id, model.geom("tip").id)
    assert pair not in _contacts(model)
    refresh_geom_derived_fields(model, spec, [model.body("obj").id])
    assert _contacts(model) == _contacts(_recompiled(spec)) == [pair]


def test_static_body_keeps_inertial_frame() -> None:
    spec = mujoco.MjSpec.from_string(_XML)
    model = spec.compile()
    shelf = model.body("shelf").id
    frame = [model.body_ipos[shelf].copy(), model.body_iquat[shelf].copy()]
    model.geom("board").size = [0.01, 0.05, 0.08]
    refresh_geom_derived_fields(model, spec, [shelf])

    spec.geom("board").size = [0.01, 0.05, 0.08]
    ref = spec.copy().compile()
    board = model.geom("board").id
    np.testing.assert_allclose(model.geom_aabb[board], ref.geom_aabb[board])
    np.testing.assert_allclose(model.geom_rbound[board], ref.geom_rbound[board])
    np.testing.assert_array_equal(model.body_ipos[shelf], frame[0])
    np.testing.assert_array_equal(model.body_iquat[shelf], frame[1])


def test_moving_simple_body_frame_is_not_moved() -> None:
    spec = mujoco.MjSpec.from_string(
        '<mujoco><worldbody><body name="egg"><freejoint/>'
        '<geom name="shell" type="ellipsoid" size="0.01 0.02 0.03"/>'
        "</body></worldbody></mujoco>"
    )
    model = spec.compile()
    egg = model.body("egg").id
    assert model.body_simple[egg]
    # Moving the only geom moves the centre of mass off the body origin.
    model.geom("shell").pos = [0.01, 0, 0]
    aabb = model.geom_aabb.copy()
    with pytest.raises(NotImplementedError, match="simple"):
        refresh_geom_derived_fields(model, spec, [egg])
    np.testing.assert_array_equal(model.geom_aabb, aabb)


@pytest.mark.parametrize("full_inertia", [False, True])
def test_explicit_inertia_scales_without_cumulative_drift(full_inertia: bool) -> None:
    spec = mujoco.MjSpec.from_string(_XML)
    body = spec.body("obj")
    body.explicitinertial = True
    body.mass, body.ipos = 1.0, [0.01, 0, 0]
    if full_inertia:
        body.fullinertia = [1e-3, 2e-3, 2e-3, 1e-4, 0, 0]
    else:
        body.inertia = [1e-3, 2e-3, 2e-3]
    model = spec.compile()
    bid = model.body("obj").id
    for mass in (0.05, 1.2, 0.5, 1.2):
        model.body_mass[bid] = mass
        model.geom("main").size = [0.02, 0.03, 0.04]
        refresh_geom_derived_fields(model, spec, [bid])
        ref_spec = spec.copy()
        ref_body = ref_spec.body("obj")
        ref_body.mass = mass
        if full_inertia:
            ref_body.fullinertia = body.fullinertia * mass
        else:
            ref_body.inertia = body.inertia * mass
        ref_spec.geom("main").size = model.geom("main").size
        ref = ref_spec.compile()
        for name in (
            "geom_aabb",
            "geom_rbound",
            "bvh_aabb",
            "body_inertia",
            "body_ipos",
            "body_iquat",
            "body_invweight0",
            "dof_invweight0",
        ):
            np.testing.assert_allclose(
                getattr(model, name), getattr(ref, name), atol=1e-12, err_msg=name
            )
