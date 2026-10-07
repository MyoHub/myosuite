# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Body scaling (OpenSim Scale Tool rules) on a synthetic leg, MyoLeg and MyoFullBody.

A uniform factor ``s`` must be an exact similarity: in a matched pose (same
angles, translations times ``s``) every site, tendon and muscle length is ``s``
times longer and every muscle force is the same. Per-axis factors must move
everything a body carries with that body's frame.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import mujoco
import numpy as np
import pytest

from myosuite.core.body_scaling import (
    RAJAGOPAL_MYOFULLBODY_SEGMENTS,
    RAJAGOPAL_MYOLEG_SEGMENTS,
    _body_scales,
    _frame_rot,
    _quat,
    _rot,
    read_opensim_scales,
    scale_bodies,
)
from myosuite.core.model_builder import ModelBuilder
from myosuite.core.muscle_conditions import _peak_force
from myosuite.tests.support.optional_deps import require_musclemimic_models

pytestmark = pytest.mark.tier1

LEG_XML = """
<mujoco>
  <compiler eulerseq="xyz"/>
  <asset>
    <mesh name="cube" vertex="-.1 -.1 -.1 .1 -.1 -.1 .1 .1 -.1 -.1 .1 -.1 -.1 -.1 .1 .1 -.1 .1 .1 .1 .1 -.1 .1 .1"/>
  </asset>
  <worldbody>
    <body name="thigh" pos="0 0 1" euler="0 0 30">
      <joint name="hip" axis="0 1 0"/>
      <geom name="thigh_cap" type="capsule" fromto="0 0 0 0 0 -0.4" size="0.05"/>
      <geom name="thigh_mesh" type="mesh" mesh="cube" pos="0 0 -0.2"/>
      <site name="origin" pos="0.03 0.01 -0.1"/>
      <frame name="fr" pos="0.02 0 -0.15" euler="0 45 0">
        <frame name="fr2" pos="0 0.01 0" axisangle="1 0 0 30">
          <site name="side" pos="0.04 0 0"/>
          <geom name="wrap" type="cylinder" size="0.02 0.03" zaxis="1 1 0"/>
        </frame>
        <geom name="pad" type="box" size="0.01 0.02 0.03" xyaxes="0 1 0 -1 0 0"/>
      </frame>
      <body name="shank" pos="0 0 -0.4">
        <joint name="knee" axis="0 1 0" range="-120 0"/>
        <joint name="knee_tx" type="slide" axis="1 0 0" range="-0.01 0.01"/>
        <geom name="shank_cap" type="capsule" pos="0 0 -0.2" size="0.04 0.2"/>
        <site name="insert" pos="0.03 0 -0.1"/>
        <site name="marker" type="capsule" fromto="0 0 -0.05 0 0 -0.15" size="0.005"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <spatial name="ten" springlength="0.3" range="0.1 0.6" limited="true">
      <site site="origin"/>
      <geom geom="wrap" sidesite="side"/>
      <site site="insert"/>
    </spatial>
  </tendon>
  <equality>
    <joint joint1="knee_tx" joint2="knee" polycoef="0 0.01 0.002 0 0"/>
  </equality>
  <actuator>
    <muscle name="vas" tendon="ten" force="-1" scale="200"/>
  </actuator>
  <keyframe>
    <key name="bent" qpos="0.1 -0.5 0.005"/>
  </keyframe>
</mujoco>
"""


def _leg() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_string(LEG_XML)


def _myoleg() -> mujoco.MjSpec:
    import myosuite  # noqa: F401, PLC0415  (registers the envs)
    from myosuite.utils import gym  # noqa: PLC0415
    from myosuite.utils.asset_path_resolver import resolve_model_xml_path  # noqa: PLC0415

    path = resolve_model_xml_path(gym.spec("myoLegWalk-v0").kwargs["model_path"])
    return mujoco.MjSpec.from_file(str(path))


def _fullbody() -> mujoco.MjSpec:
    require_musclemimic_models()
    from ml_collections import config_dict  # noqa: PLC0415

    from myosuite.integrations.musclemimic.fullbody_model import (  # noqa: PLC0415
        build_mimic_fullbody_spec,
        default_mimic_fullbody_config,
    )

    return build_mimic_fullbody_spec(
        config_dict.create(**dict(default_mimic_fullbody_config()))
    )[0]


MODELS: dict[str, Callable[[], mujoco.MjSpec]] = {
    "leg": _leg,
    "myoleg": _myoleg,
    "fullbody": _fullbody,
}


def _matched_pose(
    ref: mujoco.MjModel, scaled: mujoco.MjModel, s: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """A random pose of *ref* and the same pose of the uniformly scaled model."""
    q = ref.qpos0.copy()
    for j in range(ref.njnt):
        adr = ref.jnt_qposadr[j]
        if ref.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            lo, hi = ref.jnt_range[j] if ref.jnt_limited[j] else (-0.5, 0.5)
            q[adr] = rng.uniform(lo, hi)
        elif ref.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE:
            q[adr] += rng.uniform(-0.005, 0.005)
    q_scaled = q.copy()
    for j in range(ref.njnt):
        adr = ref.jnt_qposadr[j]
        if ref.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE:
            q_scaled[adr] *= s
        elif ref.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            q_scaled[adr : adr + 3] *= s
    assert scaled.nq == ref.nq
    return q, q_scaled


def _model_bytes(model: mujoco.MjModel) -> bytes:
    buffer = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, buffer)
    return buffer.tobytes()


@pytest.mark.parametrize("name", ["leg", "myoleg"])
def test_no_scale_leaves_the_model_bit_identical(name: str) -> None:
    spec = MODELS[name]()
    before = _model_bytes(spec.copy().compile())
    scale_bodies(spec, {})
    scale_bodies(spec, {b.name: 1.0 for b in spec.bodies[1:] if b.name})
    assert _model_bytes(spec.compile()) == before


@pytest.mark.parametrize("name", list(MODELS))
def test_resolved_orientations_match_the_compiler(name: str) -> None:
    """Euler, axis-angle, z-axis, xy-axes and nested frames resolve as MuJoCo does."""
    spec = MODELS[name]()
    model = spec.compile()

    def same(q1: np.ndarray, q2: np.ndarray) -> bool:
        return abs(abs(float(np.dot(q1, q2))) - 1.0) < 1e-9

    for body in spec.bodies[1:]:
        if body.name:
            rot = _frame_rot(spec, body.frame) @ _rot(_quat(spec, body))
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, rot.ravel())
            assert same(quat, model.body(body.name).quat), body.name
    for kind, elements, compiled in (
        ("site", spec.sites, model.site_quat),
        ("geom", spec.geoms, model.geom_quat),
    ):
        for i, element in enumerate(elements):
            if (
                kind == "geom" and element.type == mujoco.mjtGeom.mjGEOM_MESH
            ) or not np.isnan(element.fromto[0]):
                continue  # meshes are recentred, fromto overrides the orientation
            rot = _frame_rot(spec, element.frame) @ _rot(_quat(spec, element))
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, rot.ravel())
            assert same(quat, compiled[i]), f"{kind} {element.name or i}"


@pytest.mark.parametrize("name", list(MODELS))
def test_uniform_scale_is_an_exact_similarity(name: str) -> None:
    spec, s = MODELS[name](), 1.1
    ref = spec.copy().compile()
    scaled = scale_bodies(spec, {b.name: s for b in spec.bodies[1:]}).compile()
    for attr in ("nq", "nv", "nu", "nsite", "ntendon", "nbody", "neq"):
        assert getattr(scaled, attr) == getattr(ref, attr), attr

    np.testing.assert_allclose(
        scaled.body_mass.sum(), s**3 * ref.body_mass.sum(), rtol=1e-12
    )
    above_floor = ref.body_inertia > 2 * spec.compiler.boundinertia
    np.testing.assert_allclose(
        np.sort(scaled.body_inertia, axis=1)[np.sort(above_floor, axis=1)],
        s**5 * np.sort(ref.body_inertia, axis=1)[np.sort(above_floor, axis=1)],
        rtol=1e-9,
    )
    muscles = ref.actuator_gaintype == mujoco.mjtGain.mjGAIN_MUSCLE
    np.testing.assert_allclose(
        scaled.actuator_lengthrange[muscles],
        s * ref.actuator_lengthrange[muscles],
        rtol=1e-12,
    )

    rng = np.random.default_rng(0)
    for _ in range(4):
        q, q_scaled = _matched_pose(ref, scaled, s, rng)
        data, scaled_data = mujoco.MjData(ref), mujoco.MjData(scaled)
        data.qpos[:], scaled_data.qpos[:] = q, q_scaled
        data.act[:] = scaled_data.act[:] = 0.5
        mujoco.mj_forward(ref, data)
        mujoco.mj_forward(scaled, scaled_data)
        # The world is not scaled: body sites scale about the root body.
        on_body = ref.site_bodyid > 0
        np.testing.assert_allclose(
            scaled_data.site_xpos[on_body] - scaled_data.xpos[1],
            s * (data.site_xpos[on_body] - data.xpos[1]),
            rtol=1e-9,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            scaled_data.ten_length, s * data.ten_length, rtol=1e-9
        )
        np.testing.assert_allclose(
            scaled_data.actuator_length, s * data.actuator_length, rtol=1e-9
        )
        np.testing.assert_allclose(
            scaled_data.actuator_force[muscles],
            data.actuator_force[muscles],
            rtol=1e-9,
            atol=1e-9,
        )


def test_slide_joints_couplings_and_keyframes_scale_with_the_parent() -> None:
    spec = _leg()
    ref = spec.copy().compile()
    scaled = scale_bodies(
        spec, {"thigh": (1.2, 1.0, 1.0)}
    ).compile()  # knee_tx slides along thigh x
    knee_tx = ref.joint("knee_tx")
    np.testing.assert_allclose(
        scaled.jnt_range[knee_tx.id], 1.2 * ref.jnt_range[knee_tx.id]
    )
    np.testing.assert_allclose(scaled.eq_data[0, :5], 1.2 * ref.eq_data[0, :5])
    adr = knee_tx.qposadr[0]
    np.testing.assert_allclose(scaled.key_qpos[0, adr], 1.2 * ref.key_qpos[0, adr])
    np.testing.assert_allclose(
        scaled.tendon_lengthspring[0] / ref.tendon_lengthspring[0],
        scaled.tendon_range[0] / ref.tendon_range[0],
    )


@pytest.mark.parametrize("name", list(MODELS))
def test_per_axis_scale_moves_everything_with_its_body(name: str) -> None:
    spec = MODELS[name]()
    ref = spec.copy().compile()
    rng = np.random.default_rng(1)
    factors = {b.name: rng.uniform(0.8, 1.25, 3) for b in spec.bodies[1:]}
    scaled = scale_bodies(spec, factors).compile()

    def s(body_id: int) -> np.ndarray:
        return factors.get(ref.body(body_id).name, np.ones(3))

    for i in range(ref.nsite):
        np.testing.assert_allclose(
            scaled.site_pos[i], s(ref.site_bodyid[i]) * ref.site_pos[i], atol=1e-12
        )
    for j in range(ref.njnt):
        np.testing.assert_allclose(
            scaled.jnt_pos[j], s(ref.jnt_bodyid[j]) * ref.jnt_pos[j], atol=1e-12
        )
    for b in range(1, ref.nbody):
        np.testing.assert_allclose(
            scaled.body_pos[b], s(ref.body_parentid[b]) * ref.body_pos[b], atol=1e-12
        )
    # Plain ints: `numpy int in (enum, ...)` is False with MuJoCo's Linux wheels.
    skipped = {
        int(mujoco.mjtGeom.mjGEOM_MESH),
        int(mujoco.mjtGeom.mjGEOM_PLANE),
        int(mujoco.mjtGeom.mjGEOM_HFIELD),
    }
    for g in range(ref.ngeom):
        if int(ref.geom_type[g]) not in skipped:
            np.testing.assert_allclose(
                scaled.geom_pos[g], s(ref.geom_bodyid[g]) * ref.geom_pos[g], atol=1e-12
            )


@pytest.mark.parametrize("name", list(MODELS))
def test_muscles_keep_their_force_in_the_default_pose(name: str) -> None:
    """OpenSim rule: fibre and tendon slack length follow the muscle-tendon length at qpos0."""
    spec = MODELS[name]()
    ref = spec.copy().compile()
    rng = np.random.default_rng(2)
    scaled = scale_bodies(
        spec, {b.name: rng.uniform(0.85, 1.2, 3) for b in spec.bodies[1:]}
    ).compile()
    forces = []
    for model in (ref, scaled):
        data = mujoco.MjData(model)
        data.act[:] = 0.5
        mujoco.mj_forward(model, data)
        forces.append(data.actuator_force.copy())
    np.testing.assert_allclose(forces[1], forces[0], rtol=1e-9, atol=1e-9)


def test_mass_options_and_strength() -> None:
    ref = _leg().compile()
    keep = scale_bodies(_leg(), {"thigh": 1.2}, mass="keep").compile()
    np.testing.assert_allclose(keep.body_mass, ref.body_mass)
    total = scale_bodies(_leg(), {"thigh": 1.2}, total_mass=7.5).compile()
    np.testing.assert_allclose(total.body_mass.sum(), 7.5)
    # force="-1": the peak force comes from the mass, so scaling must freeze it first.
    stronger = scale_bodies(
        _leg(), {"thigh": 1.2, "shank": 1.2}, force_scale=1.5
    ).compile()
    np.testing.assert_allclose(_peak_force(stronger), 1.5 * _peak_force(ref))
    same = scale_bodies(_leg(), {"thigh": 1.2, "shank": 1.2}).compile()
    np.testing.assert_allclose(_peak_force(same), _peak_force(ref))


def test_unsupported_cases_are_refused() -> None:
    with pytest.raises(KeyError, match="nope"):
        scale_bodies(_leg(), {"nope": 1.1})
    with pytest.raises(ValueError, match="positive"):
        scale_bodies(_leg(), {"thigh": (1.0, 0.0, 1.0)})
    shared = _leg()
    shared.body("shank").add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="cube")
    with pytest.raises(NotImplementedError, match="cube"):
        scale_bodies(shared, {"thigh": 1.1})
    connected = _leg()
    connected.add_equality(
        type=mujoco.mjtEq.mjEQ_CONNECT,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        name1="shank",
        name2="world",
    )
    with pytest.raises(NotImplementedError, match="connect/weld"):
        scale_bodies(connected, {"shank": 1.1})


def test_model_builder_scale_bodies_matches_the_function() -> None:
    model, _ = (
        ModelBuilder.from_spec(_leg())
        .scale_bodies({"thigh": (1.1, 1.0, 0.9)}, mass="keep")
        .build()
    )
    direct = scale_bodies(_leg(), {"thigh": (1.1, 1.0, 0.9)}, mass="keep").compile()
    np.testing.assert_array_equal(model.site_pos, direct.site_pos)
    np.testing.assert_array_equal(
        model.actuator_lengthrange, direct.actuator_lengthrange
    )


def test_rajagopal_segments_map_onto_myofullbody() -> None:
    spec = _fullbody()
    scales = _body_scales(
        spec,
        {"torso": 1.1, "hand_r": 0.9, "femur_r": (1.0, 1.2, 1.0), "humerus_l": 1.05},
        RAJAGOPAL_MYOFULLBODY_SEGMENTS,
    )
    np.testing.assert_array_equal(scales["lumbar5"], [1.1] * 3)
    np.testing.assert_array_equal(
        scales["humphant_r"], [1.1] * 3
    )  # shoulder helpers follow the torso
    np.testing.assert_array_equal(scales["head"], [1.1] * 3)
    np.testing.assert_array_equal(scales["femur_r"], [1.0, 1.2, 1.0])
    assert "tibia_r" not in scales  # a segment missing from the scales stays unscaled
    assert "sacrum" not in scales and "pelvis" not in scales
    hand = [
        b.name
        for b in spec.bodies
        if b.name in scales and np.array_equal(scales[b.name], [0.9] * 3)
    ]
    assert len(hand) == 27 and "lunate_r" in hand
    model = scale_bodies(
        spec, {"torso": 1.1, "hand_r": 0.9}, segments=RAJAGOPAL_MYOFULLBODY_SEGMENTS
    ).compile()
    data = mujoco.MjData(model)
    for _ in range(200):
        mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.qpos))


def test_rajagopal_segments_cover_myoleg() -> None:
    spec = _myoleg()
    scales = _body_scales(
        spec, {seg: 1.1 for seg in RAJAGOPAL_MYOLEG_SEGMENTS}, RAJAGOPAL_MYOLEG_SEGMENTS
    )
    assert set(scales) == {b.name for b in spec.bodies[1:]}
    # Arms are not part of MyoLeg: their factors are ignored.
    assert _body_scales(spec, {"humerus_r": 1.3}, RAJAGOPAL_MYOLEG_SEGMENTS) == {}


SCALE_SET = """<OpenSimDocument Version="40000">
  <ScaleSet name="subject">
    <objects>
      <Scale name="s1"><scales> 1.1 1.2 1.3</scales><segment>femur_r</segment><apply>true</apply></Scale>
      <Scale name="s2"><scales> 0.9 0.9 0.9</scales><segment>torso</segment><apply>false</apply></Scale>
    </objects>
  </ScaleSet>
</OpenSimDocument>"""


def _osim(femur: str, tibia: str) -> str:
    def body(name: str, factors: str) -> str:
        return (
            f'<Body name="{name}"><attached_geometry><Mesh name="{name}_geom">'
            f"<scale_factors>{factors}</scale_factors><mesh_file>{name}.vtp</mesh_file>"
            "</Mesh></attached_geometry></Body>"
        )

    return (
        '<OpenSimDocument Version="40000"><Model name="m"><BodySet><objects>'
        f"{body('femur_r', femur)}{body('tibia_r', tibia)}</objects></BodySet></Model></OpenSimDocument>"
    )


def test_read_opensim_scales(tmp_path: Path) -> None:
    scale_set = tmp_path / "scale_set.xml"
    scale_set.write_text(SCALE_SET)
    scales = read_opensim_scales(scale_set)
    assert set(scales) == {"femur_r"}  # apply=false is skipped
    np.testing.assert_allclose(scales["femur_r"], [1.1, 1.2, 1.3])

    scaled, generic = tmp_path / "scaled.osim", tmp_path / "generic.osim"
    scaled.write_text(_osim("1.05 1.05 1.05", "2 2 2"))
    generic.write_text(_osim("1 1 1", "2 2 2"))
    np.testing.assert_allclose(read_opensim_scales(scaled)["femur_r"], [1.05] * 3)
    relative = read_opensim_scales(scaled, reference=generic)
    np.testing.assert_allclose(relative["tibia_r"], [1.0] * 3)
