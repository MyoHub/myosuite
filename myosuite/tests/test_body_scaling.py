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

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import mujoco
import numpy as np
import pytest

from myosuite.core.body_scaling import (
    _MYOFULLBODY_SEGMENTS,
    _MYOLEG_SEGMENTS,
    _body_scales,
    _frame_rot,
    _geom_bottom,
    _model_digest,
    _quat,
    _read_opensim_scales,
    _rot,
    _scale_spec,
    _segments_for,
    scale_model,
)
from myosuite.core.muscle_conditions import _peak_force
from myosuite.core.trajectory_io import MotionClip, expand_motion_clip_to_model
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

# A free-floating two-body subject: a root and a foot with a sole box.
STANDING_XML = """<mujoco><worldbody>
<geom name="ground" type="plane" size="2 2 .01"/>
<body name="root" pos="0 0 1"><freejoint name="root_free"/>
<geom type="sphere" size=".1" mass="2"/>
<body name="foot" pos="0 0 -1"><joint name="ankle" axis="0 1 0"/>
<geom name="sole" type="box" size=".1 .2 .05" mass="1"/>
</body></body></worldbody>
<keyframe><key name="stand" qpos="0 0 1 1 0 0 0 0"/></keyframe>
</mujoco>"""


def _leg() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_string(LEG_XML)


def _simple_body() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_string(
        """<mujoco><worldbody><body name="thigh"><joint/>
          <geom type="sphere" size="0.1"/>
        </body></worldbody></mujoco>"""
    )


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


def _scale_set(path: Path, scales: dict[str, tuple[float, float, float]]) -> Path:
    """Write an OpenSim ScaleSet with one ``<Scale>`` per segment."""
    rows = "".join(
        f"<Scale name='{name}'><scales>{' '.join(map(str, values))}</scales>"
        f"<segment>{name}</segment><apply>true</apply></Scale>"
        for name, values in scales.items()
    )
    path.write_text(
        "<OpenSimDocument Version='40000'><ScaleSet name='subject'><objects>"
        f"{rows}</objects></ScaleSet></OpenSimDocument>"
    )
    return path


@pytest.mark.parametrize(
    "total,expected",
    [(None, [80, 20]), (50, [40, 10])],
)
def test_compiler_total_mass_is_resolved_before_scaling(
    total: float | None,
    expected: list[float],
) -> None:
    # One kinematic tree, so total_mass rescales both bodies.
    spec = mujoco.MjSpec.from_string("""<mujoco><compiler settotalmass="30"/>
        <worldbody><body name="a"><geom size=".1" mass="1"/>
        <body name="b"><geom size=".1" mass="2"/></body></body></worldbody></mujoco>""")
    model = scale_model(spec, {"a": 2}, total_mass=total)
    np.testing.assert_allclose(model.body_mass[1:], expected)


def test_partial_keyframe_scales_slide_values() -> None:
    spec = mujoco.MjSpec.from_string("""<mujoco><worldbody>
        <body name="parent"><geom size=".1"/>
        <body name="child"><joint name="slide" type="slide" axis="1 0 0"/>
        <joint name="hinge"/><geom size=".1"/></body></body></worldbody>
        <keyframe><key qpos=".2"/></keyframe></mujoco>""")
    np.testing.assert_allclose(scale_model(spec, {"parent": 2}).key_qpos, [[0.4, 0]])


def test_unnamed_slide_keyframes_use_coordinate_addresses() -> None:
    spec = mujoco.MjSpec.from_string("""<mujoco><worldbody>
        <body name="parent"><geom size=".1"/>
        <body name="child"><joint type="slide" axis="1 0 0"/>
        <joint type="slide" axis="0 1 0"/><geom size=".1"/></body></body></worldbody>
        <keyframe><key qpos=".2 .3"/></keyframe></mujoco>""")
    model = scale_model(spec, {"parent": [2, 3, 1]})
    np.testing.assert_allclose(model.key_qpos, [[0.4, 0.9]])


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
    assert _model_bytes(scale_model(spec, {})) == before
    ones = {b.name: 1.0 for b in spec.bodies[1:] if b.name}
    assert _model_bytes(scale_model(spec, ones)) == before
    assert _model_bytes(spec.compile()) == before


@pytest.mark.parametrize("name", ["leg", "myoleg"])
def test_the_callers_spec_is_not_changed(name: str) -> None:
    spec = MODELS[name]()
    before = _model_bytes(spec.copy().compile())
    scale_model(spec, {b.name: 1.1 for b in spec.bodies[1:] if b.name}, total_mass=50)
    assert _model_bytes(spec.copy().compile()) == before


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
    scaled = scale_model(spec, {b.name: s for b in spec.bodies[1:]})
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
    # knee_tx slides along thigh x.
    scaled = scale_model(spec, {"thigh": (1.2, 1.0, 1.0)})
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


def test_slide_joint_axis_resolves_nested_frames() -> None:
    spec = mujoco.MjSpec.from_string(
        """<mujoco><worldbody><body name="parent">
          <geom type="sphere" size="0.1"/>
          <body name="child" euler="10 25 -15">
            <frame euler="0 0 35"><frame euler="15 -20 45">
              <joint name="slide" type="slide" axis="1 0 0"
                range="-0.1 0.2" ref="0.01" springref="0.02"/>
            </frame></frame>
            <geom type="sphere" size="0.1"/>
          </body>
        </body></worldbody></mujoco>"""
    )
    ref = spec.copy().compile()
    factors = np.array([1.7, 1.1, 0.8])
    axis = _rot(ref.body("child").quat) @ ref.joint("slide").axis
    expected = np.linalg.norm(factors * axis)
    scaled = scale_model(spec, {"parent": factors})
    np.testing.assert_allclose(scaled.jnt_range, ref.jnt_range * expected)
    np.testing.assert_allclose(scaled.qpos0, ref.qpos0 * expected)
    np.testing.assert_allclose(scaled.qpos_spring, ref.qpos_spring * expected)


@pytest.mark.parametrize("euler", ["0 0 0", "0 0 45", "20 35 10"])
def test_sphere_radius_uses_opensim_mean_independent_of_orientation(euler: str) -> None:
    spec = mujoco.MjSpec.from_string(
        f"""<mujoco><worldbody><body name="body"><joint/>
          <geom name="sphere" type="sphere" size="0.1" euler="{euler}"/>
        </body></worldbody></mujoco>"""
    )
    factors = np.array([2.0, 1.0, 0.8])
    model = scale_model(spec, {"body": factors})
    np.testing.assert_allclose(model.geom("sphere").size[0], 0.1 * factors.mean())


@pytest.mark.parametrize("kind", ["cylinder", "capsule"])
@pytest.mark.parametrize(
    "placement",
    ['euler="20 35 10" size="0.1 0.2"', 'fromto="0 0 0 0.2 0.3 0.4" size="0.1"'],
)
def test_cylinder_radius_uses_opensim_transverse_mean(
    kind: str, placement: str
) -> None:
    spec = mujoco.MjSpec.from_string(
        f"""<mujoco><worldbody><body name="body"><joint/>
          <frame euler="10 25 30">
            <geom name="geom" type="{kind}" {placement}/>
          </frame>
        </body></worldbody></mujoco>"""
    )
    ref = spec.copy().compile()
    factors = np.array([2.0, 1.0, 0.8])
    local_factors = np.linalg.norm(
        factors[:, None] * _rot(ref.geom("geom").quat), axis=0
    )
    model = scale_model(spec, {"body": factors})
    np.testing.assert_allclose(
        model.geom("geom").size[0], ref.geom("geom").size[0] * local_factors[:2].mean()
    )
    np.testing.assert_allclose(
        model.geom("geom").size[1], ref.geom("geom").size[1] * local_factors[2]
    )


@pytest.mark.parametrize("name", list(MODELS))
def test_per_axis_scale_moves_everything_with_its_body(name: str) -> None:
    spec = MODELS[name]()
    ref = spec.copy().compile()
    rng = np.random.default_rng(1)
    factors = {b.name: rng.uniform(0.8, 1.25, 3) for b in spec.bodies[1:]}
    scaled = scale_model(spec, factors)

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
        # A free root only moves up or down, to keep its ground clearance.
        free = ref.body_jntnum[b] > 0 and int(ref.jnt_type[ref.body_jntadr[b]]) == int(
            mujoco.mjtJoint.mjJNT_FREE
        )
        axes = slice(0, 2) if free else slice(0, 3)
        np.testing.assert_allclose(
            scaled.body_pos[b, axes],
            (s(ref.body_parentid[b]) * ref.body_pos[b])[axes],
            atol=1e-12,
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


def _coupled_leg() -> mujoco.MjSpec:
    """The leg with a knee coupling that ``qpos0`` violates, like MyoLeg's patella."""
    spec = _leg()
    spec.equalities[0].data[0] = 0.02  # knee_tx = 2 cm + ... at knee = 0
    return spec


def _assembled(model: mujoco.MjModel) -> np.ndarray:
    """``qpos0`` with each joint coupling ``q1 - q1_0 = sum c_k (q2 - q2_0)^k`` met."""
    q = model.qpos0.copy()
    for i in range(model.neq):
        if model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT and model.eq_active0[i]:
            first = model.jnt_qposadr[model.eq_obj1id[i]]
            second = model.jnt_qposadr[model.eq_obj2id[i]]
            offset = q[second] - model.qpos0[second]
            q[first] = model.qpos0[first] + np.polynomial.polynomial.polyval(
                offset, model.eq_data[i, :5]
            )
    return q


@pytest.mark.parametrize("name", [*MODELS, "coupled_leg"])
def test_muscles_keep_their_force_in_the_default_pose(name: str) -> None:
    """OpenSim rule: fibre and tendon slack length follow the muscle-tendon length
    in the default pose, assembled so that every joint coupling holds."""
    spec = _coupled_leg() if name == "coupled_leg" else MODELS[name]()
    ref = spec.copy().compile()
    rng = np.random.default_rng(2)
    scaled = scale_model(
        spec, {b.name: rng.uniform(0.85, 1.2, 3) for b in spec.bodies[1:]}
    )
    forces = []
    for model in (ref, scaled):
        data = mujoco.MjData(model)
        data.qpos[:] = _assembled(model)
        data.act[:] = 0.5
        mujoco.mj_forward(model, data)
        equality = data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        np.testing.assert_allclose(data.efc_pos[equality], 0, atol=1e-12)
        forces.append(data.actuator_force.copy())
    np.testing.assert_allclose(forces[1], forces[0], rtol=1e-9, atol=1e-9)


def test_myoleg_quadriceps_lengths_are_measured_with_the_patella_assembled() -> None:
    """MyoLeg's qpos0 leaves the patella 5 cm off its knee coupling, which skewed
    the quadriceps' fibre and tendon slack lengths."""
    spec = _myoleg()
    ref = spec.copy().compile()
    beta = ref.eq("knee_angle_beta_translation1_constraint_r")
    assert abs(beta.data[0]) > 0.05  # qpos0 violates this coupling by 5 cm
    scaled = scale_model(spec, {"femur_r": 1.12, "tibia_r": 0.95})
    lengths = []
    for model in (ref, scaled):
        data = mujoco.MjData(model)
        data.qpos[:] = _assembled(model)
        mujoco.mj_forward(model, data)
        lengths.append(data.actuator_length.copy())
    for name in ("recfem_r", "vasint_r", "vaslat_r", "vasmed_r"):
        i = ref.actuator(name).id
        np.testing.assert_allclose(
            scaled.actuator_lengthrange[i] / ref.actuator_lengthrange[i],
            lengths[1][i] / lengths[0][i],
            rtol=1e-12,
            err_msg=name,
        )


def test_mass_scales_with_volume_and_strength_is_kept() -> None:
    ref = _leg().compile()
    model = scale_model(_leg(), {"thigh": 1.2})
    thigh = ref.body("thigh").id
    np.testing.assert_allclose(model.body_mass[thigh], 1.2**3 * ref.body_mass[thigh])
    total = scale_model(_leg(), {"thigh": 1.2}, total_mass=7.5)
    np.testing.assert_allclose(total.body_mass.sum(), 7.5)
    # force="-1": the peak force comes from the mass, so scaling must freeze it first.
    same = scale_model(_leg(), {"thigh": 1.2, "shank": 1.2})
    np.testing.assert_allclose(_peak_force(same), _peak_force(ref))


# qpos0 floats 5 cm, "stand" touches the floor, "tilted" sinks a tilted sole 8.5 cm.
GROUND_XML = """<mujoco><worldbody>
<geom name="floor" type="plane" size="2 2 .01"/>
<body name="root" pos="0 0 1.1"><freejoint name="root_free"/>
<geom type="sphere" size=".1" mass="2"/>
<body name="foot" pos="0 0 -1"><joint name="ankle" axis="0 1 0"/>
<geom name="sole" type="box" size=".1 .2 .05" mass="1"/>
</body></body></worldbody>
<keyframe>
<key name="stand" qpos="0 0 1.05 1 0 0 0 0"/>
<key name="tilted" qpos="0.3 0 1.0 1 0 0 0 0.4"/>
</keyframe></mujoco>"""


def _clearance(model: mujoco.MjModel, qpos: np.ndarray, floor: str) -> float:
    """Signed distance of the subject's collision geoms to the floor (MuJoCo's collider)."""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    plane = model.geom(floor).id
    return min(
        mujoco.mj_geomDistance(model, data, g, plane, 10.0, None)
        for g in range(model.ngeom)
        if model.geom_bodyid[g] > 0
        and (model.geom_contype[g] or model.geom_conaffinity[g])
    )


@pytest.mark.parametrize(
    "scales", [{"root": 1.2, "foot": (1.1, 1.3, 1.25)}, {"foot": 0.8}]
)
def test_every_pose_keeps_its_nominal_ground_clearance(
    scales: dict[str, float | tuple[float, float, float]],
) -> None:
    """qpos0 and each keyframe are aligned on their own, not by qpos0's offset."""
    spec = mujoco.MjSpec.from_string(GROUND_XML)
    ref = spec.copy().compile()
    model = scale_model(spec, scales)
    poses = [(ref.qpos0, model.qpos0), *zip(ref.key_qpos, model.key_qpos)]
    for nominal, scaled in poses:
        assert _clearance(model, scaled, "floor") == pytest.approx(
            _clearance(ref, nominal, "floor"), abs=1e-9
        )
        # Only the root height changes.
        np.testing.assert_array_equal(np.delete(scaled, 2), np.delete(nominal, 2))


def test_myoleg_keyframes_stand_where_the_nominal_ones_do() -> None:
    spec = _myoleg()
    ref = spec.copy().compile()
    model = scale_model(spec, {b.name: 1.1 for b in spec.bodies[1:]})
    poses = [(ref.qpos0, model.qpos0), *zip(ref.key_qpos, model.key_qpos)]
    for nominal, scaled in poses:
        assert _clearance(model, scaled, "floor") == pytest.approx(
            _clearance(ref, nominal, "floor"), abs=1e-6
        )


def test_root_alignment_works_in_nested_rotated_frames() -> None:
    spec = mujoco.MjSpec.from_string("""<mujoco><worldbody>
        <geom name="floor" type="plane" size="2 2 .01"/>
        <frame pos=".1 .2 .3" euler="20 30 40"><frame pos=".1 .1 .1" euler="-10 20 30">
        <body name="root" pos="0 0 1"><freejoint/>
        <geom name="sole" type="box" size=".1 .2 .3"/>
        </body></frame></frame></worldbody></mujoco>""")
    ref = spec.copy().compile()
    model = scale_model(spec, {"root": (1.2, 0.9, 1.1)})
    assert _clearance(model, model.qpos0, "floor") == pytest.approx(
        _clearance(ref, ref.qpos0, "floor"), abs=1e-9
    )
    np.testing.assert_allclose(model.qpos0[:2], ref.qpos0[:2], atol=1e-12)


@pytest.mark.parametrize(
    "kind,size,euler,extent",
    [
        ("sphere", ".1", "0 0 0", 0.1),
        ("capsule", ".1 .2", "0 0 0", 0.3),
        ("cylinder", ".1 .2", "0 0 0", 0.2),
        ("ellipsoid", ".1 .2 .3", "0 0 0", 0.3),
        ("box", ".1 .2 .3", "0 0 0", 0.3),
        ("capsule", ".1 .2", "0 45 0", 0.1 + 0.2 / np.sqrt(2)),
        ("cylinder", ".1 .2", "0 45 0", 0.3 / np.sqrt(2)),
        ("ellipsoid", ".1 .2 .3", "0 45 0", np.sqrt(0.05)),
        ("box", ".1 .2 .3", "0 45 0", 0.4 / np.sqrt(2)),
    ],
)
def test_geom_bottom_is_the_support_of_each_shape(
    kind: str, size: str, euler: str, extent: float
) -> None:
    model = mujoco.MjModel.from_xml_string(f"""<mujoco><worldbody>
        <body pos="0 0 1"><freejoint/>
        <geom type="{kind}" size="{size}" euler="{euler}"/>
        </body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    assert _geom_bottom(model, data, 0) == pytest.approx(1 - extent)


def test_geom_bottom_uses_the_transformed_mesh_vertices() -> None:
    model = mujoco.MjModel.from_xml_string("""<mujoco><asset><mesh name="tetra"
        vertex="0 0 0 1 0 0 0 2 0 0 0 3"/></asset><worldbody>
        <body pos="0 0 1"><freejoint/>
        <geom type="mesh" mesh="tetra" pos="0 0 -.2" euler="0 45 0"/>
        </body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    # The authored vertex (1, 0, 0) is lowest after rotating about y.
    assert _geom_bottom(model, data, 0) == pytest.approx(
        1 - 0.2 - 1 / np.sqrt(2), abs=2e-7
    )


def _leg_in_scene(with_agent: bool) -> mujoco.MjSpec:
    """The leg next to a heavy static prop and, optionally, a free-floating agent."""
    spec = _leg()
    spec.worldbody.add_body(name="prop", pos=[1, 0, 0]).add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.2, 0.2, 0.2], mass=40.0
    )
    if with_agent:
        agent = spec.worldbody.add_body(name="agent", pos=[-1, 0, 1])
        agent.add_freejoint()
        agent.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.1, 0, 0], mass=60.0)
    return spec


@pytest.mark.parametrize(
    "scales,with_agent",
    [({"thigh": 1.2}, True), ({"thigh": 1.0}, True), ({}, False)],
)
def test_total_mass_rescales_only_the_subject(
    scales: dict[str, float], with_agent: bool
) -> None:
    """Static scenery and other agents in a composed scene keep their mass."""
    spec = _leg_in_scene(with_agent)
    ref = spec.copy().compile()
    model = scale_model(spec, scales, total_mass=7.5)
    subject = [model.body(name).id for name in ("thigh", "shank")]
    np.testing.assert_allclose(model.body_mass[subject].sum(), 7.5)
    for name in ("prop", "agent") if with_agent else ("prop",):
        np.testing.assert_allclose(model.body(name).mass, ref.body(name).mass)


def test_total_mass_without_a_named_subject_must_be_unambiguous() -> None:
    with pytest.raises(ValueError, match="ambiguous"):
        scale_model(_leg_in_scene(with_agent=True), {}, total_mass=7.5)


def test_unsupported_cases_are_refused() -> None:
    with pytest.raises(KeyError, match="nope"):
        scale_model(_leg(), {"nope": 1.1})
    with pytest.raises(ValueError, match="positive"):
        scale_model(_leg(), {"thigh": (1.0, 0.0, 1.0)})
    shared = _leg()
    shared.body("shank").add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="cube")
    with pytest.raises(NotImplementedError, match="cube"):
        scale_model(shared, {"thigh": 1.1})
    connected = _leg()
    connected.add_equality(
        type=mujoco.mjtEq.mjEQ_CONNECT,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        name1="shank",
        name2="world",
    )
    with pytest.raises(NotImplementedError, match="connect/weld"):
        scale_model(connected, {"shank": 1.1})


HFIELD_XML = """<mujoco><asset><hfield name="h" nrow="3" ncol="3" size="1 1 .1 .1"/></asset>
<worldbody><body name="a"><geom size=".1"/></body>
<body name="b" pos="0 0 1"><geom type="hfield" hfield="h"/></body></worldbody></mujoco>"""

TWO_AGENTS_XML = """<mujoco><compiler settotalmass="30"/><worldbody>
<body name="a"><freejoint/><geom size=".1" mass="1"/></body>
<body name="b" pos="1 0 0"><freejoint/><geom size=".1" mass="2"/></body>
</worldbody></mujoco>"""


def _shared_mesh_leg() -> mujoco.MjSpec:
    spec = _leg()
    spec.body("shank").add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="cube")
    return spec


@pytest.mark.parametrize(
    "make,scales,total_mass,error",
    [
        (_shared_mesh_leg, {"thigh": 1.1}, None, NotImplementedError),
        (_shared_mesh_leg, {"thigh": 1.1, "shank": 1.2}, None, NotImplementedError),
        (
            lambda: mujoco.MjSpec.from_string(HFIELD_XML),
            {"a": 1.1, "b": 1.1},
            None,
            NotImplementedError,
        ),
        (lambda: mujoco.MjSpec.from_string(TWO_AGENTS_XML), {}, 50.0, ValueError),
    ],
    ids=["mesh-also-unscaled", "mesh-scaled-twice", "height-field", "ambiguous-mass"],
)
def test_refused_scaling_leaves_the_spec_unchanged(
    make: Callable[[], mujoco.MjSpec],
    scales: dict[str, float],
    total_mass: float | None,
    error: type[Exception],
) -> None:
    """Everything that can refuse a scaling is checked before the first edit."""
    spec = make()
    xml, before = spec.to_xml(), _model_bytes(spec.copy().compile())
    with pytest.raises(error):
        _scale_spec(spec, scales, total_mass=total_mass)
    assert spec.to_xml() == xml
    assert _model_bytes(spec.copy().compile()) == before


@pytest.mark.parametrize("factor", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_invalid_factors_are_rejected(factor: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        scale_model(_simple_body(), {"thigh": [1.0, factor, 1.0]})
    # Malformed segment factors cannot pass silently, even for unused segments.
    with pytest.raises(ValueError, match="finite and positive"):
        _body_scales(
            _simple_body(), {"unused": factor}, {"thigh": (("thigh",), (0, 1, 2))}
        )


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_invalid_total_mass_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="total_mass must be finite and positive"):
        scale_model(_simple_body(), {"thigh": 1.2}, total_mass=value)


def test_ambiguous_segments_cannot_depend_on_mapping_order() -> None:
    with pytest.raises(ValueError, match="more than one segment"):
        _body_scales(
            _simple_body(),
            {"a": 1.1, "b": 1.2},
            {"a": (("thigh",), (0, 1, 2)), "b": (("thigh",), (0, 1, 2))},
        )


@pytest.mark.parametrize("factor", [1.0, 1.2])
def test_world_body_cannot_be_scaled(factor: float) -> None:
    with pytest.raises(KeyError, match="world"):
        scale_model(_simple_body(), {"world": factor})


@pytest.mark.parametrize("name", ["leg", "myoleg"])
def test_precompiled_mesh_scaling_has_fresh_bounds_and_bvh(name: str) -> None:
    fresh, compiled = MODELS[name](), MODELS[name]()
    compiled.compile()
    factors = (
        {"thigh": (0.8, 0.9, 1.1)} if name == "leg" else {"femur_r": (1.02, 1.07, 0.98)}
    )
    expected = scale_model(fresh, factors)
    actual = scale_model(compiled, factors)
    for field in (
        "mesh_vert",
        "geom_size",
        "geom_aabb",
        "geom_rbound",
        "bvh_aabb",
        "bvh_nodeid",
    ):
        np.testing.assert_array_equal(
            getattr(actual, field), getattr(expected, field), err_msg=field
        )
    assert _model_bytes(actual) == _model_bytes(expected)


def test_mesh_recreation_preserves_authored_data_and_names() -> None:
    spec = _leg()
    mesh = spec.mesh("cube")
    mesh.refpos = [0.01, -0.02, 0.03]
    mesh.refquat = [0.9238795325112867, 0.0, 0.0, 0.3826834323650898]
    mesh.scale = [1.1, 0.9, 1.2]
    mesh.smoothnormal = False
    mesh.maxhullvert = 6
    mesh.info = "authored mesh provenance"
    fields = (
        "name",
        "classname",
        "file",
        "content_type",
        "refpos",
        "refquat",
        "inertia",
        "smoothnormal",
        "maxhullvert",
        "uservert",
        "usernormal",
        "userface",
        "usertexcoord",
        "userfacenormal",
        "userfacetexcoord",
        "material",
        "info",
    )
    before = {field: np.array(getattr(mesh, field), copy=True) for field in fields}
    original_scale = mesh.scale.copy()
    spec.compile()
    _scale_spec(spec, {"thigh": (0.8, 0.9, 1.1)})
    replacement = spec.mesh("cube")
    assert replacement is not None
    assert spec.geom("thigh_mesh").meshname == "cube"
    for field in fields:
        np.testing.assert_array_equal(
            getattr(replacement, field), before[field], err_msg=field
        )
    np.testing.assert_allclose(replacement.scale, original_scale * [0.8, 0.9, 1.1])
    spec.compile()


def test_rajagopal_segments_map_onto_myofullbody(tmp_path: Path) -> None:
    spec = _fullbody()
    assert _segments_for(spec) is _MYOFULLBODY_SEGMENTS
    scales = _body_scales(
        spec,
        {"torso": 1.1, "hand_r": 0.9, "femur_r": (1.0, 1.2, 1.0), "humerus_l": 1.05},
        _MYOFULLBODY_SEGMENTS,
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
    path = _scale_set(
        tmp_path / "scales.xml", {"torso": (1.1,) * 3, "hand_r": (0.9,) * 3}
    )
    model = scale_model(spec, path)
    data = mujoco.MjData(model)
    for _ in range(200):
        mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.qpos))


def test_rajagopal_segments_cover_myoleg() -> None:
    spec = _myoleg()
    assert _segments_for(spec) is _MYOLEG_SEGMENTS
    scales = _body_scales(
        spec, {seg: 1.1 for seg in _MYOLEG_SEGMENTS}, _MYOLEG_SEGMENTS
    )
    assert set(scales) == {b.name for b in spec.bodies[1:]}
    # Arms are not part of MyoLeg: their factors are ignored.
    assert _body_scales(spec, {"humerus_r": 1.3}, _MYOLEG_SEGMENTS) == {}


def test_opensim_file_scales_the_mapped_bodies(tmp_path: Path) -> None:
    spec = _myoleg()
    segments = {"femur_r": (1.04, 1.08, 1.02), "tibia_l": (0.97,) * 3}
    path = _scale_set(tmp_path / "scales.xml", segments)
    by_file = scale_model(spec, path, total_mass=70)
    by_body = scale_model(
        spec,
        {
            **{b: 1.0 for bodies, _ in _MYOLEG_SEGMENTS.values() for b in bodies},
            **_body_scales(spec, segments, _MYOLEG_SEGMENTS),
        },
        total_mass=70,
    )
    assert _model_bytes(by_file) == _model_bytes(by_body)
    np.testing.assert_allclose(by_file.body_mass.sum(), 70)
    with pytest.raises(ValueError, match="No OpenSim segment mapping"):
        scale_model(_leg(), path)


def _body_frame_extent(model: mujoco.MjModel, body: str) -> np.ndarray:
    """Extent of a body's mesh vertices along the body's own axes."""
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    b = model.body(body).id
    points = []
    for g in np.flatnonzero(
        (model.geom_bodyid == b) & (model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)
    ):
        mesh = model.geom_dataid[g]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        world = (
            model.mesh_vert[start : start + count] @ data.geom_xmat[g].reshape(3, 3).T
            + data.geom_xpos[g]
        )
        points.append((world - data.xpos[b]) @ data.xmat[b].reshape(3, 3))
    return np.ptp(np.concatenate(points), axis=0)


def test_opensim_factors_land_on_the_right_axes_of_z_up_bodies(
    tmp_path: Path,
) -> None:
    """OpenSim frames are y-up; MyoLeg's root and torso are z-up with a lateral y."""
    spec = _myoleg()
    ref = spec.copy().compile()
    # OpenSim (x anterior, y up, z right): a 20% taller torso.
    taller = scale_model(
        spec, _scale_set(tmp_path / "taller.xml", {"torso": (1.0, 1.2, 1.0)})
    )
    np.testing.assert_allclose(  # mesh vertices are float32
        _body_frame_extent(taller, "torso") / _body_frame_extent(ref, "torso"),
        [1.0, 1.0, 1.2],
        rtol=1e-6,
    )
    pelvis = (1.1, 1.2, 1.3)
    resolved = _body_scales(
        spec, {"pelvis": pelvis, "torso": (1.0, 1.2, 1.0)}, _MYOLEG_SEGMENTS
    )
    np.testing.assert_array_equal(resolved["root"], [1.1, 1.3, 1.2])
    np.testing.assert_array_equal(resolved["pelvis"], pelvis)  # y-up like OpenSim
    np.testing.assert_array_equal(resolved["torso"], [1.0, 1.0, 1.2])


def test_myofullbody_root_takes_the_pelvis_factors_on_its_own_axes() -> None:
    spec = _fullbody()
    pelvis = (1.1, 1.2, 1.3)
    resolved = _body_scales(spec, {"pelvis": pelvis}, _MYOFULLBODY_SEGMENTS)
    np.testing.assert_array_equal(resolved["Full Body"], [1.3, 1.1, 1.2])
    for body in ("sacrum", "pelvis"):
        np.testing.assert_array_equal(resolved[body], pelvis)


@pytest.mark.parametrize(
    "name,segments",
    [("myoleg", _MYOLEG_SEGMENTS), ("fullbody", _MYOFULLBODY_SEGMENTS)],
)
def test_segment_axis_maps_match_the_model_frames(
    name: str, segments: dict[str, tuple[tuple[str, ...], tuple[int, int, int]]]
) -> None:
    """Each first body's axes lie along the OpenSim axes its map names.

    The femur frame is OpenSim's (its hip offset is Rajagopal's).
    """
    model = MODELS[name]().compile()
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    opensim = data.xmat[model.body("femur_r").id].reshape(3, 3)
    for segment, (bodies, axes) in segments.items():
        for body in bodies:
            rot = opensim.T @ data.xmat[model.body(body).id].reshape(3, 3)
            np.testing.assert_allclose(
                np.abs(rot).max(axis=0), 1.0, atol=1e-5, err_msg=segment
            )
            assert tuple(np.argmax(np.abs(rot), axis=0)) == axes, segment


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
    scales = _read_opensim_scales(scale_set)
    assert set(scales) == {"femur_r"}  # apply=false is skipped
    np.testing.assert_allclose(scales["femur_r"], [1.1, 1.2, 1.3])

    scaled = tmp_path / "scaled.osim"
    scaled.write_text(_osim("1.05 1.05 1.05", "2 2 2"))
    np.testing.assert_allclose(_read_opensim_scales(scaled)["femur_r"], [1.05] * 3)
    np.testing.assert_allclose(_read_opensim_scales(scaled)["tibia_r"], [2] * 3)


def test_model_digest_tracks_the_morphology() -> None:
    source = mujoco.MjSpec.from_string(STANDING_XML)
    scaled = _model_digest(scale_model(source, {"root": 1.1, "foot": 1.2}))
    same = _model_digest(scale_model(source, {"foot": [1.2] * 3, "root": [1.1] * 3}))
    assert scaled == same
    changed = mujoco.MjSpec.from_string(STANDING_XML)
    changed.geom("sole").size[0] *= 2
    assert _model_digest(scale_model(changed, {"root": 1.1, "foot": 1.2})) != scaled


def test_model_digest_is_stable_across_processes() -> None:
    code = (
        "import mujoco; from myosuite.core.body_scaling import _model_digest; "
        f"print(_model_digest(mujoco.MjSpec.from_string({STANDING_XML!r}).compile()))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True, check=True
    )
    expected = _model_digest(mujoco.MjSpec.from_string(STANDING_XML).compile())
    assert result.stdout.splitlines()[-1] == expected


def test_named_motion_replays_on_scaled_fullbody() -> None:
    source = _fullbody()
    nominal = source.copy().compile()
    model = scale_model(source, {"femur_r": 1.08})
    assert (model.nq, model.nv, model.nu, model.ntendon) == (
        nominal.nq,
        nominal.nv,
        nominal.nu,
        nominal.ntendon,
    )
    names = [model.joint(i).name for i in range(model.njnt)]
    assert names == [nominal.joint(i).name for i in range(nominal.njnt)]
    # A clip authored against the new target, stored in reversed named-joint order.
    qpos = np.repeat(model.qpos0[None], 3, axis=0)
    hinge = next(
        i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
    )
    qpos[:, model.jnt_qposadr[hinge]] += [0, 0.01, 0.02]
    indices = []
    for i in reversed(range(model.njnt)):
        start = int(model.jnt_qposadr[i])
        stop = int(model.jnt_qposadr[i + 1]) if i + 1 < model.njnt else model.nq
        indices.extend(range(start, stop))
    clip = MotionClip(
        qpos=qpos[:, indices],
        qvel=None,
        site_xpos=None,
        site_names=None,
        qpos_joint_names=list(reversed(names)),
        frequency_hz=100,
    )
    replay = expand_motion_clip_to_model(clip, model)
    np.testing.assert_allclose(replay.qpos, qpos)
    data = mujoco.MjData(model)
    for frame in replay.qpos:
        data.qpos[:] = frame
        mujoco.mj_forward(model, data)
        assert np.all(np.isfinite(data.site_xpos))
        assert np.all(np.isfinite(data.actuator_length))
