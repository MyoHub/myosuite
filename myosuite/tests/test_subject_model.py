# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Subject construction, ground alignment, provenance and replay integration."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

from myosuite.core.model_builder import ModelBuilder
from myosuite.core.subject_model import (
    GroundAlignment,
    ScaleEstimate,
    SubjectModelConfig,
    align_root_to_ground,
    build_subject_model,
    model_digest,
)
from myosuite.core.trajectory_io import MotionClip, expand_motion_clip_to_model
from myosuite.utils import checkpoint_manifest as cm

pytestmark = pytest.mark.tier1

XML = """<mujoco><worldbody>
<geom name="ground" type="plane" size="2 2 .01"/>
<body name="root" pos="0 0 1"><freejoint name="root_free"/>
<geom type="sphere" size=".1" mass="2"/>
<body name="foot" pos="0 0 -1"><joint name="ankle" axis="0 1 0"/>
<geom name="sole" type="box" size=".1 .2 .05" mass="1"/>
</body></body></worldbody>
<keyframe><key name="stand" qpos="0 0 1 1 0 0 0 0"/></keyframe>
</mujoco>"""


def _source() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_string(XML)


def test_estimate_is_validated_and_detached_from_inputs() -> None:
    factors = np.array([1.0, 1.1, 1.2])
    estimate = ScaleEstimate({"root": factors}, mass_kg=72, source="static trial")
    factors[:] = 4
    assert estimate.to_dict()["factors"] == {"root": [1, 1.1, 1.2]}
    with pytest.raises(TypeError):
        estimate.factors["root"] = 2
    for kwargs in ({"mass_kg": 0}, {"height_m": np.inf}, {"confidence": -0.1}):
        with pytest.raises(ValueError):
            ScaleEstimate({}, **kwargs)
    with pytest.raises(ValueError):
        ScaleEstimate({"root": np.nan})


def test_subject_digest_is_canonical_and_sensitive() -> None:
    config = SubjectModelConfig(ScaleEstimate({"root": 1.1, "foot": 1.2}))
    equivalent = SubjectModelConfig(
        ScaleEstimate({"foot": [1.2] * 3, "root": [1.1] * 3})
    )
    assert equivalent.digest == config.digest
    subject = build_subject_model(_source(), config)
    repeated = build_subject_model(ModelBuilder.from_spec(_source()), equivalent)
    assert subject.metadata == repeated.metadata
    assert (
        model_digest(_source().compile()) == subject.metadata["reference_model_digest"]
    )
    changed_source = _source()
    changed_source.geom("sole").size[0] *= 2
    other = build_subject_model(changed_source, config)
    assert subject.metadata["model_digest"] != other.metadata["model_digest"]
    assert (
        subject.metadata["reference_model_digest"]
        != other.metadata["reference_model_digest"]
    )
    assert subject.metadata["config_digest"] == other.metadata["config_digest"]
    json.dumps(subject.metadata, allow_nan=False)


def test_model_digest_is_stable_across_processes() -> None:
    code = (
        "import mujoco; from myosuite.core.subject_model import model_digest; "
        f"print(model_digest(mujoco.MjSpec.from_string({XML!r}).compile()))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True, check=True
    )
    assert result.stdout.splitlines()[-1] == model_digest(_source().compile())


def test_subject_mass_ground_and_keyframes_preserve_source() -> None:
    source = _source()
    before = model_digest(source.copy().compile())
    ground = GroundAlignment("root_free", ("sole",), ground_height=0.2, clearance=0.01)
    subject = build_subject_model(
        source,
        SubjectModelConfig(
            ScaleEstimate({"root": 1.2, "foot": (1.1, 1.2, 1.3)}, mass_kg=72),
            mass="keep",
            ground=ground,
        ),
    )
    assert model_digest(source.copy().compile()) == before
    assert sum(subject.model.body_mass) == pytest.approx(72)
    aligned, offset = align_root_to_ground(subject.model, ground)
    assert offset == pytest.approx(0, abs=1e-12)
    np.testing.assert_allclose(aligned, subject.model.qpos0)
    np.testing.assert_allclose(subject.model.key_qpos[0], subject.model.qpos0)
    assert subject.metadata["root_z_offset"] == pytest.approx(0.475)


@pytest.mark.parametrize(
    "kind,size,extent",
    [
        ("sphere", ".1", 0.1),
        ("capsule", ".1 .2", 0.3),
        ("cylinder", ".1 .2", 0.2),
        ("ellipsoid", ".1 .2 .3", 0.3),
        ("box", ".1 .2 .3", 0.3),
    ],
)
def test_ground_primitive_support(kind: str, size: str, extent: float) -> None:
    spec = mujoco.MjSpec.from_string(f"""<mujoco><worldbody><body name="root" pos="0 0 1">
        <freejoint name="free"/><geom name="sole" type="{kind}" size="{size}"/>
        </body></worldbody></mujoco>""")
    model = spec.compile()
    qpos, offset = align_root_to_ground(model, GroundAlignment("free", ("sole",)))
    assert qpos[2] == pytest.approx(extent)
    assert offset == pytest.approx(extent - 1)
    assert model.qpos0[2] == 1


def test_ground_rejects_wrong_root_and_unsupported_geoms() -> None:
    model = _source().compile()
    with pytest.raises(ValueError, match="outside"):
        align_root_to_ground(model, GroundAlignment("root_free", ("ground",)))
    with pytest.raises(ValueError, match="free root"):
        align_root_to_ground(model, GroundAlignment("ankle", ("sole",)))


@pytest.mark.parametrize(
    "kind,size,extent",
    [
        ("capsule", ".1 .2", 0.1 + 0.2 / np.sqrt(2)),
        ("cylinder", ".1 .2", 0.3 / np.sqrt(2)),
        ("ellipsoid", ".1 .2 .3", np.sqrt(0.05)),
        ("box", ".1 .2 .3", 0.4 / np.sqrt(2)),
    ],
)
def test_ground_support_uses_geom_rotation(kind: str, size: str, extent: float) -> None:
    spec = mujoco.MjSpec.from_string(f"""<mujoco><worldbody><body name="root" pos="0 0 1">
        <freejoint name="free"/><geom name="sole" type="{kind}" size="{size}" euler="0 45 0"/>
        </body></worldbody></mujoco>""")
    model = spec.compile()
    qpos, _ = align_root_to_ground(model, GroundAlignment("free", ("sole",)))
    assert qpos[2] == pytest.approx(extent)


def test_ground_support_uses_transformed_mesh_vertices() -> None:
    spec = mujoco.MjSpec.from_string("""<mujoco><asset><mesh name="tetra"
        vertex="0 0 0 1 0 0 0 2 0 0 0 3"/></asset><worldbody>
        <body name="root" pos="0 0 1"><freejoint name="free"/>
        <geom name="sole" type="mesh" mesh="tetra" pos="0 0 -.2" euler="0 45 0"/>
        </body></worldbody></mujoco>""")
    model = spec.compile()
    qpos, _ = align_root_to_ground(model, GroundAlignment("free", ("sole",)))
    # The authored vertex (1, 0, 0) is lowest after rotating about y.
    assert qpos[2] == pytest.approx(0.2 + 1 / np.sqrt(2), abs=2e-7)


def test_subject_ground_alignment_in_nested_rotated_frames() -> None:
    spec = mujoco.MjSpec.from_string("""<mujoco><worldbody>
        <frame pos=".1 .2 .3" euler="20 30 40"><frame pos=".1 .1 .1" euler="-10 20 30">
        <body name="root" pos="0 0 1"><freejoint name="free"/>
        <geom name="sole" type="box" size=".1 .2 .3"/>
        </body></frame></frame></worldbody></mujoco>""")
    alignment = GroundAlignment("free", ("sole",), clearance=0.003)
    subject = build_subject_model(
        spec,
        SubjectModelConfig(
            ScaleEstimate({"root": 1.2}),
            ground=alignment,
        ),
    )
    _, remaining = align_root_to_ground(subject.model, alignment)
    assert remaining == pytest.approx(0, abs=1e-12)


def test_checkpoint_morphology_is_checked_even_without_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subject = build_subject_model(
        _source(), SubjectModelConfig(ScaleEstimate({"foot": 1.1}))
    )
    with pytest.warns(UserWarning, match="morphology"):
        assert not cm.check_contract("test", tmp_path, subject_model=subject)
    with pytest.raises(ValueError, match="morphology"):
        cm.check_contract("test", tmp_path, subject_model=subject, strict=True)
    contract = {"obs_shape": [1], "action_shape": [1], "ctrl_dt": 0.01}
    monkeypatch.setattr(cm, "env_contract", lambda _: contract)
    cm.write_manifest(tmp_path, "test", contract=contract, subject_model=subject)
    assert cm.read_manifest(tmp_path)["subject_model"] == subject.metadata
    assert cm.check_contract("test", tmp_path, subject_model=subject, strict=True)
    monkeypatch.setattr(cm, "env_contract", lambda _: {**contract, "obs_shape": [2]})
    with pytest.warns(UserWarning, match="may not work"):
        assert not cm.check_contract("test", tmp_path, subject_model=subject)
    with pytest.warns(UserWarning, match="morphology"):
        assert not cm.check_contract("test", tmp_path)
    other = build_subject_model(
        _source(), SubjectModelConfig(ScaleEstimate({"foot": 1.2}))
    )
    with pytest.raises(ValueError, match="morphology"):
        cm.check_contract("test", tmp_path, subject_model=other, strict=True)


def test_named_motion_replays_on_scaled_fullbody() -> None:
    from myosuite.tests.test_body_scaling import _fullbody

    source = _fullbody()
    nominal = source.copy().compile()
    subject = build_subject_model(
        source, SubjectModelConfig(ScaleEstimate({"femur_r": 1.08}))
    )
    model = subject.model
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
