# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Explicit body scaling of skin influences and export fidelity."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from myosuite.core.body_scaling import (
    RAJAGOPAL_MYOFULLBODY_SEGMENTS,
    _body_scales,
    scale_bodies,
)
from myosuite.viz.skin import Skin, SkinPose, load_skin, load_skn, save_skn
from myosuite.viz.skin_glb import read_skin_glb, write_skin_glb

pytestmark = pytest.mark.tier1

_XML = """
<mujoco><worldbody>
  <body name="upper" pos=".1 0 1.2" euler="20 -10 30">
    <joint name="hip" pos=".02 0 -.04"/><geom size=".05"/>
    <body name="lower" pos=".03 .02 -.4" euler="15 25 -10">
      <joint name="knee" pos=".02 .01 -.1" axis="0 1 0"/>
      <geom size=".05"/>
    </body>
  </body>
</worldbody></mujoco>"""


def _data(
    model: mujoco.MjModel, qpos: tuple[float, float] | None = None
) -> mujoco.MjData:
    data = mujoco.MjData(model)
    if qpos is not None:
        data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    return data


def _skin(model: mujoco.MjModel) -> Skin:
    data = _data(model)
    ids = [model.body(n).id for n in ("upper", "lower")]
    rot = data.xmat[ids[0]].reshape(3, 3)
    local = np.array([[0.1, 0.08, 0.1], [-0.1, -0.08, -0.2], [0.07, -0.1, -0.35]])
    return Skin(
        (local @ rot.T + data.xpos[ids[0]]).astype(np.float32),
        np.zeros((3, 2), np.float32),
        np.array([[0, 1, 2]], np.int32),
        ["upper", "lower"],
        data.xpos[ids].astype(np.float32),
        data.xquat[ids].astype(np.float32),
        [np.array([0, 1], np.int32), np.array([1, 2], np.int32)],
        [np.array([1.0, 0.3], np.float32), np.array([0.7, 1.0], np.float32)],
    )


def _expected(
    skin: Skin,
    data: mujoco.MjData,
    model: mujoco.MjModel,
    factors: dict[str, np.ndarray],
) -> np.ndarray:
    vertices = np.zeros_like(skin.vert, dtype=float)
    total = np.zeros(len(vertices))
    for name, pos, quat, ids, weights in zip(
        skin.bone_names, skin.bindpos, skin.bindquat, skin.vertid, skin.vertweight
    ):
        bind = np.zeros(9)
        mujoco.mju_quat2Mat(bind, quat.astype(float) / np.linalg.norm(quat))
        local = (skin.vert[ids] - pos) @ bind.reshape(3, 3)
        bone = model.body(name).id
        world = (local * factors[name]) @ data.xmat[bone].reshape(3, 3).T
        world += data.xpos[bone]
        np.add.at(vertices, ids, weights[:, None] * world)
        np.add.at(total, ids, weights)
    return vertices / total[:, None]


def test_anisotropic_scaling_uses_each_bones_local_frame(tmp_path: Path) -> None:
    spec = mujoco.MjSpec.from_string(_XML)
    reference = spec.copy().compile()
    skin = _skin(reference)
    save_skn(skin, tmp_path / "source.skn")
    skin = load_skn(tmp_path / "source.skn")
    before = skin.vert.copy(), skin.bindpos.copy(), skin.bindquat.copy()
    factors = {"upper": np.array([1.4, 0.8, 1.2]), "lower": np.array([0.9, 1.3, 1.6])}
    scaled = scale_bodies(spec, factors).compile()
    pose = SkinPose.bind(skin, scaled, body_scales=factors)
    for qpos in (None, (0.6, -0.8)):
        data = _data(scaled, qpos)
        np.testing.assert_allclose(
            pose.vertices(data), _expected(skin, data, scaled, factors), atol=2e-7
        )
        assert (
            np.max(
                np.abs(pose.vertices(data) - SkinPose.bind(skin, scaled).vertices(data))
            )
            > 0.03
        )
    for actual, original in zip((skin.vert, skin.bindpos, skin.bindquat), before):
        np.testing.assert_array_equal(actual, original)


def test_uniform_scaling_survives_skn_and_glb_exports(tmp_path: Path) -> None:
    spec = mujoco.MjSpec.from_string(_XML)
    skin = _skin(spec.copy().compile())
    factors = {"upper": 1.25, "lower": 1.25}
    model = scale_bodies(spec, factors).compile()
    pose = SkinPose.bind(skin, model, body_scales=factors)
    baked = pose.bake(model)
    rest = _data(model)
    np.testing.assert_allclose(baked.bindpos, rest.xpos[pose.body_ids], atol=1e-7)
    np.testing.assert_allclose(baked.bindquat, rest.xquat[pose.body_ids], atol=1e-7)
    save_skn(baked, tmp_path / "scaled.skn")
    write_skin_glb(pose, model, tmp_path / "scaled.glb")
    data = _data(model, (0.6, -0.8))
    for back in (
        load_skn(tmp_path / "scaled.skn"),
        load_skin(tmp_path / "scaled.glb", model),
    ):
        np.testing.assert_allclose(
            SkinPose.bind(back, model).vertices(data), pose.vertices(data), atol=3e-7
        )


def test_inexact_export_requires_explicit_opt_in(tmp_path: Path) -> None:
    spec = mujoco.MjSpec.from_string(_XML)
    skin = _skin(spec.copy().compile())
    factors = {"upper": [1.4, 0.8, 1.2], "lower": [0.9, 1.3, 1.6]}
    model = scale_bodies(spec, factors).compile()
    pose = SkinPose.bind(skin, model, body_scales=factors)
    with pytest.raises(ValueError, match="cannot be represented"):
        pose.bake(model)
    with pytest.raises(ValueError, match="cannot be represented"):
        write_skin_glb(pose, model, tmp_path / "scaled.glb")
    assert not (tmp_path / "scaled.glb").exists()
    with pytest.warns(UserWarning, match="articulated poses"):
        write_skin_glb(pose, model, tmp_path / "scaled.glb", allow_approximate=True)
    back = read_skin_glb(tmp_path / "scaled.glb", model)
    rest = _data(model)
    np.testing.assert_allclose(
        SkinPose.bind(back, model).vertices(rest), pose.vertices(rest), atol=3e-7
    )
    data = _data(model, (0.6, -0.8))
    assert (
        np.max(np.abs(SkinPose.bind(back, model).vertices(data) - pose.vertices(data)))
        > 0.005
    )


def test_scale_mapping_validates_names_and_identity() -> None:
    model = mujoco.MjModel.from_xml_string(_XML)
    skin, data = _skin(model), _data(model, (0.6, -0.8))
    np.testing.assert_array_equal(
        SkinPose.bind(skin, model, body_scales={"upper": 1.0}).vertices(data),
        SkinPose.bind(skin, model).vertices(data),
    )
    with pytest.raises(KeyError, match="absent"):
        SkinPose.bind(skin, model, body_scales={"absent": 1.2})
    with pytest.raises(ValueError, match="positive"):
        SkinPose.bind(skin, model, body_scales={"upper": 0.0})


def test_bundled_skin_uses_resolved_segment_scales(tmp_path: Path) -> None:
    from myosuite.tests.test_body_scaling import _fullbody

    spec = _fullbody()
    reference = spec.copy().compile()
    segments = RAJAGOPAL_MYOFULLBODY_SEGMENTS
    scales = {"torso": [1.1, 0.9, 1.2], "hand_r": 0.85, "femur_r": [1.2, 0.8, 1.1]}
    resolved = _body_scales(spec, scales, segments)
    scaled = scale_bodies(spec, scales, segments=segments).compile()
    skin = load_skn("fullbody")
    pose = SkinPose.bind(skin, scaled, body_scales=resolved)
    source_pose = SkinPose.bind(skin, reference)
    factors = np.array(
        [resolved.get(scaled.body(int(i)).name, np.ones(3)) for i in pose.body_ids]
    )
    np.testing.assert_allclose(
        pose._local, source_pose._local * factors[pose._bone], atol=1e-12
    )
    data = _data(scaled)
    assert np.isfinite(pose.vertices(data)).all()
    assert (
        np.max(np.abs(pose.vertices(data) - SkinPose.bind(skin, scaled).vertices(data)))
        > 0.02
    )
    with pytest.warns(UserWarning, match="segment-local"):
        baked = pose.bake(scaled, allow_approximate=True)
    save_skn(baked, tmp_path / "fullbody.skn")
    write_skin_glb(baked, scaled, tmp_path / "fullbody.glb")
    for back in (
        load_skn(tmp_path / "fullbody.skn"),
        read_skin_glb(tmp_path / "fullbody.glb", scaled),
    ):
        np.testing.assert_allclose(
            SkinPose.bind(back, scaled).vertices(data), pose.vertices(data), atol=3e-6
        )
