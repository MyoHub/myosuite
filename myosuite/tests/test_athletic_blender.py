# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Athletic Blender studio rig: lighting contract and stage export."""

from __future__ import annotations

import json
import math

import mujoco
import pytest

pytestmark = pytest.mark.tier2

from myosuite.envs.myo.athletic.blender import (
    RING_LANES,
    STUDIO_LIGHTS,
    WORLD_STRENGTH,
    lighting_spec,
    render_athletic,
    running_ring_paths,
    write_stage,
)

_XML = """
<mujoco>
  <asset>
    <mesh name="tri" vertex="0 0 0  0.2 0 0  0 0.1 0  0 0 0.2"/>
  </asset>
  <worldbody>
    <body name="pelvis" pos="0 0 1">
      <freejoint/>
      <geom name="bone_mesh" type="mesh" mesh="tri"/>
      <geom name="collider" type="capsule" size="0.04 0.1"/>
    </body>
    <body name="marker" pos="0.3 0 1">
      <geom name="marker" type="sphere" size="0.05"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_studio_rig_models_the_body_instead_of_washing_it():
    spec = lighting_spec("studio")
    by_name = {light.name: light for light in STUDIO_LIGHTS}
    assert set(by_name) == {"Key", "Fill", "Rim", "Bounce", "Hair"}
    assert by_name["Key"].size > by_name["Rim"].size
    assert by_name["Fill"].color[2] > by_name["Key"].color[2]
    assert by_name["Key"].color[0] > by_name["Fill"].color[0]
    assert by_name["Key"].power > by_name["Fill"].power * 3
    assert spec["world_strength"] == pytest.approx(WORLD_STRENGTH)
    assert spec["world_strength"] < 0.1
    assert spec["view_transform"] == "AgX"
    with pytest.raises(ValueError, match="lighting"):
        lighting_spec("headlight")  # type: ignore[arg-type]


def test_stage_exports_visual_mesh_and_studio_lights(tmp_path):
    model = mujoco.MjModel.from_xml_string(_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    snap = (data.geom_xpos.copy(), data.geom_xmat.copy(), data.qpos[:3].copy())
    meta = write_stage(model, [snap, snap], tmp_path, up_axis="z", fps=30, preview=True)
    text = (tmp_path / "geoms.obj").read_text()
    assert "o bone_mesh" in text
    assert "o marker" in text
    # The capsule shares the pelvis with the bone mesh, so it is not a second shell.
    assert "collider" not in text
    assert [light["name"] for light in meta["lighting"]["lights"]] == [
        "Key",
        "Fill",
        "Rim",
        "Bounce",
        "Hair",
    ]
    saved = json.loads((tmp_path / "render.json").read_text())
    assert saved["camera_offset"][2] > 0  # Z-up fullbody
    assert saved["track"]["kind"] == "ring"
    assert saved["track"]["n_lanes"] == RING_LANES
    assert len(saved["frames"]) == 2
    assert len(saved["frames"][0]) == 2


def test_running_ring_draws_a_line_on_every_lane_boundary():
    ring = running_ring_paths((0.0, 0.1, 1.0), "z")
    assert len(ring["lines"]) == RING_LANES + 1
    inner = ring["rubber"][0]
    outer = ring["rubber"][1]
    assert len(inner) == len(outer) > 80
    # The outer curb sits farther from the infield centre than the inner curb.
    cx, cy, _ = ring["center"]

    def span(pts: list[tuple[float, float, float]]) -> float:
        return max(math.hypot(p[0] - cx, p[1] - cy) for p in pts)

    assert span(outer) > span(inner) + 1.0


def test_elastic_render_does_not_launch_blender(tmp_path, monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("blender was launched")

    monkeypatch.setattr(
        "myosuite.envs.myo.athletic.factory.AthleticModularTaskEnv", fail
    )
    with pytest.raises(NotImplementedError, match="elastic"):
        render_athletic("sprint", tmp_path, tendons="elastic")
