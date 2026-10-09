# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Body skin: ``.skn`` parsing and posing against MuJoCo's own skinning."""

from __future__ import annotations

import struct
from pathlib import Path

import mujoco
import numpy as np
import pytest

from myosuite.viz.skin import FULLBODY_SKIN, SkinPose, load_skn

pytestmark = pytest.mark.tier1

_XML = """
<mujoco><worldbody>
  <body name="upper"><joint type="ball"/><geom size=".05"/>
    <body name="lower" pos="0 0 -.4"><joint axis="0 1 0"/><geom size=".05"/></body>
  </body>
</worldbody></mujoco>"""


def write_skn(path: Path, bones: dict[str, tuple]) -> np.ndarray:
    """Write a box skin spanning both bodies; returns its bind vertices."""
    vert = np.array(
        [
            [x, y, z]
            for z in (0.1, -0.2, -0.5)
            for x in (-0.1, 0.1)
            for y in (-0.1, 0.1)
        ],
        np.float32,
    )
    face = np.array([[i, i + 1, i + 2] for i in range(0, len(vert) - 2, 2)], np.int32)
    texcoord = np.zeros((len(vert), 2), np.float32)
    out = struct.pack("4i", len(vert), len(vert), len(face), len(bones))
    out += vert.tobytes() + texcoord.tobytes() + face.tobytes()
    for name, (pos, quat, ids, weights) in bones.items():
        out += name.encode().ljust(40, b"\0")
        out += np.asarray(pos, np.float32).tobytes()
        out += np.asarray(quat, np.float32).tobytes()
        out += struct.pack("i", len(ids)) + np.asarray(ids, np.int32).tobytes()
        out += np.asarray(weights, np.float32).tobytes()
    path.write_bytes(out)
    return vert


def _bones() -> dict[str, tuple]:
    # The middle ring is shared; weights are deliberately unnormalised.
    return {
        "upper": ([0, 0, 0], [1, 0, 0, 0], range(8), [1] * 4 + [0.6] * 4),
        "lower": ([0, 0, -0.4], [0.9, 0.1, 0.3, 0], range(4, 12), [0.8] * 4 + [2] * 4),
    }


def _posed(path: Path, inflate: float) -> tuple[mujoco.MjModel, mujoco.MjData]:
    spec = mujoco.MjSpec.from_string(_XML)
    skin = spec.add_skin()
    skin.file, skin.inflate = str(path), inflate
    model = spec.compile()
    data = mujoco.MjData(model)
    data.qpos[:] = [0.9, 0.2, -0.3, 0.25, 0.8]
    mujoco.mj_normalizeQuat(model, data.qpos)
    mujoco.mj_forward(model, data)
    return model, data


def test_load_skn_round_trips_and_rejects_truncated_files(tmp_path: Path) -> None:
    path = tmp_path / "box.skn"
    vert = write_skn(path, _bones())
    skin = load_skn(path)
    np.testing.assert_array_equal(skin.vert, vert)
    assert skin.bone_names == ["upper", "lower"]
    np.testing.assert_array_equal(skin.vertid[1], np.arange(4, 12))
    path.write_bytes(path.read_bytes()[:-3])
    with pytest.raises(ValueError, match="Truncated"):
        load_skn(path)


@pytest.mark.parametrize("inflate", [0.0, 0.02])
def test_pose_matches_mujoco_skinning(tmp_path: Path, inflate: float) -> None:
    """Posed vertices equal MuJoCo's ``mjv_updateSkin`` output."""
    path = tmp_path / "box.skn"
    write_skn(path, _bones())
    model, data = _posed(path, inflate)
    scene = mujoco.MjvScene(model, 100)
    mujoco.mjv_updateScene(
        model,
        data,
        mujoco.MjvOption(),
        None,
        mujoco.MjvCamera(),
        mujoco.mjtCatBit.mjCAT_ALL,
        scene,
    )
    expected = np.array(scene.skinvert[: 3 * model.skin_vertnum[0]]).reshape(-1, 3)
    posed = SkinPose.bind(load_skn(path), model, inflate).vertices(data)
    np.testing.assert_allclose(posed, expected, atol=1e-5)
    # The tolerance is tight enough to catch an unposed (bind-pose) skin.
    assert np.abs(load_skn(path).vert - expected).max() > 1e-2


def test_bind_names_missing_bones(tmp_path: Path) -> None:
    path = tmp_path / "box.skn"
    bones = _bones()
    bones["femur_r"] = bones.pop("lower")
    write_skn(path, bones)
    with pytest.raises(ValueError, match="femur_r"):
        SkinPose.bind(load_skn(path), mujoco.MjModel.from_xml_string(_XML))


def test_bundled_fullbody_skin_parses() -> None:
    skin = load_skn("fullbody")
    assert load_skn(FULLBODY_SKIN).vert.shape == skin.vert.shape == (14517, 3)
    assert len(skin.bone_names) == 58 and {"pelvis", "head", "toes_r"} <= set(
        skin.bone_names
    )
