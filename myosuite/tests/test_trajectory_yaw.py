# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Yaw augmentation preserves native site kinematics in both matrix layouts."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from myosuite.core.trajectory_yaw import write_rotated_clip_npz

pytestmark = pytest.mark.tier1


def _native_kinematics(
    model: mujoco.MjModel, qpos: np.ndarray, qvel: np.ndarray
) -> dict:
    data = mujoco.MjData(model)
    fields: dict[str, list[np.ndarray]] = {
        "site_xpos": [],
        "site_xmat": [],
        "site_velocity": [],
        "cvel": [],
        "subtree_com": [],
    }
    for q, v in zip(qpos, qvel):
        data.qpos[:], data.qvel[:] = q, v
        mujoco.mj_forward(model, data)
        for name in fields:
            if name != "site_velocity":
                fields[name].append(getattr(data, name).copy())
        velocity = np.empty((model.nsite, 6))
        for site in range(model.nsite):
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_SITE, site, velocity[site], 0
            )
        fields["site_velocity"].append(velocity)
    return {name: np.stack(values) for name, values in fields.items()}


@pytest.mark.parametrize("matrix_layout", [False, True])
@pytest.mark.parametrize("yaw", [np.pi / 2, -0.73])
def test_rotated_auxiliary_arrays_match_native_forward(
    tmp_path: Path, matrix_layout: bool, yaw: float
) -> None:
    model = mujoco.MjModel.from_xml_string("""
      <mujoco><worldbody><body pos="1 2 1"><freejoint/>
        <geom size=".1"/><site pos=".2 .1 .3"/>
        <body pos=".2 .1 .4"><joint axis="0 1 0"/>
          <geom size=".1"/><site pos=".2 -.1 .3"/>
        </body></body></worldbody></mujoco>
    """)
    rng = np.random.default_rng(7)
    qpos = np.tile(model.qpos0, (4, 1))
    qpos[:, :3] += rng.normal(0, 0.1, (4, 3))
    qpos[:, 3:7] = rng.normal(size=(4, 4))
    qpos[:, 3:7] /= np.linalg.norm(qpos[:, 3:7], axis=1, keepdims=True)
    qpos[:, -1] = rng.normal(0, 0.1, 4)
    qvel = rng.normal(size=(4, model.nv))
    fields = _native_kinematics(model, qpos, qvel)
    if matrix_layout:
        fields["site_xmat"] = fields["site_xmat"].reshape(4, model.nsite, 3, 3)
    source = tmp_path / "source.npz"
    np.savez(source, qpos=qpos, qvel=qvel, **fields, frequency=np.array(100.0))

    path = write_rotated_clip_npz(source, tmp_path / "rotated.npz", yaw)
    with np.load(path, allow_pickle=False) as rotated:
        expected = _native_kinematics(model, rotated["qpos"], rotated["qvel"])
        for name in fields:
            np.testing.assert_allclose(
                rotated[name],
                expected[name].reshape(fields[name].shape),
                atol=1e-12,
                err_msg=name,
            )
        # Free-joint angular velocity remains in the body frame.
        np.testing.assert_array_equal(rotated["qvel"][:, 3:], qvel[:, 3:])
        assert rotated["site_xmat"].shape == fields["site_xmat"].shape
        assert rotated["frequency"].item() == 100.0
