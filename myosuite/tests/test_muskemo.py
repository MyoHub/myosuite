# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""MuSkeMo interchange preserves compiled anatomy and MuJoCo geometry."""

import json

import mujoco
import myo_sim
import numpy as np
import pytest

from myosuite.integrations.muskemo import export_model, joint_sweep, model_fingerprint
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings

pytestmark = pytest.mark.tier1


@pytest.mark.parametrize(
    "name,count", [("myoelbow", 6), ("myoarm_r", 63), ("myolegs", 80)]
)
def test_compiled_muscles_wraps_and_mesh_transforms(tmp_path, name, count):
    model = myo_sim.load_spec(name).compile()
    fingerprint = model_fingerprint(model)
    document = export_model(model, tmp_path / "model.json")
    assert len(document["muscles"]) == count
    assert sum(len(t["elements"]) for t in document["tendons"]) == model.nwrap
    assert len(document["joints"]) == model.njnt
    assert len(document["equalities"]) == model.neq
    assert json.loads((tmp_path / "model.json").read_text()) == document
    data = mujoco.MjData(model)
    project_joint_couplings(JointCouplings.from_model(model), data.qpos)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(document["frames"][0]["geom_pos"], data.geom_xpos)
    np.testing.assert_allclose(document["frames"][0]["site_pos"], data.site_xpos)
    mesh = next(g for g in document["geoms"] if g["type"] == "MESH")
    mid = model.geom_dataid[mesh["id"]]
    np.testing.assert_array_equal(
        mesh["faces"],
        model.mesh_face[
            model.mesh_faceadr[mid] : model.mesh_faceadr[mid] + model.mesh_facenum[mid]
        ],
    )
    assert model_fingerprint(model) == fingerprint


def test_elbow_range_length_derivatives(tmp_path):
    model = myo_sim.load_spec("myoelbow").compile()
    poses = joint_sweep(model, "r_elbow_flex", 31)
    document = export_model(model, tmp_path / "sweep.json", qpos=poses)
    data = mujoco.MjData(model)
    for pose, frame in zip(poses[1:-1], document["frames"][1:-1]):
        lengths = []
        for sign in (-1, 1):
            data.qpos[:] = pose + sign * 1e-6
            mujoco.mj_forward(model, data)
            lengths.append(data.actuator_length.copy())
        derivative = (lengths[1] - lengths[0]) / 2e-6
        np.testing.assert_allclose(
            np.asarray(frame["moment_arms"])[:, 0], derivative, atol=1e-7, rtol=1e-5
        )


def test_couplings_and_joint_types_are_not_collapsed(tmp_path):
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="root"><freejoint name="free"/><geom size=".1"/>
        <body name="ball_body" pos="0 0 1"><joint name="ball" type="ball"/><geom size=".1"/>
          <body name="a"><joint name="driver" ref="20" range="-60 60"/><geom size=".1"/>
            <body name="b"><joint name="follower" type="slide" ref=".1" range="-.5 .5"/>
              <geom size=".1"/></body></body></body></body></worldbody>
      <equality><joint joint1="follower" joint2="driver" polycoef=".02 .3 .4 0 0"/></equality>
      </mujoco>""")
    poses = np.tile(model.qpos0, (2, 1))
    poses[1, model.joint("driver").qposadr[0]] += 0.2
    document = export_model(model, tmp_path / "coupled.json", qpos=poses)
    assert {j["type"] for j in document["joints"]} == {"FREE", "BALL", "HINGE", "SLIDE"}
    follower = model.joint("follower").qposadr[0]
    np.testing.assert_allclose(
        [f["qpos"][follower] for f in document["frames"]],
        [0.12, 0.12 + 0.3 * 0.2 + 0.4 * 0.2**2],
    )
    with pytest.raises(ValueError, match="independent"):
        joint_sweep(model, "follower")


@pytest.mark.parametrize("poses", [np.zeros((0, 1)), np.zeros((3, 2)), [[np.nan]]])
def test_invalid_poses_rejected(tmp_path, poses):
    model = myo_sim.load_spec("myoelbow").compile()
    with pytest.raises(ValueError, match="qpos"):
        export_model(model, tmp_path / "bad.json", qpos=poses)
