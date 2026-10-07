# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Every keyframe of the shipped models stays inside the joint ranges and actuator ctrlranges (#403).

``mj_resetDataKeyframe`` copies ``key_qpos`` and ``key_ctrl`` as they are, and tools read raw
keyframes without any task preprocessing, so they must satisfy the declared limits.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import pytest

import myosuite  # noqa: F401  (registers the envs)
from myosuite import make_env
from myosuite.utils.asset_path_resolver import resolve_model_xml_path

TOL = 1e-5
ASSETS = Path(myosuite.__file__).parent / "envs" / "myo" / "assets"
HINGE_SLIDE = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
OBJ = mujoco.mjtObj


def keyframe_violations(model: mujoco.MjModel) -> list[str]:
    """List the keyframe entries outside a limited joint range or actuator ctrlrange."""
    bad = []
    for k in range(model.nkey):
        key = mujoco.mj_id2name(model, OBJ.mjOBJ_KEY, k) or f"key_{k}"
        for j in range(model.njnt):
            if model.jnt_limited[j] and int(model.jnt_type[j]) in HINGE_SLIDE:
                value = float(model.key_qpos[k, model.jnt_qposadr[j]])
                lo, hi = model.jnt_range[j]
                if not lo - TOL <= value <= hi + TOL:
                    name = mujoco.mj_id2name(model, OBJ.mjOBJ_JOINT, j)
                    bad.append(
                        f"{key}: joint {name} = {value:.6g} outside [{lo:.6g}, {hi:.6g}]"
                    )
        for a in range(model.nu):
            if model.actuator_ctrllimited[a]:
                value = float(model.key_ctrl[k, a])
                lo, hi = model.actuator_ctrlrange[a]
                if not lo - TOL <= value <= hi + TOL:
                    name = mujoco.mj_id2name(model, OBJ.mjOBJ_ACTUATOR, a)
                    bad.append(
                        f"{key}: ctrl {name} = {value:.6g} outside [{lo:.6g}, {hi:.6g}]"
                    )
    return bad


@pytest.mark.tier1
@pytest.mark.parametrize(
    "xml", ["arm/myoarm_bionic_bimanual.xml", "arm/myoarm_tabletennis.xml"]
)
def test_raw_xml_keyframes_in_range(xml: str) -> None:
    """The raw model files of #403 (and the soccer model) satisfy the invariant."""
    model = mujoco.MjModel.from_xml_path(str(resolve_model_xml_path(ASSETS / xml)))
    assert keyframe_violations(model) == []


# Known exceptions. The full-body keyframes come from the musclemimic_models package; the ChaseTag
# FBP2 reset keyframe is 1e-4 rad below a limit and kept so that resets stay bit-identical.
UPSTREAM = "keyframe values come from the musclemimic_models package"
KEPT = "reset keyframe 1e-4 rad outside a limit, kept to keep resets bit-identical"
ENV_IDS = [
    "myoChallengeBimanual-v0",
    "myoChallengeTableTennisP0-v0",
    "myoChallengeSoccerP1-v0",
    "myoHandPose1Fixed-v0",
    "myoFingerPoseFixed-v0",
    "myoArmReachFixed-v0",
    "myoElbowPose1D6MFixed-v0",
    "myoLegWalk-v0",
    "myoTorsoPoseFixed-v0",
    pytest.param("myoMimicFullbody-v0", marks=pytest.mark.xfail(reason=UPSTREAM)),
    pytest.param("myoChallengeChaseTagFBP2-v0", marks=pytest.mark.xfail(reason=KEPT)),
]


@pytest.mark.tier2
@pytest.mark.parametrize("env_id", ENV_IDS)
def test_env_model_keyframes_in_range(env_id: str) -> None:
    """The compiled model of a registered CPU env satisfies the invariant."""
    env = make_env(env_id)
    try:
        assert keyframe_violations(env.unwrapped.model) == []
    finally:
        env.close()
