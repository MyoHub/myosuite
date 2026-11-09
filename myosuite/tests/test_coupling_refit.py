# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Refitting clips retargeted on a negated knee coupling (``coupling_refit``).

Two legs under a pelvis; each knee drives an axial shank rotation through
MyoFullBody's ``knee_angle_rotation3`` polynomial. The "old" model negates the
left polynomial, as musclemimic_models 1.0.5 did. As in MyoFullBody, the free
joint can sit on a container body with the pelvis welded below it.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from myosuite.physics.coupling_refit import (
    CouplingRefitConfig,
    refit_stale_couplings,
    stale_joint_couplings,
)
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings

pytestmark = pytest.mark.tier1

_POLY = np.array([0.0, 0.3695, -0.1695, 0.0252, 0.0])
_LEG = """
<body name="femur_{s}" pos="0 {y} -0.1">
  <joint name="hip_flexion_{s}" axis="0 1 0" range="-1.2 1.2"/>
  <joint name="hip_adduction_{s}" axis="1 0 0" range="-0.5 0.5"/>
  <joint name="hip_rotation_{s}" axis="0 0 1" range="-0.7 0.7"/>
  <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.05" mass="5"/>
  <body name="tibia_{s}" pos="0 0 -0.4">
    <joint name="knee_{s}" axis="0 1 0" range="0 2"/>
    <joint name="knee_rotation_{s}" axis="0 0 1" range="-0.3 0.3"/>
    <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.04" mass="3"/>
    <body name="calcn_{s}" pos="0 0 -0.4">
      <joint name="ankle_{s}" axis="0 1 0" range="-0.7 0.5"/>
      <geom type="box" pos="0.05 0 -0.03" size="0.1 0.04 0.02" mass="1"/>
      <body name="toes_{s}" pos="0.15 0 -0.04">
        <geom size="0.02" mass="0.1"/>
      </body>
    </body>
  </body>
</body>
"""


def _model(left_sign: float, container: bool = True) -> mujoco.MjModel:
    poly = " ".join(f"{c:g}" for c in _POLY)
    left = " ".join(f"{left_sign * c:g}" for c in _POLY)
    pelvis = f"""
    <body name="pelvis" pos="0 0 {0.0 if container else 1.0}">
      {"" if container else "<freejoint/>"}
      <geom size="0.1" mass="10"/>
      {_LEG.format(s="l", y=0.1)}
      {_LEG.format(s="r", y=-0.1)}
    </body>"""
    if container:
        pelvis = f'<body name="root" pos="0 0 1"><freejoint/><geom size="0.05" mass="1"/>{pelvis}</body>'
    return mujoco.MjModel.from_xml_string(f"""
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    {pelvis}
  </worldbody>
  <equality>
    <joint joint1="knee_rotation_l" joint2="knee_l" polycoef="{left}"/>
    <joint joint1="knee_rotation_r" joint2="knee_r" polycoef="{poly}"/>
  </equality>
</mujoco>""")


def _gait(model: mujoco.MjModel, frames: int = 24) -> np.ndarray:
    """A swing-like clip on *model*'s couplings: knees 0.1-1.3 rad, hips and ankles moving."""
    t = np.linspace(0.0, 2 * np.pi, frames)
    qpos = np.tile(model.qpos0, (frames, 1))
    qpos[:, :3] = [0.0, 0.0, 1.0]
    for s, phase in (("l", 0.0), ("r", np.pi)):
        a = {
            n: model.joint(f"{n}_{s}").qposadr[0]
            for n in ("hip_flexion", "hip_adduction", "hip_rotation", "knee", "ankle")
        }
        qpos[:, a["hip_flexion"]] = 0.5 * np.sin(t + phase)
        qpos[:, a["hip_adduction"]] = 0.08 * np.cos(t + phase)
        qpos[:, a["hip_rotation"]] = 0.1 * np.sin(2 * t + phase)
        qpos[:, a["knee"]] = 0.7 - 0.6 * np.cos(t + phase)
        qpos[:, a["ankle"]] = 0.25 * np.sin(t + phase + 0.5)
    project_joint_couplings(JointCouplings.from_model(model), qpos)
    return qpos


def _bodies(
    model: mujoco.MjModel, qpos: np.ndarray, names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    ids = [model.body(n).id for n in names]
    pos, quat = np.empty((len(qpos), len(ids), 3)), np.empty((len(qpos), len(ids), 4))
    for t, frame in enumerate(qpos):
        data.qpos[:] = frame
        mujoco.mj_kinematics(model, data)
        pos[t], quat[t] = data.xpos[ids], data.xquat[ids]
    return pos, quat


def _angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.degrees(2 * np.arccos(np.clip(np.abs((a * b).sum(-1)), 0.0, 1.0)))


def test_detects_only_the_negated_coupling() -> None:
    new, old = _model(1.0), _model(-1.0)
    clip = _gait(new)
    assert stale_joint_couplings(new, clip) == []
    assert stale_joint_couplings(old, clip) == ["knee_rotation_l"]
    # A straight knee is on both polynomials: nothing to refit.
    straight = clip.copy()
    straight[:, [new.joint(n).qposadr[0] for n in ("knee_l", "knee_rotation_l")]] = 0.0
    assert stale_joint_couplings(old, straight) == []


def test_clip_on_the_couplings_is_returned_unchanged() -> None:
    new = _model(1.0)
    clip = _gait(new)
    np.testing.assert_array_equal(refit_stale_couplings(new, clip), clip)


def test_refit_recovers_the_fit_on_the_model_couplings() -> None:
    pytest.importorskip("mink")
    new, old = _model(1.0), _model(-1.0)
    truth = _gait(new)
    # GMR on the old coupling, targeting the true bodies, then the refit back on the model's.
    retargeted = refit_stale_couplings(old, truth)
    refit = refit_stale_couplings(new, retargeted)
    projected = retargeted.copy()
    project_joint_couplings(JointCouplings.from_model(new), projected)

    names = ["femur_l", "tibia_l", "calcn_l", "toes_l"]
    _, q_true = _bodies(new, truth, names)
    err = {
        k: _angle_deg(_bodies(new, q, names)[1], q_true).max(axis=0)
        for k, q in (
            ("retargeted", retargeted),
            ("projected", projected),
            ("refit", refit),
        )
    }
    # The old fit shares the twist between thigh and shank; projection keeps the thigh's share
    # and over-rotates the shank and foot.
    assert err["retargeted"][0] > 3.0 and err["retargeted"][1] > 3.0
    assert err["projected"][1:].min() > 5.0
    # First order only: the knee twist reaches 0.24 rad here, so 1.5 deg remain.
    assert err["refit"].max() < 2.5
    assert err["refit"].max() < 0.25 * err["projected"].max()


@pytest.mark.parametrize("container", [True, False])
def test_refit_moves_only_the_stale_limb_onto_the_couplings(container: bool) -> None:
    pytest.importorskip("mink")
    new, old = _model(1.0, container), _model(-1.0, container)
    retargeted = refit_stale_couplings(old, _gait(new))
    refit = refit_stale_couplings(new, retargeted)
    left = [
        new.joint(n).qposadr[0]
        for n in (
            "hip_flexion_l",
            "hip_adduction_l",
            "hip_rotation_l",
            "knee_l",
            "knee_rotation_l",
            "ankle_l",
        )
    ]
    others = np.setdiff1d(np.arange(new.nq), left)
    np.testing.assert_array_equal(refit[:, others], retargeted[:, others])
    assert np.abs(refit[:, left] - retargeted[:, left]).max() > 0.05
    on_couplings = refit.copy()
    project_joint_couplings(JointCouplings.from_model(new), on_couplings)
    np.testing.assert_allclose(refit, on_couplings, rtol=0, atol=1e-12)
    assert stale_joint_couplings(new, refit) == []


def test_limb_without_frame_costs_raises() -> None:
    pytest.importorskip("mink")
    new, old = _model(1.0), _model(-1.0)
    with pytest.raises(ValueError, match="frame cost"):
        refit_stale_couplings(
            old, _gait(new), CouplingRefitConfig(frame_costs={"pelvis": (1.0, 1.0)})
        )


def test_invalid_qpos_raises() -> None:
    model = _model(1.0)
    for bad in (np.zeros((3, model.nq - 1)), np.full((3, model.nq), np.nan)):
        with pytest.raises(ValueError, match="qpos"):
            refit_stale_couplings(model, bad)


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="equality_cost"):
        CouplingRefitConfig(equality_cost=0.0)
    with pytest.raises(ValueError, match="max_iters"):
        CouplingRefitConfig(max_iters=0)
