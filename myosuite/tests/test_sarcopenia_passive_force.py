# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Sarcopenia scales a muscle's passive force with its peak force.

MuJoCo's passive muscle force is the peak force ``biasprm[2]`` times the passive
force-length curve, so a muscle that loses half its peak force loses half of both.
"""

from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np
import pytest

from myosuite import make_env
from myosuite.core.model_builder import ModelBuilder
from myosuite.core.muscle_conditions import (
    apply_sarcopenia_to_model,
    apply_sarcopenia_to_spec,
)

pytestmark = pytest.mark.tier1

# An explicit-force model and an automatic-force (force="-1") one.
PAIRS = (
    ("myoLegWalk-v0", "myoSarcLegWalk-v0"),
    ("myoFingerPoseRandom-v0", "myoSarcFingerPoseRandom-v0"),
)

_XML = """
<mujoco>
  <worldbody>
    <site name="s0" pos="0 0 .5"/>
    <body>
      <joint name="j" type="hinge" axis="1 0 0" range="-1 1"/>
      <geom size=".1" mass="1"/>
      <site name="s1" pos="0 .2 .1"/>
    </body>
  </worldbody>
  <tendon><spatial name="t"><site site="s0"/><site site="s1"/></spatial></tendon>
  <actuator>
    <muscle name="explicit" tendon="t" force="100"/>
    <muscle name="automatic" tendon="t" force="-1" scale="200"/>
    <position name="position" joint="j" kp="10" kv="2"/>
    <general name="affine_bias" tendon="t" dyntype="muscle" gaintype="muscle"
             biastype="affine" gainprm="0.75 1.05 100 200 0.5 1.6 1.5 1.3 1.2"
             biasprm="1 2 3"/>
  </actuator>
</mujoco>
"""


def _stretched_poses(model: mujoco.MjModel, n: int, seed: int) -> np.ndarray:
    """Poses with every limited hinge or slide joint drawn within its range."""
    rng = np.random.default_rng(seed)
    base = model.key_qpos[0] if model.nkey else model.qpos0
    poses = np.tile(base, (n, 1))
    for j in range(model.njnt):
        if model.jnt_type[j] in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            if model.jnt_limited[j]:
                lo, hi = model.jnt_range[j]
                poses[:, model.jnt_qposadr[j]] = rng.uniform(lo, hi, n)
    return poses


def _forces(model: mujoco.MjModel, qpos: np.ndarray, act: float) -> np.ndarray:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.act[:] = act
    mujoco.mj_forward(model, data)
    return data.actuator_force.copy()


def _assert_half(base: mujoco.MjModel, sarc: mujoco.MjModel) -> None:
    """Passive force (act 0) and active force (act 1 minus act 0) both halve."""
    muscle = base.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
    passive_rows = 0
    for qpos in _stretched_poses(base, 10, 0):
        passive_b, passive_s = _forces(base, qpos, 0.0), _forces(sarc, qpos, 0.0)
        active_b = _forces(base, qpos, 1.0) - passive_b
        active_s = _forces(sarc, qpos, 1.0) - passive_s
        np.testing.assert_allclose(active_s[muscle], 0.5 * active_b[muscle], rtol=1e-9)
        np.testing.assert_allclose(
            passive_s[muscle], 0.5 * passive_b[muscle], rtol=1e-9, atol=1e-12
        )
        passive_rows += int(np.sum(np.abs(passive_b[muscle]) > 1.0))
    assert passive_rows > 0, "no pose stretched a muscle into its passive range"


@pytest.mark.parametrize(("base_id", "sarc_id"), PAIRS)
def test_sarc_ids_halve_passive_and_active_force(base_id: str, sarc_id: str) -> None:
    """CPU ids (``SarcopeniaWrapper``): both force components halve."""
    base, sarc = make_env(base_id), make_env(sarc_id)
    _assert_half(base.unwrapped.model, sarc.unwrapped.model)
    base.close()
    sarc.close()


@pytest.mark.parametrize(("base_id", "_sarc_id"), PAIRS)
def test_spec_sarcopenia_halves_passive_and_active_force(
    base_id: str, _sarc_id: str
) -> None:
    """The spec path (mjlab twins, MJX, model recipes) matches the model path."""
    path = gym.spec(base_id).kwargs["model_path"]
    base, _ = ModelBuilder.from_xml_file(path).build()
    _, spec = ModelBuilder.from_xml_file(path).build()
    _assert_half(base, apply_sarcopenia_to_spec(spec).compile())


def test_only_muscle_bias_rows_hold_a_peak_force() -> None:
    """A non-muscle bias (position servo, affine bias of a muscle gain) is unchanged."""
    spec = mujoco.MjSpec.from_string(_XML)
    model = spec.compile()
    acc0 = model.actuator_acc0[model.actuator("automatic").id]
    expected_peak = {"explicit": 50.0, "automatic": 0.5 * 200.0 / acc0}
    for scaled in (
        _scaled_model(model),
        apply_sarcopenia_to_spec(mujoco.MjSpec.from_string(_XML)).compile(),
    ):
        for name, peak in expected_peak.items():
            i = scaled.actuator(name).id
            assert scaled.actuator_gainprm[i, 2] == pytest.approx(peak)
            assert scaled.actuator_biasprm[i, 2] == pytest.approx(peak)
        for name in ("position", "affine_bias"):
            i = model.actuator(name).id
            np.testing.assert_array_equal(
                scaled.actuator_biasprm[i], model.actuator_biasprm[i]
            )


def _scaled_model(model: mujoco.MjModel) -> mujoco.MjModel:
    scaled = mujoco.MjModel.from_xml_string(_XML)
    apply_sarcopenia_to_model(scaled, force_scale=0.5)
    assert scaled.nu == model.nu
    return scaled
