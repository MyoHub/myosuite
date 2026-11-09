# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""CPU resets on the joint couplings (``mjEQ_JOINT``) and the knee moment arms there.

MuJoCo's own equality residuals are the reference. The projection itself is tested
in myo_sim (``myo_sim.physics.joint_coupling``).
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from myosuite import make_env
from myosuite.tests.support.joint_couplings import joint_equality_residuals
from myosuite.tests.support.optional_deps import require_musclemimic_models

pytestmark = pytest.mark.tier1

# Envs that reset through MyoGymnasiumEnv.reset (the MuscleMimic models).
_BASE_RESET_ENVS = (
    "myoMimicFullbody-v0",
    "myoMimicBimanual-v0",
    "myoFullBodyWaypoint-v0",
)


@pytest.mark.parametrize("env_id", _BASE_RESET_ENVS)
def test_cpu_resets_start_on_the_couplings(env_id: str) -> None:
    """``qpos0``, the keyframe and the Waypoint reset noise left the patella up to
    5 cm off its knee coupling; ``MyoGymnasiumEnv.reset`` now projects it."""
    require_musclemimic_models()
    env = make_env(env_id)
    model, data = env.unwrapped.model, env.unwrapped.data
    for seed in range(3):
        env.reset(seed=seed)
        pos, vel = joint_equality_residuals(model, data.qpos, data.qvel)
        assert np.abs(pos).max() < 1e-12
        assert np.abs(vel).max() < 1e-12
    env.close()


def _dense_moment(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    moment = np.zeros((model.nu, model.nv))
    mujoco.mju_sparse2dense(
        moment,
        data.actuator_moment,
        data.moment_rownnz,
        data.moment_rowadr,
        data.moment_colind,
    )
    return moment


@pytest.mark.parametrize("side", ["r", "l"])
def test_reset_knee_moment_arms_match_finite_differences_along_the_coupling(
    side: str,
) -> None:
    """At reset, ``actuator_moment`` describes the knee mechanism the muscles can reach.

    A muscle's moment arm about the knee along its coupling,
    ``sum_d dL/dq_d dq_d/dq_knee``, equals the central difference of
    ``actuator_length`` between states on the coupling. With the patella 5 cm off
    it, the analytic moment arms were those of an unreachable pose (rectus femoris
    -3 mm instead of +44 mm).
    """
    require_musclemimic_models()
    env = make_env("myoMimicFullbody-v0")
    model, data = env.unwrapped.model, env.unwrapped.data
    env.reset(seed=0)
    knee = model.joint(f"knee_angle_{side}").id
    knee_q, knee_v = model.jnt_qposadr[knee], model.jnt_dofadr[knee]
    eqs = [
        e
        for e in range(model.neq)
        if model.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT and model.eq_obj2id[e] == knee
    ]
    assert len(eqs) == 7  # tibia translations and rotations, patella
    poly = np.polynomial.polynomial

    def on_coupling(angle: float) -> np.ndarray:
        qpos = data.qpos.copy()
        qpos[knee_q] = angle
        for e in eqs:
            adr = model.jnt_qposadr[model.eq_obj1id[e]]
            offset = poly.polyval(angle - model.qpos0[knee_q], model.eq_data[e, :5])
            qpos[adr] = model.qpos0[adr] + offset
        return qpos

    angle = float(data.qpos[knee_q])
    tangent = np.zeros(model.nv)
    tangent[knee_v] = 1.0
    for e in eqs:
        slope = poly.polyder(model.eq_data[e, :5])
        tangent[model.jnt_dofadr[model.eq_obj1id[e]]] = poly.polyval(
            angle - model.qpos0[knee_q], slope
        )
    analytic = _dense_moment(model, data) @ tangent

    step, probe, lengths = 1e-6, mujoco.MjData(model), []
    for a in (angle + step, angle - step):
        probe.qpos[:] = on_coupling(a)
        mujoco.mj_fwdPosition(model, probe)
        lengths.append(probe.actuator_length.copy())
    numeric = (lengths[0] - lengths[1]) / (2 * step)
    assert (np.abs(numeric) > 1e-3).sum() >= 10  # the muscles crossing this knee
    np.testing.assert_allclose(analytic, numeric, rtol=0, atol=1e-7)
    env.close()
