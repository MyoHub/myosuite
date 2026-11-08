# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Joint couplings (``mjEQ_JOINT``) and CPU resets on their polynomials.

MuJoCo's own equality residuals are the reference, so the checks do not reuse
the arithmetic of :func:`project_joint_couplings`.
"""

from __future__ import annotations

import re

import mujoco
import numpy as np
import pytest

from myosuite import make_env
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings
from myosuite.tests.support.joint_couplings import joint_equality_residuals
from myosuite.tests.support.optional_deps import (
    require_musclemimic_models,
    skip_if_hf_gated,
)
from myosuite.utils import gym

pytestmark = pytest.mark.tier1

# Every CPU registration, scanned from the live gym registry as test_registry.py does.
_ALL_ENV_IDS = sorted(
    env_id for env_id in gym.envs.registry if env_id.startswith(("myo", "motor"))
)
# Projected states are on the couplings to rounding (< 1e-15); the smallest offset a
# reset left (myoLegDirectional*, SoccerP1) was 3.5e-5.
_RESET_ATOL = 1e-9
# MyoChallenge envs still reset off their couplings: their resets are a competition
# contract (PyPI baselines, published policies) and change only with maintainer sign-off.
_OFF_COUPLING_CHALLENGES = re.compile(
    r"myo(Fati|Sarc)?Challenge(Bimanual|ChaseTag(FBP2|FBVs|P1|P2|P2eval)"
    r"|OslRun(Fixed|Random)|Relocate(P1|P2|P2eval)|Soccer(P1|P2)|TableTennis(P0|P1|P2))-v0"
)


def _reset_case(env_id: str) -> object:
    if not _OFF_COUPLING_CHALLENGES.fullmatch(env_id):
        return env_id
    reason = "MyoChallenge reset off its joint couplings (pending maintainer sign-off)"
    return pytest.param(env_id, marks=pytest.mark.xfail(reason=reason, strict=True))


# A slider with a nonzero ref drives a hinge (all five coefficients), which drives
# a second hinge; that chained equality is listed first. One hinge is locked
# without a master, and one equality is inactive.
_XML = """
<mujoco>
  <worldbody>
    <body>
      <joint name="master" type="slide" axis="1 0 0" ref="0.3"/>
      <geom size="0.1" mass="1"/>
      <body pos="0 0 0.3">
        <joint name="dep" axis="0 0 1" ref="-0.2"/>
        <geom size="0.1" mass="1"/>
        <body pos="0 0 0.3">
          <joint name="chained" axis="0 1 0" ref="0.1"/>
          <geom size="0.1" mass="1"/>
          <body pos="0 0 0.3">
            <joint name="locked" axis="1 0 0" ref="0.05"/>
            <geom size="0.1" mass="1"/>
            <body pos="0 0 0.3">
              <joint name="loose" axis="0 1 0"/>
              <geom size="0.1" mass="1"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <equality>
    <joint joint1="chained" joint2="dep" polycoef="0.02 -0.7 0.3 0.1 -0.05"/>
    <joint joint1="dep" joint2="master" polycoef="0.1 0.5 -0.4 0.2 0.3"/>
    <joint joint1="locked" polycoef="-0.15 0 0 0 0"/>
    <joint joint1="loose" joint2="master" polycoef="0.3 1 0 0 0" active="false"/>
  </equality>
</mujoco>
"""


def _random_state(
    model: mujoco.MjModel, rng: np.random.Generator, batch: tuple[int, ...] = ()
) -> tuple[np.ndarray, np.ndarray]:
    return (
        model.qpos0 + rng.uniform(-0.5, 0.5, (*batch, model.nq)),
        rng.uniform(-1.0, 1.0, (*batch, model.nv)),
    )


def test_projection_puts_every_active_coupling_on_its_polynomial() -> None:
    """Dependents, chained and locked joints land on their equalities; nothing else moves."""
    model = mujoco.MjModel.from_xml_string(_XML)
    couplings = JointCouplings.from_model(model)
    assert couplings.level_ends == (1, 2)  # "dep" before "chained"
    qpos, qvel = _random_state(model, np.random.default_rng(0))
    assert np.abs(joint_equality_residuals(model, qpos, qvel)[0]).max() > 0.1

    projected, projected_vel = qpos.copy(), qvel.copy()
    project_joint_couplings(couplings, projected, projected_vel)

    pos, vel = joint_equality_residuals(model, projected, projected_vel)
    assert len(pos) == 3  # the inactive equality has no row
    np.testing.assert_allclose(pos, 0.0, atol=1e-12)
    np.testing.assert_allclose(vel, 0.0, atol=1e-12)
    for name in ("master", "loose"):
        q_adr = model.jnt_qposadr[model.joint(name).id]
        v_adr = model.jnt_dofadr[model.joint(name).id]
        assert projected[q_adr] == qpos[q_adr]
        assert projected_vel[v_adr] == qvel[v_adr]


def test_torch_batch_matches_numpy() -> None:
    """The mjlab path (batched Torch tensors) gives the CPU numbers."""
    torch = pytest.importorskip("torch")
    model = mujoco.MjModel.from_xml_string(_XML)
    couplings = JointCouplings.from_model(model)
    qpos, qvel = _random_state(model, np.random.default_rng(1), (8,))
    qpos_t, qvel_t = torch.tensor(qpos), torch.tensor(qvel)
    project_joint_couplings(couplings.map_arrays(torch.as_tensor), qpos_t, qvel_t)
    for q, v in zip(qpos, qvel):
        project_joint_couplings(couplings, q, v)
    np.testing.assert_allclose(qpos_t.numpy(), qpos, rtol=0, atol=1e-15)
    np.testing.assert_allclose(qvel_t.numpy(), qvel, rtol=0, atol=1e-15)


@pytest.mark.parametrize(
    "equalities",
    [
        '<joint joint1="b" joint2="a"/><joint joint1="b" joint2="c"/>',
        '<joint joint1="b" joint2="a"/><joint joint1="a" joint2="b"/>',
    ],
    ids=["two-masters", "cycle"],
)
def test_rejects_over_constrained_and_cyclic_couplings(equalities: str) -> None:
    xml = f"""
    <mujoco>
      <worldbody>
        <body>
          <joint name="a"/><joint name="b" axis="0 1 0"/><joint name="c" axis="1 0 0"/>
          <geom size="0.1"/>
        </body>
      </worldbody>
      <equality>{equalities}</equality>
    </mujoco>
    """
    with pytest.raises(ValueError):
        JointCouplings.from_model(mujoco.MjModel.from_xml_string(xml))


@pytest.mark.parametrize("env_id", [_reset_case(env_id) for env_id in _ALL_ENV_IDS])
def test_every_env_resets_on_its_joint_couplings(env_id: str) -> None:
    """Every registered env starts its episodes with the coupled joints on their polynomials.

    Keyframes, ``qpos0`` and reset noise left them off: the patella of the full body
    5 cm (``qpos0``), the patella rotation of the leg walks 0.74 rad (keyframe 2).
    MuJoCo pulled them back over the first steps, and ``actuator_moment`` described
    a pose the coupled joints cannot reach until then.
    """
    try:
        env = make_env(env_id)
    except Exception as exc:
        skip_if_hf_gated(env_id, exc)
        raise
    try:
        for seed in range(3):
            env.reset(seed=seed)
            u = env.unwrapped  # read after reset: a scene swap replaces model and data
            pos, vel = joint_equality_residuals(u.model, u.data.qpos, u.data.qvel)
            worst = (np.max(np.abs(pos), initial=0.0), np.max(np.abs(vel), initial=0.0))
            assert (
                max(worst) < _RESET_ATOL
            ), f"seed {seed}: |efc_pos|, |efc_vel| {worst}"
    except Exception as exc:
        skip_if_hf_gated(env_id, exc)
        raise
    finally:
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
