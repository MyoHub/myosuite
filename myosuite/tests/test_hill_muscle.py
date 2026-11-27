# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Facade contracts and MuJoCo parity checks for ``myosuite.physics.hill_muscle``."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import mujoco
import numpy as np
import pytest

from myosuite.core.model_builder import ModelBuilder
from myosuite.physics import hill_muscle
from myosuite.physics.hill_muscle import (
    HillMuscleModel,
    MuscleParams,
    dense_actuator_moment,
    muscle_activation_dot,
    muscle_actuator_force,
    muscle_bias,
    muscle_gain,
    step_muscle_activation,
    strip_muscles_for_joint_torque_sim,
)
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings

pytestmark = pytest.mark.tier1

_XML = """
<mujoco>
  <worldbody>
    <site name="origin" pos="0 0 .1"/>
    <body><joint name="hinge" axis="0 1 0" range="-1 1"/>
      <geom type="capsule" fromto="0 0 0 .3 0 0" size=".02" mass="1"/>
      <site name="insertion" pos=".2 0 .05"/></body>
  </worldbody>
  <tendon><spatial name="path"><site site="origin"/><site site="insertion"/></spatial></tendon>
  <actuator><muscle name="m" tendon="path" force="-1" scale="200"/></actuator>
</mujoco>
"""


def _synthetic_params(nu: int = 2) -> MuscleParams:
    return MuscleParams(
        tau_act=np.full(nu, 0.01),
        tau_deact=np.full(nu, 0.04),
        tausmooth=np.full(nu, 0.2),
        act_limited=np.ones(nu, dtype=bool),
        act_range=np.tile([0.0, 1.0], (nu, 1)),
        length_range=np.tile([0.1, 0.2], (nu, 1)),
        acc0=np.full(nu, 0.02),
        gain_prm=np.ones((nu, 9)),
        bias_prm=np.ones((nu, 9)),
    )


def test_facade_is_lazy_and_reports_actionable_import_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code = f"""
import importlib.abc
import sys

class MissingHill(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'myo_sim.physics.hill_muscle':
            raise ModuleNotFoundError(fullname, name=fullname)

sys.meta_path.insert(0, MissingHill())

import mujoco
import numpy as np
from myosuite.physics import HillMuscleModel, strip_muscles_for_joint_torque_sim
from myosuite.physics.hill_muscle import (
    MuscleParams,
    dense_actuator_moment,
    muscle_activation_dot,
    muscle_actuator_force,
    muscle_bias,
    muscle_gain,
    step_muscle_activation,
)

assert 'myo_sim.physics.hill_muscle' not in sys.modules
assert 'torch' not in sys.modules

spec = mujoco.MjSpec.from_string({_XML!r})
model = spec.copy().compile()
data = mujoco.MjData(model)
params = MuscleParams(
    tau_act=np.ones(1),
    tau_deact=np.ones(1),
    tausmooth=np.ones(1),
    act_limited=np.ones(1, dtype=bool),
    act_range=np.zeros((1, 2)),
    length_range=np.zeros((1, 2)),
    acc0=np.ones(1),
    gain_prm=np.ones((1, 9)),
    bias_prm=np.ones((1, 9)),
)
vec = np.zeros(1)
calls = (
    lambda: MuscleParams.from_model(model),
    lambda: HillMuscleModel(model),
    lambda: muscle_activation_dot(vec, vec, params),
    lambda: step_muscle_activation(vec, vec, params, 0.002),
    lambda: muscle_gain(vec, vec, params),
    lambda: muscle_bias(vec, params),
    lambda: muscle_actuator_force(vec, vec, vec, params),
    lambda: dense_actuator_moment(model, data),
    lambda: strip_muscles_for_joint_torque_sim(spec),
)
for call in calls:
    try:
        call()
    except ImportError as exc:
        assert 'MyoHub/myo_sim#200' in str(exc) and 'Install' in str(exc)
    else:
        raise AssertionError('missing API did not raise ImportError')
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr

    def broken(name: str) -> object:
        raise ModuleNotFoundError("unrelated dependency defect", name="broken_dep")

    monkeypatch.setattr(hill_muscle.importlib, "import_module", broken)
    spec = mujoco.MjSpec.from_string(_XML)
    with pytest.raises(ModuleNotFoundError) as exc_info:
        strip_muscles_for_joint_torque_sim(spec)
    assert exc_info.value.name == "broken_dep"


def test_facade_forwards_all_calls_to_myo_sim(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
    raw_params = _synthetic_params(2)

    class StubHillModel:
        def __init__(
            self,
            model: Any = None,
            *,
            params: Any = None,
            nv: int | None = None,
            dt: float = 0.002,
        ) -> None:
            recorded.append(
                (
                    "HillMuscleModel",
                    (model,),
                    {"params": params, "nv": nv, "dt": dt},
                )
            )
            self.params = params if params is not None else raw_params
            self.nu = 2
            self.nv = nv if nv is not None else 3
            self.dt = dt

        def generalized_forces(
            self, length: Any, velocity: Any, act: Any, moment_arm: Any
        ) -> np.ndarray:
            recorded.append(
                ("generalized_forces", (length, velocity, act, moment_arm), {})
            )
            return np.full(self.nv, 4.5)

    def _record(name: str, ret: Any) -> Any:
        def _fn(*args: Any, **kwargs: Any) -> Any:
            recorded.append((name, args, kwargs))
            return ret

        return _fn

    stub = SimpleNamespace(
        MuscleParams=SimpleNamespace(
            from_model=_record("from_model", SimpleNamespace(**raw_params.__dict__))
        ),
        HillMuscleModel=StubHillModel,
        muscle_activation_dot=_record("muscle_activation_dot", np.array([0.1, 0.2])),
        step_muscle_activation=_record(
            "step_muscle_activation", (np.array([0.3, 0.4]), np.array([0.2, 0.3]))
        ),
        muscle_gain=_record("muscle_gain", np.array([-10.0, -20.0])),
        muscle_bias=_record("muscle_bias", np.array([-1.0, -2.0])),
        muscle_actuator_force=_record(
            "muscle_actuator_force", np.array([-11.0, -22.0])
        ),
        dense_actuator_moment=_record("dense_actuator_moment", np.ones((2, 3))),
        strip_muscles_for_joint_torque_sim=_record(
            "strip_muscles_for_joint_torque_sim", "stripped_spec"
        ),
    )
    monkeypatch.setattr(hill_muscle.importlib, "import_module", lambda name: stub)

    spec = mujoco.MjSpec.from_string(_XML)
    model = spec.copy().compile()
    data = mujoco.MjData(model)

    extracted = MuscleParams.from_model(model)
    assert isinstance(extracted, MuscleParams)
    np.testing.assert_array_equal(extracted.tau_act, raw_params.tau_act)

    vec = np.array([0.2, 0.5])
    np.testing.assert_allclose(muscle_activation_dot(vec, vec, extracted), [0.1, 0.2])
    nxt, mean = step_muscle_activation(vec, vec, extracted, 0.002, substeps=3)
    np.testing.assert_allclose(nxt, [0.3, 0.4])
    np.testing.assert_allclose(mean, [0.2, 0.3])
    assert recorded[-1][2] == {"substeps": 3}

    np.testing.assert_allclose(muscle_gain(vec, vec, extracted), [-10.0, -20.0])
    np.testing.assert_allclose(muscle_bias(vec, extracted), [-1.0, -2.0])
    np.testing.assert_allclose(
        muscle_actuator_force(vec, vec, vec, extracted), [-11.0, -22.0]
    )
    assert dense_actuator_moment(model, data).shape == (2, 3)

    hill = HillMuscleModel(params=extracted, nv=3, dt=0.005)
    assert hill.nu == 2 and hill.nv == 3 and hill.dt == 0.005
    assert isinstance(hill.params, MuscleParams)
    np.testing.assert_allclose(
        hill.generalized_forces(vec, vec, vec, np.ones((2, 3))), [4.5, 4.5, 4.5]
    )

    out = strip_muscles_for_joint_torque_sim(spec, add_joint_motors=False)
    assert out == "stripped_spec"
    assert recorded[-1][2] == {"add_joint_motors": False}


@pytest.mark.parametrize("fragment", ["hand", "arm"])
def test_hill_muscle_exact_parity_against_mujoco_c(fragment: str) -> None:
    pytest.importorskip("myo_sim.physics.hill_muscle")
    model, _ = ModelBuilder().attach_fragment(fragment).build()
    data = mujoco.MjData(model)
    rng = np.random.default_rng(123 if fragment == "hand" else 456)

    qpos = model.qpos0.copy()
    for j in range(model.njnt):
        qadr = int(model.jnt_qposadr[j])
        if model.jnt_limited[j]:
            lo, hi = model.jnt_range[j]
            qpos[qadr] = rng.uniform(
                0.3 * lo + 0.7 * qpos[qadr], 0.3 * hi + 0.7 * qpos[qadr]
            )
    project_joint_couplings(JointCouplings.from_model(model), qpos)

    data.qpos[:] = qpos
    data.qvel[:] = rng.normal(scale=0.25, size=model.nv)
    data.act[:] = rng.uniform(0.05, 0.95, size=model.nu)
    data.ctrl[:] = rng.uniform(0.0, 1.0, size=model.nu)
    mujoco.mj_forward(model, data)

    params = MuscleParams.from_model(model)
    hill = HillMuscleModel(model)

    act_dot = muscle_activation_dot(data.ctrl, data.act, params)
    np.testing.assert_allclose(act_dot, data.act_dot, atol=1e-12, rtol=1e-12)

    gain = muscle_gain(data.actuator_length, data.actuator_velocity, params)
    bias = muscle_bias(data.actuator_length, params)
    force = muscle_actuator_force(
        data.actuator_length, data.actuator_velocity, data.act, params
    )
    np.testing.assert_allclose(force, data.actuator_force, atol=1e-11, rtol=1e-11)
    assert np.all(gain <= 0.0)
    assert np.all(bias <= 0.0)

    moment = dense_actuator_moment(model, data)
    qfrc = hill.generalized_forces(
        data.actuator_length, data.actuator_velocity, data.act, moment
    )
    np.testing.assert_allclose(qfrc, data.qfrc_actuator, atol=1e-11, rtol=1e-11)

    next_act, mean_act = step_muscle_activation(
        data.act, data.ctrl, params, dt=float(model.opt.timestep), substeps=1
    )
    np.testing.assert_allclose(mean_act, data.act, atol=1e-15)
    np.testing.assert_allclose(
        next_act,
        np.clip(data.act + model.opt.timestep * data.act_dot, 0.0, 1.0),
        atol=1e-12,
    )


def test_strip_muscles_for_joint_torque_sim_preserves_kinematics_and_couplings() -> (
    None
):
    pytest.importorskip("myo_sim.physics.hill_muscle")
    orig, spec = ModelBuilder().attach_fragment("arm").build()
    stripped_spec = strip_muscles_for_joint_torque_sim(spec, add_joint_motors=True)
    stripped = stripped_spec.compile()

    assert stripped.ntendon == 0
    assert stripped.nbody == orig.nbody
    assert stripped.nsite == orig.nsite
    assert stripped.neq == orig.neq
    assert np.all(stripped.actuator_trntype == int(mujoco.mjtTrn.mjTRN_JOINT))
    assert np.all(stripped.actuator_dyntype == int(mujoco.mjtDyn.mjDYN_NONE))
    assert stripped.nu == orig.nv - orig.neq
