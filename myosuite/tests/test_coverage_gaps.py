# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Coverage-gap tests for muscle_conditions, event_terms, registry, trajectory_io, and reward_terms."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytestmark = pytest.mark.tier1

# Import only available at module level so ClassVar is recognised by @dataclass.
pytest.importorskip("mujoco", reason="mujoco required for registry tests")

from myosuite.core.config import (  # noqa: E402
    ActuatorGroupSpec,
    BackendConfig,
    GoalSpec,
    ObsSpec,
    RewardSpec,
    TaskConfig,
    VariantSpec,
)


@dataclass
class _MyoVariantTask(TaskConfig):
    """Task with myo-prefixed id and non-empty config_delta for registry variant tests."""

    model: str = "elbow_standard"
    obs: ObsSpec = field(default_factory=lambda: ObsSpec(keys=["joint_pos"]))
    goal: GoalSpec = field(
        default_factory=lambda: GoalSpec(
            target_type="joint_angles",
            randomize=False,
            range={"r_elbow_flex": (1.0, 1.0)},
        )
    )
    reward: RewardSpec = field(default_factory=lambda: RewardSpec(terms=["pose"]))
    actuators: list[ActuatorGroupSpec] = field(
        default_factory=lambda: [ActuatorGroupSpec()]
    )
    max_episode_steps: int = 10
    backend: BackendConfig = field(
        default_factory=lambda: BackendConfig(n_substeps=2, ctrl_dt=0.002)
    )
    some_flag: bool = False
    variants: ClassVar[list[VariantSpec]] = [
        VariantSpec(suffix="Sarc", config_delta={"some_flag": True}),
    ]


# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------


class _FakeAccessor:
    """Minimal EnvAccessor stub for event-term tests."""

    def __init__(self, n_act: int = 4) -> None:
        self._n_act = n_act
        self._act = np.zeros(n_act)

    def muscle_act(self) -> np.ndarray:
        return self._act.copy()

    def array_module(self) -> Any:
        return np


class _FakeAccessorWithModel(_FakeAccessor):
    """Accessor that also exposes a mock MjModel (for sarcopenia tests)."""

    def __init__(self, n_act: int = 4, n_actuators: int = 4) -> None:
        super().__init__(n_act)
        self.model = MagicMock()
        self.model.actuator_gainprm = np.ones((n_actuators, 10))


# ---------------------------------------------------------------------------
# muscle_conditions.CumulativeFatigue
# ---------------------------------------------------------------------------


def _make_fatigue_model(n: int) -> CumulativeFatigue:  # noqa: F821
    """Build a CumulativeFatigue with n uniform muscle actuators (no real MjModel needed)."""
    import mujoco
    from myosuite.core.muscle_conditions import CumulativeFatigue

    mock = MagicMock()
    mock.opt.timestep = 0.002
    mock.nu = n
    mock.actuator_dyntype = np.full(n, mujoco.mjtDyn.mjDYN_MUSCLE)
    mock.actuator_dynprm = np.tile([0.01, 0.04] + [0.0] * 8, (n, 1))
    return CumulativeFatigue(mock, use_uniform_params=True)


def _fatigue_stepper(backend: str, n: int) -> Any:
    """Return ``step(TL, dt) -> MA`` for the numpy or torch 3CC-r model (tau 10/40 ms)."""
    if backend == "numpy":
        f = _make_fatigue_model(n)
        return lambda tl, dt: f.compute_act(tl.copy(), dt=dt)[0].copy()
    torch = pytest.importorskip("torch")
    from myosuite.core.muscle_conditions import TorchFatigueState

    t = TorchFatigueState(num_envs=1, n_muscles=n)
    return lambda tl, dt: (
        t.step(torch.tensor(tl[None], dtype=torch.float32), dt)[0].double().numpy()
    )


# Control steps of the registered myoFati* envs (10, 20 and 25 ms).
_FATIGUE_CTRL_DTS = (0.01, 0.02, 0.025)


def _rest_protocol(n: int = 3) -> np.ndarray:
    """Target loads at 20 ms: (10 s at 50%, 10 s at 20%, 10 s at rest) x 2."""
    block = np.repeat([0.5, 0.2, 0.0], 500)
    return np.tile(np.tile(block, 2)[:, None], (1, n))


def _rest_protocol_trace(backend: str, loads: np.ndarray, dt: float = 0.02) -> dict:
    """MA, MF before and after each step of the uniform-parameter model.

    Starts with MF = 0.3 so that the recovery rate is visible from the start.
    """
    from myosuite.core.muscle_conditions import _UNIFORM_PARAMS, TorchFatigueState

    n = loads.shape[1]
    mf0 = np.full(n, 0.3)
    if backend == "numpy":
        cpu = _make_fatigue_model(n)
        cpu.reset(fatigue_reset_vec=mf0)

        def step(tl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            cpu.compute_act(tl.copy(), dt=dt)
            return cpu.MA.copy(), cpu.MF.copy()

        state = (cpu.MA.copy(), cpu.MF.copy())
    else:
        torch = pytest.importorskip("torch")
        gpu = TorchFatigueState(1, n, **{k: _UNIFORM_PARAMS[k] for k in "FRr"})
        gpu.reset(fatigue_reset_vec=mf0)

        def step(tl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            gpu.step(torch.tensor(tl[None], dtype=torch.float32), dt)
            return gpu.MA[0].double().numpy(), gpu.MF[0].double().numpy()

        state = (gpu.MA[0].double().numpy(), gpu.MF[0].double().numpy())
    ma, mf = [state[0]], [state[1]]
    for tl in loads:
        new_ma, new_mf = step(tl)
        ma.append(new_ma)
        mf.append(new_mf)
    return {"MA": np.array(ma), "MF": np.array(mf), "dt": dt, **_UNIFORM_PARAMS}


def _effective_recovery_multiplier(trace: dict) -> np.ndarray:
    """``rR / R`` of each step, from ``dMF = (F MA - rR MF) dt``."""
    ma, mf = trace["MA"], trace["MF"]
    dmf_dt = np.diff(mf, axis=0) / trace["dt"]
    return (trace["F"] * ma[:-1] - dmf_dt) / (trace["R"] * mf[:-1])


class TestCumulativeFatigue:
    def test_init_compartments(self) -> None:
        f = _make_fatigue_model(6)
        assert f.MA.shape == (6,)
        assert f.MF.shape == (6,)
        assert f.MR.shape == (6,)
        assert np.all(f.MA == 0.0)
        assert np.all(f.MF == 0.0)
        assert np.all(f.MR == 1.0)

    def test_step_output_shape(self) -> None:
        f = _make_fatigue_model(3)
        out = f.step(np.ones(3) * 0.5, dt=0.01)
        assert out.shape == (3,)

    def test_step_output_within_bounds(self) -> None:
        f = _make_fatigue_model(4)
        for _ in range(50):
            out = f.step(np.random.rand(4), dt=0.01)
        assert np.all(out >= 0.0)
        assert np.all(out <= 1.0)

    def test_step_monotone_accumulation(self) -> None:
        f = _make_fatigue_model(2)
        # MA should increase from zero when excitation > MA
        f.step(np.ones(2), dt=1.0)
        assert np.all(f.MA > 0.0), "Active compartment must grow"

    def test_reset_restores_initial_state(self) -> None:
        f = _make_fatigue_model(3)
        for _ in range(20):
            f.step(np.ones(3) * 0.8, dt=0.01)
        f.reset()
        assert np.all(f.MA == 0.0)
        assert np.all(f.MF == 0.0)
        assert np.all(f.MR == 1.0)

    def test_custom_rates(self) -> None:
        f = _make_fatigue_model(2)
        f.set_FatigueCoefficient(0.1)
        f.set_RecoveryCoefficient(0.05)
        assert f.F == pytest.approx(0.1)
        assert f.R == pytest.approx(0.05)

    @pytest.mark.parametrize("dt", _FATIGUE_CTRL_DTS)
    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_step_from_rest_does_not_overshoot_command(
        self, backend: str, dt: float
    ) -> None:
        # LD * dt reaches 5 at a control step; an explicit Euler step then
        # drove MA from rest to 1.0 for a 0.25 command.
        tl = np.array([0.05, 0.1, 0.25, 0.5, 0.8, 1.0])
        ma = _fatigue_stepper(backend, tl.size)(tl, dt)
        assert np.all(ma > 0.0)
        assert np.all(ma <= tl + 1e-6), f"MA {ma} overshoots TL {tl}"

    @pytest.mark.parametrize("dt", _FATIGUE_CTRL_DTS)
    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_step_never_crosses_command(self, backend: str, dt: float) -> None:
        # MA moves towards TL and stops short of it (on the way down, only the
        # fatigue drain F * MA * dt < 3e-4 may take it below TL).
        step = _fatigue_stepper(backend, 8)
        rng = np.random.default_rng(0)
        ma = np.zeros(8)
        for _ in range(300):
            tl = rng.uniform(0.0, 1.0, 8)
            new = step(tl, dt)
            rising = ma < tl
            assert np.all(new[rising] <= tl[rising] + 1e-6)
            assert np.all(new[~rising] >= tl[~rising] - 1e-3)
            ma = new

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_rest_multiplier_applies_only_at_rest(self, backend: str) -> None:
        """``r`` multiplies recovery only when TL == 0 (Rakshit et al. 2021, Eq. 7).

        On the 20% steps that follow 50% the muscle relaxes (MA >= TL) under a
        non-zero load, which the model used to treat as rest.
        """
        loads = _rest_protocol()
        trace = _rest_protocol_trace(backend, loads)
        mult = _effective_recovery_multiplier(trace)
        rest = loads == 0.0
        relaxing = (trace["MA"][:-1] >= loads) & ~rest
        assert relaxing.sum() > 100, "protocol must relax under load"
        np.testing.assert_allclose(mult[rest], trace["r"], rtol=0.05)
        np.testing.assert_allclose(mult[~rest], 1.0, atol=0.3)

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_negative_command_counts_as_rest(self, backend: str) -> None:
        """A negative command (``[-1, 1]`` muscle ctrl ranges, clamped to zero
        excitation by MuJoCo) recovers like TL = 0."""
        rest = _rest_protocol_trace(backend, np.zeros((100, 3)))
        negative = _rest_protocol_trace(backend, np.full((100, 3), -1.0))
        np.testing.assert_allclose(negative["MF"], rest["MF"], atol=1e-7)
        mult = _effective_recovery_multiplier(negative)
        np.testing.assert_allclose(mult, negative["r"], rtol=0.05)

    def test_rest_protocol_cpu_matches_torch(self) -> None:
        """CPU and torch agree on load / relax / rest cycles."""
        pytest.importorskip("torch")
        loads = _rest_protocol()
        cpu = _rest_protocol_trace("numpy", loads)
        gpu = _rest_protocol_trace("torch", loads)
        for k in ("MA", "MF"):
            np.testing.assert_allclose(gpu[k], cpu[k], atol=1e-5, err_msg=k)

    def test_sustained_command_fatigue_matches_fine_step(self) -> None:
        # At the 20 ms control step MF must match the same ODE integrated at
        # the 2 ms physics step. Overshooting MA to TL used to select the
        # recovery rate r * R (then applied at MA >= TL) on every other step
        # and under-counted MF (-9%).
        coarse, fine = _make_fatigue_model(1), _make_fatigue_model(1)
        tl = np.full(1, 0.3)
        for _ in range(1500):  # 30 s
            coarse.compute_act(tl, dt=0.02)
            for _ in range(10):
                fine.compute_act(tl, dt=0.002)
        np.testing.assert_allclose(coarse.MF, fine.MF, rtol=0.02)

    def test_per_muscle_params_ignore_attach_prefix_and_side_suffix(self) -> None:
        """Prefixed and side-suffixed muscle names get their group's F / R / r.

        mjlab scenes prefix actuators (``robot/BIClong``); ``myohand_r.xml`` and
        the left arm of full-body models suffix the side (``ECRL_r``, ``BIClong_l``).
        """
        import mujoco
        from myosuite.core.muscle_conditions import (
            MUSCLE_FATIGUE_PARAMS,
            CumulativeFatigue,
        )

        def fatigue(names: list[str]) -> CumulativeFatigue:
            mock = MagicMock()
            mock.opt.timestep = 0.002
            mock.nu = len(names)
            mock.actuator_dyntype = np.full(len(names), mujoco.mjtDyn.mjDYN_MUSCLE)
            mock.actuator_dynprm = np.tile([0.01, 0.04] + [0.0] * 8, (len(names), 1))
            mock.actuator.side_effect = lambda i: SimpleNamespace(name=names[i])
            return CumulativeFatigue(mock)

        got = fatigue(
            ["robot/BIClong", "ECRL_r", "robot/ECRL_r", "BIClong_l", "robot/soleus_r"]
        )
        want = fatigue(["BIClong", "ECRL", "ECRL", "BIClong", "soleus_r"])
        assert np.all(want.F != MUSCLE_FATIGUE_PARAMS["Default"]["F"])
        for p in ("F", "R", "r"):
            np.testing.assert_array_equal(getattr(got, p), getattr(want, p), p)

    def test_get_effort_is_distance_to_target_load(self) -> None:
        """get_effort() (legacy API, JAX twins) is ||MA - TL|| after compute_act."""
        f = _make_fatigue_model(3)
        tl = np.array([0.2, 0.6, 1.0])
        f.compute_act(tl, dt=0.02)
        assert f.get_effort() == pytest.approx(float(np.linalg.norm(f.MA - tl)))
        assert f.get_effort() > 0.0

    def test_torch_nan_excitation_does_not_poison_state(self) -> None:
        """One NaN excitation gives C = 0 on torch as on CPU; the state stays finite."""
        torch = pytest.importorskip("torch")
        import mujoco
        from myosuite.core.muscle_conditions import CumulativeFatigue, TorchFatigueState

        model = mujoco.MjModel.from_xml_string(_FATIGUE_MUSCLE_XML)
        cpu = CumulativeFatigue(model)
        gpu = TorchFatigueState.from_mj_model(model, num_envs=2)
        rng = np.random.default_rng(0)
        for k in range(40):
            tl = rng.uniform(0.0, 1.0, cpu.na)
            if k == 5:
                tl[0] = np.nan
            cpu.compute_act(tl.copy(), dt=0.02)
            gpu.step(torch.tensor(np.stack([tl, np.nan_to_num(tl)])), 0.02)
            assert bool(torch.isfinite(gpu.MA).all())
            np.testing.assert_allclose(gpu.MA[0].numpy(), cpu.MA, atol=1e-5)
            np.testing.assert_allclose(gpu.MF[0].numpy(), cpu.MF, atol=1e-5)

    def test_torch_state_dtype_is_fixed(self) -> None:
        """A float64 excitation no longer switches the float32 state to float64."""
        torch = pytest.importorskip("torch")
        from myosuite.core.muscle_conditions import TorchFatigueState

        f32 = TorchFatigueState(num_envs=1, n_muscles=3)
        f64 = TorchFatigueState(num_envs=1, n_muscles=3)
        tl = torch.tensor([[0.2, 0.5, 0.9]], dtype=torch.float64)
        out = f64.step(tl, 0.02)
        f32.step(tl.float(), 0.02)
        assert (
            f64.MA.dtype == f64.MR.dtype == f64.MF.dtype == out.dtype == torch.float32
        )
        torch.testing.assert_close(f64.MA, f32.MA)

    def test_uniform_params_match_across_backends(self) -> None:
        """use_uniform_params: the v2.4 row and the model's tau on CPU and torch."""
        torch = pytest.importorskip("torch")
        import mujoco
        from myosuite.core.muscle_conditions import (
            MUSCLE_FATIGUE_PARAMS,
            CumulativeFatigue,
            TorchFatigueState,
        )

        model = mujoco.MjModel.from_xml_string(_FATIGUE_MUSCLE_XML)
        cpu = CumulativeFatigue(model, use_uniform_params=True)
        gpu = TorchFatigueState.from_mj_model(model, 1, use_uniform_params=True)
        v24 = MUSCLE_FATIGUE_PARAMS["Default_v2_4"]
        for p in ("F", "R", "r"):
            np.testing.assert_allclose(getattr(cpu, p), v24[p])
        np.testing.assert_allclose(gpu._tauact.numpy(), model.actuator_dynprm[:, 0])
        np.testing.assert_allclose(gpu._taudeact.numpy(), model.actuator_dynprm[:, 1])
        rng = np.random.default_rng(1)
        for _ in range(200):
            tl = rng.uniform(0.0, 1.0, cpu.na)
            cpu.compute_act(tl, dt=0.02)
            gpu.step(torch.tensor(tl[None], dtype=torch.float32), 0.02)
        np.testing.assert_allclose(gpu.MA[0].numpy(), cpu.MA, atol=1e-5)
        np.testing.assert_allclose(gpu.MF[0].numpy(), cpu.MF, atol=1e-5)

    def test_torch_reset_vec_matches_cpu(self) -> None:
        """A fixed fatigue vector resets the chosen envs like the CPU model."""
        torch = pytest.importorskip("torch")
        from myosuite.core.muscle_conditions import TorchFatigueState

        cpu = _make_fatigue_model(3)
        gpu = TorchFatigueState(num_envs=4, n_muscles=3)
        for _ in range(30):
            cpu.compute_act(np.full(3, 0.7), dt=0.02)
            gpu.step(torch.full((4, 3), 0.7), 0.02)
        before = gpu.MF.clone()
        vec = np.array([0.1, 0.4, 0.8])
        cpu.reset(fatigue_reset_vec=vec)
        gpu.reset(torch.tensor([1, 3]), fatigue_reset_vec=vec)
        for name in ("MA", "MR", "MF"):
            got = getattr(gpu, name)[[1, 3]].double().numpy()
            np.testing.assert_allclose(
                got, np.tile(getattr(cpu, name), (2, 1)), atol=1e-7
            )
        torch.testing.assert_close(gpu.MF[[0, 2]], before[[0, 2]])

    def test_torch_reset_random_matches_cpu_distribution(self) -> None:
        """Random resets: seeded, per env, and the CPU construction of the state.

        CPU and torch draw from different generators, so the states are compared
        in distribution: ``MF = 1 - nf`` and ``MA / (MA + MR) = ap`` are uniform.
        """
        torch = pytest.importorskip("torch")
        from myosuite.core.muscle_conditions import TorchFatigueState

        n = 3
        gpu = TorchFatigueState(num_envs=4000, n_muscles=n)
        gpu.reset(fatigue_reset_random=True, generator=torch.Generator().manual_seed(0))
        again = TorchFatigueState(num_envs=4000, n_muscles=n)
        again.reset(
            fatigue_reset_random=True, generator=torch.Generator().manual_seed(0)
        )
        torch.testing.assert_close(gpu.MF, again.MF)
        total = (gpu.MA + gpu.MR + gpu.MF).double().numpy()
        np.testing.assert_allclose(total, 1.0, atol=1e-6)
        assert len(torch.unique(gpu.MF[:, 0])) > 3900  # one draw per env

        cpu = _make_fatigue_model(n)
        rng = np.random.default_rng(0)
        states = []
        for _ in range(4000):
            cpu.reset(fatigue_reset_random=True, np_random=rng)
            states.append((cpu.MA.copy(), cpu.MR.copy(), cpu.MF.copy()))
        ma, mr, mf = (np.stack(s) for s in zip(*states))
        q = np.linspace(0.05, 0.95, 19)
        for cpu_x, gpu_x in (
            (mf, gpu.MF.double().numpy()),
            (ma, gpu.MA.double().numpy()),
            (ma / (ma + mr), (gpu.MA / (gpu.MA + gpu.MR)).double().numpy()),
        ):
            np.testing.assert_allclose(
                np.quantile(gpu_x, q), np.quantile(cpu_x, q), atol=0.02
            )

    def test_torch_reset_rejects_invalid_options(self) -> None:
        """Both options at once, or a vector of the wrong length, raise as on CPU."""
        torch = pytest.importorskip("torch")
        from myosuite.core.muscle_conditions import TorchFatigueState

        gpu = TorchFatigueState(num_envs=2, n_muscles=3)
        with pytest.raises(ValueError, match="Cannot pass"):
            gpu.reset(fatigue_reset_vec=[0.1] * 3, fatigue_reset_random=True)
        with pytest.raises(ValueError, match="length"):
            gpu.reset(fatigue_reset_vec=[0.1, 0.2])
        torch.testing.assert_close(gpu.MR, torch.ones(2, 3))


# Three muscles; the first two have non-default activation time constants.
_FATIGUE_MUSCLE_XML = """
<mujoco>
  <worldbody>
    <body>
      <joint name="j" type="hinge" range="-1 1" limited="true"/>
      <geom size="0.1"/>
    </body>
  </worldbody>
  <actuator>
    <muscle name="m0" joint="j" timeconst="0.02 0.08"/>
    <muscle name="m1" joint="j" timeconst="0.005 0.03"/>
    <muscle name="m2" joint="j"/>
  </actuator>
</mujoco>
"""


def _active_muscle_force(model: Any) -> np.ndarray:
    """Active muscle force at qpos0: actuator force at act=1 minus act=0."""
    import mujoco

    data = mujoco.MjData(model)
    forces = []
    for act in (1.0, 0.0):
        data.act[:] = act
        mujoco.mj_forward(model, data)
        forces.append(data.actuator_force.copy())
    return forces[0] - forces[1]


class TestApplySarcopenia:
    # One muscle with automatic peak force (force="-1", the default) and one explicit.
    _XML = """
        <mujoco>
          <worldbody>
            <site name="s0" pos="0 0 0"/>
            <body pos="0.05 0 -0.3">
              <joint name="j" type="hinge" axis="0 1 0" limited="true" range="-1 1"/>
              <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.2"/>
              <site name="s1" pos="0.05 0 -0.1"/>
            </body>
          </worldbody>
          <tendon><spatial name="t"><site site="s0"/><site site="s1"/></spatial></tendon>
          <actuator>
            <muscle name="auto" tendon="t"/>
            <muscle name="explicit" tendon="t" force="80"/>
          </actuator>
        </mujoco>
        """

    def test_apply_sarcopenia_to_model_scales_gainprm(self) -> None:
        from myosuite.core.muscle_conditions import apply_sarcopenia_to_model

        model = MagicMock()
        model.actuator_gainprm = np.ones((5, 10))
        apply_sarcopenia_to_model(model, force_scale=0.5)
        assert np.all(model.actuator_gainprm[:, 2] == pytest.approx(0.5))

    def test_apply_sarcopenia_to_model_default_scale(self) -> None:
        from myosuite.core.muscle_conditions import apply_sarcopenia_to_model

        model = MagicMock()
        model.actuator_gainprm = np.full((3, 10), 2.0)
        apply_sarcopenia_to_model(model)
        assert np.all(model.actuator_gainprm[:, 2] == pytest.approx(1.0))

    def test_apply_sarcopenia_to_spec_matches_model(self) -> None:
        """Spec-level sarcopenia compiles to the same peak forces as the CPU path."""
        import mujoco

        from myosuite.core.muscle_conditions import (
            apply_sarcopenia_to_model,
            apply_sarcopenia_to_spec,
        )

        cpu_model = mujoco.MjSpec.from_string(self._XML).compile()
        apply_sarcopenia_to_model(cpu_model, force_scale=0.5)

        spec = mujoco.MjSpec.from_string(self._XML)
        returned = apply_sarcopenia_to_spec(spec, force_scale=0.5)
        assert returned is spec
        np.testing.assert_allclose(
            spec.compile().actuator_gainprm[:, 2], cpu_model.actuator_gainprm[:, 2]
        )
        assert cpu_model.actuator_gainprm[1, 2] == pytest.approx(40.0)

    def test_apply_sarcopenia_scales_auto_peak_force(self) -> None:
        """Both paths halve the active force, including force="-1" muscles."""
        import mujoco

        from myosuite.core.muscle_conditions import (
            apply_sarcopenia_to_model,
            apply_sarcopenia_to_spec,
        )

        healthy = _active_muscle_force(mujoco.MjSpec.from_string(self._XML).compile())
        assert np.all(np.abs(healthy) > 1.0)

        cpu_model = mujoco.MjSpec.from_string(self._XML).compile()
        apply_sarcopenia_to_model(cpu_model, force_scale=0.5)
        spec = mujoco.MjSpec.from_string(self._XML)
        apply_sarcopenia_to_spec(spec, force_scale=0.5)

        for model in (cpu_model, spec.compile()):
            np.testing.assert_allclose(_active_muscle_force(model), 0.5 * healthy)


# ---------------------------------------------------------------------------
# myo_event_terms: apply_sarcopenia
# ---------------------------------------------------------------------------


class TestApplySarcopeniaEventTerm:
    def test_apply_sarcopenia_scales_model(self) -> None:
        from myosuite.terms.base_event import apply_sarcopenia

        acc = _FakeAccessorWithModel(n_actuators=4)
        state: dict = {}
        apply_sarcopenia(acc, state, force_scale=0.5)
        assert np.all(acc.model.actuator_gainprm[:, 2] == pytest.approx(0.5))

    def test_apply_sarcopenia_no_model_attr_is_noop(self) -> None:
        from myosuite.terms.base_event import apply_sarcopenia

        acc = _FakeAccessor()
        state: dict = {}
        result = apply_sarcopenia(acc, state, force_scale=0.5)
        assert result is state  # unchanged

    def test_apply_sarcopenia_returns_state(self) -> None:
        from myosuite.terms.base_event import apply_sarcopenia

        acc = _FakeAccessorWithModel()
        state = {"key": 42}
        result = apply_sarcopenia(acc, state)
        assert result is state
        assert result["key"] == 42


# ---------------------------------------------------------------------------
# core/trajectory_io: resolve_motion_path additional branches
# ---------------------------------------------------------------------------


def _write_npz(path: Path, nq: int, nv: int) -> None:
    np.savez(
        path,
        qpos=np.zeros((4, nq), dtype=np.float64),
        qvel=np.zeros((4, nv), dtype=np.float64),
        site_xpos=np.zeros((4, 3), dtype=np.float64),
        frequency=np.array(120.0, dtype=np.float64),
    )


class TestResolveMotionPath:
    def test_empty_string_raises(self) -> None:
        from myosuite.core.trajectory_io import resolve_motion_path

        with pytest.raises(ValueError, match="empty"):
            resolve_motion_path("")

    def test_absolute_path_resolved(self, tmp_path: Path) -> None:
        from myosuite.core.trajectory_io import resolve_motion_path

        p = tmp_path / "clip.npz"
        _write_npz(p, nq=3, nv=3)
        got = resolve_motion_path(str(p))
        assert got == p.resolve()

    def test_cwd_relative_path_resolved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from myosuite.core.trajectory_io import resolve_motion_path

        p = tmp_path / "relclip.npz"
        _write_npz(p, nq=2, nv=2)
        monkeypatch.chdir(tmp_path)
        got = resolve_motion_path("relclip.npz")
        assert got == p.resolve()

    def test_cache_root_with_npz_extension(self, tmp_path: Path) -> None:
        from myosuite.core.trajectory_io import resolve_motion_path

        env_name = "TestEnv"
        cache_file = tmp_path / env_name / "gmr" / "clip.npz"
        cache_file.parent.mkdir(parents=True)
        _write_npz(cache_file, nq=2, nv=2)
        got = resolve_motion_path("clip.npz", env_name=env_name, cache_root=tmp_path)
        assert got == cache_file.resolve()

    def test_cache_root_auto_appends_npz(self, tmp_path: Path) -> None:
        from myosuite.core.trajectory_io import resolve_motion_path

        env_name = "TestEnv2"
        cache_file = tmp_path / env_name / "gmr" / "clip.npz"
        cache_file.parent.mkdir(parents=True)
        _write_npz(cache_file, nq=2, nv=2)
        # Pass without .npz extension → should auto-append
        got = resolve_motion_path("clip", env_name=env_name, cache_root=tmp_path)
        assert got == cache_file.resolve()

    def test_not_found_raises(self, tmp_path: Path) -> None:
        from myosuite.core.trajectory_io import resolve_motion_path

        with pytest.raises(FileNotFoundError):
            resolve_motion_path("nonexistent.npz", cache_root=tmp_path)


class TestLoadMotionClip:
    def test_full_clip_loaded(self, tmp_path: Path) -> None:
        from myosuite.core.trajectory_io import load_motion_clip

        p = tmp_path / "full.npz"
        _write_npz(p, nq=5, nv=4)
        clip = load_motion_clip(p, expected_nq=5, expected_nv=4)
        assert clip.qpos.shape == (4, 5)
        assert clip.qvel is not None and clip.qvel.shape == (4, 4)
        assert clip.site_xpos is not None
        assert clip.frequency_hz == pytest.approx(120.0)

    def test_missing_qpos_raises(self, tmp_path: Path) -> None:
        from myosuite.core.trajectory_io import load_motion_clip

        p = tmp_path / "no_qpos.npz"
        np.savez(p, qvel=np.zeros((4, 3)))
        with pytest.raises(KeyError, match="qpos"):
            load_motion_clip(p, expected_nq=3, expected_nv=3)

    def test_1d_qpos_raises(self, tmp_path: Path) -> None:
        from myosuite.core.trajectory_io import load_motion_clip

        p = tmp_path / "rank1.npz"
        np.savez(p, qpos=np.zeros(4))
        with pytest.raises(ValueError, match="rank-2"):
            load_motion_clip(p, expected_nq=4, expected_nv=3)

    def test_qvel_wrong_width_raises(self, tmp_path: Path) -> None:
        from myosuite.core.trajectory_io import load_motion_clip

        p = tmp_path / "bad_qvel.npz"
        np.savez(p, qpos=np.zeros((4, 3)), qvel=np.zeros((4, 5)))
        with pytest.raises(ValueError, match="qvel width mismatch"):
            load_motion_clip(p, expected_nq=3, expected_nv=3)

    def test_no_optional_keys(self, tmp_path: Path) -> None:
        from myosuite.core.trajectory_io import load_motion_clip

        p = tmp_path / "minimal.npz"
        np.savez(p, qpos=np.zeros((4, 3)))
        clip = load_motion_clip(p, expected_nq=3, expected_nv=3)
        assert clip.qvel is None
        assert clip.site_xpos is None
        assert clip.frequency_hz is None

    def test_bad_frequency_handled(self, tmp_path: Path) -> None:
        from myosuite.core.trajectory_io import load_motion_clip

        p = tmp_path / "bad_freq.npz"
        # frequency as a multi-element array → reshape(()) fails → None
        np.savez(p, qpos=np.zeros((4, 3)), frequency=np.array([1.0, 2.0]))
        clip = load_motion_clip(p, expected_nq=3, expected_nv=3)
        assert clip.frequency_hz is None


# ---------------------------------------------------------------------------
# registry: make_env MJX/mjlab import-error paths
# ---------------------------------------------------------------------------


class TestMakeEnvBackends:
    def test_make_env_mjx_import_error(self) -> None:
        from myosuite.core.registry import make_env

        with patch.dict("sys.modules", {"mujoco_playground": None}):
            with pytest.raises(ImportError, match="mujoco_playground"):
                make_env("NonExistent-v0", backend="mjx")

    def test_make_env_mjx_forwards_overrides(self) -> None:
        """Keyword overrides reach mujoco_playground as config_overrides."""
        import types

        from myosuite.core.registry import make_env

        calls: list[tuple[str, Any]] = []

        def _load(
            env_id: str, config: Any = None, config_overrides: Any = None
        ) -> None:
            calls.append((env_id, config_overrides))

        registry = types.ModuleType("mujoco_playground.registry")
        registry.load = _load  # type: ignore[attr-defined]
        playground = types.ModuleType("mujoco_playground")
        playground.registry = registry  # type: ignore[attr-defined]
        with patch.dict(
            "sys.modules",
            {"mujoco_playground": playground, "mujoco_playground.registry": registry},
        ):
            make_env("Task-v0", backend="mjx", num_envs=8)
            make_env("Task-v0", backend="mjx")
        assert calls == [("Task-v0", {"num_envs": 8}), ("Task-v0", None)]

    def test_make_env_mjlab_import_error(self) -> None:
        from myosuite.core.registry import make_env

        with patch.dict("sys.modules", {"mjlab": None, "mjlab.envs": None}):
            with pytest.raises(ImportError, match="mjlab"):
                make_env("NonExistent-v0", backend="mjlab")


# ---------------------------------------------------------------------------
# registry: variant expansion with "myo" prefix and non-empty config_delta
# ---------------------------------------------------------------------------


class TestRegistryVariantPaths:
    def test_myo_prefix_variant_id_format(self) -> None:
        """A 'myo'-prefixed base_env_id generates 'myoSarc...-v0' variant id."""
        import gymnasium as gym
        from myosuite.core.registry import register_task

        base_id = "myoTestElbowVariant-v0"
        register_task(_MyoVariantTask(), env_id=base_id)
        assert base_id in gym.envs.registry
        variant_id = "myoSarcTestElbowVariant-v0"
        assert variant_id in gym.envs.registry

    def test_nonempty_config_delta_registers_variant(self) -> None:
        """VariantSpec with non-empty config_delta must register the variant env_id."""
        import gymnasium as gym
        from myosuite.core.registry import _ENV_REGISTRY, register_task

        base_id = "myoTestElbowDelta-v0"
        register_task(_MyoVariantTask(), env_id=base_id)
        variant_id = "myoSarcTestElbowDelta-v0"
        assert variant_id in _ENV_REGISTRY
        assert variant_id in gym.envs.registry


# ---------------------------------------------------------------------------
# walk_env_reward (myosuite.terms.base_reward)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# MuscleActionTerm (mjlab-style)
# ---------------------------------------------------------------------------


class TestMuscleActionTerm:
    def _make_term(self, normalize: bool = True) -> Any:
        from myosuite.terms.base_action import (
            MuscleActionTerm,
            MuscleActionTermCfg,
        )

        mock_entity = MagicMock()
        mock_entity.num_actuators = 4
        mock_env = MagicMock()
        mock_env.scene = {"robot": mock_entity}
        cfg = MuscleActionTermCfg(entity_name="robot", normalize=normalize)
        return MuscleActionTerm(cfg, mock_env), mock_entity

    def test_action_dim(self) -> None:
        term, entity = self._make_term()
        assert term.action_dim == 4

    def test_process_actions_normalize_true(self) -> None:
        term, _ = self._make_term(normalize=True)
        actions = np.array([[-1.0, 0.0, 1.0, 2.0]])
        term.process_actions(actions)
        expected = 1.0 / (1.0 + np.exp(-5.0 * (actions - 0.5)))
        np.testing.assert_allclose(term._processed, expected)

    def test_process_actions_normalize_false(self) -> None:
        term, _ = self._make_term(normalize=False)
        actions = np.array([[-1.0, 0.5, 1.0, 2.0]])
        term.process_actions(actions)
        expected = np.clip(actions, 0, 1)
        np.testing.assert_allclose(term._processed, expected)

    def test_apply_actions_calls_set_ctrl(self) -> None:
        term, entity = self._make_term()
        actions = np.array([[0.2, 0.4, 0.6, 0.8]])
        term.process_actions(actions)
        term.apply_actions()
        entity.set_ctrl.assert_called_once()

    def test_apply_actions_no_op_when_not_processed(self) -> None:
        term, entity = self._make_term()
        term.apply_actions()  # _processed is None
        entity.set_ctrl.assert_not_called()


class TestWalkEnvReward:
    def _make_task_state(self) -> dict:
        qpos = np.zeros(30)
        qpos[3] = 1.0  # quaternion w=1
        return {
            "qpos": qpos,
            "height": 1.0,
            "com_vel": np.array([0.0, 1.0]),
        }

    def _make_accessor(self) -> _FakeAccessor:
        return _FakeAccessor()

    def test_returns_required_keys(self) -> None:
        from myosuite.terms.base_reward import walk_env_reward

        acc = self._make_accessor()
        state = self._make_task_state()
        result = walk_env_reward(
            acc,
            state,
            hip_flex_indices=(10, 15),
            hip_angle_indices=(11, 16, 12, 17),
            target_rot=np.array([1.0, 0.0, 0.0, 0.0]),
            target_vel=(0.0, 1.2),
            min_height=0.8,
            max_rot=0.8,
        )
        for key in (
            "vel_reward",
            "done",
            "solved",
            "cyclic_hip",
            "ref_rot",
            "joint_angle_rew",
            "dense",
        ):
            assert key in result, f"Missing key: {key}"

    def test_terminates_on_low_height(self) -> None:
        from myosuite.terms.base_reward import walk_env_reward

        acc = self._make_accessor()
        state = self._make_task_state()
        state["height"] = 0.1  # below min_height=0.8
        result = walk_env_reward(
            acc,
            state,
            hip_flex_indices=(10, 15),
            hip_angle_indices=(11, 16, 12, 17),
            target_rot=np.array([1.0, 0.0, 0.0, 0.0]),
            min_height=0.8,
            max_rot=0.8,
        )
        assert float(result["done"]) != 0.0

    def test_com_vel_indices_branch(self) -> None:
        from myosuite.terms.base_reward import walk_env_reward

        acc = self._make_accessor()
        state = self._make_task_state()
        state["com_vel"] = np.array([0.0, 0.5, 1.0])  # 3-element vector
        result = walk_env_reward(
            acc,
            state,
            hip_flex_indices=(10, 15),
            hip_angle_indices=(11, 16, 12, 17),
            target_rot=np.array([1.0, 0.0, 0.0, 0.0]),
            com_vel_indices=(0, 2),  # vx=com_vel[0], vy=com_vel[2]
        )
        assert "vel_reward" in result

    def test_com_height_index_branch(self) -> None:
        from myosuite.terms.base_reward import walk_env_reward

        acc = self._make_accessor()
        state = self._make_task_state()
        state["height_like"] = np.array([0.5, 1.2, 0.9])
        result = walk_env_reward(
            acc,
            state,
            hip_flex_indices=(10, 15),
            hip_angle_indices=(11, 16, 12, 17),
            target_rot=np.array([1.0, 0.0, 0.0, 0.0]),
            com_height_index=1,  # use height_like[1]=1.2
        )
        assert "dense" in result
