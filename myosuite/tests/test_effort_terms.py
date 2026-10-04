# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Effort terms (``myosuite.terms.effort``), the accessor data they read and the
joint-limit terms (``joint_penalty`` / ``joint_limit_violation``) on real models."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
import pytest

import myosuite  # noqa: F401
from myosuite.core.muscle_conditions import CumulativeFatigue
from myosuite.physics.muscle import (
    MuscleParams,
    active_force_length,
    passive_force,
)
from myosuite.terms import effort
from myosuite.terms.base_reward import joint_penalty
from myosuite.terms.base_termination import joint_limit_violation

pytestmark = pytest.mark.tier1

_SIGMA, _RHO = effort.UMBERGER_SPECIFIC_TENSION, effort.UMBERGER_DENSITY


def _env(env_id: str, steps: int = 5, seed: int = 0) -> Any:
    env = gym.make(env_id).unwrapped
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        env.step(rng.uniform(-1, 1, env.action_space.shape).astype(np.float32))
    return env


class _ArrayAccessor:
    """EnvAccessor over fixed arrays (numpy, or torch float64 for parity tests)."""

    def __init__(self, xp: Any, dt: float = 0.01, **arrays: Any) -> None:
        self._xp, self._dt, self._a = xp, dt, arrays

    @classmethod
    def snapshot(cls, acc: Any, xp: Any = np) -> _ArrayAccessor:
        """Copy the state a real CPU accessor exposes, converted to *xp*."""
        conv = (lambda v: v) if xp is np else (lambda v: xp.as_tensor(np.asarray(v)))
        params = acc.muscle_params().map(lambda _, v: conv(v))
        qids, ranges = acc.joint_range()
        return cls(
            xp,
            acc.dt(),
            joint_pos=conv(acc.joint_pos()),
            muscle_act=conv(acc.muscle_act()),
            muscle_force=conv(acc.muscle_force()),
            muscle_length=conv(acc.muscle_length()),
            muscle_velocity=conv(acc.muscle_velocity()),
            qfrc_actuator=conv(acc.qfrc_actuator()),
            muscle_params=params,
            joint_range=(conv(qids), conv(ranges)),
        )

    def __getattr__(self, name: str) -> Any:
        if name in self.__dict__.get("_a", {}):
            return lambda: self._a[name]
        raise AttributeError(name)

    def dt(self) -> float:
        return self._dt

    def array_module(self) -> Any:
        return self._xp


# ---------------------------------------------------------------------------
# Muscle curves and accessor data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env_id", ["myoElbowPose1D6MExoRandom-v0", "myoArmReachRandom-v0"]
)
def test_muscle_curves_match_mujoco(env_id: str) -> None:
    """The xp ports of MuJoCo's force-length curves equal mju_muscleGain / mju_muscleBias."""
    env = gym.make(env_id).unwrapped
    m, p = env.model, env._accessor.muscle_params()
    ids = np.flatnonzero(m.actuator_gaintype == mujoco.mjtGain.mjGAIN_MUSCLE)
    # The arm's biasprm (passive curve) differs from its gainprm (active curve).
    for norm_len in np.linspace(0.0, 2.6, 131):
        length = p.length_range_lo + (norm_len - p.range_lo) * p.optimal_length
        fl = active_force_length(np.full(len(ids), norm_len), p.lmin, p.lmax, np)
        fp = passive_force(length, p, np)
        for k, i in enumerate(ids):
            lr, acc0 = m.actuator_lengthrange[i], m.actuator_acc0[i]
            gain = mujoco.mju_muscleGain(
                length[k], 0.0, lr, acc0, m.actuator_gainprm[i, :9]
            )
            bias = mujoco.mju_muscleBias(length[k], lr, acc0, m.actuator_biasprm[i, :9])
            assert fl[k] == pytest.approx(-gain / p.peak_force[k], abs=1e-12)
            assert fp[k] == pytest.approx(-bias, rel=1e-12, abs=1e-9)


def test_cpu_accessor_muscle_state_excludes_motors() -> None:
    """Muscle getters skip the exo motor (actuator 0) and match MuJoCo's muscle force."""
    env = _env("myoElbowPose1D6MExoRandom-v0")
    m, d, acc = env.model, env.data, env._accessor
    assert m.nu == 7 and m.actuator_gaintype[0] != mujoco.mjtGain.mjGAIN_MUSCLE
    np.testing.assert_array_equal(acc.muscle_force(), d.actuator_force[1:])
    np.testing.assert_array_equal(acc.muscle_length(), d.actuator_length[1:])
    np.testing.assert_array_equal(acc.muscle_velocity(), d.actuator_velocity[1:])
    p = acc.muscle_params()
    np.testing.assert_array_equal(p.act_ids, np.arange(6))
    np.testing.assert_array_equal(p.peak_force, m.actuator_gainprm[1:, 2])
    force = [
        mujoco.mju_muscleGain(
            d.actuator_length[i],
            d.actuator_velocity[i],
            m.actuator_lengthrange[i],
            m.actuator_acc0[i],
            m.actuator_gainprm[i, :9],
        )
        * d.act[m.actuator_actadr[i]]
        + mujoco.mju_muscleBias(
            d.actuator_length[i],
            m.actuator_lengthrange[i],
            m.actuator_acc0[i],
            m.actuator_biasprm[i, :9],
        )
        for i in range(1, 7)
    ]
    np.testing.assert_allclose(acc.muscle_force(), force, rtol=1e-12)
    np.testing.assert_array_equal(acc.qfrc_actuator(), d.qfrc_actuator)


def test_cpu_joint_range_skips_free_root() -> None:
    """joint_range() lists the limited hinge/slide joints at their qpos addresses."""
    env = gym.make("myoLegWalk-v0").unwrapped
    m = env.model
    qids, ranges = env._accessor.joint_range()
    scalar = np.isin(
        m.jnt_type, (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
    )
    keep = scalar & m.jnt_limited.astype(bool)
    np.testing.assert_array_equal(qids, m.jnt_qposadr[keep])
    np.testing.assert_array_equal(ranges, m.jnt_range[keep])
    assert m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE and qids.min() >= 7


# ---------------------------------------------------------------------------
# joint_penalty / joint_limit_violation on real models (nq != nu)
# ---------------------------------------------------------------------------


def test_joint_limit_terms_use_joint_range_on_elbow_with_exo() -> None:
    """Elbow + exo: nq = 1, nu = 7; the elbow range is [0, 2.269] rad, the ctrl ranges [0, 1]."""
    env = gym.make("myoElbowPose1D6MExoRandom-v0").unwrapped
    acc, hi = env._accessor, float(env.model.jnt_range[0, 1])
    assert env.model.nq == 1 and env.model.nu == 7

    def at(q: float) -> tuple[float, bool]:
        env.data.qpos[0] = q
        return float(joint_penalty(acc, {})["joint_penalty"]), bool(
            joint_limit_violation(acc, {})
        )

    assert at(1.5) == (0.0, False)  # beyond every ctrl range, inside the joint range
    assert at(0.5) == (0.0, False)
    pen, out = at(2.2)  # inside the outer 5 % of the range
    assert not out and pen == pytest.approx(-50.0 * (2.2 - 0.95 * hi))
    assert at(hi + 0.05)[1]
    assert at(-0.05)[1]


def test_joint_limit_violation_on_floating_base_leg() -> None:
    """Leg: the free root (qpos 0..6) is ignored and a knee beyond its range is caught."""
    env = gym.make("myoLegWalk-v0").unwrapped
    env.reset(seed=0)
    acc, m = env._accessor, env.model
    env.data.qpos[2] = 50.0  # root height: unlimited
    assert not joint_limit_violation(acc, {})
    knee = m.joint("knee_angle_r")
    env.data.qpos[knee.qposadr[0]] = knee.range[0] - 0.1
    assert joint_limit_violation(acc, {})
    assert joint_penalty(acc, {})["joint_penalty"] < 0.0


# ---------------------------------------------------------------------------
# numpy / torch parity of every term
# ---------------------------------------------------------------------------


def _assert_same(got: Any, want: Any, msg: str) -> None:
    """Torch (or numpy) *got* equals numpy *want*: exactly for flags, else to 1e-6."""
    got = got.numpy() if hasattr(got, "numpy") else np.asarray(got)
    want = np.asarray(want)
    if want.dtype == bool or got.dtype == bool:
        np.testing.assert_array_equal(got, want, err_msg=msg)
    else:
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6, err_msg=msg)


def _all_terms(acc: Any, fatigue: Any, arm_dofs: Any) -> dict[str, dict[str, Any]]:
    return {
        "power_abs": effort.muscle_mechanical_power(acc, {}),
        "power_pos": effort.muscle_mechanical_power(acc, {}, mode="positive"),
        "metabolic_2003": effort.metabolic_energy_rate(acc, {}),
        "metabolic_2010": effort.metabolic_energy_rate(
            acc, {}, version="2010", fast_twitch_fraction=0.3, aerobic_factor=1.0
        ),
        "consumed_endurance": effort.consumed_endurance(
            acc, {}, shoulder_dof_ids=arm_dofs, max_shoulder_torque=5.0
        ),
        "fatigue": effort.fatigue_effort(acc, {"fatigue": fatigue}),
        "discomfort": effort.joint_limit_discomfort(acc, {}),
        "joint_penalty": joint_penalty(acc, {}),
    }


@pytest.mark.parametrize(
    "env_id", ["myoElbowPose1D6MExoRandom-v0", "myoLegWalk-v0", "myoArmReachRandom-v0"]
)
def test_terms_numpy_torch_parity(env_id: str) -> None:
    """Every effort term gives the same values (<= 1e-6) with numpy and torch inputs."""
    torch = pytest.importorskip("torch")
    env = _env(env_id, steps=8)
    acc_np = _ArrayAccessor.snapshot(env._accessor, np)
    acc_t = _ArrayAccessor.snapshot(env._accessor, torch)
    rng = np.random.default_rng(1)
    na = len(acc_np.muscle_act())
    fat_np = SimpleNamespace(
        MA=rng.random(na), MF=rng.random(na) * 0.3, TL=rng.random(na)
    )
    fat_t = SimpleNamespace(**{k: torch.as_tensor(v) for k, v in vars(fat_np).items()})
    dofs = [0] if env.model.nv < 3 else [0, 1, 2]
    out_np = _all_terms(acc_np, fat_np, dofs)
    out_t = _all_terms(acc_t, fat_t, torch.as_tensor(dofs))
    for term, values in out_np.items():
        for key, value in values.items():
            _assert_same(out_t[term][key], value, f"{term}/{key}")


def test_terms_batched_torch_equals_per_env_numpy() -> None:
    """Batched torch inputs (N, ...) give the per-env numpy values."""
    torch = pytest.importorskip("torch")
    snaps = [
        _ArrayAccessor.snapshot(_env("myoLegWalk-v0", steps=s)._accessor)
        for s in (3, 6, 9)
    ]
    stack = {
        name: torch.as_tensor(np.stack([s._a[name] for s in snaps]))
        for name in (
            "joint_pos",
            "muscle_act",
            "muscle_force",
            "muscle_length",
            "muscle_velocity",
            "qfrc_actuator",
        )
    }
    params = snaps[0]._a["muscle_params"].map(lambda _, v: torch.as_tensor(v))
    qids, ranges = snaps[0]._a["joint_range"]
    batched = _ArrayAccessor(
        torch,
        0.01,
        muscle_params=params,
        joint_range=(torch.as_tensor(qids), torch.as_tensor(ranges)),
        **stack,
    )
    fatigue = SimpleNamespace(MA=torch.full((3, 80), 0.2), MF=torch.full((3, 80), 0.1))
    out_b = _all_terms(batched, fatigue, torch.as_tensor([6, 7, 8]))
    for i, snap in enumerate(snaps):
        fat_i = SimpleNamespace(MA=np.full(80, 0.2), MF=np.full(80, 0.1))
        out_i = _all_terms(snap, fat_i, [6, 7, 8])
        for term, values in out_i.items():
            for key, value in values.items():
                got = out_b[term][key]
                _assert_same(
                    got[i] if getattr(got, "ndim", 0) else got, value, f"{term}/{key}"
                )


# ---------------------------------------------------------------------------
# Metabolic model: published constants and the OpenSim reference implementation
# ---------------------------------------------------------------------------

_L0 = 0.1  # m
_F0 = 1.0 * _SIGMA / (_RHO * _L0)  # N, gives a muscle mass of exactly 1 kg
_PRM = np.array(
    [0.5, 1.6, _F0, 1.0, 0.5, 1.6, 10.0, 1.3, 1.2]
)  # MuJoCo muscle defaults, vmax 10
_LR = np.array([0.05, 0.05 + 1.1 * _L0])  # lengthrange for range [0.5, 1.6]


def _one_muscle(norm_len: float, v_norm: float, act: float) -> _ArrayAccessor:
    """One MuJoCo muscle of mass 1 kg at a given normalised length / velocity / activation."""
    length = _LR[0] + (norm_len - _PRM[0]) * _L0
    vel = v_norm * _L0
    force = mujoco.mju_muscleGain(
        length, vel, _LR, 1.0, _PRM
    ) * act + mujoco.mju_muscleBias(length, _LR, 1.0, _PRM)
    params = MuscleParams(
        peak_force=np.array([_F0]),
        optimal_length=np.array([_L0]),
        length_range_lo=_LR[:1],
        range_lo=_PRM[:1],
        lmin=_PRM[4:5],
        lmax=_PRM[5:6],
        vmax=_PRM[6:7],
        fvmax=_PRM[8:9],
        passive_force=np.array([_F0]),
        passive_optimal_length=np.array([_L0]),
        passive_range_lo=_PRM[:1],
        passive_lmax=_PRM[5:6],
        fpmax=_PRM[7:8],
        act_ids=np.array([0]),
    )
    return _ArrayAccessor(
        np,
        muscle_act=np.array([act]),
        muscle_force=np.array([force]),
        muscle_length=np.array([length]),
        muscle_velocity=np.array([vel]),
        muscle_params=params,
    )


@pytest.mark.parametrize(("f_ft", "expected"), [(0.0, 25.0), (0.5, 89.0), (1.0, 153.0)])
def test_metabolic_isometric_heat_rate_matches_umberger_2003(
    f_ft: float, expected: float
) -> None:
    """Maximal isometric heat rate at L0: 1.28 %FT + 25 W/kg (Umberger et al. 2003).

    25 W/kg for slow-twitch and 153 W/kg for fast-twitch muscle (anaerobic, S = 1);
    times S = 1.5 under aerobic conditions. No work is done (v = 0).
    """
    acc = _one_muscle(1.0, 0.0, 1.0)
    out = effort.metabolic_energy_rate(
        acc, {}, fast_twitch_fraction=f_ft, aerobic_factor=1.0
    )
    assert out["metabolic_rate"] == pytest.approx(expected, rel=1e-12)
    assert out["metabolic_work_rate"] == 0.0
    aerobic = effort.metabolic_energy_rate(acc, {}, fast_twitch_fraction=f_ft)
    assert aerobic["metabolic_rate"] == pytest.approx(1.5 * expected, rel=1e-12)
    # Beyond L0 the heat scales with 0.4 + 0.6 F_iso.
    stretched = _one_muscle(1.3, 0.0, 1.0)
    f_iso = active_force_length(np.array([1.3]), _PRM[4], _PRM[5], np)[0]
    out = effort.metabolic_energy_rate(
        stretched, {}, fast_twitch_fraction=f_ft, aerobic_factor=1.0
    )
    assert out["metabolic_heat_rate"] == pytest.approx(
        expected * (0.4 + 0.6 * f_iso), rel=1e-12
    )


def test_metabolic_shortening_heat_relation() -> None:
    """Shortening heat grows linearly with speed (Hill 1938) with Umberger's coefficients.

    alpha_S(ST) = 100 / vmax_ST and alpha_S(FT) = 153 / vmax_FT (vmax_ST = vmax_FT / 2.5);
    slow-twitch shortening heat saturates at 100 W/kg above vmax_ST.
    """
    vmax_ft = _PRM[6]
    vmax_st = vmax_ft / 2.5

    def shortening_heat(f_ft: float, v_norm: float) -> float:
        rate = lambda v: effort.metabolic_energy_rate(  # noqa: E731
            _one_muscle(1.0, v, 1.0), {}, fast_twitch_fraction=f_ft, aerobic_factor=1.0
        )["metabolic_heat_rate"]
        return float(rate(v_norm) - rate(0.0))

    assert shortening_heat(0.0, -0.5 * vmax_st) == pytest.approx(50.0, rel=1e-12)
    assert shortening_heat(0.0, -vmax_st) == pytest.approx(100.0, rel=1e-12)
    assert shortening_heat(0.0, -vmax_ft) == pytest.approx(
        100.0, rel=1e-12
    )  # saturated
    assert shortening_heat(1.0, -0.5 * vmax_ft) == pytest.approx(76.5, rel=1e-12)
    assert shortening_heat(1.0, -vmax_ft) == pytest.approx(153.0, rel=1e-12)
    assert shortening_heat(0.5, -vmax_st) == pytest.approx(
        0.5 * 100.0 + 0.5 * 153.0 / 2.5, rel=1e-12
    )
    # Lengthening heat: alpha_L = 4 alpha_S(ST) (2003) or 0.3 alpha_S(ST) (2010), times A.
    for version, ratio in (("2003", 4.0), ("2010", 0.3)):
        out = effort.metabolic_energy_rate(
            _one_muscle(1.0, 0.2, 1.0),
            {},
            version=version,
            fast_twitch_fraction=0.0,
            aerobic_factor=1.0,
            forbid_negative_total_rate=False,
        )
        expected = 25.0 + ratio * 100.0 / vmax_st * 0.2
        assert out["metabolic_heat_rate"] == pytest.approx(expected, rel=1e-12)


def test_metabolic_work_rate_is_active_fiber_power() -> None:
    """w = -F_CE v: the active fiber force (no passive part) times the shortening speed."""
    acc = _one_muscle(1.2, -2.0, 0.6)
    length, vel = acc.muscle_length()[0], acc.muscle_velocity()[0]
    f_active = -mujoco.mju_muscleGain(length, vel, _LR, 1.0, _PRM) * 0.6
    out = effort.metabolic_energy_rate(acc, {})
    assert out["metabolic_work_rate"] == pytest.approx(-f_active * vel, rel=1e-12)
    # Umberger 2010 counts positive work only.
    stretch = _one_muscle(1.2, 2.0, 0.6)
    assert (
        effort.metabolic_energy_rate(stretch, {}, version="2010")["metabolic_work_rate"]
        == 0.0
    )
    assert effort.metabolic_energy_rate(stretch, {})["metabolic_work_rate"] < 0.0


def _opensim_umberger(
    exc: float,
    act: float,
    norm_len: float,
    vel: float,
    l_opt: float,
    f_iso: float,
    f_active: float,
    mass: float,
    vmax: float,
    slow: float,
    s: float,
    include_neg: bool,
) -> float:
    """Scalar transcription of OpenSim Umberger2010MuscleMetabolicsProbe (one muscle, W).

    Defaults of the probe: no Bhargava recruitment, minimum heat 1 W/kg, negative total
    power forbidden; ``include_neg`` is ``include_negative_mechanical_work``.
    """
    a_dep = exc if exc > act else (exc + act) / 2
    am_un = 128 * (1 - slow) + 25
    am = (
        s
        * a_dep**0.6
        * (am_un if norm_len <= 1.0 else 0.4 * am_un + 0.6 * am_un * f_iso)
    )
    v_norm = vel / l_opt
    alpha_fast, alpha_slow = 153 / vmax, 100 / (vmax / 2.5)
    if v_norm <= 0:
        tmp_slow = min(-alpha_slow * v_norm, 100.0)
        tmp_fast = alpha_fast * v_norm * (1 - slow)
        sdot = s * a_dep**2 * (tmp_slow * slow - tmp_fast)
    else:
        sdot = s * a_dep * (4.0 if include_neg else 0.3) * alpha_slow * v_norm
    if norm_len > 1.0:
        sdot *= f_iso
    f_active = max(f_active, 0.0)
    wdot = -f_active * vel if (include_neg or vel <= 0) else 0.0
    wdot /= mass
    total = am + sdot + wdot
    if total < 0:
        sdot -= total
    heat = max(am + sdot, 1.0)
    return mass * (heat + wdot)


@pytest.mark.parametrize("env_id", ["myoElbowPose1D6MExoRandom-v0", "myoLegWalk-v0"])
@pytest.mark.parametrize("version", ["2003", "2010"])
def test_metabolic_matches_opensim_reference(env_id: str, version: str) -> None:
    """Per-muscle rates equal the OpenSim probe's algorithm, on real states with excitation."""
    env = _env(env_id, steps=12, seed=3)
    m, d, acc = env.model, env.data, env._accessor
    p = acc.muscle_params()
    ids = np.flatnonzero(m.actuator_gaintype == mujoco.mjtGain.mjGAIN_MUSCLE)
    exc = np.random.default_rng(0).uniform(0, 1, len(ids))
    f_ft = np.linspace(0.2, 0.8, len(ids))
    out = effort.metabolic_energy_rate(
        acc, {"muscle_excitation": exc}, version=version, fast_twitch_fraction=f_ft
    )
    for k, i in enumerate(ids):
        lr, acc0, prm = (
            m.actuator_lengthrange[i],
            m.actuator_acc0[i],
            m.actuator_gainprm[i, :9],
        )
        length, vel = d.actuator_length[i], d.actuator_velocity[i]
        expected = _opensim_umberger(
            exc=exc[k],
            act=d.act[m.actuator_actadr[i]],
            norm_len=prm[0] + (length - lr[0]) / p.optimal_length[k],
            vel=vel,
            l_opt=p.optimal_length[k],
            f_iso=-mujoco.mju_muscleGain(length, 0.0, lr, acc0, prm) / p.peak_force[k],
            f_active=-mujoco.mju_muscleGain(length, vel, lr, acc0, prm)
            * d.act[m.actuator_actadr[i]],
            mass=p.peak_force[k] / _SIGMA * _RHO * p.optimal_length[k],
            vmax=prm[6],
            slow=1 - f_ft[k],
            s=1.5,
            include_neg=version == "2003",
        )
        assert out["metabolic_rate_per_muscle"][k] == pytest.approx(
            expected, rel=1e-9, abs=1e-9
        )
    assert out["metabolic_rate"] == pytest.approx(
        out["metabolic_rate_per_muscle"].sum()
    )


# ---------------------------------------------------------------------------
# Consumed Endurance
# ---------------------------------------------------------------------------


def _paper_endurance(pct: float) -> float:
    """Hincapié-Ramos et al. (2014), Eqs. 1-2: E = 1236.5 / (%MVC - 15)^0.618 - 72.5 (s)."""
    return 1236.5 / math.pow(pct - 15.0, 0.618) - 72.5


def test_endurance_time_formula() -> None:
    strength = np.array([0.10, 0.15, 0.30, 0.50, 1.00, 1.50])
    got = effort.endurance_time(strength, np)
    assert np.isinf(got[:2]).all()  # <= 15 %: sustainable indefinitely
    np.testing.assert_allclose(
        got[2:5], [_paper_endurance(p) for p in (30.0, 50.0, 100.0)]
    )
    assert got[2] == pytest.approx(159.4368, abs=1e-4)
    assert (
        got[4] == pytest.approx(6.90, abs=1e-2) and got[5] == got[4]
    )  # clipped at 100 %


def test_consumed_endurance_worked_example() -> None:
    """Holding the arm at 30 % of Max_Torque for 60 s consumes 37.63 % of the endurance.

    CE = interaction time / E(average strength) * 100 (Eq. 7) = 60 / 159.44 * 100.
    The per-step term at the same torque spends 100 dt / E per step.
    """
    dt, steps = 0.01, 6000
    torque = 0.3 * effort.CE_MAX_SHOULDER_TORQUE_MALE
    acc = _ArrayAccessor(np, dt, qfrc_actuator=np.array([0.0, torque, 0.0, 9.0]))
    step = effort.consumed_endurance(acc, {}, shoulder_dof_ids=[0, 1, 2])
    assert step["shoulder_torque"] == pytest.approx(torque)
    assert step["shoulder_strength"] == pytest.approx(0.3)
    assert step["endurance_time"] == pytest.approx(_paper_endurance(30.0))
    assert step["ce_step"] == pytest.approx(100 * dt / _paper_endurance(30.0))
    episode = effort.consumed_endurance_episode(np.full(steps, 0.3), dt, np)
    assert episode["interaction_time"] == pytest.approx(60.0)
    assert episode["consumed_endurance"] == pytest.approx(
        100 * 60.0 / _paper_endurance(30.0)
    )
    assert episode["consumed_endurance"] == pytest.approx(37.6325, abs=1e-4)
    assert step["ce_step"] * steps == pytest.approx(episode["consumed_endurance"])
    # The paper averages the torque: half the time at 20 %, half at 40 % scores as 30 %.
    varying = effort.consumed_endurance_episode(
        np.repeat([0.2, 0.4], steps // 2), dt, np
    )
    assert varying["consumed_endurance"] == pytest.approx(episode["consumed_endurance"])
    # Below 15 % nothing is consumed.
    rest = effort.consumed_endurance_episode(np.full(steps, 0.1), dt, np)
    assert rest["consumed_endurance"] == 0.0


def test_consumed_endurance_reads_shoulder_qfrc_on_arm() -> None:
    """On the MyoSuite arm the shoulder torque is the qfrc_actuator norm at the shoulder dofs."""
    env = _env("myoArmReachRandom-v0", steps=10)
    m = env.model
    dofs = [
        m.joint(n).dofadr[0]
        for n in ("elv_angle_r", "shoulder_elv_r", "shoulder_rot_r")
    ]
    out = effort.consumed_endurance(env._accessor, {}, shoulder_dof_ids=dofs)
    assert out["shoulder_torque"] == pytest.approx(
        np.linalg.norm(env.data.qfrc_actuator[dofs])
    )
    assert out["shoulder_torque"] > 0.0


# ---------------------------------------------------------------------------
# Fatigue effort and joint-limit discomfort
# ---------------------------------------------------------------------------


def test_fatigue_effort_reads_3ccr_state() -> None:
    """CPU CumulativeFatigue and the torch TorchFatigueState give the same effort."""
    torch = pytest.importorskip("torch")
    from myosuite.core.muscle_conditions import TorchFatigueState

    env = gym.make("myoFatiElbowPose1D6MRandom-v0").unwrapped
    acc, model = env._accessor, env.model
    cpu = CumulativeFatigue(model, frame_skip=10)
    gpu = TorchFatigueState.from_mj_model(model, num_envs=2)
    excitation = np.linspace(0.3, 1.0, cpu.na)
    for _ in range(300):
        cpu.compute_act(excitation.copy(), dt=0.02)
        gpu.step(torch.as_tensor(np.stack([excitation] * 2), dtype=torch.float32), 0.02)
    out = effort.fatigue_effort(acc, {"fatigue": cpu})
    assert out["fatigue_mf"] == pytest.approx(cpu.MF.mean()) and out["fatigue_mf"] > 0.0
    assert out["fatigue_mf_max"] == pytest.approx(cpu.MF.max())
    assert out["fatigue_effort"] == pytest.approx(cpu.get_effort())
    assert out["dense"] == pytest.approx(-cpu.MF.mean())
    torch_acc = _ArrayAccessor(torch)
    out_t = effort.fatigue_effort(torch_acc, {"fatigue": gpu})
    assert "fatigue_effort" not in out_t  # TorchFatigueState keeps no target load
    np.testing.assert_allclose(
        out_t["fatigue_mf"].numpy(), [cpu.MF.mean()] * 2, rtol=1e-4
    )
    target = torch.as_tensor(np.stack([excitation] * 2), dtype=torch.float32)
    out_t = effort.fatigue_effort(torch_acc, {"fatigue": gpu, "fatigue_target": target})
    np.testing.assert_allclose(
        out_t["fatigue_effort"].numpy(), [cpu.get_effort()] * 2, rtol=1e-3
    )


def test_joint_limit_discomfort_is_smooth_and_uses_joint_range() -> None:
    def discomfort(x: float) -> float:
        acc = _ArrayAccessor(
            np,
            joint_pos=np.array([9.0, -1.0 + 2.0 * x]),
            joint_range=(np.array([1]), np.array([[-1.0, 1.0]])),
        )
        return float(
            effort.joint_limit_discomfort(acc, {}, margin=0.1)["joint_limit_discomfort"]
        )

    for x in (0.1, 0.5, 0.9):
        assert discomfort(x) == pytest.approx(0.0, abs=1e-12)
    assert discomfort(0.0) == pytest.approx(1.0) and discomfort(1.0) == pytest.approx(
        1.0
    )
    assert discomfort(0.05) == pytest.approx(0.25)
    # C1 at the margin: both one-sided slopes vanish (O(h) for the quadratic side).
    h = 1e-7
    assert (discomfort(0.1 + h) - discomfort(0.1)) / h == pytest.approx(0.0, abs=1e-4)
    assert (discomfort(0.1) - discomfort(0.1 - h)) / h == pytest.approx(0.0, abs=1e-4)
