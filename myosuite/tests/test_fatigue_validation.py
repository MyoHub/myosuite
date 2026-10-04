# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Validation of the 3CC-r fatigue model against published endurance times.

A sustained isometric contraction at a target load ``TL`` (fraction of MVC) is
held with the repo's own :class:`CumulativeFatigue` update until task failure,
defined as in Frey-Law, Looft & Heitsman (2012, J Biomech 45:1803): "when the
sum of the resting and active states fall below target levels, MR + MA < TL".
The endurance time (ET) is compared with

* the closed-form 3CC solution for a sustained load (``MA = TL``):
  ``ET = -ln(1 - R (1 - TL) / (F TL)) / R`` (implementation check), and
* the joint-specific power models of Frey-Law & Avin (2010, Ergonomics 53:109,
  Table 2), ``ET = b0 * TL**b1`` (ET in s, TL as a fraction), fitted in log-log
  space to 369 data points from 194 studies.

The recovery multiplier ``r`` only acts when ``MA >= TL``, so a sustained
contraction validates ``F`` and ``R``. See ``docs/source/fatigue_validation.rst``
for the full table.
"""

from __future__ import annotations

import functools

import mujoco
import numpy as np
import pytest

from myosuite.core.muscle_conditions import MUSCLE_FATIGUE_PARAMS, CumulativeFatigue

pytestmark = pytest.mark.tier1

# Frey-Law & Avin (2010), Table 2, power model ET = b0 * TL**b1.
POWER_MODELS = {
    "General": (21.92, -1.98),
    "Ankle": (34.71, -2.06),
    "Trunk": (22.69, -2.27),
    "Elbow": (17.98, -2.21),
    "Grip": (33.55, -1.61),
    "Knee": (19.38, -1.88),
    "Shoulder": (14.86, -1.83),
}

# Frey-Law, Looft & Heitsman (2012), Table 1: optimal (F, R) per joint, fitted to
# the power models above with the same failure definition.
FREY_LAW_2012 = {
    "Ankle": (0.00589, 0.00058),
    "Knee": (0.01500, 0.00149),
    "Trunk": (0.00755, 0.00075),
    "Shoulder": (0.01820, 0.00168),
    "Elbow": (0.00912, 0.00094),
    "Grip": (0.00980, 0.00064),
    "General": (0.00970, 0.00091),
}

# Parameter rows with a joint-level counterpart in Frey-Law & Avin (2010).
# Functional-muscle-group and sex-specific rows (Rakshit et al. 2021) have no
# matching empirical curve and are only tabulated in the docs.
JOINT_ROWS = {
    "Default": "General",
    "Default_v2_4": "General",
    "Elbow": "Elbow",
    "Hand": "Grip",
    "Wrist": "Grip",
    "Finger": "Grip",
    "Wrist-Flexor": "Grip",  # Rakshit et al. (2021) general handgrip (G/GEN)
    "Knee": "Knee",
    "Knee-Extensor": "Knee",
    "Ankle": "Ankle",
    "Toe": "Ankle",
    "Shoulder": "Shoulder",
}

DT = 0.02  # control step of the myoFati* envs; ET is 1e2-1e3 s
LOADS = np.round(np.arange(0.2, 0.91, 0.1), 2)
PINNED = LOADS <= 0.8  # loads compared with the power models
T_MAX = 2500.0  # s; above twice every pinned empirical ET


def _row_params(row: str) -> tuple[float, float, float]:
    """``(F, R, r)`` of a parameter row with the Default fallbacks of the lookup."""
    p, default = MUSCLE_FATIGUE_PARAMS[row], MUSCLE_FATIGUE_PARAMS["Default"]
    return p.get("F", default["F"]), p.get("R", default["R"]), p.get("r", default["r"])


def _muscle_model(n: int) -> mujoco.MjModel:
    """``n`` muscle actuators with MuJoCo's default 10 / 40 ms time constants."""
    actuators = "".join(
        f'<general name="m{i}" joint="j" dyntype="muscle" dynprm="0.01 0.04"/>'
        for i in range(n)
    )
    return mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><joint name="j"/><geom size="0.1"/></body>'
        f"</worldbody><actuator>{actuators}</actuator></mujoco>"
    )


def endurance_times(
    params: list[tuple[float, float, float]], loads: np.ndarray
) -> np.ndarray:
    """Endurance time of each parameter set at each sustained load.

    Args:
        params: ``(F, R, r)`` per parameter set.
        loads: Target loads (fraction of MVC).

    Returns:
        ET in seconds, shape ``(len(params), len(loads))``; ``inf`` if the
        load is still held after :data:`T_MAX`.
    """
    F, R, r = (np.repeat(np.asarray(p), loads.size) for p in zip(*params))
    target = np.tile(loads, len(params))
    fatigue = CumulativeFatigue(_muscle_model(target.size))
    fatigue.set_FatigueCoefficient(F)
    fatigue.set_RecoveryCoefficient(R)
    fatigue.set_RecoveryMultiplier(r)
    et = np.full(target.size, np.inf)
    for k in range(1, int(T_MAX / DT) + 1):
        ma, mr, _ = fatigue.compute_act(target, dt=DT)
        failed = np.isinf(et) & (ma + mr < target)
        et[failed] = k * DT
        if np.isfinite(et).all():
            break
    return et.reshape(len(params), loads.size)


@functools.cache
def _simulated() -> dict[str, np.ndarray]:
    """ET of every repo row and of every Frey-Law (2012) joint, one batch."""
    keys = list(MUSCLE_FATIGUE_PARAMS) + [f"FL2012-{j}" for j in FREY_LAW_2012]
    params = [_row_params(row) for row in MUSCLE_FATIGUE_PARAMS]
    params += [(F, R, 1.0) for F, R in FREY_LAW_2012.values()]
    return dict(zip(keys, endurance_times(params, LOADS), strict=True))


def closed_form_et(F: float, R: float, load: np.ndarray) -> np.ndarray:
    """Sustained-load 3CC endurance time; ``inf`` at or below ``R / (F + R)``."""
    x = 1.0 - R * (1.0 - load) / (F * load)
    return np.where(x > 0, -np.log(np.where(x > 0, x, 1.0)) / R, np.inf)


def power_model_et(joint: str, load: np.ndarray) -> np.ndarray:
    """Frey-Law & Avin (2010) endurance time of *joint* at *load*."""
    b0, b1 = POWER_MODELS[joint]
    return b0 * load**b1


@pytest.mark.parametrize("row", list(MUSCLE_FATIGUE_PARAMS))
def test_endurance_time_matches_closed_form(row: str) -> None:
    """The integrated update reproduces the analytic sustained-load ET.

    Only loads at least 5 points above the asymptote ``R / (F + R)`` are checked;
    closer to it ET diverges and is ill-conditioned. The 2 % covers the explicit
    Euler step of the compartments and the ~50 ms rise of MA to TL.
    """
    F, R, _ = _row_params(row)
    expected = closed_form_et(F, R, LOADS)
    check = (LOADS >= R / (F + R) + 0.05) & (expected < T_MAX)
    np.testing.assert_allclose(
        _simulated()[row][check], expected[check], rtol=0.02, atol=0.1
    )


# Factor between the model and the empirical ET allowed at 20-80% MVC. The power
# models were fitted in log space, so the band is a ratio. It bounds the published
# optimal 3CC fits (FREY_LAW_2012, every joint: 0.54x to 1.39x of the curves, the
# worst being grip at 80% MVC; the authors report mean relative errors of -6.9% to
# 16.2%), so a row outside it fits its joint worse than the published model does.
# At 90% MVC even the published fits are 46-71% short (a constant F gives
# ET ~ (1 - TL) / (F TL) there), so that load is tabulated in the docs, not pinned.
_RATIO_TOL = 2.0

_HAND = (
    "Rakshit et al. (2021) hand F=0.01227 vs Frey-Law et al. (2012) grip F=0.00980: "
    "ET 0.43x the grip curve at 80% MVC (published grip fit: 0.54x)"
)
_KNOWN_DEVIATIONS = {
    "Shoulder": "copies the Knee row (F=0.00825, R=0.00076): ET 2.1-2.5x the "
    "shoulder curve at 20-60% MVC; Frey-Law et al. (2012) fit F=0.01820, R=0.00168",
    "Ankle": "Rakshit et al. (2021) ankle F=0.01485 is 2.5x the Frey-Law et al. "
    "(2012) ankle F: ET 0.32-0.49x the ankle curve at 60-80% MVC",
    "Toe": "copies the Ankle row",
    "Hand": _HAND,
    "Wrist": "copies the Hand row; " + _HAND,
    "Finger": "copies the Hand row; " + _HAND,
    "Wrist-Flexor": "Rakshit et al. (2021) general handgrip F=0.01235: ET 0.43x "
    "the grip curve at 80% MVC",
}


def _cases() -> list:
    cases = [pytest.param(f"FL2012-{j}", j, id=f"FL2012-{j}") for j in FREY_LAW_2012]
    for row, joint in JOINT_ROWS.items():
        marks = ()
        if row in _KNOWN_DEVIATIONS:
            marks = pytest.mark.xfail(reason=_KNOWN_DEVIATIONS[row], strict=True)
        cases.append(pytest.param(row, joint, id=f"{row}-vs-{joint}", marks=marks))
    # Torso muscles are not in MUSCLE_FMG and fall back to Default.
    cases.append(pytest.param("Default", "Trunk", id="Default-vs-Trunk"))
    return cases


@pytest.mark.parametrize(("key", "joint"), _cases())
def test_endurance_time_matches_frey_law_avin(key: str, joint: str) -> None:
    """Endurance times stay within a factor of 2 of the empirical joint curve."""
    model = _simulated()[key][PINNED]
    empirical = power_model_et(joint, LOADS[PINNED])
    ratio = model / empirical
    assert np.all((ratio <= _RATIO_TOL) & (ratio >= 1 / _RATIO_TOL)), (
        f"{key} vs {joint}: ET {np.round(model, 1)} s, empirical "
        f"{np.round(empirical, 1)} s at {LOADS[PINNED]} MVC"
    )
