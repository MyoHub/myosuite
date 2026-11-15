# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Muscle effort metrics of recorded rollouts (``myosuite.biomechanics.effort``)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from myosuite.biomechanics.effort import (
    MUSCLE_SATURATION_MARGIN,
    cocontraction_index,
    muscle_saturated,
    muscle_saturation,
    muscle_saturation_bounds,
    squared_activation,
    wasted_torque,
)
from myosuite.terms.base_action import sigmoid_muscle_activation

pytestmark = pytest.mark.tier1

_M = MUSCLE_SATURATION_MARGIN


# Saturation cases from #588 (Mark Colley).
@pytest.mark.parametrize("margin", [0.01, _M, 0.2])
def test_sigmoid_bounds_give_margin_excitation(margin: float) -> None:
    lo, hi = muscle_saturation_bounds(sigmoid=True, margin=margin)
    excitation = sigmoid_muscle_activation(np.array([lo, hi]), np)
    np.testing.assert_allclose(excitation, [margin, 1.0 - margin], rtol=1e-12)


def test_identity_and_clamped_bounds() -> None:
    assert muscle_saturation_bounds(sigmoid=False) == (_M, 1.0 - _M)
    # A [-1, 1] clamp before the sigmoid caps excitation at 0.924: flat beyond it.
    lo, hi = muscle_saturation_bounds(sigmoid=True, clip=(-1.0, 1.0))
    assert lo == pytest.approx(0.5 - math.log(19.0) / 5.0)
    assert hi == 1.0
    assert muscle_saturation_bounds(sigmoid=False, clip=(-1.0, 1.0)) == (_M, 1.0 - _M)


@pytest.mark.parametrize("margin", [0.0, 0.5, -0.1])
def test_margin_is_validated(margin: float) -> None:
    with pytest.raises(ValueError, match="margin"):
        muscle_saturation_bounds(sigmoid=True, margin=margin)


def test_mask() -> None:
    a = np.array([-0.5, 0.0, 0.5, 1.0, 1.25])
    np.testing.assert_array_equal(
        muscle_saturated(a, 0.0, 1.0), [True, True, False, True, True]
    )


def test_saturation_switches_at_the_bounds() -> None:
    eps = 1e-3
    for bounds in (
        muscle_saturation_bounds(sigmoid=True),
        muscle_saturation_bounds(sigmoid=True, clip=(-1.0, 1.0)),
        muscle_saturation_bounds(sigmoid=False),
    ):
        lo, hi = bounds
        for value, expected in (
            (lo - eps, 1.0),
            (lo + eps, 0.0),
            (hi - eps, 0.0),
            (hi + eps, 1.0),
        ):
            actions = np.full((2, 4, 3), value)
            np.testing.assert_array_equal(
                muscle_saturation(actions, bounds), [expected, expected]
            )


def test_saturation_fraction_of_excitations() -> None:
    # 3 of 8 commands within 0.05 of 0 or 1.
    e = np.array([[0.0, 0.5, 0.97, 0.3], [0.2, 0.04, 0.6, 0.9]])
    assert muscle_saturation(e) == pytest.approx(3 / 8)


def test_squared_activation() -> None:
    a = np.array([[[1.0, 0.0], [0.5, 0.5]], [[0.0, 0.0], [1.0, 1.0]]])
    np.testing.assert_allclose(squared_activation(a), [0.75, 1.0])


def test_cocontraction_of_an_antagonist_pair() -> None:
    # Equal and opposite moment arms: CCI = 2 min(F1, F2) / (F1 + F2).
    arms = np.array([[0.02], [-0.02]])
    forces = np.array([[100.0, 100.0], [100.0, 0.0], [300.0, 100.0]])
    np.testing.assert_allclose(wasted_torque(arms, forces), [4.0, 0.0, 4.0])
    cci = cocontraction_index(arms, forces)
    assert cci.shape == (1,)
    np.testing.assert_allclose(cci, [8.0 / 14.0])
    # Synergists (same sign) never co-contract; a DOF without torque gives 0.
    same = np.array([[0.02, 0.0], [0.01, 0.0]])
    np.testing.assert_allclose(cocontraction_index(same, forces), [0.0, 0.0])


def test_numpy_and_torch_agree() -> None:
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    a = rng.uniform(-3.0, 3.0, (2, 10, 6))
    arms = rng.normal(0.0, 0.02, (2, 10, 6, 3))
    forces = rng.uniform(0.0, 500.0, (2, 10, 6))
    t = torch.as_tensor
    bounds = muscle_saturation_bounds(sigmoid=True, clip=(-1.0, 1.0))
    pairs = (
        (muscle_saturation(t(a), bounds), muscle_saturation(a, bounds)),
        (squared_activation(t(a)), squared_activation(a)),
        (wasted_torque(t(arms), t(forces)), wasted_torque(arms, forces)),
        (cocontraction_index(t(arms), t(forces)), cocontraction_index(arms, forces)),
    )
    for got, want in pairs:
        assert torch.is_tensor(got)
        np.testing.assert_allclose(got.numpy(), want, rtol=1e-6)
