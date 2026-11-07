# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Saturation bounds of the muscle action maps (MyoHub/myosuite#566)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from myosuite.terms.base_action import (
    MUSCLE_SATURATION_MARGIN,
    muscle_saturated,
    muscle_saturation_bounds,
    muscle_saturation_excess,
    sigmoid_muscle_activation,
)

pytestmark = pytest.mark.tier1

_M = MUSCLE_SATURATION_MARGIN


@pytest.mark.parametrize("margin", [0.01, _M, 0.2])
def test_sigmoid_bounds_give_margin_excitation(margin: float) -> None:
    lo, hi = muscle_saturation_bounds(sigmoid=True, margin=margin)
    excitation = sigmoid_muscle_activation(np.array([lo, hi]), np)
    np.testing.assert_allclose(excitation, [margin, 1.0 - margin], rtol=1e-12)


def test_identity_and_clamped_bounds() -> None:
    assert muscle_saturation_bounds(sigmoid=False) == (_M, 1.0 - _M)
    # CPU twins clamp to [-1, 1] before the sigmoid, which caps excitation at 0.924:
    # beyond the clamp the map is flat.
    lo, hi = muscle_saturation_bounds(sigmoid=True, clip=(-1.0, 1.0))
    assert lo == pytest.approx(0.5 - math.log(19.0) / 5.0)
    assert hi == 1.0
    # A wider clamp (identity map, MuJoCo clamps muscle ctrl to [0, 1]) changes nothing.
    assert muscle_saturation_bounds(sigmoid=False, clip=(-1.0, 1.0)) == (_M, 1.0 - _M)


@pytest.mark.parametrize("margin", [0.0, 0.5, -0.1])
def test_margin_is_validated(margin: float) -> None:
    with pytest.raises(ValueError, match="margin"):
        muscle_saturation_bounds(sigmoid=True, margin=margin)


def test_mask_and_excess() -> None:
    lo, hi = 0.0, 1.0
    a = np.array([-0.5, 0.0, 0.5, 1.0, 1.25])
    np.testing.assert_array_equal(
        muscle_saturated(a, lo, hi), [True, True, False, True, True]
    )
    np.testing.assert_allclose(
        muscle_saturation_excess(a, lo, hi, np), [0.5, 0.0, 0.0, 0.0, 0.25]
    )


def test_numpy_and_torch_agree() -> None:
    torch = pytest.importorskip("torch")
    lo, hi = muscle_saturation_bounds(sigmoid=True, clip=(-1.0, 1.0))
    a = np.linspace(-3.0, 3.0, 60).reshape(3, -1)
    t = torch.as_tensor(a)
    np.testing.assert_array_equal(
        muscle_saturated(t, lo, hi).numpy(), muscle_saturated(a, lo, hi)
    )
    np.testing.assert_allclose(
        muscle_saturation_excess(t, lo, hi, torch).numpy(),
        muscle_saturation_excess(a, lo, hi, np),
    )
