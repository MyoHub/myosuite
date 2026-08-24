# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Regression test for https://github.com/MyoHub/myosuite/issues/401.

Pristine models stepped from their shipped reset state under zero control
must not inject energy and explode (e.g. mjWARN_BADQACC, super-physical
qacc). Covers every registered model recipe (`list_recipes()`), not just the
hand_pen / hand_sar recipes that carry free-floating manipulation objects
analogous to the legacy myohand_pen.xml / myohand_sar.xml models named in
the issue.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from myosuite.core.model_builder import build_from_recipe, list_recipes
from myosuite.core.model_recipes import _MUSCLEMIMIC_NAMES, _musclemimic_build

pytestmark = pytest.mark.tier1

SUPERPHYSICAL_QACC = 1e6
HORIZON = 400
BAD_WARNINGS = {
    "mjWARN_BADQACC",
    "mjWARN_BADQPOS",
    "mjWARN_BADQVEL",
    "mjWARN_BADCTRL",
}


def _build(recipe_name: str) -> mujoco.MjModel:
    if recipe_name in _MUSCLEMIMIC_NAMES:
        model, _ = _musclemimic_build(recipe_name)
    else:
        model, _ = build_from_recipe(recipe_name)
    return model


def _engine_warnings(data: mujoco.MjData) -> dict[str, int]:
    return {
        name: int(data.warning[int(getattr(mujoco.mjtWarning, name))].number)
        for name in dir(mujoco.mjtWarning)
        if name.startswith("mjWARN_")
        and int(data.warning[int(getattr(mujoco.mjtWarning, name))].number)
    }


@pytest.mark.parametrize("recipe_name", list_recipes())
def test_zero_control_rollout_stays_bounded(recipe_name: str) -> None:
    """Pristine reset state + zero control must not blow up within 400 steps."""
    try:
        model = _build(recipe_name)
    except (FileNotFoundError, ImportError) as e:
        pytest.skip(f"dependency unavailable for {recipe_name!r}: {e}")
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    if model.nu:
        data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    max_abs_qacc = float(np.max(np.abs(data.qacc))) if data.qacc.size else 0.0
    for _ in range(HORIZON):
        if model.nu:
            data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)
        assert np.all(np.isfinite(data.qpos))
        assert np.all(np.isfinite(data.qvel))
        if data.qacc.size:
            max_abs_qacc = max(max_abs_qacc, float(np.max(np.abs(data.qacc))))

    warnings = _engine_warnings(data)
    assert not (
        set(warnings) & BAD_WARNINGS
    ), f"engine instability warnings: {warnings}"
    assert max_abs_qacc < SUPERPHYSICAL_QACC, f"max(abs(qacc))={max_abs_qacc:.3e}"
