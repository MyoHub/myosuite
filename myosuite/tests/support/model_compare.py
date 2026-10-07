# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Bit-exact comparison of compiled MuJoCo models."""

from __future__ import annotations

import mujoco
import numpy as np


def model_arrays(model: mujoco.MjModel) -> dict[str, np.ndarray]:
    """Every numpy array attribute of *model*."""
    names = (name for name in dir(model) if not name.startswith("_"))
    values = {name: getattr(model, name) for name in names}
    return {k: v for k, v in values.items() if isinstance(v, np.ndarray)}


def assert_same_model(a: mujoco.MjModel, b: mujoco.MjModel) -> None:
    """Every array attribute and every ``opt``/``stat`` field is identical."""
    arrays_a, arrays_b = model_arrays(a), model_arrays(b)
    assert arrays_a.keys() == arrays_b.keys()
    for name, value in arrays_a.items():
        np.testing.assert_array_equal(value, arrays_b[name], err_msg=name)
    for struct in ("opt", "stat"):
        for name in dir(getattr(a, struct)):
            if not name.startswith("_"):
                np.testing.assert_array_equal(
                    getattr(getattr(a, struct), name),
                    getattr(getattr(b, struct), name),
                    err_msg=f"{struct}.{name}",
                )
