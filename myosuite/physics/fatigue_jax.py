# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""The JAX 3CC-r fatigue model; it lives in :mod:`myo_sim.physics.fatigue_jax`."""

from myo_sim.physics.fatigue_jax import (  # noqa: F401
    CumulativeFatigue,
    cumulative_fatigue_step,
)
