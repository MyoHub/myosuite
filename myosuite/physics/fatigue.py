# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""The 3CC-r fatigue model; it lives in :mod:`myo_sim.physics.fatigue`."""

from myo_sim.physics.fatigue import (  # noqa: F401
    FATIGUE_REST_THRESHOLD,
    MUSCLE_FATIGUE_PARAMS,
    CumulativeFatigue,
    TorchFatigueState,
    three_cc_r_step,
    torch_three_cc_r_step,
)
