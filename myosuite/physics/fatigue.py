# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
#
# Backward-compatibility shim: the canonical implementation now lives in
# myosuite.core.muscle_conditions.  Import from there; this module
# re-exports for legacy code that does `from myosuite.physics.fatigue import ...`.

from myosuite.core.muscle_conditions import (  # noqa: F401
    MUSCLE_FATIGUE_PARAMS,
    CumulativeFatigue,
)
