# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""CPU Gymnasium registration for the athletic task×plant matrix."""

from __future__ import annotations

from myosuite.envs.myo.athletic.catalog import register_athletic_matrix

# myofullbody_simple has a ModularTaskEnv model recipe; bolt sprinter is mjlab-only.
register_athletic_matrix(
    plants=("myofullbody_simple",),
    backends={"cpu"},
)
