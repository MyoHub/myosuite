# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Compatibility re-exports for spatial enclosure geometry and MDP term adapters."""

from __future__ import annotations

from myosuite.physics.spatial_enclosure import (
    enclosure_margins,
    excess_contact_force,
    min_enclosure_clearance,
)
from myosuite.terms.base_obs import boundary_proximity_obs
from myosuite.terms.base_reward import (
    enclosure_clearance_reward,
    enclosure_contact_penalty,
)
from myosuite.terms.base_termination import enclosure_violation

__all__ = [
    "boundary_proximity_obs",
    "enclosure_clearance_reward",
    "enclosure_contact_penalty",
    "enclosure_margins",
    "enclosure_violation",
    "excess_contact_force",
    "min_enclosure_clearance",
]
