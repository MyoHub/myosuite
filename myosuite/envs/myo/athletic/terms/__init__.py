# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Shared athletic reward/termination helpers (CPU + mjlab)."""

from __future__ import annotations

from typing import Any

__all__ = [
    "lane_reward_keys",
    "default_lane_reward_terms",
]


def lane_reward_keys() -> tuple[str, ...]:
    """Canonical reward keys matching msk_envs LaneEnv / AthleticSpec."""
    return ("rew_vel", "rew_mid_lane", "rew_limit", "rew_spring", "rew_damper", "rew_muscle_passive")


def default_lane_reward_terms(spec: Any) -> dict[str, float]:
    """Map AthleticSpec λ fields onto msk reward keys."""
    return {
        "rew_vel": float(spec.lambda_vel),
        "rew_mid_lane": float(spec.lambda_mid_lane),
        "rew_limit": float(spec.lambda_limit),
        "rew_spring": float(spec.lambda_spring),
        "rew_damper": float(spec.lambda_damper),
        "rew_muscle_passive": float(spec.lambda_muscle_passive),
    }
