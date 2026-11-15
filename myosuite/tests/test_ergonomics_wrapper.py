# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for ErgonomicsWrapper musculoskeletal telemetry and FatigueWrapper reuse."""

from __future__ import annotations

import numpy as np
import pytest

from myosuite import make_env
from myosuite.envs.ergonomics_wrapper import ErgonomicsConfig, ErgonomicsWrapper
from myosuite.envs.wrappers import FatigueWrapper, apply_features, wrapper_spec

pytestmark = pytest.mark.tier1

_EXPECTED_KEYS = {
    "instantaneous_effort",
    "smoothed_effort",
    "cumulative_fatigue",
    "peak_muscle_fatigue",
    "muscle_saturation",
    "joint_limit_strain",
}


def test_ergonomics_wrapper_tracks_effort_fatigue_saturation_and_strain() -> None:
    env = ErgonomicsWrapper(make_env("myoElbowPose1D6MRandom-v0"))
    try:
        _, info0 = env.reset(seed=0)
        assert set(info0["ergonomics"]) == _EXPECTED_KEYS
        assert all(0.0 <= v <= 1.0 for v in info0["ergonomics"].values())
        assert info0["ergonomics"]["cumulative_fatigue"] == 0.0

        mid = np.full(env.action_space.shape, 0.5, dtype=env.action_space.dtype)
        _, _, _, _, info_mid = env.step(mid)
        assert info_mid["ergonomics"]["muscle_saturation"] == 0.0

        high = np.ones(env.action_space.shape, dtype=env.action_space.dtype)
        for _ in range(15):
            _, _, _, _, info_hi = env.step(high)

        ergo_hi = info_hi["ergonomics"]
        assert all(0.0 <= v <= 1.0 for v in ergo_hi.values())
        assert ergo_hi["instantaneous_effort"] > 0.25
        assert ergo_hi["smoothed_effort"] > 0.15
        assert ergo_hi["cumulative_fatigue"] > 0.0
        assert ergo_hi["peak_muscle_fatigue"] >= ergo_hi["cumulative_fatigue"]
        assert ergo_hi["muscle_saturation"] == pytest.approx(1.0)

        _, info_reset = env.reset(seed=1)
        assert info_reset["ergonomics"]["cumulative_fatigue"] == 0.0
    finally:
        env.close()


def test_ergonomics_wrapper_reuses_host_fatigue_and_supports_wrapper_spec() -> None:
    base = FatigueWrapper(make_env("myoElbowPose1D6MRandom-v0"))
    wrapped = apply_features(
        base, [wrapper_spec(ErgonomicsWrapper, reward_penalty_weight=0.2)]
    )
    try:
        assert wrapped._fatigue_observer is None
        wrapped.reset(seed=3)
        act = np.ones(wrapped.action_space.shape, dtype=wrapped.action_space.dtype)
        _, reward_penalized, _, _, info = wrapped.step(act)
        host_mf = np.asarray(wrapped.get_wrapper_attr("muscle_fatigue").MF)
        assert info["ergonomics"]["cumulative_fatigue"] == pytest.approx(
            float(np.mean(host_mf))
        )
        assert isinstance(reward_penalized, float)
    finally:
        wrapped.close()


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"effort_ema_alpha": 0.0},
        {"effort_ema_alpha": 1.5},
        {"rom_margin_fraction": 0.0},
        {"passive_torque_scale": -1.0},
        {"saturation_margin": 0.5},
        {"reward_penalty_weight": -0.1},
    ],
)
def test_ergonomics_config_validates_bounds(bad_kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ErgonomicsConfig(**bad_kwargs)
