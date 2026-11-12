# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""SI and body-normalized push conditions resolve against a native model."""

import numpy as np
import pytest

from myosuite.tests.support.push_env import PushEnv, make_push_protocol as _protocol

pytestmark = pytest.mark.tier1


def test_normalized_force_time_and_nominal_phase_record_actual_quantization() -> None:
    env = PushEnv()
    env.reset(seed=0)
    cfg = _protocol(
        force_n=None,
        force_bw=0.1,
        duration_s=None,
        duration_tau=0.25,
        start_s=None,
        phase=0.25,
        stride_period_s=0.1,
        cycles_before_push=1,
    )
    resolved = cfg.resolve(env)
    assert resolved["force_world_or_body_n"] == pytest.approx([0.1 * 2 * 9.81, 0, 0])
    assert resolved["actual_duration_s"] >= resolved["requested_duration_s"]
    assert resolved["actual_duration_s"] - resolved["requested_duration_s"] < 0.01
    assert abs(resolved["actual_start_s"] - 0.125) <= 0.005 + 1e-12
    assert resolved["phase_origin"] == "reset_clock_nominal_stride"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"leg_length_m": 0},
        {"force_bw": 0.1},
        {"direction": (0, 0, 0)},
        {"phase": 0.5},
        {"frame": "camera"},
        {"duration_s": np.nan},
    ],
)
def test_invalid_protocol_fails_before_control(kwargs: dict) -> None:
    env = PushEnv()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        _protocol(**kwargs).resolve(env)
    assert env.data.time == 0
