# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Actual dose and recovery censoring checked on native paired trajectories."""

from dataclasses import replace

import numpy as np
import pytest

from myosuite.tests.support.push_env import PushEnv, make_push_protocol
from myosuite.utils.paired_replay import ReplayPolicy
from myosuite.utils.push_metrics import RecoveryCriteria, summarize_trial
from myosuite.utils.push_recovery import run_paired_push

pytestmark = pytest.mark.tier1
CRITERIA = RecoveryCriteria(min_height_L=0.01, speed_error_natural=0.01, hold_s=0.03)


def _trial(force=4):
    return run_paired_push(
        PushEnv(),
        ReplayPolicy.stateless(lambda obs: np.zeros(1)),
        make_push_protocol(force_n=force, forward_world=(1, 0, 0)),
        seed=0,
    )


def test_native_impulse_and_no_recovery_without_velocity_correction() -> None:
    result = summarize_trial(_trial(), CRITERIA)
    assert result["eligible"] and result["complete_dose"]
    assert result["delivered_scalar_impulse_ns"] == pytest.approx(0.4)
    assert result["delivered_world_impulse_ns"] == pytest.approx([0.4, 0, 0])
    assert result["delivered_dose_fraction"] == pytest.approx(1)
    assert result["recovery_censored"] and not result["recovered"]
    assert result["baseline"]["mean_squared_activation"] is None
    assert result["baseline"]["foot_contact_fraction"] is None


def test_zero_force_control_recovers_immediately_and_has_zero_effect() -> None:
    result = summarize_trial(_trial(0), CRITERIA)
    assert result["recovered"] and result["recovery_time_s"] == 0
    assert result["delivered_scalar_impulse_ns"] == 0
    assert result["max_speed_error_natural"] == 0
    assert result["paired_mean_speed_delta_mps"] == 0


def test_native_early_ending_reports_partial_dose_instead_of_scheduled_dose() -> None:
    class EarlyEnding(PushEnv):
        def step(self, action):
            obs, reward, _, _, info = super().step(action)
            return obs, reward, self.data.time >= 0.15 - 1e-10, False, info

    trial = run_paired_push(
        EarlyEnding(),
        ReplayPolicy.stateless(lambda obs: np.zeros(1)),
        make_push_protocol(),
        seed=0,
    )
    result = summarize_trial(trial, CRITERIA)
    assert not result["eligible"] and not result["recovered"]
    assert result["exposed_transitions"] == 5 and not result["complete_dose"]
    assert result["delivered_scalar_impulse_ns"] == pytest.approx(0.2)
    assert result["scheduled_impulse_ns"] == pytest.approx(0.4)


def test_failure_before_push_is_ineligible_and_unexposed() -> None:
    trial = _trial()

    # Preserve a real native prefix representing a terminated episode.
    def prefix(trace):
        return replace(
            trace,
            states={k: v[:6] for k, v in trace.states.items()},
            actions=trace.actions[:5],
            observations=trace.observations[:6],
            rewards=trace.rewards[:5],
            applied_wrench=trace.applied_wrench[:5],
            terminated=True,
        )

    result = summarize_trial(
        replace(trial, baseline=prefix(trial.baseline), pushed=prefix(trial.pushed)),
        CRITERIA,
    )
    assert not result["eligible"] and result["exposed_transitions"] == 0
    assert result["delivered_scalar_impulse_ns"] == 0
    assert result["recovery_time_s"] is None and not result["recovered"]


def test_recovery_requires_contiguous_hold_and_excludes_later_fall() -> None:
    trial = _trial(0)
    states = {k: v.copy() for k, v in trial.pushed.states.items()}
    states["root_rotation"][23] = np.diag([1, -1, -1])
    result = summarize_trial(
        replace(trial, pushed=replace(trial.pushed, states=states)), CRITERIA
    )
    assert result["pushed"]["fall"] and not result["recovered"]
    assert result["recovery_censored"]


@pytest.mark.parametrize(
    "kwargs", [{"hold_s": 0}, {"heading_error_rad": np.nan}, {"max_tilt_rad": 4}]
)
def test_invalid_criteria(kwargs) -> None:
    with pytest.raises(ValueError):
        RecoveryCriteria(**kwargs)
