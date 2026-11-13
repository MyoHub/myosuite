# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Contact phase uses baseline events, jitter remains explicitly paired."""

import mujoco
import numpy as np
import pytest

from myosuite.tests.support.push_env import PushEnv, make_push_protocol
from myosuite.utils.paired_replay import ReplayPolicy
from myosuite.utils.push_metrics import RecoveryCriteria, summarize_trial
from myosuite.utils.push_recovery import run_paired_push
from myosuite.utils.push_scheduling import ContactPhase
from myosuite import make_env

pytestmark = pytest.mark.tier1


def test_phase_uses_measured_bracket_and_suppresses_short_contact_chatter() -> None:
    env = PushEnv()
    cfg = make_push_protocol(steps=100, foot_bodies=("pelvis",)).resolve(env)
    mask = np.zeros((101, 1), bool)
    mask[0:10] = True  # Initial standing contact is not a heel strike.
    mask[12:13] = True  # Insufficient contact and swing dwell.
    mask[20:30] = True
    mask[50:60] = True
    phase = ContactPhase(("pelvis",), 0.25, stride_index=0, min_contact_s=0.02)
    resolved = phase.schedule({"foot_contact_mask": mask}, cfg)
    assert resolved["heel_strike_steps"] == [20, 50]
    assert resolved["start_step"] == 28  # nearest step of 20 + .25*30
    assert resolved["measured_stride_s"] == pytest.approx(0.3)
    assert resolved["phase_origin"] == "baseline_heel_strike"


def test_missing_native_second_heel_strike_is_unavailable_not_recovered() -> None:
    trial = run_paired_push(
        PushEnv(),
        ReplayPolicy.stateless(lambda obs: np.zeros(1)),
        make_push_protocol(foot_bodies=("pelvis",)),
        seed=0,
        contact_phase=ContactPhase(("pelvis",), 0.5, stride_index=0),
    )
    result = summarize_trial(trial, RecoveryCriteria(min_height_L=0.01))
    assert not trial.protocol["phase_available"]
    assert result["ineligible_reason"] == "contact_phase_unavailable"
    assert result["exposed_transitions"] == 0 and not result["complete_dose"]
    assert not result["recovered"] and result["delivered_scalar_impulse_ns"] == 0
    np.testing.assert_array_equal(
        trial.baseline.states["qpos"], trial.pushed.states["qpos"]
    )


def test_native_velocity_jitter_is_seeded_fresh_and_identical_in_both_arms() -> None:
    class Refreshable(PushEnv):
        def forward(self):
            mujoco.mj_forward(self.model, self.data)
            return (
                np.r_[self.data.qpos, self.data.qvel].astype(np.float32),
                0,
                False,
                False,
                {},
            )

    policy = ReplayPolicy.stateless(lambda obs: np.zeros(1))
    trials = [
        run_paired_push(
            Refreshable(),
            policy,
            make_push_protocol(force_n=0),
            seed=seed,
            velocity_jitter_natural=0.02,
        )
        for seed in (1, 1, 2)
    ]
    np.testing.assert_array_equal(
        trials[0].baseline.states["qvel"], trials[1].baseline.states["qvel"]
    )
    assert not np.array_equal(
        trials[0].baseline.states["qvel"][0], trials[2].baseline.states["qvel"][0]
    )
    for trial in trials:
        np.testing.assert_array_equal(
            trial.baseline.states["qvel"], trial.pushed.states["qvel"]
        )
        np.testing.assert_array_equal(
            trial.baseline.observations[0, 7:],
            trial.baseline.states["qvel"][0].astype(np.float32),
        )
        assert trial.baseline.states["time_s"][0] == 0
        assert trial.protocol["initial_velocity_jitter"]["std_natural"] == 0.02


def test_jitter_without_refresh_contract_fails_before_dynamics() -> None:
    env = PushEnv()
    with pytest.raises(ValueError, match="forward"):
        run_paired_push(
            env,
            ReplayPolicy.stateless(lambda obs: np.zeros(1)),
            make_push_protocol(),
            seed=0,
            velocity_jitter_natural=0.02,
        )
    assert env.data.time == 0 and np.all(env.data.qvel == 0)


def test_native_muscle_trace_separates_commands_states_and_force() -> None:
    env = make_env("myoElbowPose1D6MFixed-v0")
    try:
        body = env.unwrapped.model.body(int(env.unwrapped.model.jnt_bodyid[0])).name
        cfg = make_push_protocol(
            body=body, steps=5, start_s=0, duration_s=0.01, force_n=0
        )
        trial = run_paired_push(
            env, ReplayPolicy.stateless(lambda obs: np.ones(6)), cfg, seed=0
        )
        state = trial.baseline.states
        assert state["muscle_excitation"].shape == (6, 6)
        assert state["muscle_activation"].shape == (6, 6)
        assert state["muscle_force_n"].shape == (6, 6)
        assert np.any(state["muscle_excitation"][1] != state["muscle_activation"][1])
        assert np.any(state["muscle_force_n"][1] != 0)
        np.testing.assert_array_equal(state["muscle_activation"], state["act"])
    finally:
        env.close()


@pytest.mark.parametrize("tolerance", [0.2, np.float64(0.2)])
def test_stationary_baseline_cannot_pass_declared_walking_command_eligibility(
    tolerance,
) -> None:
    cfg = make_push_protocol(force_n=0)
    trial = run_paired_push(
        PushEnv(), ReplayPolicy.stateless(lambda obs: np.zeros(1)), cfg, seed=0
    )
    result = summarize_trial(
        trial,
        RecoveryCriteria(
            min_height_L=0.01,
            target_speed_mps=1,
            target_speed_tolerance_mps=tolerance,
            require_baseline_tracking=True,
        ),
    )
    assert result["ineligible_reason"] == "baseline_does_not_track_command"
    assert not result["recovered"] and not result["baseline"]["tracks_target_speed"]
    assert result["baseline"]["post_push_mean_target_speed_error_mps"] == pytest.approx(
        1
    )


def test_post_push_foot_placement_is_unavailable_without_new_contact_entry() -> None:
    cfg = make_push_protocol(steps=100, start_s=0.7, force_n=0, foot_bodies=("pelvis",))
    trial = run_paired_push(
        PushEnv(), ReplayPolicy.stateless(lambda obs: np.zeros(1)), cfg, seed=0
    )
    result = summarize_trial(trial, RecoveryCriteria(min_height_L=0.01))
    assert result["post_push_foot_placement"][0]["baseline"] is None
    assert result["post_push_foot_placement"][0]["world_position_delta_m"] is None


def test_native_termination_before_dose_has_no_post_push_tracking_samples() -> None:
    class EarlyEnding(PushEnv):
        def step(self, action):
            obs, reward, _, _, info = super().step(action)
            return obs, reward, self.data.time >= 0.05 - 1e-10, False, info

    trial = run_paired_push(
        EarlyEnding(),
        ReplayPolicy.stateless(lambda obs: np.zeros(1)),
        make_push_protocol(),
        seed=0,
    )
    criteria = RecoveryCriteria(
        min_height_L=0.01, target_speed_mps=0, require_baseline_tracking=True
    )
    result = summarize_trial(trial, criteria)
    for arm in ("baseline", "pushed"):
        assert result[arm]["post_push_mean_target_speed_error_mps"] is None
        assert result[arm]["tracks_target_speed"] is None
    assert not result["eligible"] and result["exposed_transitions"] == 0
    assert result["ineligible_reason"] == "baseline_fall_or_incomplete"
