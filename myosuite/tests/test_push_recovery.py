# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Native paired push, impulse, normalization and controller-state contracts."""

import numpy as np
import pytest

from myosuite.utils.paired_replay import ReplayPolicy
from myosuite.utils.push_recovery import run_paired_push
from myosuite.tests.support.push_env import PushEnv, make_push_protocol as _protocol

pytestmark = pytest.mark.tier1


def test_native_force_window_matches_analytic_impulse_and_restores_initial_state() -> (
    None
):
    env = PushEnv()
    trial = run_paired_push(
        env, ReplayPolicy.stateless(lambda obs: np.zeros(1)), _protocol(), seed=0
    )
    active = np.flatnonzero(trial.pushed.applied_wrench[:, 0])
    np.testing.assert_array_equal(active, np.arange(10, 20))
    np.testing.assert_allclose(
        trial.pushed.states["qvel"][-1, 0] - trial.baseline.states["qvel"][-1, 0],
        0.2,
        atol=1e-12,
    )
    assert trial.protocol["impulse_ns"] == pytest.approx(0.4)
    assert env.data.time == 0 and np.all(env.data.qvel == 0)
    assert len(trial.initial_integration_state) > env.model.nq + env.model.nv
    assert trial.initial_rng_state["env_and_state_owners"]


def test_zero_push_is_bit_identical_for_both_arms() -> None:
    trial = run_paired_push(
        PushEnv(),
        ReplayPolicy.stateless(lambda obs: np.zeros(1)),
        _protocol(force_n=0.0),
        seed=2,
    )
    for key in trial.baseline.states:
        np.testing.assert_array_equal(
            trial.baseline.states[key], trial.pushed.states[key]
        )
    np.testing.assert_array_equal(trial.baseline.actions, trial.pushed.actions)


def test_configured_ground_contacts_measure_native_body_weight() -> None:
    trial = run_paired_push(
        PushEnv(),
        ReplayPolicy.stateless(lambda obs: np.zeros(1)),
        _protocol(steps=100, start_s=0.5, force_n=0.0, foot_bodies=("pelvis",)),
        seed=0,
    )
    assert trial.baseline.states["foot_contact_mask"][-1, 0]
    assert trial.baseline.states["vertical_grf_n"][-1] == pytest.approx(
        2 * 9.81, abs=0.05
    )


def test_missing_controller_state_is_detected_as_unpaired() -> None:
    history = []

    def action(obs):
        history.append(1)
        return np.array([len(history) * 0.01], np.float32)

    with pytest.raises(ValueError, match="pre-push"):
        run_paired_push(PushEnv(), ReplayPolicy.stateless(action), _protocol(), seed=0)


def test_background_target_wrench_is_rejected_before_rollout() -> None:
    class BackgroundForce(PushEnv):
        def reset(self, **kwargs):
            result = super().reset(**kwargs)
            self.data.xfrc_applied[self.model.body("pelvis").id, 0] = 3.0
            return result

    env = BackgroundForce()
    with pytest.raises(ValueError, match="background wrench"):
        run_paired_push(
            env, ReplayPolicy.stateless(lambda obs: np.zeros(1)), _protocol(), seed=0
        )
    assert env.data.time == 0
    assert env.data.xfrc_applied[1, 0] == 3.0
