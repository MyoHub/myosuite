# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for the SB3 all-env sweep helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from myosuite.utils.sb3_sweep import (
    WinnerPpoConfig,
    budget_for,
    improved,
    improvement_threshold,
    is_excluded_env_id,
    parse_act_dim,
    solved_from_info,
)


pytestmark = pytest.mark.tier2


def test_is_excluded_env_id() -> None:
    assert is_excluded_env_id("myoSarcElbowPose1D6MFixed-v0")
    assert is_excluded_env_id("myoFatiHandReachFixed-v0")
    assert is_excluded_env_id("myoReafHandObjHoldFixed-v0")
    assert not is_excluded_env_id("myoElbowPose1D6MFixed-v0")
    assert not is_excluded_env_id("motorFingerPoseFixed-v0")


@pytest.mark.parametrize(
    ("act_dim", "base", "escalate", "expected"),
    [
        (5, 100_000, False, 100_000),
        (20, 100_000, False, 150_000),
        (50, 100_000, False, 200_000),
        (216, 100_000, False, 1_000_000),
        (354, 100_000, False, 1_000_000),
        (5, 100_000, True, 1_000_000),
        (20, 100_000, True, 1_500_000),
        (50, 100_000, True, 2_000_000),
        (354, 100_000, True, 3_000_000),
    ],
)
def test_budget_for(act_dim: int, base: int, escalate: bool, expected: int) -> None:
    assert budget_for(act_dim, base, escalate=escalate) == expected


def test_improved_matches_legacy_summary_rule() -> None:
    # Absolute floor dominates near-zero mimic rewards.
    assert not improved(1e-5, 2e-5, eps_abs=0.5, eps_rel=0.05)
    # Relative floor for large |before|.
    assert not improved(-221.5, -214.7, eps_abs=0.5, eps_rel=0.05)
    assert improved(-131.6, 656.4, eps_abs=0.5, eps_rel=0.05)
    assert improved(50.0, 640.0, eps_abs=0.5, eps_rel=0.05)


def test_improvement_threshold() -> None:
    assert improvement_threshold(0.0) == 0.5
    assert improvement_threshold(-221.5) == pytest.approx(11.075)
    assert improvement_threshold(10.0) == 0.5


def test_parse_act_dim() -> None:
    assert parse_act_dim("obs=(17,) act=(5,)") == 5
    assert parse_act_dim("obs=(120,) act=(39,)") == 39


def test_winner_ppo_config_divides_rollout() -> None:
    cfg8 = WinnerPpoConfig.for_n_envs(8)
    assert cfg8.n_steps == 1024
    assert cfg8.batch_size == 1024
    assert (8 * cfg8.n_steps) % cfg8.batch_size == 0
    assert cfg8.learning_rate == 2e-5
    assert cfg8.ent_coef == 1e-5
    assert cfg8.net_arch == (256, 256)
    assert cfg8.vec_normalize
    assert cfg8.no_early_stop

    cfg4 = WinnerPpoConfig.for_n_envs(4)
    assert cfg4.n_steps == 1024
    assert (4 * cfg4.n_steps) % cfg4.batch_size == 0


def test_solved_from_info() -> None:
    assert solved_from_info({"solved": True})
    assert solved_from_info({"rwd_dict": {"solved": 1.0}})
    assert not solved_from_info({})
    assert not solved_from_info({"solved": False})


def test_summary_schema_roundtrip(tmp_path: Path) -> None:
    row = {
        "env_id": "myoElbowPose1D6MFixed-v0",
        "status": "pass",
        "act_dim": 6,
        "timesteps": 100_000,
        "reward_before": -10.0,
        "reward_after": 20.0,
        "delta": 30.0,
        "threshold": 0.5,
        "success_after_pct": 100.0,
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps([row], indent=2))
    loaded = json.loads(path.read_text())
    assert loaded[0]["status"] == "pass"
    assert improved(loaded[0]["reward_before"], loaded[0]["reward_after"])
