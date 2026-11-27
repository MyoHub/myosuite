# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Data-only native artifacts and honest seed-level interval denominators."""

import csv
import json

import numpy as np
import pytest
from scipy.stats import binomtest

from myosuite.tests.support.push_env import PushEnv, make_push_protocol
from myosuite.utils.paired_replay import ReplayPolicy
from myosuite.utils.push_metrics import RecoveryCriteria
from myosuite.utils.push_recovery import run_paired_push
from myosuite.utils.push_results import (
    save_trial,
    summarize_conditions,
    write_json,
    write_trials_csv,
)

pytestmark = pytest.mark.tier1


def test_native_artifacts_load_without_pickle_and_preserve_paired_data(
    tmp_path,
) -> None:
    trial = run_paired_push(
        PushEnv(),
        ReplayPolicy.stateless(lambda obs: np.zeros(1)),
        make_push_protocol(force_n=0),
        seed=0,
    )
    row = save_trial(
        tmp_path,
        0,
        "zero",
        trial,
        RecoveryCriteria(min_height_L=0.01, hold_s=0.03),
        elapsed_s=0.1,
    )
    data = np.load(tmp_path / "trial_00000.npz", allow_pickle=False)
    np.testing.assert_array_equal(data["baseline_qpos"], data["pushed_qpos"])
    np.testing.assert_array_equal(
        data["initial_integration_state"], trial.initial_integration_state
    )
    assert json.loads((tmp_path / "trial_00000.initial.json").read_text())["rng"][
        "env_and_state_owners"
    ]
    write_json(tmp_path / "result.json", row)
    write_trials_csv(tmp_path / "trials.csv", [row])
    with (tmp_path / "trials.csv").open() as stream:
        assert next(csv.DictReader(stream))["recovered"] == "True"
    with pytest.raises(FileExistsError):
        save_trial(tmp_path, 0, "zero", trial, RecoveryCriteria(), elapsed_s=0.1)
    with pytest.raises(ValueError):
        write_json(tmp_path / "nan.json", {"invalid": np.nan})
    assert not (tmp_path / "nan.json").exists()


def _rows():
    return [
        {
            "condition_id": "push",
            "seed": i,
            "initial_state_sha256": str(i),
            "metrics": {
                "eligible": True,
                "exposed_transitions": 5,
                "recovered": i != 0,
                "complete_dose": True,
                "pushed": {"fall": i == 0},
                "paired_mean_speed_delta_mps": float(i - 1),
            },
        }
        for i in range(3)
    ]


def test_seed_level_intervals_match_scipy_and_do_not_pool_conditions() -> None:
    rows = _rows()
    second = [dict(row, condition_id="other") for row in rows]
    result = summarize_conditions(rows + second, independent_jitter=True, seed=3)
    item = result["conditions"]["push"]
    ci = binomtest(2, 3).proportion_ci(method="exact")
    assert item["recovery_exact_ci"] == pytest.approx([ci.low, ci.high])
    assert item["eligible_exposed"] == 3 and item["unique_seeds"] == 3
    assert item["paired_mean_speed_delta_bootstrap_ci"] == [-1.0, 1.0]
    assert result == summarize_conditions(
        rows + second, independent_jitter=True, seed=3
    )
    assert "overall_ci" not in result


@pytest.mark.parametrize("duplicate", ["seed", "initial_state_sha256", None])
def test_deterministic_or_duplicated_states_never_generate_intervals(duplicate) -> None:
    rows = _rows()
    if duplicate:
        rows[1][duplicate] = rows[0][duplicate]
    item = summarize_conditions(rows, independent_jitter=duplicate is not None)[
        "conditions"
    ]["push"]
    assert not item["intervals_available"] and item["recovery_exact_ci"] is None


def test_ineligible_baselines_are_separate_from_conditional_recovery_denominator() -> (
    None
):
    rows = _rows()
    rows[0]["metrics"]["eligible"] = False
    item = summarize_conditions(rows, independent_jitter=True)["conditions"]["push"]
    assert item["trials"] == 3 and item["eligible_exposed"] == 2
    assert item["recovery_rate_given_eligible_exposure"] == 1
