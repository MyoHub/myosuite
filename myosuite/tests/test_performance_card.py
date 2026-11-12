# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Seed-level statistics, observed crossing/censoring and JSON boundary checks."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from myosuite.utils.performance_card import RunRecord, summarize_runs

pytestmark = pytest.mark.tier1


def _record(seed: int = 0, **changes: Any) -> dict[str, Any]:
    data = dict(
        config_id="small",
        env_id="task",
        seed=seed,
        contract_id="physics-task-eval",
        startup_s=2,
        collection_s=4,
        learning_s=3,
        total_wall_s=12,
        transitions=100,
        peak_memory_bytes=200,
        quality_metric="success",
        quality_points=[
            dict(wall_s=2, transitions=0, value=0.2),
            dict(wall_s=8, transitions=80, value=0.96),
            dict(wall_s=10, transitions=100, value=0.9),
        ],
        target=0.95,
        clock_origin="script_entry",
        memory_scope="torch_allocations",
    )
    data.update(changes)
    return data


def test_observed_crossings_censoring_and_analytical_seed_bootstrap() -> None:
    records = [
        RunRecord.from_dict(_record()),
        RunRecord.from_dict(
            _record(
                1,
                startup_s=4,
                collection_s=6,
                learning_s=1,
                total_wall_s=15,
                peak_memory_bytes=400,
                quality_points=[dict(wall_s=12, transitions=100, value=0.8)],
            )
        ),
    ]
    card = summarize_runs(records, bootstrap_samples=2000, seed=7)
    assert card == summarize_runs(
        list(reversed(records)), bootstrap_samples=2000, seed=7
    )
    config = card["configs"]["small"]
    assert config["stats"]["startup_s"] == {"n": 2, "mean": 3, "ci": [2, 4]}
    assert config["stats"]["peak_memory_bytes"]["mean"] == 300
    assert config["stats"]["transitions_per_wall_s"]["mean"] == pytest.approx(7.5)
    reached, censored = config["runs"]
    assert reached["first_observed_crossing"] == dict(
        wall_s=8, transitions=80, value=0.96
    )
    assert reached["last_quality_point"]["value"] == 0.9  # No sustained-success claim.
    assert censored["status"] == "right_censored"
    assert censored["last_quality_point"]["wall_s"] == 12  # Not process exit at 15.
    assert card["shared_observed_horizon"] == dict(wall_s=10, transitions=100)
    assert config["restricted_first_crossing_cost"]["wall_s"] == {
        "n": 2,
        "mean": 9,
        "ci": [8, 10],
    }
    assert config["restricted_first_crossing_cost"]["transitions"]["mean"] == 90


def test_shared_horizon_includes_every_configuration_and_single_seed_has_no_ci() -> (
    None
):
    other = RunRecord.from_dict(
        _record(
            config_id="large",
            quality_points=[dict(wall_s=5, transitions=40, value=0.1)],
        )
    )
    card = summarize_runs([RunRecord.from_dict(_record()), other])
    assert card["shared_observed_horizon"] == dict(wall_s=5, transitions=40)
    for config in card["configs"].values():
        assert config["restricted_first_crossing_cost"]["wall_s"] == {
            "n": 1,
            "mean": 5,
            "ci": None,
        }
    unobserved = summarize_runs(
        [RunRecord.from_dict(_record(quality_points=[], collection_s=0))]
    )
    config = unobserved["configs"]["small"]
    assert config["runs"][0]["status"] == "unobserved"
    assert config["restricted_first_crossing_cost"] is None
    assert config["stats"]["transitions_per_collection_s"] is None
    json.dumps(unobserved, allow_nan=False)


def test_lower_is_better_and_duplicate_seed_rejection() -> None:
    run = RunRecord.from_dict(_record(target=0.3, higher_is_better=False))
    assert (
        summarize_runs([run])["configs"]["small"]["runs"][0]["first_observed_crossing"][
            "wall_s"
        ]
        == 2
    )
    with pytest.raises(ValueError, match="duplicate"):
        summarize_runs([run, run])


@pytest.mark.parametrize(
    "change",
    [
        dict(total_wall_s=0),
        dict(startup_s=13),
        dict(peak_memory_bytes=-1),
        dict(transitions=1.5),
        dict(seed=True),
        dict(env_id=" "),
        dict(target=float("nan")),
        dict(higher_is_better="true"),
        dict(quality_points=[dict(wall_s=13, transitions=1, value=0)]),
        dict(quality_points=[dict(wall_s=1, transitions=101, value=0)]),
        dict(quality_points=[dict(wall_s=2, transitions=0, value=float("inf"))]),
        dict(
            quality_points=[
                dict(wall_s=2, transitions=1, value=0),
                dict(wall_s=1, transitions=2, value=0),
            ]
        ),
        dict(
            quality_points=[
                dict(wall_s=1, transitions=2, value=0),
                dict(wall_s=2, transitions=1, value=0),
            ]
        ),
    ],
)
def test_invalid_run_boundaries(change: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        RunRecord.from_dict(_record(**change))


@pytest.mark.parametrize(
    "change",
    [
        dict(env_id="other"),
        dict(contract_id="other"),
        dict(quality_metric="other"),
        dict(target=0.8),
        dict(higher_is_better=False),
        dict(clock_origin="process_start"),
        dict(memory_scope="process_rss"),
    ],
)
def test_mixed_contracts_are_rejected(change: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="mixed"):
        summarize_runs(
            [RunRecord.from_dict(_record()), RunRecord.from_dict(_record(1, **change))]
        )


def test_cli_reads_multiple_files_and_rejects_incompatible_inputs(
    tmp_path: Path,
) -> None:
    repo = Path(__file__).resolve().parents[2]
    paths = [tmp_path / "single.json", tmp_path / "list.json"]
    paths[0].write_text(json.dumps(_record()), encoding="utf-8")
    paths[1].write_text(json.dumps([_record(1)]), encoding="utf-8")
    output = tmp_path / "card.json"
    command = [
        sys.executable,
        str(repo / "scripts/report_performance.py"),
        *map(str, paths),
        "--output",
        str(output),
    ]
    env = dict(os.environ, PYTHONPATH=str(repo))
    process = subprocess.run(command, env=env, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert json.loads(output.read_text())["configs"]["small"]["n_seeds"] == 2
    paths[1].write_text(
        json.dumps(_record(1, contract_id="different")), encoding="utf-8"
    )
    process = subprocess.run(command, env=env, capture_output=True, text=True)
    assert process.returncode == 2
    assert "mixed" in process.stderr
