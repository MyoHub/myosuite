# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Data-only performance summaries, with independent training seeds as replicates."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

import numpy as np
from scipy.stats import bootstrap


def _finite(value: float, name: str, *, nonnegative: bool = True) -> None:
    if isinstance(value, bool) or not isfinite(value) or (nonnegative and value < 0):
        raise ValueError(
            f"{name} must be finite" + (" and nonnegative" if nonnegative else "")
        )


def _integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True)
class QualityPoint:
    """One evaluation result at an observed wall time and transition count."""

    wall_s: float
    transitions: int
    value: float

    def __post_init__(self) -> None:
        _finite(self.wall_s, "wall_s")
        _integer(self.transitions, "transitions")
        _finite(self.value, "value", nonnegative=False)


@dataclass(frozen=True)
class RunRecord:
    """One independent training run; clock and memory scopes are producer-defined.

    ``contract_id`` binds physics, task and evaluation protocol. Phase durations
    are disjoint and lie within total wall time. Quality events share its clock.
    """

    config_id: str
    env_id: str
    seed: int
    contract_id: str
    startup_s: float
    collection_s: float
    learning_s: float
    total_wall_s: float
    transitions: int
    peak_memory_bytes: int
    quality_metric: str
    quality_points: tuple[QualityPoint, ...]
    target: float
    clock_origin: str
    memory_scope: str
    higher_is_better: bool = True

    def __post_init__(self) -> None:
        for key in (
            "config_id",
            "env_id",
            "contract_id",
            "quality_metric",
            "clock_origin",
            "memory_scope",
        ):
            if (
                not isinstance(getattr(self, key), str)
                or not getattr(self, key).strip()
            ):
                raise ValueError(f"{key} must be a nonempty string")
        for key in ("seed", "transitions", "peak_memory_bytes"):
            _integer(getattr(self, key), key)
        for key in ("startup_s", "collection_s", "learning_s", "total_wall_s"):
            _finite(getattr(self, key), key)
        _finite(self.target, "target", nonnegative=False)
        if not isinstance(self.higher_is_better, bool):
            raise ValueError("higher_is_better must be a boolean")
        if (
            self.total_wall_s <= 0
            or self.startup_s + self.collection_s + self.learning_s
            > self.total_wall_s + 1e-9
        ):
            raise ValueError(
                "total_wall_s must be positive and contain all disjoint phases"
            )
        object.__setattr__(self, "quality_points", tuple(self.quality_points))
        previous = (0.0, 0)
        for point in self.quality_points:
            if not isinstance(point, QualityPoint):
                raise ValueError("quality_points must contain QualityPoint records")
            if point.wall_s < previous[0] or point.transitions < previous[1]:
                raise ValueError(
                    "quality events must be ordered in wall time and transitions"
                )
            if point.wall_s > self.total_wall_s or point.transitions > self.transitions:
                raise ValueError("quality events must lie within run totals")
            previous = point.wall_s, point.transitions

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> RunRecord:
        """Parse one JSON-compatible run, rejecting unknown or missing fields."""
        values = dict(record)
        values["quality_points"] = tuple(
            QualityPoint(**p) for p in values["quality_points"]
        )
        return cls(**values)


def summarize_runs(
    records: Sequence[RunRecord],
    *,
    confidence: float = 0.95,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Summarize compatible configurations using seed-level percentile bootstrap.

    Crossings are first *observed* events, never interpolated or sustained-success
    claims. Unreached runs are right-censored at their final evaluation, not at
    process exit. Restricted costs cap every run at a shared observed horizon;
    they include failures and do not estimate a continuous-time survival curve.
    No quality observations disables these costs for the entire comparison.

    Args:
        records: Runs with unique (config_id, seed), and one shared contract.
        confidence: Confidence level of the bootstrap mean intervals.
        bootstrap_samples: Number of resamples of independent training seeds.
        seed: Bootstrap random seed, independent of the recorded training seeds.

    Returns:
        JSON-compatible contract, per-config means/intervals and per-seed events.
        A single seed has no confidence interval. No successful-only mean
        time-to-target is reported.
    """
    if not records:
        raise ValueError("at least one run is required")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie between zero and one")
    _integer(bootstrap_samples, "bootstrap_samples")
    _integer(seed, "seed")
    if bootstrap_samples < 2:
        raise ValueError("bootstrap_samples must be at least two")
    contract_keys = (
        "env_id",
        "contract_id",
        "quality_metric",
        "target",
        "higher_is_better",
        "clock_origin",
        "memory_scope",
    )
    contract = {key: getattr(records[0], key) for key in contract_keys}
    groups: dict[str, list[RunRecord]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for run in records:
        if any(getattr(run, key) != value for key, value in contract.items()):
            raise ValueError("mixed task, physics, quality, clock or memory contracts")
        if (run.config_id, run.seed) in seen:
            raise ValueError("duplicate training seed within a configuration")
        seen.add((run.config_id, run.seed))
        groups[run.config_id].append(run)
    horizons = None
    if all(run.quality_points for run in records):
        horizons = {
            key: min(getattr(run.quality_points[-1], key) for run in records)
            for key in ("wall_s", "transitions")
        }
    rng = np.random.default_rng(seed)

    def mean(values: Sequence[float]) -> dict[str, Any]:
        array = np.asarray(values, dtype=float)
        interval = None
        if len(array) > 1:
            ci = bootstrap(
                (array,),
                np.mean,
                confidence_level=confidence,
                n_resamples=bootstrap_samples,
                method="percentile",
                random_state=rng,
            ).confidence_interval
            interval = [float(ci.low), float(ci.high)]
        return {"n": len(array), "mean": float(array.mean()), "ci": interval}

    configs = {}
    for config_id, runs in sorted(groups.items()):
        runs.sort(key=lambda run: run.seed)
        stats = {
            key: mean([getattr(run, key) for run in runs])
            for key in (
                "startup_s",
                "collection_s",
                "learning_s",
                "total_wall_s",
                "transitions",
                "peak_memory_bytes",
            )
        }
        stats["transitions_per_wall_s"] = mean(
            [run.transitions / run.total_wall_s for run in runs]
        )
        stats["transitions_per_collection_s"] = (
            mean([run.transitions / run.collection_s for run in runs])
            if all(run.collection_s > 0 for run in runs)
            else None
        )
        events = []
        for run in runs:
            crossing = next(
                (
                    p
                    for p in run.quality_points
                    if (
                        p.value >= run.target
                        if run.higher_is_better
                        else p.value <= run.target
                    )
                ),
                None,
            )
            last = run.quality_points[-1] if run.quality_points else None
            events.append(
                {
                    "seed": run.seed,
                    "status": "reached"
                    if crossing
                    else "right_censored"
                    if last
                    else "unobserved",
                    "first_observed_crossing": asdict(crossing) if crossing else None,
                    "last_quality_point": asdict(last) if last else None,
                    "quality_points": [asdict(p) for p in run.quality_points],
                }
            )
        restricted = None
        if horizons is not None:
            restricted = {
                key: mean(
                    [
                        min(event["first_observed_crossing"][key], horizon)
                        if event["first_observed_crossing"]
                        else horizon
                        for event in events
                    ]
                )
                for key, horizon in horizons.items()
            }
        configs[config_id] = {
            "n_seeds": len(runs),
            "reached": sum(e["status"] == "reached" for e in events),
            "stats": stats,
            "restricted_first_crossing_cost": restricted,
            "runs": events,
        }
    return {
        "schema_version": 1,
        "contract": contract,
        "confidence": confidence,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "interval_method": "percentile bootstrap of independent training seeds",
        "shared_observed_horizon": horizons,
        "configs": configs,
    }
