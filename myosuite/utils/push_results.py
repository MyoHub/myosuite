# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Data-only push artifacts and per-condition seeded-trial uncertainty."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest, bootstrap

from myosuite.utils.deployment_manifest import artifact_record
from myosuite.utils.push_metrics import RecoveryCriteria, summarize_trial
from myosuite.utils.push_recovery import PairedPushTrial


def _json_array(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Only data-only controller/RNG state is exportable: {type(value)}")


def write_json(path: Path, document: Any) -> None:
    """Write finite JSON exclusively, without silently overwriting artifacts."""
    text = json.dumps(document, default=_json_array, allow_nan=False, indent=2)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text + "\n")


def save_trial(
    directory: Path,
    trial_id: int,
    condition_id: str,
    trial: PairedPushTrial,
    criteria: RecoveryCriteria,
    *,
    elapsed_s: float,
) -> dict[str, Any]:
    """Save aligned native traces and explicit initial controller/history/RNG data.

    Native integration state is portable only for the same model/runtime. This is
    an experiment artifact, not a serializer for arbitrary Python env objects;
    the complete Python wrapper replay graph remains an in-memory snapshot.
    """
    name = f"trial_{trial_id:05d}"
    arrays = {"initial_integration_state": trial.initial_integration_state}
    for arm in ("baseline", "pushed"):
        trace = getattr(trial, arm)
        arrays.update({f"{arm}_{k}": v for k, v in trace.states.items()})
        for key in ("actions", "observations", "rewards", "applied_wrench"):
            arrays[f"{arm}_{key}"] = getattr(trace, key)
    if any(np.asarray(a).dtype.hasobject for a in arrays.values()):
        raise TypeError("Object arrays are not valid push artifacts")
    state_path = directory / f"{name}.initial.json"
    write_json(
        state_path,
        {
            "controller": trial.initial_controller_state,
            "rng": trial.initial_rng_state,
            "native_state_spec": "mjSTATE_INTEGRATION",
            "scope": "native integration and explicit controller/RNG data; Python env graph is in-memory only",
        },
    )
    trace_path = directory / f"{name}.npz"
    with trace_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return {
        "trial_id": trial_id,
        "condition_id": condition_id,
        "seed": trial.seed,
        "initial_state_sha256": hashlib.sha256(
            np.ascontiguousarray(trial.initial_integration_state).tobytes()
        ).hexdigest(),
        "elapsed_s": elapsed_s,
        "protocol": trial.protocol,
        "metrics": summarize_trial(trial, criteria),
        "artifacts": {p.name: artifact_record(p) for p in (trace_path, state_path)},
    }


def summarize_conditions(
    rows: list[dict],
    *,
    independent_jitter: bool,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute marginal condition summaries across independent initial-state seeds.

    Deterministic sweeps are descriptive. Duplicated seeds/initial native states
    disable intervals. Conditions sharing a seed are correlated and are never
    pooled as independent Bernoulli trials. Recovery intervals condition on an
    eligible baseline and exposure; dose completeness is reported separately.
    """
    if not 0 < confidence < 1 or not isinstance(resamples, int) or resamples < 1:
        raise ValueError("Declare valid confidence and positive resample count")
    conditions = {}
    for name in dict.fromkeys(row["condition_id"] for row in rows):
        group = [row for row in rows if row["condition_id"] == name]
        independent = (
            independent_jitter
            and len(group) >= 2
            and len({r["seed"] for r in group}) == len(group)
            and len({r["initial_state_sha256"] for r in group}) == len(group)
        )
        eligible = [
            r
            for r in group
            if r["metrics"]["eligible"] and r["metrics"]["exposed_transitions"] > 0
        ]
        successes = sum(r["metrics"]["recovered"] for r in eligible)
        item = {
            "trials": len(group),
            "unique_seeds": len({r["seed"] for r in group}),
            "unique_initial_states": len({r["initial_state_sha256"] for r in group}),
            "eligible_exposed": len(eligible),
            "recovered": successes,
            "complete_doses": sum(r["metrics"]["complete_dose"] for r in group),
            "pushed_falls": sum(r["metrics"]["pushed"]["fall"] for r in group),
            "recovery_rate_given_eligible_exposure": successes / len(eligible)
            if eligible
            else None,
            "independent_seed_design": bool(independent),
            "intervals_available": bool(independent and eligible),
            "recovery_exact_ci": None,
            "paired_mean_speed_delta_bootstrap_ci": None,
        }
        if independent and eligible:
            ci = binomtest(successes, len(eligible)).proportion_ci(
                confidence_level=confidence, method="exact"
            )
            item["recovery_exact_ci"] = [float(ci.low), float(ci.high)]
            if len(eligible) >= 2:
                effects = np.array(
                    [r["metrics"]["paired_mean_speed_delta_mps"] for r in eligible]
                )
                ci = bootstrap(
                    (effects,),
                    np.mean,
                    confidence_level=confidence,
                    n_resamples=resamples,
                    method="percentile",
                    random_state=np.random.default_rng(
                        seed
                    ),  # Also supports SciPy <1.15.
                ).confidence_interval
                item["paired_mean_speed_delta_bootstrap_ci"] = [
                    float(ci.low),
                    float(ci.high),
                ]
        conditions[name] = item
    return {
        "confidence": confidence,
        "bootstrap_method": "percentile_of_paired_speed_differences",
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "binomial_method": "Clopper-Pearson_exact",
        "pooling": "none; shared-seed conditions are correlated",
        "conditions": conditions,
    }


def write_trials_csv(path: Path, rows: list[dict]) -> None:
    """Write flat outcome columns; full protocols/arrays remain in JSON/NPZ."""
    flattened = []
    for row in rows:
        flat = {k: row[k] for k in ("trial_id", "condition_id", "seed", "elapsed_s")}
        for key, value in row["metrics"].items():
            if isinstance(value, dict):
                flat.update(
                    {
                        f"{key}.{k}": v
                        for k, v in value.items()
                        if not isinstance(v, (dict, list))
                    }
                )
            elif not isinstance(value, list):
                flat[key] = value
        flattened.append(flat)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(dict.fromkeys(k for row in flattened for k in row))
        )
        writer.writeheader()
        writer.writerows(flattened)
