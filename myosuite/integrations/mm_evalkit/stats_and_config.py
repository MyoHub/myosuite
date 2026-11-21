# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Lazy adapters for ``mm_evalkit.stats`` and ``PyYAML`` policy config serialization."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np


def compute_rsm_rsa(pred_ch: np.ndarray, exp_ch: np.ndarray) -> tuple[float, float]:
    """Delegate RSM + RSA to ``mm_evalkit.stats.rsa`` (raises ``ImportError`` if absent)."""
    from mm_evalkit.stats.rsa import compute_rsm, rsa

    sp, pe = rsa(compute_rsm(pred_ch.T), compute_rsm(exp_ch.T))
    return float(sp), float(pe)


def compute_wilcoxon_comparison(
    *,
    label_a: str,
    label_b: str,
    shared_units: list[str],
    channels_all: list[str],
    per_channel_a: dict[str, list[float]],
    per_channel_b: dict[str, list[float]],
    scalar_vectors: Sequence[tuple[str, list[float], list[float]]],
    unit_name: str,
    clean_json_fn: Callable[[Any], Any],
) -> dict[str, Any]:
    """Delegate paired Wilcoxon + Holm tables to ``mm_evalkit.stats.wilcoxon``."""
    from mm_evalkit.stats.wilcoxon import (
        MetricFamily,
        MetricSamples,
        wilcoxon_table,
    )

    families: list[Any] = []
    if channels_all:
        samples_ch = {
            label_a: MetricSamples.from_per_channel(
                per_channel_a, shared_units, channels_all
            ),
            label_b: MetricSamples.from_per_channel(
                per_channel_b, shared_units, channels_all
            ),
        }
        families.append(
            MetricFamily(
                name="r_opt_lag",
                channel_name="channel",
                samples=samples_ch,
                comparisons=[("Model A vs Model B", label_a, label_b)],
            )
        )
    for metric_name, va, vb in scalar_vectors:
        families.append(
            MetricFamily(
                name=metric_name,
                channel_name="all",
                samples={
                    label_a: MetricSamples.from_vector(va, shared_units),
                    label_b: MetricSamples.from_vector(vb, shared_units),
                },
                comparisons=[("Model A vs Model B", label_a, label_b)],
            )
        )
    table_df, ch_df = wilcoxon_table(families, unit_name=unit_name)
    summary_rows = [
        {k: clean_json_fn(v) for k, v in row.items()}
        for row in table_df.to_dict(orient="records")
    ]
    per_channel_rows = [
        {k: clean_json_fn(v) for k, v in row.items()}
        for row in ch_df.to_dict(orient="records")
    ]
    return {
        "backend": "mm_evalkit.stats.wilcoxon",
        "unit_name": unit_name,
        "shared_units": len(shared_units),
        "summary_rows": summary_rows,
        "per_channel_rows": per_channel_rows,
    }


def dump_policies_yaml(normalized: Sequence[Mapping[str, Any]]) -> str:
    """Serialize normalized policy specs with ``PyYAML`` (raises ``ImportError`` if absent)."""
    import yaml

    return str(
        yaml.safe_dump(
            {"policies": list(normalized)}, sort_keys=False, allow_unicode=True
        )
    )


def compare_emg_reports_mm_evalkit(
    reports_a: Sequence[Mapping[str, Any]],
    reports_b: Sequence[Mapping[str, Any]],
    *,
    label_a: str = "model_a",
    label_b: str = "model_b",
) -> dict[str, Any]:
    """Compare EMG reports via ``mm_evalkit.stats.wilcoxon`` when installed, else SciPy."""
    from myosuite.biomechanics.emg_validation import compare_emg_validation_reports

    try:
        import mm_evalkit.stats.wilcoxon  # noqa: F401
    except ImportError:
        return compare_emg_validation_reports(
            reports_a, reports_b, label_a=label_a, label_b=label_b
        )
    return compare_emg_validation_reports(
        reports_a,
        reports_b,
        label_a=label_a,
        label_b=label_b,
        stats_backend=compute_wilcoxon_comparison,
    )
