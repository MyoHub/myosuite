# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Explicit quality decisions and narrowly allowlisted aggregate exports."""

from __future__ import annotations

import ipaddress
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from myosuite.integrations.capture.retargeting import MarkerRetargetResult

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}\Z")
_DOI = re.compile(r"10\.[0-9]{4,9}/[A-Za-z0-9._;()/:+-]+\Z")
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
_ERROR_METRICS = frozenset(
    {
        "marker_rms_m",
        "marker_max_m",
        "sole_penetration_max_m",
        "root_tracking_rms_m",
        "equality_error_max",
        "rms_m",
        "max_error_m",
    }
)
_CONDITION_METRICS = frozenset({"fit_condition_number", "geometry_condition_number"})
_COUNT_METRICS = frozenset(
    {
        "frames",
        "converged_frames",
        "failed_frames",
        "valid_marker_observations",
        "missing_marker_observations",
        "fit_jacobian_rank",
        "reference_geometry_rank",
        "observed_geometry_rank",
        "evaluations",
        "geometric_contact_pair_count",
        "geometric_contact_frame_count",
    }
)
_METRICS = _ERROR_METRICS | _CONDITION_METRICS | _COUNT_METRICS | {"converged_fraction"}
_SCOPE = (
    "Aggregate numerical diagnostics only; coordinates, model parameters, scale "
    "factors, rigid transforms, residual vectors and frame arrays are excluded.",
    "Thresholds are caller supplied and establish no clinical or dynamics validity.",
    "Geometric contact counts, when supplied, are heuristics and not measured forces.",
    "Attribution records describe source-reported terms and grant no redistribution rights.",
)


def _token(value: Any, label: str) -> str:
    if not isinstance(value, str) or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{label} must be an identifier without paths or whitespace.")
    return value


def _public_url(value: Any) -> str:
    if not isinstance(value, str) or any(ord(char) <= 32 for char in value):
        raise ValueError("Source URLs must be public HTTP(S) references.")
    try:
        parts = urlsplit(value)
        host = parts.hostname or ""
        port = parts.port
    except ValueError as error:
        raise ValueError("Source URLs must be public HTTP(S) references.") from error
    if (
        parts.scheme not in {"http", "https"}
        or not host
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
        or "\\" in value
        or host.lower() == "localhost"
        or host.lower().endswith((".localhost", ".local", ".internal"))
        or port not in (None, 80, 443)
    ):
        raise ValueError(
            "Source URLs must exclude credentials, queries and local hosts."
        )
    try:
        address = ipaddress.ip_address(host)
    except ValueError as error:
        if "." not in host:
            raise ValueError("Source URLs must identify public hosts.") from error
    else:
        if not address.is_global:
            raise ValueError("Source URLs must not identify local network addresses.")
    return value


@dataclass(frozen=True)
class SourceAttribution:
    """An explicitly approved source record, separate from arbitrary provenance.

    Citations are DOI identifiers or public HTTP(S) URLs. Hashes are named SHA256
    digests. Free-text source definitions and local input paths stay local.
    """

    source_id: str
    license: str
    citations: tuple[str, ...] = ()
    source_urls: tuple[str, ...] = ()
    source_version: str | None = None
    hashes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _token(self.source_id, "source_id")
        _token(self.license, "license")
        if self.source_version is not None:
            _token(self.source_version, "source_version")
        if isinstance(self.citations, str) or isinstance(self.source_urls, str):
            raise ValueError(
                "Citations and source_urls must be sequences of references."
            )
        citations = tuple(self.citations)
        for citation in citations:
            if not isinstance(citation, str) or _DOI.fullmatch(citation) is None:
                _public_url(citation)
        urls = tuple(_public_url(url) for url in self.source_urls)
        if not isinstance(self.hashes, Mapping):
            raise ValueError("hashes must map identifiers to SHA256 digests.")
        hashes = {}
        for name, digest in self.hashes.items():
            _token(name, "hash name")
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise ValueError("Source hashes must be SHA256 hexadecimal digests.")
            hashes[name] = digest.lower()
        object.__setattr__(self, "citations", citations)
        object.__setattr__(self, "source_urls", urls)
        object.__setattr__(self, "hashes", MappingProxyType(hashes))

    def metadata(self) -> dict[str, Any]:
        """Return only the explicitly approved attribution fields."""
        return {
            "source_id": self.source_id,
            "license": self.license,
            "citations": list(self.citations),
            "source_urls": list(self.source_urls),
            "source_version": self.source_version,
            "hashes": dict(self.hashes),
        }


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class QualityThresholds:
    """Optional caller-chosen upper errors and minimum convergence fraction."""

    marker_rms_m: float | None = None
    marker_max_m: float | None = None
    sole_penetration_max_m: float | None = None
    root_tracking_rms_m: float | None = None
    equality_error_max: float | None = None
    min_converged_fraction: float | None = None

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value is None:
                continue
            number = _number(value)
            if number is None or number < 0:
                raise ValueError("Quality thresholds must be finite and nonnegative.")
            if name == "min_converged_fraction" and number > 1:
                raise ValueError("min_converged_fraction must be between zero and one.")
            object.__setattr__(self, name, number)


def _collect_metrics(
    diagnostics: Mapping[str, Any],
) -> tuple[dict[str, float | int], list[str]]:
    metrics: dict[str, float | int] = {}
    invalid = set()
    for name in sorted(_METRICS & diagnostics.keys()):
        number = _number(diagnostics[name])
        if (
            number is None
            or number < 0
            or (name in _CONDITION_METRICS and number < 1)
            or (name == "converged_fraction" and number > 1)
            or (name in _COUNT_METRICS and not number.is_integer())
            or (name == "frames" and number == 0)
        ):
            invalid.add(name)
        else:
            metrics[name] = int(number) if name in _COUNT_METRICS else number
    frames, converged, failed = (
        metrics.get(name) for name in ("frames", "converged_frames", "failed_frames")
    )
    if frames is not None:
        for name, value in (("converged_frames", converged), ("failed_frames", failed)):
            if value is not None and value > frames:
                invalid.add(name)
        if converged is not None and not invalid.intersection(
            {"frames", "converged_frames"}
        ):
            expected_failed = frames - converged
            if failed is not None and failed != expected_failed:
                invalid.add("failed_frames")
            else:
                metrics["failed_frames"] = expected_failed
            fraction = converged / frames
            if "converged_fraction" in metrics and not math.isclose(
                metrics["converged_fraction"], fraction, rel_tol=1e-12, abs_tol=1e-12
            ):
                invalid.add("converged_fraction")
            elif "converged_fraction" not in invalid:
                metrics["converged_fraction"] = fraction
    for name in invalid:
        metrics.pop(name, None)
    return metrics, sorted(invalid)


def _quality(
    metrics: Mapping[str, float | int],
    invalid: list[str],
    thresholds: QualityThresholds,
) -> dict[str, Any]:
    checks = {}
    for name, threshold in asdict(thresholds).items():
        if threshold is None:
            continue
        metric = "converged_fraction" if name == "min_converged_fraction" else name
        observed = metrics.get(metric)
        comparison = "at_least" if name == "min_converged_fraction" else "at_most"
        if metric in invalid:
            status, reason = "fail", "invalid_metric"
        elif observed is None:
            status, reason = "unknown", "missing_metric"
        else:
            passed = (
                observed >= threshold
                if comparison == "at_least"
                else observed <= threshold
            )
            status, reason = (
                ("pass", "within_threshold")
                if passed
                else ("fail", "threshold_exceeded")
            )
        checks[metric] = {
            "status": status,
            "observed": observed,
            "threshold": threshold,
            "comparison": comparison,
            "reason": reason,
        }
    statuses = {check["status"] for check in checks.values()}
    if invalid or "fail" in statuses:
        status = "fail"
    elif not checks or "unknown" in statuses:
        status = "unknown"
    else:
        status = "pass"
    return {
        "status": status,
        "accepted": status == "pass",
        "checks": checks,
        "invalid_metrics": invalid,
    }


def evaluate_quality(
    diagnostics: Mapping[str, Any], thresholds: QualityThresholds
) -> dict[str, Any]:
    """Evaluate requested limits; missing evidence stays unknown and unaccepted.

    Invalid allowlisted metrics fail, including negative penetration/error values,
    nonfinite numbers and inconsistent frame counts. Other input fields are ignored.
    """
    if not isinstance(diagnostics, Mapping) or not isinstance(
        thresholds, QualityThresholds
    ):
        raise TypeError("A diagnostics mapping and QualityThresholds are required.")
    metrics, invalid = _collect_metrics(diagnostics)
    return _quality(metrics, invalid, thresholds)


def aggregate_validation_report(
    result: MarkerRetargetResult | Mapping[str, Any],
    *,
    thresholds: QualityThresholds,
    attributions: tuple[SourceAttribution, ...] = (),
    protocol_id: str | None = None,
) -> dict[str, Any]:
    """Build an aggregate-only report without copying result provenance or arrays."""
    if isinstance(result, Mapping):
        diagnostics = result
    else:
        from myosuite.integrations.capture.retargeting import MarkerRetargetResult

        if not isinstance(result, MarkerRetargetResult):
            raise TypeError("Expected MarkerRetargetResult or a diagnostics mapping.")
        diagnostics = result.diagnostics
    if not isinstance(thresholds, QualityThresholds):
        raise TypeError("thresholds must be QualityThresholds.")
    if protocol_id is not None:
        _token(protocol_id, "protocol_id")
    if not isinstance(attributions, tuple) or any(
        not isinstance(source, SourceAttribution) for source in attributions
    ):
        raise TypeError("attributions must be an explicit tuple of SourceAttribution.")
    metrics, invalid = _collect_metrics(diagnostics)
    return {
        "schema": 1,
        "report_kind": "aggregate_capture_validation",
        "protocol_id": protocol_id,
        "metrics": metrics,
        "thresholds": asdict(thresholds),
        "quality": _quality(metrics, invalid, thresholds),
        "sources": [source.metadata() for source in attributions],
        "scope": list(_SCOPE),
    }


def write_aggregate_report(report: Mapping[str, Any], path: str | Path) -> Path:
    """Revalidate/rebuild the allowlisted schema before writing JSON.

    Unknown fields or modified decisions are rejected, rather than serializing
    arbitrary additions to a previously safe report. This does not clear data rights.
    """
    expected = {
        "schema",
        "report_kind",
        "protocol_id",
        "metrics",
        "thresholds",
        "quality",
        "sources",
        "scope",
    }
    if not isinstance(report, Mapping) or set(report) != expected:
        raise ValueError("Unexpected aggregate report schema.")
    if report["schema"] != 1 or report["report_kind"] != "aggregate_capture_validation":
        raise ValueError("Unsupported aggregate report schema.")
    metrics, threshold_values, quality = (
        report["metrics"],
        report["thresholds"],
        report["quality"],
    )
    if not isinstance(metrics, Mapping) or not set(metrics).issubset(_METRICS):
        raise ValueError("Aggregate metrics contain unsupported fields.")
    if not isinstance(threshold_values, Mapping) or set(threshold_values) != set(
        asdict(QualityThresholds())
    ):
        raise ValueError("Unexpected threshold schema.")
    if not isinstance(quality, Mapping) or set(quality) != {
        "status",
        "accepted",
        "checks",
        "invalid_metrics",
    }:
        raise ValueError("Unexpected quality schema.")
    invalid = quality["invalid_metrics"]
    if not isinstance(invalid, list) or any(
        not isinstance(name, str) or name not in _METRICS for name in invalid
    ):
        raise ValueError("Unexpected invalid-metric identifiers.")
    sources = report["sources"]
    source_fields = {
        "source_id",
        "license",
        "citations",
        "source_urls",
        "source_version",
        "hashes",
    }
    if not isinstance(sources, list) or any(
        not isinstance(source, Mapping) or set(source) != source_fields
        for source in sources
    ):
        raise ValueError("Unexpected source-attribution schema.")
    rebuilt = aggregate_validation_report(
        {**metrics, **dict.fromkeys(invalid)},
        thresholds=QualityThresholds(**threshold_values),
        attributions=tuple(SourceAttribution(**source) for source in sources),
        protocol_id=report["protocol_id"],
    )
    if dict(report) != rebuilt:
        raise ValueError(
            "Aggregate report was modified or contains invalid scalar fields."
        )
    serialized = json.dumps(rebuilt, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")
    return path
