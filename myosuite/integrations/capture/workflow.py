# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Local capture interchange and configurable subject-model workflow CLI."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal
from collections.abc import Sequence

import mujoco
import numpy as np
from gymnasium.error import Error as GymnasiumError

from myosuite import make_env
from myosuite.core.subject_model import ScaleEstimate, SubjectModel
from myosuite.integrations.capture.landmarks import read_named_landmarks
from myosuite.integrations.capture.markers import read_c3d
from myosuite.integrations.capture.protocols import TargetProtocol
from myosuite.integrations.capture.reporting import (
    QualityThresholds,
    SourceAttribution,
    aggregate_validation_report,
    write_aggregate_report,
)
from myosuite.integrations.capture.retargeting import (
    load_retargeted_clip,
    motion_diagnostics,
)
from myosuite.integrations.capture.trial_io import load_marker_trial, save_marker_trial


@dataclass(frozen=True)
class ModelSource:
    """Reference an existing registered environment or locally licensed MJCF."""

    kind: Literal["env", "xml"]
    value: str

    def __post_init__(self) -> None:
        if (
            self.kind not in {"env", "xml"}
            or not isinstance(self.value, str)
            or not self.value.strip()
        ):
            raise ValueError("Model source requires kind env/xml and a nonempty value.")

    def spec(self) -> mujoco.MjSpec:
        """Construct a source spec using the normal environment or MJCF loader."""
        if self.kind == "xml":
            return mujoco.MjSpec.from_file(str(Path(self.value).resolve()))
        with make_env(self.value) as env:
            return env.unwrapped._mj_spec.copy()


def _json_read(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object.")
    return value


def _distinct_output(output: str | Path, *inputs: str | Path) -> None:
    if Path(output).resolve() in {Path(path).resolve() for path in inputs}:
        raise ValueError(
            "Output would overwrite a workflow input; choose a different path."
        )


def _json_write(path: str | Path, value: dict[str, Any]) -> Path:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return path


def inspect_trial(path: str | Path) -> dict[str, Any]:
    """Summarize a versioned local trial without emitting observed coordinates."""
    trial = load_marker_trial(path)
    valid = np.isfinite(trial.positions).all(axis=-1)
    return {
        "schema": 1,
        "scope": "local inspection; not a publication or redistribution grant",
        "frames": len(trial.positions),
        "labels": list(trial.labels),
        "valid_fraction_by_marker": dict(
            zip(trial.labels, valid.mean(axis=0).tolist())
        ),
        "units": trial.units,
        "frequency_hz": trial.frequency_hz,
        "point_semantics": trial.point_semantics,
        "subject_id": trial.subject_id,
        "definition_source": trial.definition_source,
        "definition_license": trial.definition_license,
        "metadata_fields": sorted(trial.metadata()),
    }


def dependency_status(source_format: str) -> dict[str, Any]:
    """Check reader availability without importing native packages or assets."""
    requirements = {
        "trial": (),
        "csv": (),
        "c3d": ("ezc3d",),
        "trc": ("myosuite.integrations.capture.trc",),
        "b3d": ("myosuite.integrations.capture.native_sources", "nimblephysics"),
        "bioamass": ("myosuite.integrations.capture.native_sources", "opensim"),
        "skel": ("myosuite.integrations.capture.native_sources", "skel", "torch"),
    }
    if source_format not in requirements:
        raise ValueError(f"Unknown source format: {source_format}.")
    missing = [
        name
        for name in requirements[source_format]
        if importlib.util.find_spec(name) is None
    ]
    return {
        "format": source_format,
        "available": not missing,
        "missing_modules": missing,
        "check": "module discovery only; native import/ABI compatibility is checked when reading",
        "next_step": "Evaluate native inputs in their compatible licensed environment, then save_marker_trial for transfer; source permissions still apply.",
    }


def estimate_trial(
    trial_path: str | Path,
    protocol_path: str | Path,
    output: str | Path,
    *,
    calibration_frames: Sequence[int] = (0,),
) -> Path:
    """Save a local scale estimate and complete evidence from explicit calibration."""
    _distinct_output(output, trial_path, protocol_path)
    trial = load_marker_trial(trial_path)
    protocol = TargetProtocol.load(protocol_path)
    result = protocol.estimate_scales(trial, calibration_frames)
    return _json_write(
        output,
        {
            "schema": 1,
            "scope": "local fitting evidence; may reconstruct licensed inputs",
            "protocol": protocol.to_dict(),
            "estimate": result.estimate.to_dict(),
            "evidence": result.evidence,
        },
    )


def build_target(
    source: ModelSource,
    protocol_path: str | Path,
    estimate_path: str | Path,
    output: str | Path,
) -> Path:
    """Validate construction and save a local rebuild recipe, without copying assets."""
    _distinct_output(output, protocol_path, estimate_path)
    if source.kind == "xml":
        _distinct_output(output, source.value)
    protocol = TargetProtocol.load(protocol_path)
    payload = _json_read(estimate_path)
    if payload.get("schema") != 1 or payload.get("protocol") != protocol.to_dict():
        raise ValueError(
            "Estimate and target protocol differ; estimate again with this profile."
        )
    estimate = ScaleEstimate(**payload["estimate"])
    subject = protocol.build_subject(source.spec(), estimate)
    if source.kind == "xml":
        source = ModelSource("xml", str(Path(source.value).resolve()))
    return _json_write(
        output,
        {
            "schema": 1,
            "scope": "local subject recipe; input/output licences apply",
            "model_source": {"kind": source.kind, "value": source.value},
            "protocol": protocol.to_dict(),
            "estimate": estimate.to_dict(),
            "scale_evidence": payload["evidence"],
            "model_digest": subject.metadata["model_digest"],
        },
    )


def _load_target(
    path: str | Path,
) -> tuple[TargetProtocol, SubjectModel, dict[str, Any]]:
    payload = _json_read(path)
    if payload.get("schema") != 1:
        raise ValueError("Unsupported local build recipe schema.")
    protocol = TargetProtocol.from_dict(payload["protocol"])
    source = ModelSource(**payload["model_source"])
    subject = protocol.build_subject(
        source.spec(), ScaleEstimate(**payload["estimate"])
    )
    if subject.metadata["model_digest"] != payload["model_digest"]:
        raise ValueError("Model differs from the saved build; rebuild before fitting.")
    return protocol, subject, payload


def retarget_trial(
    trial_path: str | Path,
    build_path: str | Path,
    output: str | Path,
    *,
    tolerance_m: float = 1e-4,
    max_evaluations: int = 200,
    keep_failed: bool = False,
) -> Path:
    """Fit against the rebuilt subject and save local motion plus full provenance."""
    _distinct_output(output, trial_path, build_path)
    protocol, subject, payload = _load_target(build_path)
    trial = load_marker_trial(trial_path)
    calibration_subject = payload["scale_evidence"]["trial"].get("subject_id")
    if (
        calibration_subject is not None
        and trial.subject_id is not None
        and calibration_subject != trial.subject_id
    ):
        raise ValueError("Calibration and motion identify different subjects.")
    fitted = protocol.retarget(
        subject,
        trial,
        tolerance_m=tolerance_m,
        max_evaluations=max_evaluations,
        strict=not keep_failed,
        scale_evidence=payload["scale_evidence"],
    )
    return fitted.save(output)


def validate_trial(
    trial_path: str | Path,
    build_path: str | Path,
    clip_path: str | Path,
    output: str | Path,
    *,
    thresholds: QualityThresholds,
    attributions: tuple[SourceAttribution, ...] = (),
) -> dict[str, Any]:
    """Replay locally, then export only approved aggregate metrics and attribution."""
    _distinct_output(output, trial_path, build_path, clip_path)
    protocol, subject, _ = _load_target(build_path)
    trial = load_marker_trial(trial_path)
    protocol.validate_against_model(subject.model, trial)
    clip = load_retargeted_clip(clip_path, subject)
    if clip.frequency_hz != trial.frequency_hz:
        raise ValueError(
            "Clip and trial sampling rates differ; resample explicitly before fitting."
        )
    diagnostics = motion_diagnostics(
        subject.model,
        clip,
        trial,
        protocol.marker_sites,
        free_joint=protocol.free_joint,
        sole_geoms=protocol.sole_geoms,
        root_marker=protocol.root_marker,
        ground_height_m=protocol.ground_height_m,
    )
    with np.load(clip_path, allow_pickle=False) as archive:
        provenance = json.loads(str(archive["provenance_json"]))
    if provenance.get("target_protocol") != protocol.to_dict():
        raise ValueError("Clip and build target protocols differ.")
    if provenance.get("trial") != trial.metadata():
        raise ValueError("Clip and validation trial provenance differ.")
    if (
        provenance.get("marker_sha256")
        != hashlib.sha256(np.asarray(trial.positions).tobytes()).hexdigest()
    ):
        raise ValueError("Clip and validation marker coordinates differ.")
    if (
        provenance.get("fitted_qpos_sha256")
        != hashlib.sha256(np.asarray(clip.qpos).tobytes()).hexdigest()
        or provenance.get("fitted_frequency_hz") != clip.frequency_hz
    ):
        raise ValueError(
            "Saved fitted motion was changed; retarget before using its solver summary."
        )
    summary = provenance.get("solver_summary", {})
    if summary and summary.get("frames") != len(clip.qpos):
        raise ValueError("Saved solver frame count differs from the replay.")
    for key in ("frames", "converged_frames"):
        if key in summary:
            diagnostics[key] = summary[key]
    report = aggregate_validation_report(
        diagnostics,
        thresholds=thresholds,
        attributions=attributions,
        protocol_id="protocol-"
        + hashlib.sha256(
            json.dumps(protocol.to_dict(), sort_keys=True).encode()
        ).hexdigest(),
    )
    write_aggregate_report(report, output)
    return report


def _export(args: argparse.Namespace) -> Path:
    status = dependency_status(args.format)
    if not status["available"]:
        raise ImportError(
            f"Missing modules: {', '.join(status['missing_modules'])}. {status['next_step']}"
        )
    if args.format == "trial":
        return save_marker_trial(load_marker_trial(args.input), args.output)
    if (
        args.rotation is None
        or not args.definition_source
        or not args.definition_license
        or not args.point_semantics
    ):
        raise ValueError(
            "Raw export requires --rotation, --definition-source, --definition-license and --point-semantics."
        )
    kwargs = {
        "rotation_to_model": np.array(args.rotation).reshape(3, 3),
        "translation_to_model_m": args.translation,
        "definition_source": args.definition_source,
        "definition_license": args.definition_license,
    }
    if args.format == "csv":
        if args.units is None:
            raise ValueError("CSV export requires explicit --units.")
        trial = read_named_landmarks(
            args.input, units=args.units, source=str(args.input), **kwargs
        )
    elif args.format == "c3d":
        definitions = {
            key: kwargs.pop(key) for key in ("definition_source", "definition_license")
        }
        trial = replace(read_c3d(args.input, **kwargs), **definitions)
    elif args.format == "trc":
        reader = importlib.import_module("myosuite.integrations.capture.trc").read_trc
        trial = reader(args.input, **kwargs)
    else:
        readers = importlib.import_module(
            "myosuite.integrations.capture.native_sources"
        )
        if args.format == "b3d":
            trial = readers.read_addbiomechanics(args.input, args.trial_index, **kwargs)
        else:
            if args.motion is None or not args.body_names:
                raise ValueError("BioAmass export requires --motion and --body-names.")
            trial = readers.read_bioamass(
                args.input, args.motion, body_names=args.body_names, **kwargs
            )
    return save_marker_trial(
        replace(
            trial, point_semantics=args.point_semantics, subject_id=args.subject_id
        ),
        args.output,
    )


def _source(args: argparse.Namespace) -> ModelSource:
    return (
        ModelSource("env", args.env)
        if args.env
        else ModelSource("xml", str(args.model.resolve()))
    )


def _thresholds(args: argparse.Namespace) -> QualityThresholds:
    return QualityThresholds(
        **{field.name: getattr(args, field.name) for field in fields(QualityThresholds)}
    )


def _attributions(path: Path | None) -> tuple[SourceAttribution, ...]:
    if path is None:
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Attributions must be a JSON list of approved source records.")
    return tuple(SourceAttribution(**item) for item in payload)


def _quality_options(parser: argparse.ArgumentParser) -> None:
    for field in fields(QualityThresholds):
        parser.add_argument("--" + field.name.replace("_", "-"), type=float)
    parser.add_argument("--attributions", type=Path)


def main(argv: Sequence[str] | None = None) -> None:
    """Inspect, export, estimate, build, retarget and validate local capture inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor")
    doctor.add_argument(
        "--format",
        choices=("trial", "csv", "trc", "c3d", "b3d", "bioamass", "skel"),
        default="trial",
    )
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--trial", type=Path, required=True)
    export = commands.add_parser("export")
    export.add_argument("--input", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument(
        "--format",
        choices=("trial", "csv", "trc", "c3d", "b3d", "bioamass"),
        required=True,
    )
    export.add_argument("--units", choices=("m", "cm", "mm"))
    export.add_argument("--rotation", type=float, nargs=9)
    export.add_argument("--translation", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    export.add_argument("--definition-source")
    export.add_argument("--definition-license")
    export.add_argument("--point-semantics")
    export.add_argument("--subject-id")
    export.add_argument("--motion", type=Path)
    export.add_argument("--body-names", nargs="+")
    export.add_argument("--trial-index", type=int, default=0)
    for name in ("estimate", "build", "retarget", "validate", "run"):
        command = commands.add_parser(name)
        if name != "build":
            command.add_argument("--trial", type=Path, required=True)
        if name in {"estimate", "build", "run"}:
            command.add_argument("--profile", type=Path, required=True)
        if name in {"estimate", "run"}:
            command.add_argument(
                "--calibration-frames", type=int, nargs="+", default=[0]
            )
        if name in {"build", "run"}:
            source = command.add_mutually_exclusive_group(required=True)
            source.add_argument("--env")
            source.add_argument("--model", type=Path)
        if name == "build":
            command.add_argument("--estimate", type=Path, required=True)
        if name in {"retarget", "validate"}:
            command.add_argument("--build", type=Path, required=True)
        if name in {"retarget", "run"}:
            command.add_argument("--tolerance-m", type=float, default=1e-4)
            command.add_argument("--max-evaluations", type=int, default=200)
            command.add_argument("--keep-failed", action="store_true")
        if name == "run":
            command.add_argument("--motion-trial", type=Path)
        if name == "validate":
            command.add_argument("--clip", type=Path, required=True)
        if name in {"validate", "run"}:
            _quality_options(command)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            report = dependency_status(args.format)
            print(json.dumps(report, indent=2))
            if not report["available"]:
                parser.exit(1)
            return
        if args.command == "inspect":
            print(json.dumps(inspect_trial(args.trial), indent=2))
            return
        if args.command == "export":
            print(_export(args))
            return
        if args.command == "estimate":
            print(
                estimate_trial(
                    args.trial,
                    args.profile,
                    args.output,
                    calibration_frames=args.calibration_frames,
                )
            )
            return
        if args.command == "build":
            print(build_target(_source(args), args.profile, args.estimate, args.output))
            return
        if args.command == "retarget":
            print(
                retarget_trial(
                    args.trial,
                    args.build,
                    args.output,
                    tolerance_m=args.tolerance_m,
                    max_evaluations=args.max_evaluations,
                    keep_failed=args.keep_failed,
                )
            )
            return
        if args.command == "run":
            output = args.output.resolve()
            motion_trial = args.motion_trial or args.trial
            estimate = estimate_trial(
                args.trial,
                args.profile,
                output / "estimate.local.json",
                calibration_frames=args.calibration_frames,
            )
            build = build_target(
                _source(args), args.profile, estimate, output / "build.local.json"
            )
            clip = retarget_trial(
                motion_trial,
                build,
                output / "motion.local.npz",
                tolerance_m=args.tolerance_m,
                max_evaluations=args.max_evaluations,
                keep_failed=args.keep_failed,
            )
            report = validate_trial(
                motion_trial,
                build,
                clip,
                output / "aggregate.json",
                thresholds=_thresholds(args),
                attributions=_attributions(args.attributions),
            )
        else:
            report = validate_trial(
                args.trial,
                args.build,
                args.clip,
                args.output,
                thresholds=_thresholds(args),
                attributions=_attributions(args.attributions),
            )
        print(json.dumps(report, indent=2))
        status = report["quality"]["status"]
        if status != "pass":
            parser.exit(1 if status == "fail" else 2)
    except (
        ValueError,
        OSError,
        ImportError,
        KeyError,
        TypeError,
        GymnasiumError,
    ) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
