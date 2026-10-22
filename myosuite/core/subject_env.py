# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Portable morphology and task settings for registered subject environments."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, asdict, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import mujoco
import numpy as np

from myosuite.core.model_builder import ModelBuilder
from myosuite.core.subject_model import (
    GroundAlignment,
    ScaleEstimate,
    SubjectModel,
    SubjectModelConfig,
    _digest,
    align_root_to_ground,
    build_subject_model,
    model_digest,
)


@dataclass(frozen=True)
class SubjectEnvConfig:
    """Explicit task parameters accompanying a subject morphology.

    Args:
        model: Scaling recipe, including named root and sole ground alignment.
        min_height: Fall threshold for centre-of-mass height, in world metres.
        reset_keyframe: Authored keyframe used as the reset pose and velocity.
        contact_margin: Optional sole collision margin in metres; None preserves
            the authored values. Margins are never inferred from scale factors.
        joint_offsets: Reset offsets U(lo, hi) by scalar joint name, in radians
            for hinges and metres for slides. Empty means deterministic reset.
        reset_qpos: Optional retargeted reset pose in the model's qpos order.
        reset_qvel: Optional reset velocity in the model's qvel order.
    """

    model: SubjectModelConfig
    min_height: float
    reset_keyframe: int = 2
    contact_margin: float | None = None
    joint_offsets: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    reset_qpos: tuple[float, ...] | None = None
    reset_qvel: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.model.ground is None:
            raise ValueError(
                "A registered subject task requires explicit ground alignment."
            )
        if not np.isfinite(self.min_height) or self.min_height <= 0:
            raise ValueError("min_height must be finite and positive (metres).")
        if not isinstance(self.reset_keyframe, int) or self.reset_keyframe < 0:
            raise ValueError("reset_keyframe must be a nonnegative integer.")
        if self.contact_margin is not None and (
            not np.isfinite(self.contact_margin) or self.contact_margin < 0
        ):
            raise ValueError("contact_margin must be finite and nonnegative (metres).")
        offsets = {}
        for name, bounds in sorted(self.joint_offsets.items()):
            values = np.asarray(bounds, dtype=float)
            if (
                values.shape != (2,)
                or not np.all(np.isfinite(values))
                or values[0] > values[1]
            ):
                raise ValueError(f"Invalid reset bounds for joint {name!r}.")
            offsets[name] = (float(values[0]), float(values[1]))
        object.__setattr__(self, "joint_offsets", MappingProxyType(offsets))
        for name in ("reset_qpos", "reset_qvel"):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value, dtype=float)
                if array.ndim != 1 or not np.all(np.isfinite(array)):
                    raise ValueError(
                        f"{name} must be a finite one-dimensional sequence."
                    )
                object.__setattr__(self, name, tuple(float(x) for x in array))

    def __deepcopy__(self, memo: dict[int, Any]) -> SubjectEnvConfig:
        return self

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return (
            type(self),
            (
                self.model,
                self.min_height,
                self.reset_keyframe,
                self.contact_margin,
                dict(self.joint_offsets),
                self.reset_qpos,
                self.reset_qvel,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a versioned JSON recipe suitable for CLI/run provenance."""
        return {
            "schema": 1,
            "model": self.model.to_dict(),
            "min_height": float(self.min_height),
            "reset_keyframe": self.reset_keyframe,
            "contact_margin": None
            if self.contact_margin is None
            else float(self.contact_margin),
            "joint_offsets": {
                name: list(values) for name, values in self.joint_offsets.items()
            },
            "reset_qpos": None if self.reset_qpos is None else list(self.reset_qpos),
            "reset_qvel": None if self.reset_qvel is None else list(self.reset_qvel),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SubjectEnvConfig:
        """Restore and validate a versioned JSON recipe."""
        data = dict(value)
        if data.pop("schema", None) != 1:
            raise ValueError("Unsupported subject environment schema.")
        model = dict(data.pop("model"))
        if model.pop("schema", None) != 1:
            raise ValueError("Unsupported subject model schema.")
        estimate = ScaleEstimate(**model.pop("estimate"))
        ground = model.pop("ground", None)
        morphology = SubjectModelConfig(
            estimate=estimate,
            ground=None if ground is None else GroundAlignment(**ground),
            **model,
        )
        return cls(model=morphology, **data)

    @classmethod
    def load(cls, path: str | Path) -> SubjectEnvConfig:
        """Load a JSON subject recipe used by training and evaluation CLIs."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @property
    def digest(self) -> str:
        """Canonical task/morphology identity, independent of mapping order."""
        return _digest(self.to_dict())


def build_subject_env_model(
    source: mujoco.MjSpec | ModelBuilder, config: SubjectEnvConfig
) -> SubjectModel:
    """Build the shared leg task model before either backend initializes state.

    The selected reset keyframe is aligned in its own pose, independently of
    neutral qpos0. Scalar reset offsets are validated against the built topology.
    """
    spec = source.build_spec() if isinstance(source, ModelBuilder) else source.copy()
    terrain = spec.geom("terrain")
    if terrain is not None:
        terrain.pos = np.array([0.0, 0.0, -10.0])
        terrain.rgba[3] = 0.0
    subject = build_subject_model(spec, config.model)
    model, spec = subject.model, subject.spec
    if config.reset_keyframe >= model.nkey:
        raise ValueError(
            f"reset_keyframe {config.reset_keyframe} is absent from the model."
        )
    ground = config.model.ground
    assert ground is not None
    if config.contact_margin is not None:
        for name in ground.sole_geoms:
            spec.geom(name).margin = config.contact_margin
    for name in config.joint_offsets:
        joint = model.joint(name)
        if int(joint.type[0]) not in (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        ):
            raise ValueError(f"Reset offsets require scalar joints, got {name!r}.")
    initial = model.key_qpos[config.reset_keyframe].copy()
    velocity = model.key_qvel[config.reset_keyframe].copy()
    for name, size, target in (
        ("reset_qpos", model.nq, initial),
        ("reset_qvel", model.nv, velocity),
    ):
        value = getattr(config, name)
        if value is not None:
            if len(value) != size:
                raise ValueError(f"{name} requires {size} values, got {len(value)}.")
            target[:] = value
    for joint in range(model.njnt):
        kind = int(model.jnt_type[joint])
        if kind in (int(mujoco.mjtJoint.mjJNT_FREE), int(mujoco.mjtJoint.mjJNT_BALL)):
            start = int(model.jnt_qposadr[joint]) + (
                3 if kind == int(mujoco.mjtJoint.mjJNT_FREE) else 0
            )
            if not np.isclose(np.linalg.norm(initial[start : start + 4]), 1, atol=1e-6):
                raise ValueError("reset_qpos requires unit joint quaternions.")
    initial, offset = align_root_to_ground(model, ground, initial)
    spec.keys[config.reset_keyframe].qpos = initial
    spec.keys[config.reset_keyframe].qvel = velocity
    model = spec.copy().compile()
    metadata = copy.deepcopy(subject.metadata)
    metadata.update(
        {
            "subject_env_config": config.to_dict(),
            "task_digest": config.digest,
            "reset_root_z_offset": offset,
            "model_digest": model_digest(model),
            "cache_key": _digest(
                {"model": metadata["reference_model_digest"], "config": config.digest}
            ),
        }
    )
    return SubjectModel(model, spec, metadata)


def subject_metadata(value: Any) -> dict[str, Any] | None:
    """Read subject provenance from a CPU env, mjlab env, or mjlab config."""
    host = getattr(value, "unwrapped", value)
    metadata = getattr(host, "subject_model_metadata", None)
    if metadata is not None:
        return copy.deepcopy(metadata)
    cfg = getattr(host, "cfg", host)
    event = getattr(cfg, "events", {}).get("subject_provenance")
    return None if event is None else copy.deepcopy(event.params["metadata"])


def with_subject_features(metadata: dict[str, Any], features: Any) -> dict[str, Any]:
    """Include muscle-command/condition configuration in policy admission identity.

    The compiled-model digest describes the shared model before feature wrappers.
    The additional feature digest identifies static strength edits and dynamic
    stages. Subject feature parameters must have a stable JSON representation.
    """

    def normalize(value: Any) -> Any:
        if is_dataclass(value) and not isinstance(value, type):
            return normalize(asdict(value))
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Mapping):
            return {str(key): normalize(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [normalize(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(
            "Subject feature parameters must be JSON-serializable for provenance."
        )

    records = [
        {"entry_point": str(spec.entry_point), "kwargs": normalize(spec.kwargs or {})}
        for spec in features
    ]
    result = copy.deepcopy(metadata)
    result["features"] = records
    result["feature_digest"] = _digest({"features": records})
    return result


def parse_subject_args(argv: list[str]) -> tuple[SubjectEnvConfig | None, list[str]]:
    """Extract a single --subject JSON path before backend CLI parsing."""
    remaining = []
    subject = None
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--subject" or arg.startswith("--subject="):
            if subject is not None:
                raise ValueError("Pass --subject once.")
            if arg == "--subject":
                index += 1
                if index == len(argv):
                    raise ValueError("--subject requires a JSON path.")
                path = argv[index]
            else:
                path = arg.split("=", 1)[1]
            subject = SubjectEnvConfig.load(path)
        else:
            remaining.append(arg)
        index += 1
    return subject, remaining
