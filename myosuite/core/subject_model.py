# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Source-neutral subject model construction and morphology provenance."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import mujoco
import numpy as np
from numpy.typing import ArrayLike

from myosuite.core.body_scaling import _body_scales, _factors, _frame_rot, scale_bodies
from myosuite.core.model_builder import ModelBuilder


@dataclass(frozen=True)
class ScaleEstimate:
    """Validated factors and optional evidence from any scale-estimation source.

    Args:
        factors: Body or segment name to positive local-axis factors.
        height_m: Measured height, recorded as provenance (not inferred scaling).
        mass_kg: Subject mass, used as the target total mass by the builder.
        source: Source identifier, for example an OpenSim ScaleSet filename.
        confidence: Optional source-reported confidence in [0, 1].
        warnings: Source-reported limitations, retained in model metadata.
    """

    factors: Mapping[str, float | ArrayLike]
    height_m: float | None = None
    mass_kg: float | None = None
    source: str = "unspecified"
    confidence: float | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        factors = {}
        for name, value in self.factors.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Scale names must be nonempty strings.")
            factors[name] = tuple(float(v) for v in _factors(value))
        for field_name, scalar in (
            ("height_m", self.height_m),
            ("mass_kg", self.mass_kg),
        ):
            if scalar is not None and (not np.isfinite(scalar) or scalar <= 0):
                raise ValueError(f"{field_name} must be finite and positive.")
            if scalar is not None:
                object.__setattr__(self, field_name, float(scalar))
        if self.confidence is not None and (
            not np.isfinite(self.confidence) or not 0 <= self.confidence <= 1
        ):
            raise ValueError("confidence must be finite and between zero and one.")
        if self.confidence is not None:
            object.__setattr__(self, "confidence", float(self.confidence))
        if not isinstance(self.source, str) or not all(
            isinstance(w, str) for w in self.warnings
        ):
            raise ValueError("source and warnings must be strings.")
        object.__setattr__(self, "factors", MappingProxyType(factors))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable source evidence, with canonical axis factors."""
        return {
            "factors": {
                name: _factors(value).tolist() for name, value in self.factors.items()
            },
            "height_m": self.height_m,
            "mass_kg": self.mass_kg,
            "source": self.source,
            "confidence": self.confidence,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class GroundAlignment:
    """Place the lowest selected sole geom on a horizontal ground plane.

    Args:
        free_joint: Free root joint whose world-z translation can be changed.
        sole_geoms: Explicit collision geom names belonging to that root tree.
        ground_height: World-z plane height in metres.
        clearance: Nonnegative sole clearance above the plane, in metres.
    """

    free_joint: str
    sole_geoms: tuple[str, ...]
    ground_height: float = 0.0
    clearance: float = 0.0

    def __post_init__(self) -> None:
        if not self.sole_geoms:
            raise ValueError("Ground alignment requires at least one sole geom.")
        if not np.isfinite(self.ground_height):
            raise ValueError("ground_height must be finite.")
        if not np.isfinite(self.clearance) or self.clearance < 0:
            raise ValueError("clearance must be finite and nonnegative.")
        object.__setattr__(self, "sole_geoms", tuple(self.sole_geoms))


def align_root_to_ground(
    model: mujoco.MjModel,
    alignment: GroundAlignment,
    qpos: ArrayLike | None = None,
) -> tuple[np.ndarray, float]:
    """Return aligned qpos and its world-z offset, without editing the model.

    Uses actual primitive/mesh support points, after forward kinematics. All
    selected geoms must descend from the named free root. Contact margins and
    task termination thresholds remain explicit task parameters in metres.
    """
    joint = model.joint(alignment.free_joint)
    if joint.type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Ground alignment requires a free root joint.")
    data = mujoco.MjData(model)
    if qpos is not None:
        values = np.asarray(qpos, dtype=float)
        if values.shape != (model.nq,) or not np.all(np.isfinite(values)):
            raise ValueError(f"qpos must be finite with shape ({model.nq},).")
        data.qpos[:] = values
    mujoco.mj_forward(model, data)
    heights = []
    for name in alignment.sole_geoms:
        geom = model.geom(name)
        body = int(geom.bodyid[0])
        root_body = int(joint.bodyid[0])
        while body != root_body and body != 0:
            body = int(model.body_parentid[body])
        if body != root_body:
            raise ValueError(f"Sole geom {name!r} is outside the free root tree.")
        heights.append(_geom_bottom(model, data, geom.id))
    offset = alignment.ground_height + alignment.clearance - min(heights)
    result = data.qpos.copy()
    result[int(joint.qposadr[0]) + 2] += offset
    return result, float(offset)


def _geom_bottom(model: mujoco.MjModel, data: mujoco.MjData, index: int) -> float:
    kind = int(model.geom_type[index])
    size = model.geom_size[index]
    vertical = data.geom_xmat[index].reshape(3, 3)[2]
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        extent = size[0]
    elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        extent = size[0] + size[1] * abs(vertical[2])
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        extent = size[1] * abs(vertical[2]) + size[0] * np.linalg.norm(vertical[:2])
    elif kind == mujoco.mjtGeom.mjGEOM_BOX:
        extent = np.abs(vertical) @ size
    elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        extent = np.linalg.norm(vertical * size)
    elif kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = int(model.geom_dataid[index])
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        return float(
            data.geom_xpos[index, 2]
            + np.min(model.mesh_vert[start : start + count] @ vertical)
        )
    else:
        raise NotImplementedError(f"Unsupported sole geom type {kind}.")
    return float(data.geom_xpos[index, 2] - extent)


@dataclass(frozen=True)
class SubjectModelConfig:
    """Reusable scaling recipe for a CPU or GPU model-building pipeline.

    Args:
        estimate: Source-neutral factors and subject mass.
        segments: Optional segment-to-first-body mapping, as in scale_bodies.
        mass: Scale masses by volume or preserve their relative distribution.
        force_scale: Explicit positive multiplier of peak muscle strength.
        ground: Optional neutral-pose root and keyframe vertical alignment.
    """

    estimate: ScaleEstimate
    segments: Mapping[str, Sequence[str]] | None = None
    mass: Literal["volume", "keep"] = "volume"
    force_scale: float = 1.0
    ground: GroundAlignment | None = None

    def __post_init__(self) -> None:
        if self.mass not in ("volume", "keep"):
            raise ValueError("mass must be 'volume' or 'keep'.")
        if not np.isfinite(self.force_scale) or self.force_scale <= 0:
            raise ValueError("force_scale must be finite and positive.")
        if self.segments is not None:
            object.__setattr__(
                self,
                "segments",
                MappingProxyType(
                    {name: tuple(bodies) for name, bodies in self.segments.items()}
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable, versioned construction recipe."""
        ground = self.ground
        return {
            "schema": 1,
            "estimate": self.estimate.to_dict(),
            "segments": None
            if self.segments is None
            else {name: list(bodies) for name, bodies in self.segments.items()},
            "mass": self.mass,
            "force_scale": float(self.force_scale),
            "ground": None
            if ground is None
            else {
                "free_joint": ground.free_joint,
                "sole_geoms": list(ground.sole_geoms),
                "ground_height": float(ground.ground_height),
                "clearance": float(ground.clearance),
            },
        }

    @property
    def digest(self) -> str:
        """SHA-256 of the recipe, independent of mapping order and scalar syntax."""
        return _digest(self.to_dict())


def _digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def model_digest(model: mujoco.MjModel) -> str:
    """Hash compiled model contents, including assets, within a MuJoCo version.

    The binary format is version-specific; provenance records MuJoCo's version
    and a different version conservatively counts as a different morphology.
    """
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, buffer)
    return hashlib.sha256(mujoco.__version__.encode() + buffer.tobytes()).hexdigest()


@dataclass
class SubjectModel:
    """Built model and spec with a snapshot of its construction provenance.

    Rebuild after further model edits; metadata describes the model at build time.
    """

    model: mujoco.MjModel
    spec: mujoco.MjSpec
    metadata: dict[str, Any] = field(repr=False)

    @property
    def body_scales(self) -> dict[str, tuple[float, float, float]]:
        """Resolved body factors for explicit scale-aware skin binding."""
        return {
            name: tuple(values) for name, values in self.metadata["body_scales"].items()
        }


def build_subject_model(
    source: mujoco.MjSpec | ModelBuilder, config: SubjectModelConfig
) -> SubjectModel:
    """Build a fresh subject model without mutating the source or registration.

    Build this target first, then retarget motion and bind skins against it.
    The returned metadata can be stored with clips/checkpoints or cache entries.
    It includes both the input model identity and resolved per-body factors.
    """
    spec = source.build_spec() if isinstance(source, ModelBuilder) else source.copy()
    reference = spec.copy().compile()
    body_scales = _body_scales(spec, config.estimate.factors, config.segments)
    scale_bodies(
        spec,
        config.estimate.factors,
        segments=config.segments,
        mass=config.mass,
        total_mass=config.estimate.mass_kg,
        force_scale=config.force_scale,
    )
    model = spec.copy().compile()
    offset = 0.0
    if config.ground is not None:
        _, offset = align_root_to_ground(model, config.ground)
        joint = model.joint(config.ground.free_joint)
        body = spec.body(model.body(int(joint.bodyid[0])).name)
        body.pos = np.asarray(body.pos) + _frame_rot(spec, body.frame).T @ np.array(
            [0.0, 0.0, offset]
        )
        address = int(joint.qposadr[0])
        for index, key in enumerate(spec.keys):
            values = model.key_qpos[index].copy()
            values[address + 2] += offset
            key.qpos = values
        model = spec.copy().compile()
    recipe = config.to_dict()
    metadata = {
        "schema": 1,
        "config": recipe,
        "config_digest": config.digest,
        "reference_model_digest": model_digest(reference),
        "model_digest": model_digest(model),
        "mujoco_version": mujoco.__version__,
        "body_scales": {name: value.tolist() for name, value in body_scales.items()},
        "root_z_offset": offset,
    }
    return SubjectModel(model=model, spec=spec, metadata=metadata)


__all__ = [
    "GroundAlignment",
    "ScaleEstimate",
    "SubjectModel",
    "SubjectModelConfig",
    "align_root_to_ground",
    "build_subject_model",
    "model_digest",
]
