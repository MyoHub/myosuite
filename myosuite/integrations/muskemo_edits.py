# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Apply selected source-bound anatomy edits without rebuilding the source model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from myosuite.integrations.muskemo import _inertia, model_fingerprint

PATCH_SCHEMA = 1


def _compile_source(spec: mujoco.MjSpec) -> mujoco.MjModel:
    body_count = len(spec.bodies)
    model = spec.compile()
    if model.nbody < body_count:
        raise ValueError(
            "Static-body fusion is unsupported for model patches and archives; "
            "disable fusestatic and export the source again"
        )
    return model


def _fields(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} must contain exactly {sorted(expected)}")
    return value


def _array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        array = np.asarray(value)
        if array.dtype.kind not in "iuf":
            raise ValueError
        array = array.astype(float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must contain real numbers") from exc
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{label} must have shape {shape} and be finite")
    return array


def _mass_inertia(entry: dict[str, Any]) -> tuple[float, np.ndarray, np.ndarray]:
    mass = float(_array(entry["mass"], (), "mass"))
    com = _array(entry["com"], (3,), "com")
    inertia = _array(entry["inertia"], (3, 3), "inertia")
    scale = float(np.max(np.abs(inertia)))
    if mass <= 0:
        raise ValueError("mass must be positive")
    if not np.allclose(inertia, inertia.T, atol=1e-12 * scale, rtol=1e-12):
        raise ValueError("inertia must be symmetric")
    inertia = 0.5 * inertia + 0.5 * inertia.T
    principal = np.linalg.eigvalsh(inertia)
    if not np.isfinite(principal).all() or principal[0] <= 0:
        raise ValueError("inertia must be positive definite")
    normalized = principal / scale
    if normalized[2] > normalized[0] + normalized[1] + 32 * np.finfo(float).eps:
        raise ValueError(
            "inertia must satisfy the principal-moment triangle inequality"
        )
    return mass, com, inertia


def apply_model_patch(spec: mujoco.MjSpec, patch: dict[str, Any]) -> mujoco.MjSpec:
    """Validate and apply named site/inertia edits to a copy of a source spec.

    Args:
        spec: Original model, including its assets and simulation settings.
        patch: Schema-1 document containing ``schema``, ``fingerprint``,
            ``mujoco_version``, ``sites`` and ``inertias``. Site ``pos`` and body
            ``com`` are body-local metres; ``inertia`` is a symmetric 3x3 tensor
            about the COM, expressed in body axes, in kg m².

    Returns:
        A new spec whose edits have compiled successfully. The caller's spec is
        never compiled or modified, including when validation/compilation fails.

    Raises:
        ValueError: Wrong source/version, unknown or duplicate edits, invalid
            numbers/inertias, a compiler-rejected edit, or an ignored inertia.
    """
    _fields(
        patch, {"schema", "fingerprint", "mujoco_version", "sites", "inertias"}, "patch"
    )
    if (
        not isinstance(patch["schema"], int)
        or isinstance(patch["schema"], bool)
        or patch["schema"] != PATCH_SCHEMA
    ):
        raise ValueError("Unsupported model patch schema")
    if patch["mujoco_version"] != mujoco.__version__:
        raise ValueError("Patch MuJoCo version does not match this installation")
    candidate = spec.copy()
    source = _compile_source(candidate)
    if patch["fingerprint"] != model_fingerprint(source):
        raise ValueError("Patch fingerprint does not match the source model")
    validated: dict[str, list[tuple[Any, ...]]] = {"sites": [], "inertias": []}
    for kind, fields, lookup in (
        ("sites", {"name", "pos"}, source.site),
        ("inertias", {"name", "mass", "com", "inertia"}, source.body),
    ):
        entries = patch[kind]
        if not isinstance(entries, list):
            raise ValueError(f"{kind} must be a list")
        seen = set()
        for entry in entries:
            _fields(entry, fields, kind)
            name = entry["name"]
            if not isinstance(name, str) or not name:
                raise ValueError(f"{kind} require a nonempty source name")
            if name in seen:
                raise ValueError(f"Duplicate {kind} edit for {name!r}")
            seen.add(name)
            try:
                obj = lookup(name)
            except KeyError as exc:
                raise ValueError(f"Unknown {kind} name {name!r}") from exc
            if kind == "inertias" and obj.id == 0:
                raise ValueError("The world body's inertia cannot be edited")
            values = (
                (_array(entry["pos"], (3,), "pos"),)
                if kind == "sites"
                else _mass_inertia(entry)
            )
            validated[kind].append((name, obj.id, *values))
    for name, sid, pos in validated["sites"]:
        site = candidate.site(name)
        if site.frame is not None:
            site.set_frame(site.parent.add_frame())
            site.quat = source.site_quat[sid]
            site.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT
        site.pos = pos
    for name, _, mass, com, inertia in validated["inertias"]:
        body = candidate.body(name)
        principal, axes = np.linalg.eigh(inertia)
        if np.linalg.det(axes) < 0:
            axes[:, 0] *= -1
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, axes.ravel())
        body.explicitinertial = True
        body.mass, body.ipos = mass, com
        body.fullinertia = [np.nan, 0, 0, 0, 0, 0]
        body.inertia, body.iquat = principal, quat
        body.ialt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT
    result = _compile_source(candidate)
    for _, sid, pos in validated["sites"]:
        if not (
            result.site_bodyid[sid] == source.site_bodyid[sid]
            and np.allclose(result.site_pos[sid], pos, rtol=1e-9, atol=1e-12)
            and np.isclose(
                abs(np.dot(result.site_quat[sid], source.site_quat[sid])),
                1,
                rtol=0,
                atol=1e-12,
            )
        ):
            raise ValueError(
                "Compiler changed or ignored the requested body-local site edit"
            )
    for _, bid, mass, com, inertia in validated["inertias"]:
        actual = np.asarray(_inertia(result.body_inertia[bid], result.body_iquat[bid]))
        if not (
            np.isclose(result.body_mass[bid], mass, rtol=1e-9, atol=0)
            and np.allclose(result.body_ipos[bid], com, rtol=1e-9, atol=1e-12)
            and np.allclose(
                actual, inertia, rtol=1e-9, atol=1e-12 * np.max(np.abs(inertia))
            )
        ):
            raise ValueError(
                "Compiler settings changed or ignored the requested inertia"
            )
    return candidate


def mesh_inertia(path: str | Path, density: float) -> dict[str, Any]:
    """Mass, local COM and inertia of a homogeneous solid mesh in metre coordinates.

    Args:
        path: One closed, consistently oriented mesh in the target body's axes.
        density: Positive homogeneous density, in kg/m³.

    Returns:
        ``mass``, ``com`` and ``inertia`` fields for an inertia patch entry.

    Raises:
        ImportError: The optional ``trimesh`` package is unavailable.
        ValueError: Invalid density, a scene/non-volume mesh, or invalid inertia.

    Mesh coordinates must already be body-local metres. This does not infer
    tissue composition, repair holes, or validate intersecting/overlapping solids.
    """
    value = float(_array(density, (), "density"))
    if value <= 0:
        raise ValueError("density must be positive")
    try:
        import trimesh  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Install myosuite[muskemo] to compute optional mesh inertia"
        ) from exc
    mesh = trimesh.load_mesh(Path(path), process=True)
    if not isinstance(mesh, trimesh.Trimesh) or not mesh.is_volume:
        raise ValueError(
            "A watertight, consistently oriented positive-volume mesh is required"
        )
    mesh.density = value
    result = dict(
        mass=float(mesh.mass),
        com=mesh.center_mass.tolist(),
        inertia=mesh.moment_inertia.tolist(),
    )
    _mass_inertia(result)
    return result
