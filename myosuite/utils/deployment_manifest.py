"""Data-only provenance for a policy, its physical model and inference contract.

The physical digest is exact and MuJoCo-version-specific. It includes compiled
assets and all public model/option arrays, unlike the retargeting kinematic hash.
Source asset hashes can additionally be supplied when source-byte identity matters.
Unknown units, history and normalization are recorded as unknown, never inferred.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import warnings
from collections.abc import Iterator, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import mujoco
import numpy as np

SCHEMA_VERSION = 1
MODEL_HASH_ALGORITHM = "mujoco-public-fields-v1"


def _model_fields(obj: Any, prefix: str = "") -> Iterator[tuple[str, bytes]]:
    # MJB embeds machine-local paths; use the compiled contents instead (PR #525).
    for name in sorted(dir(obj)):
        if name.startswith("_") or name in ("paths", "npaths", "nbuffer"):
            continue
        if name.endswith("_pathadr"):
            continue
        value = getattr(obj, name)
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            yield (
                prefix + name,
                str((array.dtype.str, array.shape)).encode() + array.tobytes(),
            )
        elif isinstance(value, bytes):
            yield prefix + name, value
        elif isinstance(value, (bool, int, float)):
            yield prefix + name, repr(value).encode()
        elif not callable(value) and type(value).__module__.startswith("mujoco"):
            yield from _model_fields(value, f"{prefix}{name}.")


def model_identity(model: mujoco.MjModel) -> dict[str, Any]:
    """Return exact physical and separate geometry-only compatibility identities.

    Asset path relocation does not change the physical digest; compiled mesh,
    texture, heightfield, inertia, muscle, equality and solver edits do. Runtime
    randomization must be recorded separately from this nominal-model identity.
    """
    from myosuite.biomechanics.retarget import kinematic_fingerprint

    digest = hashlib.sha256()
    for name, value in _model_fields(model):
        for part in (name.encode(), value):
            digest.update(len(part).to_bytes(8, "little"))
            digest.update(part)
    return {
        "algorithm": MODEL_HASH_ALGORITHM,
        "physical_sha256": digest.hexdigest(),
        "mujoco_version": mujoco.__version__,
        "kinematic_fingerprint": kinematic_fingerprint(model, (), ()),
        "kinematic_scope": "body-tree-joints-couplings; no landmark or sole targets",
    }


def _relative_path(name: str) -> str:
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or ":" in name
    ):
        raise ValueError(f"Artifact name must be a relative POSIX path: {name!r}")
    return path.as_posix()


def artifact_record(
    path: str | Path, *, source: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Hash file bytes, optionally recording a pinned Hugging Face source.

    ``source`` uses ``repo_id``, ``filename``, ``revision`` (tag or commit, never
    main/master/HEAD) and optional ``resolved_commit``. No download is performed.
    """
    from myosuite.biomechanics.retarget import file_sha256

    path = Path(path)
    result: dict[str, Any] = {
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }
    if source is not None:
        if not all(source.get(k) for k in ("repo_id", "filename", "revision")):
            raise ValueError(
                "HF source requires repo_id, filename and a pinned revision"
            )
        if source["revision"].lower() in ("main", "master", "head"):
            raise ValueError(
                "HF source requires a pinned tag or commit, not a default branch"
            )
        result["source"] = dict(source)
    return result


def _native_settings(model: mujoco.MjModel) -> dict[str, Any]:
    return {
        name: value.tolist() if isinstance(value, np.ndarray) else value
        for name in sorted(dir(model.opt))
        if not name.startswith("_")
        and isinstance(
            value := getattr(model.opt, name), (np.ndarray, bool, int, float)
        )
    }


def build_deployment_manifest(
    model: mujoco.MjModel,
    *,
    interface: Mapping[str, Any],
    control_dt: float,
    artifacts: Mapping[str, str | Path] | None = None,
    references: Mapping[str, Mapping[str, Any]] | None = None,
    checkpoint: Mapping[str, Any] | None = None,
    normalization: Mapping[str, Any] | None = None,
    backend: str = "cpu",
    runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe an actual compiled deployment model and explicit inference I/O.

    Args:
        model: Model actually used, including nominal physical edits.
        interface: Ordered observation/action terms, units, shapes and history.
            Unknown fields should be null; list order is significant. Also record
            action mappings, stateful command bindings and wrapper configuration.
        control_dt: Seconds between policy actions, an integer physics step multiple.
        artifacts: Bundle-relative path -> actual local file (checkpoint/ONNX/assets).
        references: Named ``artifact_record`` records for motion/calibration inputs.
        checkpoint: Training checkpoint byte identity/source; provenance only, it
            need not be shipped with an exported inference policy.
        normalization: Frozen statistics/embedding description, or unknown if omitted.
        backend: Runtime backend name. Package versions are recorded without imports.
        runtime: Explicit target runtime versions/configuration (e.g. WASM engine,
            ONNX Runtime Web and browser); kept separate from build-time packages.

    Returns:
        JSON-safe manifest. No weights, pickle payloads or absolute paths are embedded.
    """
    physics_dt = float(model.opt.timestep)
    ratio = float(control_dt) / physics_dt
    if (
        not np.isfinite(ratio)
        or ratio < 1
        or not np.isclose(ratio, round(ratio), rtol=0, atol=1e-8)
    ):
        raise ValueError(
            "control_dt must be a positive integer multiple of physics timestep"
        )
    dependencies = {}
    for package in (
        "myosuite",
        "myo_sim",
        "mujoco",
        "mjlab",
        "mujoco-warp",
        "torch",
        "onnx",
        "onnxruntime",
        "numpy",
        "mjswan",
        "mjviser",
    ):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = None
    result = {
        "schema": SCHEMA_VERSION,
        "model": model_identity(model),
        "interface": dict(interface),
        "normalization": dict(normalization)
        if normalization is not None
        else {"kind": "unknown"},
        "timing": {
            "physics_dt": physics_dt,
            "control_dt": float(control_dt),
            "decimation": round(ratio),
        },
        "physics_options": _native_settings(model),
        "runtime": {
            "backend": backend,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": dependencies,
            "target": dict(runtime or {}),
        },
        "artifacts": {
            _relative_path(name): artifact_record(path)
            for name, path in (artifacts or {}).items()
        },
        "references": dict(references or {}),
        "checkpoint": dict(checkpoint) if checkpoint is not None else None,
    }
    return json.loads(json.dumps(result, allow_nan=False))


def env_deployment_manifest(env: Any, **kwargs: Any) -> dict[str, Any]:
    """Describe a CPU env without inventing per-term units or history semantics.

    Pass ``interface`` and ``normalization`` explicitly for a fully specified
    deployment. The default records the env's ordered obs keys, spaces and
    actuator names; units/history/action mapping remain unknown.
    """
    from myosuite.utils.checkpoint_manifest import _shape

    unwrapped = env.unwrapped
    model = unwrapped.model
    interface = kwargs.pop(
        "interface",
        {
            "obs_shape": _shape(env.observation_space),
            "action_shape": _shape(env.action_space),
            "observations": list(getattr(unwrapped, "obs_keys", ())),
            "actions": [model.actuator(i).name or None for i in range(model.nu)],
            "units": None,
            "history": None,
            "action_mapping": None,
        },
    )
    dt = next(
        getattr(unwrapped, key) for key in ("dt", "_ctrl_dt") if hasattr(unwrapped, key)
    )
    return build_deployment_manifest(
        model, interface=interface, control_dt=float(dt), **kwargs
    )


def write_deployment_manifest(
    artifact: str | Path, manifest: Mapping[str, Any]
) -> Path:
    """Write the data-only ``<artifact>.manifest.json`` sidecar."""
    path = Path(str(artifact) + ".manifest.json")
    path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return path


def validate_deployment_manifest(
    manifest: Mapping[str, Any],
    *,
    current: Mapping[str, Any] | None = None,
    artifact_root: str | Path | None = None,
    allow_mismatch: bool = False,
) -> list[str]:
    """Reject mismatched model, I/O, normalization, timing, references or bytes.

    ``current`` compares every compatibility field; runtime backend and optional
    dependency versions are provenance only (CPU/GPU/browser deliberately differ).
    The exact model hash is tied to the recorded MuJoCo version. Artifact integrity
    can also be checked without constructing an env. Old checkpoint manifests have
    no deployment section and their caller keeps the schema-1 warning behavior.
    An explicit override warns and returns every mismatch rather than hiding it.
    """
    mismatches = []
    if manifest.get("schema") != SCHEMA_VERSION:
        mismatches.append("unsupported deployment schema")
    for field in (
        "model",
        "interface",
        "normalization",
        "timing",
        "physics_options",
        "runtime",
        "artifacts",
        "references",
    ):
        if field not in manifest:
            mismatches.append(f"missing {field}")
    if current is not None:
        for field in (
            "model",
            "interface",
            "normalization",
            "timing",
            "physics_options",
            "references",
        ):
            if manifest.get(field) != current.get(field):
                mismatches.append(field)
    if artifact_root is not None:
        for name, expected in manifest.get("artifacts", {}).items():
            path = Path(artifact_root) / _relative_path(name)
            if not path.is_file() or artifact_record(path) != {
                k: expected[k] for k in ("sha256", "size_bytes")
            }:
                mismatches.append(f"artifact {name}")
    if mismatches:
        message = "Deployment provenance mismatch: " + ", ".join(mismatches)
        if not allow_mismatch:
            raise ValueError(
                message
                + ". Pass allow_mismatch=True only after reviewing the differences."
            )
        warnings.warn(message + " (explicit override)", stacklevel=2)
    return mismatches


def validate_policy_manifest(
    checkpoint: str | Path,
    env: Any,
    *,
    current: Mapping[str, Any] | None = None,
    allow_mismatch: bool = False,
) -> list[str]:
    """Validate a checkpoint/ONNX sidecar before loading its executable payload.

    Legacy manifests remain readable. Automatic env validation checks the actual
    physical model, timing and env-declared I/O. Normalization is authenticated by
    artifact bytes, not independently reconstructed. For custom I/O/normalizers or
    motion references supply ``current`` from the actual configured deployment;
    otherwise recorded references cannot be verified and are rejected.
    """
    from myosuite.utils.checkpoint_manifest import read_manifest

    checkpoint = Path(checkpoint)
    sidecar = Path(str(checkpoint) + ".manifest.json")
    if sidecar.is_file():
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    else:
        folder_manifest = read_manifest(checkpoint.parent)
        manifest = (folder_manifest or {}).get("deployment")
    if manifest is None:
        return []
    missing = []
    if checkpoint.name not in manifest.get("artifacts", {}):
        message = f"Deployment provenance has no artifact hash for {checkpoint.name}"
        if not allow_mismatch:
            raise ValueError(message)
        warnings.warn(message + " (explicit override)", stacklevel=2)
        missing.append(f"missing artifact {checkpoint.name}")
    if current is None:
        current = env_deployment_manifest(
            env, normalization=manifest.get("normalization")
        )
    return missing + validate_deployment_manifest(
        manifest,
        current=current,
        artifact_root=checkpoint.parent,
        allow_mismatch=allow_mismatch,
    )
