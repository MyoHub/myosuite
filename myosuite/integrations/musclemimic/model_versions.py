# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Select the ``musclemimic_models`` release a MuscleMimic model is built as.

``musclemimic_models`` 1.0.6 fixed the left-knee coupling (two equality
polynomials had the wrong sign) and moved six muscle-wrap geoms. Checkpoints
trained on 1.0.5, such as ``amathislab/mm-10m-2``, fail on the fixed model. The
two releases differ only in the 13 attribute values below, so either model is
rebuilt bit-exactly from the other by setting them on the ``MjSpec``.

The version is ``config.model_version`` if set, else the environment variable
``MYOSUITE_MUSCLEMIMIC_MODELS_VERSION`` (for envs that build their config
internally), else :data:`DEFAULT_MODELS_VERSION`. It applies to the packaged
MJCF only; an explicit ``config.model_path`` is built as it is.
"""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version as _dist_version
import os
from pathlib import Path
import re
from typing import Any

import mujoco

DEFAULT_MODELS_VERSION = "1.0.6"
MODELS_VERSION_ENV_VAR = "MYOSUITE_MUSCLEMIMIC_MODELS_VERSION"

# The release each published checkpoint was trained on; unlisted ones get the
# default. MyoSuite's myoMimicFullbody-v0 walking_medium06 baseline was trained on
# 1.0.6. List new checkpoints too, so they keep their release if the default moves.
# TODO(#mm-10m-3): amathislab/mm-10m-3 does not exist yet; check the repo id once it is published.
PUBLISHED_CHECKPOINT_MODELS_VERSIONS: dict[str, str] = {
    "amathislab/mm-10m-2": "1.0.5",
    "amathislab/mm-10m-3": "1.0.6",
}

_EQ_POLYCOEF = "eq_polycoef"
_JOINT_RANGE = "joint_range"
_GEOM_POS = "geom_pos"
_GEOM_SIZE0 = "geom_size0"

# Every attribute value that differs between the releases (compare/v1.0.5...v1.0.6).
_MODELS_VALUES: dict[str, dict[str, dict[str, Any]]] = {
    "1.0.5": {
        _EQ_POLYCOEF: {
            "knee_angle_translation2_constraint_l": (
                -7.69254e-11,
                -0.00587971,
                0.00125622,
                2.61846e-06,
                -6.24355e-07,
            ),
            "knee_angle_rotation3_constraint_l": (
                -1.08939e-08,
                -0.369499,
                0.169478,
                -0.0251643,
                -3.50498e-07,
            ),
        },
        _JOINT_RANGE: {
            "knee_angle_rotation3_r": (-0.262788, 0.262788),
            "knee_angle_rotation3_l": (-0.262788, 0.262788),
        },
        _GEOM_POS: {
            "DELT1hh_ellipsoid_DELT1": (-0.0179974, -0.0075035, -0.0580664),
            "DELT1hh_ellipsoid_DELT1_left": (-0.0179974, -0.0075035, 0.0580664),
            "EDCL_torus_wrap": (-0.0128, -0.0133, 0.0148),
            "EDCL_torus_wrap_left": (-0.0128, -0.0133, -0.0148),
            "EDCM_torus_wrap": (-0.0007, -0.0117, 0.0191),
            "EDCM_torus_wrap_left": (-0.0007, -0.0117, -0.0191),
        },
        _GEOM_SIZE0: {"back_cylinder_l": 0.17},
    },
    "1.0.6": {
        _EQ_POLYCOEF: {
            "knee_angle_translation2_constraint_l": (
                7.69254e-11,
                0.00587971,
                -0.00125622,
                -2.61846e-06,
                6.24355e-07,
            ),
            "knee_angle_rotation3_constraint_l": (
                1.08939e-08,
                0.369499,
                -0.169478,
                0.0251643,
                3.50498e-07,
            ),
        },
        _JOINT_RANGE: {
            "knee_angle_rotation3_r": (-0.262788, 0.263),
            "knee_angle_rotation3_l": (-0.262788, 0.263),
        },
        _GEOM_POS: {
            "DELT1hh_ellipsoid_DELT1": (-0.0279974, -0.0075035, -0.0080664),
            "DELT1hh_ellipsoid_DELT1_left": (-0.0279974, -0.0075035, 0.0080664),
            "EDCL_torus_wrap": (-0.0098, -0.0133, 0.0148),
            "EDCL_torus_wrap_left": (-0.0098, -0.0133, -0.0148),
            "EDCM_torus_wrap": (-0.0097, -0.0117, 0.0191),
            "EDCM_torus_wrap_left": (-0.0097, -0.0117, -0.0191),
        },
        _GEOM_SIZE0: {"back_cylinder_l": 0.166},
    },
}

SUPPORTED_MODELS_VERSIONS = tuple(_MODELS_VALUES)

# Env ids whose body is built from the musclemimic_models MJCF (MuscleMimic full body and bimanual,
# directional locomotion, full-body ChaseTag); other envs use MyoSuite's own models.
_MODELS_ENV_ID = re.compile(r"Mimic|FullBodyDirectional|ChaseTagFB")

# Field of a checkpoint folder's manifest.json that records the release (see env_models_version).
MANIFEST_FIELD = "musclemimic_models_version"


def installed_models_version() -> str | None:
    """Return the installed ``musclemimic_models`` version, or ``None`` if absent."""
    try:
        return _dist_version("musclemimic-models")
    except PackageNotFoundError:
        return None


def resolve_models_version(config: Any = None) -> str:
    """Return the ``musclemimic_models`` release a model is built as.

    Args:
        config: Optional config with a ``model_version`` entry.

    Returns:
        ``config.model_version``, else ``$MYOSUITE_MUSCLEMIMIC_MODELS_VERSION``,
        else :data:`DEFAULT_MODELS_VERSION`.

    Raises:
        ValueError: If the version is not one of :data:`SUPPORTED_MODELS_VERSIONS`.
    """
    requested = (
        getattr(config, "model_version", None)
        or os.environ.get(MODELS_VERSION_ENV_VAR)
        or DEFAULT_MODELS_VERSION
    )
    if requested not in _MODELS_VALUES:
        raise ValueError(
            f"musclemimic_models version {requested!r} is not supported; "
            f"choose one of {SUPPORTED_MODELS_VERSIONS}."
        )
    return requested


def uses_musclemimic_models(env_id: str) -> bool:
    """Whether the env builds its body from the ``musclemimic_models`` MJCF."""
    return bool(_MODELS_ENV_ID.search(env_id))


def env_models_version(env_id: str) -> str | None:
    """The ``musclemimic_models`` release an env builds its body as, for its checkpoint manifest.

    Args:
        env_id: Registered env id.

    Returns:
        :func:`resolve_models_version` for envs built from ``musclemimic_models``, ``None`` for the
        others.

    Raises:
        RuntimeError: If the env uses ``musclemimic_models`` but the package is not installed: the
            env would then build MyoSuite's approximate ``myo_sim`` composition instead.
    """
    if not uses_musclemimic_models(env_id):
        return None
    if installed_models_version() is None:
        raise RuntimeError(
            f"{env_id} builds its body from musclemimic_models, which is not installed; install "
            "'myosuite[musclemimic]' so the recorded release is the one the env uses."
        )
    return resolve_models_version()


def _manifest_models_version(ref: str) -> str | None:
    """The release recorded in the manifest.json of a local checkpoint file or folder, if any."""
    path = Path(ref)
    if not ref or not path.exists():
        return None
    manifest = (path if path.is_dir() else path.parent) / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        recorded = json.loads(manifest.read_text()).get(MANIFEST_FIELD)
    except (OSError, ValueError):
        return None
    return recorded if recorded in _MODELS_VALUES else None


def checkpoint_models_version(checkpoint_ref: str | Path | None) -> str:
    """Return the ``musclemimic_models`` release a checkpoint was trained on.

    Args:
        checkpoint_ref: ``hf://owner/repo``, a Hugging Face snapshot path, or any
            local path; ``None`` for no checkpoint.

    Returns:
        The release of a published checkpoint in
        :data:`PUBLISHED_CHECKPOINT_MODELS_VERSIONS`, else the release recorded in the
        ``manifest.json`` next to a local checkpoint, else :func:`resolve_models_version`.
    """
    ref = str(checkpoint_ref or "").replace("\\", "/")
    for repo_id, models_version in PUBLISHED_CHECKPOINT_MODELS_VERSIONS.items():
        # The repo id as whole path components; snapshots spell it models--owner--repo.
        names = (re.escape(repo_id), re.escape("models--" + repo_id.replace("/", "--")))
        if re.search(rf"(^|[/:])({'|'.join(names)})($|[/@])", ref):
            return models_version
    return (
        _manifest_models_version(str(checkpoint_ref or "")) or resolve_models_version()
    )


def apply_models_version(spec: mujoco.MjSpec, models_version: str) -> int:
    """Edit a spec built from the installed ``musclemimic_models`` into *models_version*.

    Elements the spec lacks (e.g. legs in the bimanual model) are skipped.

    Args:
        spec: Spec loaded from the installed ``musclemimic_models`` MJCF.
        models_version: One of :data:`SUPPORTED_MODELS_VERSIONS`.

    Returns:
        Number of edited elements (0 when *models_version* is installed).

    Raises:
        RuntimeError: If the installed release is not a supported one, so the
            requested model cannot be rebuilt from it.
    """
    installed = installed_models_version()
    if models_version == installed:
        return 0
    if installed not in _MODELS_VALUES:
        raise RuntimeError(
            f"Cannot build musclemimic_models {models_version} from the installed "
            f"{installed}: only {SUPPORTED_MODELS_VERSIONS} are known. Install "
            f"musclemimic_models=={DEFAULT_MODELS_VERSION}."
        )
    return _set_models_values(spec, models_version)


def _set_models_values(spec: mujoco.MjSpec, models_version: str) -> int:
    """Set the recorded values of *models_version* on the elements *spec* has."""
    values = _MODELS_VALUES[models_version]
    edited = 0
    for name, coef in values[_EQ_POLYCOEF].items():
        if (eq := spec.equality(name)) is not None:
            eq.data[: len(coef)] = coef
            edited += 1
    for name, limits in values[_JOINT_RANGE].items():
        if (joint := spec.joint(name)) is not None:
            joint.range = limits
            edited += 1
    for name, pos in values[_GEOM_POS].items():
        if (geom := spec.geom(name)) is not None:
            geom.pos = pos
            edited += 1
    for name, size0 in values[_GEOM_SIZE0].items():
        if (geom := spec.geom(name)) is not None:
            geom.size[0] = size0
            edited += 1
    return edited


__all__ = [
    "DEFAULT_MODELS_VERSION",
    "MANIFEST_FIELD",
    "MODELS_VERSION_ENV_VAR",
    "PUBLISHED_CHECKPOINT_MODELS_VERSIONS",
    "SUPPORTED_MODELS_VERSIONS",
    "apply_models_version",
    "checkpoint_models_version",
    "env_models_version",
    "installed_models_version",
    "resolve_models_version",
    "uses_musclemimic_models",
]
