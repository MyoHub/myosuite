"""Manifest of a published checkpoint and the contract hash that ties it to an env.

A policy only works on the observations, actions and control step it was trained with. The
*contract* records exactly those; its short hash is stored in ``manifest.json`` next to the
checkpoint and compared with the current env when a downloaded checkpoint is used.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import myosuite

if TYPE_CHECKING:
    from myosuite.core.subject_model import SubjectModel

MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 1


def _shape(space: Any) -> list[int] | dict[str, Any]:
    """Shape of a Box space, or the shapes of a Dict space's entries."""
    if hasattr(space, "spaces"):
        return {k: _shape(v) for k, v in space.spaces.items()}
    return list(space.shape)


def env_contract(env_id: str) -> dict[str, Any]:
    """Describe what a policy sees and does on the CPU env of *env_id*.

    Args:
        env_id: Registered env id.

    Returns:
        ``obs_shape``, ``action_shape``, ``ctrl_dt`` and, where the env has them, the
        observation ``obs_keys``.
    """
    from myosuite import make_env  # noqa: PLC0415

    env = make_env(env_id)
    try:
        unwrapped = env.unwrapped
        dt = next(
            (
                getattr(unwrapped, a)
                for a in ("dt", "_ctrl_dt")
                if hasattr(unwrapped, a)
            ),
            None,
        )
        contract: dict[str, Any] = {
            "obs_shape": _shape(env.observation_space),
            "action_shape": _shape(env.action_space),
            "ctrl_dt": None if dt is None else round(float(dt), 6),
        }
        if hasattr(unwrapped, "obs_keys"):
            contract["obs_keys"] = list(unwrapped.obs_keys)
        return contract
    finally:
        env.close()


def contract_hash(contract: dict[str, Any]) -> str:
    """Return a 12-character hash of a contract (stable across runs and platforms)."""
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _git_commit() -> str | None:
    """Commit of the MyoSuite checkout, or ``None`` outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(myosuite.__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def write_manifest(
    checkpoint_dir: str | Path,
    env_id: str,
    *,
    contract: dict[str, Any] | None = None,
    success: float | None = None,
    steps: int | None = None,
    note: str | None = None,
    subject_model: SubjectModel | None = None,
) -> Path:
    """Write ``manifest.json`` into a checkpoint folder.

    Args:
        checkpoint_dir: Folder holding the ``model_*.pt`` files.
        env_id: Registered env id the policy was trained on.
        contract: Contract to record; defaults to the current ``env_contract(env_id)``.
        success: Deterministic success rate in percent, if measured.
        steps: Environment steps the policy was trained for, if known.
        note: Free text (for example how success is scored).
        subject_model: Built subject whose morphology provenance is recorded.

    Returns:
        Path of the written manifest. For envs whose body is built from ``musclemimic_models``
        it also records that release (``musclemimic_models_version``): its physics differ between
        releases while the contract does not, and loaders build the recorded one.
    """
    checkpoint_dir = Path(checkpoint_dir)
    contract = env_contract(env_id) if contract is None else contract
    manifest = {
        "schema": SCHEMA_VERSION,
        "env_id": env_id,
        "contract": contract,
        "contract_hash": contract_hash(contract),
        "myosuite_version": myosuite.__version__,
        "myosuite_commit": _git_commit(),
        "created": datetime.now(timezone.utc).date().isoformat(),
        "success_percent": success,
        "train_steps": steps,
        "files": sorted(p.name for p in checkpoint_dir.glob("model_*.pt")),
        "note": note,
    }
    from myosuite.integrations.musclemimic.model_versions import (
        MANIFEST_FIELD,
        env_models_version,
    )  # noqa: PLC0415

    models_version = env_models_version(env_id)
    if models_version is not None:
        manifest[MANIFEST_FIELD] = models_version
    if subject_model is not None:
        manifest["subject_model"] = subject_model.metadata
    path = checkpoint_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def read_manifest(checkpoint_dir: str | Path) -> dict[str, Any] | None:
    """Return the manifest of a checkpoint folder, or ``None`` if it has none."""
    path = Path(checkpoint_dir) / MANIFEST_NAME
    return json.loads(path.read_text()) if path.is_file() else None


def check_contract(
    env_id: str,
    checkpoint_dir: str | Path,
    *,
    subject_model: SubjectModel | None = None,
    strict: bool = False,
) -> bool:
    """Warn when a checkpoint's recorded contract differs from the current env.

    Args:
        env_id: Registered env id the checkpoint is about to be used on.
        checkpoint_dir: Folder with the checkpoint and its manifest.
        subject_model: Current subject, or None for nominal morphology.
        strict: Raise ValueError instead of warning on a morphology mismatch.

    Returns:
        ``False`` for a morphology or contract mismatch. With nominal morphology,
        a missing manifest/contract retains the legacy ``True`` result.
    """
    manifest = read_manifest(checkpoint_dir)
    recorded = None if manifest is None else manifest.get("subject_model")
    expected = None if subject_model is None else subject_model.metadata
    if (None if recorded is None else recorded.get("model_digest")) != (
        None if expected is None else expected.get("model_digest")
    ):
        message = (
            f"The checkpoint for {env_id} has incompatible or unrecorded subject "
            "morphology. Nominal-policy results do not establish subject-specific "
            "validity; retrain or explicitly validate the policy on this model."
        )
        if strict:
            raise ValueError(message)
        warnings.warn(message, stacklevel=2)
        return False
    if manifest is None or not manifest.get("contract"):
        return True
    current = contract_hash(env_contract(env_id))
    if current == manifest["contract_hash"]:
        return True
    warnings.warn(
        f"The checkpoint for {env_id} was trained on contract {manifest['contract_hash']} "
        f"(MyoSuite {manifest.get('myosuite_version')}), but the env now has {current}: "
        "observations, actions or control step changed, so the policy may not work. "
        "Retrain it or use the checkpoint release that matches your MyoSuite version.",
        stacklevel=2,
    )
    return False
