"""Manifest of a published checkpoint and the contract hash that ties it to an env.

A policy only works on the observations, actions and control step it was trained with. The
*contract* records exactly those; its short hash is stored in ``manifest.json`` next to the
checkpoint and compared with the current env when a downloaded checkpoint is used.

A policy trained through an action map (:mod:`myosuite.utils.action_map`) does not command one
action per muscle. Its checkpoint records the map under :data:`ACTION_MAP_FIELD` (in the
checkpoint's ``infos`` and, if written, in ``manifest.json``); :func:`action_map_kind` finds
it, so plain loaders can refuse such checkpoints without importing any method.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

import myosuite

MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 1
# Action map a checkpoint needs: ``{"manifest": {"kind": ..., ...}, "env_steps": ...}``.
ACTION_MAP_FIELD = "action_map"
# Checkpoints written before ACTION_MAP_FIELD: ``infos`` key -> kind.
_LEGACY_INFOS_KEYS = {"dynsyn": "dynsyn"}
# Earlier DynSyn-only action term in ``params/env.yaml``: its class and a key only it has.
_LEGACY_ENV_CLASS, _LEGACY_ENV_KEY = "DynSynActionCfg", "start_steps"


class CfgLoader(yaml.SafeLoader):
    """SafeLoader that reads ``!!python/*`` tags (e.g. tuples) as plain data."""


def _construct_plain(loader: yaml.SafeLoader, _suffix: str, node: yaml.Node) -> Any:
    if isinstance(node, yaml.SequenceNode):
        return tuple(loader.construct_sequence(node, deep=True))
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    return loader.construct_scalar(node)


CfgLoader.add_multi_constructor("tag:yaml.org,2002:python/", _construct_plain)


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
    action_map: dict[str, Any] | None = None,
) -> Path:
    """Write ``manifest.json`` into a checkpoint folder.

    Args:
        checkpoint_dir: Folder holding the ``model_*.pt`` files.
        env_id: Registered env id the policy was trained on.
        contract: Contract to record; defaults to the current ``env_contract(env_id)``.
        success: Deterministic success rate in percent, if measured.
        steps: Environment steps the policy was trained for, if known.
        note: Free text (for example how success is scored).
        action_map: Manifest of the action map the policy acts through, if any.

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
    if action_map is not None:
        manifest[ACTION_MAP_FIELD] = action_map
    from myosuite.integrations.musclemimic.model_versions import (
        MANIFEST_FIELD,
        env_models_version,
    )  # noqa: PLC0415

    models_version = env_models_version(env_id)
    if models_version is not None:
        manifest[MANIFEST_FIELD] = models_version
    path = checkpoint_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def read_manifest(checkpoint_dir: str | Path) -> dict[str, Any] | None:
    """Return the manifest of a checkpoint folder, or ``None`` if it has none."""
    path = Path(checkpoint_dir) / MANIFEST_NAME
    return json.loads(path.read_text()) if path.is_file() else None


def check_contract(env_id: str, checkpoint_dir: str | Path) -> bool:
    """Warn when a checkpoint's recorded contract differs from the current env.

    Args:
        env_id: Registered env id the checkpoint is about to be used on.
        checkpoint_dir: Folder with the checkpoint and its manifest.

    Returns:
        ``False`` when the contract hashes differ, ``True`` otherwise (also when the folder
        has no manifest, or the manifest records no contract, as for the MuscleMimic policy).
    """
    manifest = read_manifest(checkpoint_dir)
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


def env_yaml_action_term(env: dict[str, Any]) -> dict[str, Any] | None:
    """The action-map term of a run's ``params/env.yaml`` (read with :class:`CfgLoader`).

    Returns:
        The term's fields (``action_map``, ``step_offset``, ...; the earlier DynSyn-only
        term: ``groups``, ``start_steps``, ...), or ``None`` if the run used no map.
    """
    for term in (env.get("actions") or {}).values():
        if isinstance(term, dict) and (
            ACTION_MAP_FIELD in term or {_LEGACY_ENV_KEY, "groups"} <= term.keys()
        ):
            return term
    return None


def action_map_kind(infos: dict[str, Any], checkpoint: str | Path) -> str | None:
    """Kind of the action map a checkpoint was trained through, if any.

    Args:
        infos: The checkpoint's ``infos`` (``torch.load(checkpoint)["infos"]``).
        checkpoint: ``model_*.pt`` path; its folder's ``manifest.json`` and the run's
            ``params/env.yaml`` are checked too.

    Returns:
        The kind (e.g. ``"dynsyn"``), or ``None`` for one action per muscle.
    """
    if ACTION_MAP_FIELD in infos:
        return infos[ACTION_MAP_FIELD]["manifest"]["kind"]
    for key, kind in _LEGACY_INFOS_KEYS.items():
        if key in infos:
            return kind
    folder = Path(checkpoint).parent
    manifest = read_manifest(folder)
    if manifest is not None and ACTION_MAP_FIELD in manifest:
        return manifest[ACTION_MAP_FIELD]["kind"]
    env_yaml = folder / "params" / "env.yaml"
    if not env_yaml.is_file():
        return None
    text = env_yaml.read_text(errors="replace")
    if ACTION_MAP_FIELD not in text and _LEGACY_ENV_CLASS not in text:
        return None  # skip parsing the env.yaml of plain runs
    term = env_yaml_action_term(yaml.load(text, Loader=CfgLoader) or {})
    if term is None:
        return None
    return term[ACTION_MAP_FIELD]["kind"] if ACTION_MAP_FIELD in term else "dynsyn"
