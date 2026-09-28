# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Common Hugging Face IO helpers for MyoSuite."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_MUSCLEMIMIC_CACHE_ENV_VARS = (
    "MUSCLEMIMIC_CONVERTED_AMASS_PATH",
    "CONVERTED_AMASS_PATH",
)


@dataclass(frozen=True)
class HfRef:
    """Parsed Hugging Face reference."""

    repo_id: str
    subpath: str


def parse_hf_ref(path: str) -> HfRef:
    """Parse ``hf://owner/repo[/subpath]``.

    Args:
        path: Hugging Face-style path.

    Returns:
        Parsed repository id and optional subpath.
    """
    if not path.startswith("hf://"):
        raise ValueError(f"Not an hf path: {path}")
    raw = path[len("hf://") :].strip("/")
    if not raw:
        raise ValueError("hf:// path is empty")
    parts = raw.split("/")
    if len(parts) < 2:
        raise ValueError(
            "hf:// path must include owner/repo, e.g. " "hf://amathislab/mm-10m-2"
        )
    repo_id = "/".join(parts[:2])
    subpath = "/".join(parts[2:])
    return HfRef(repo_id=repo_id, subpath=subpath)


def resolve_hf_snapshot(
    hf_path: str,
    repo_type: str = "model",
) -> Path:
    """Resolve a Hugging Face path to a local snapshot path."""
    ref = parse_hf_ref(hf_path)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as err:
        raise ImportError(
            "Hugging Face path requires huggingface_hub. Install with: "
            "pip install huggingface_hub or "
            "pip install 'MyoSuite[musclemimic]'."
        ) from err
    snapshot_dir = Path(
        snapshot_download(
            repo_id=ref.repo_id,
            repo_type=repo_type,
        )
    ).resolve()
    local = snapshot_dir / ref.subpath if ref.subpath else snapshot_dir
    if not local.exists():
        raise FileNotFoundError(f"HF subpath not found in snapshot: {local}")
    return local


BASELINES_REPO_ID = "myohub/myosuite-3-baselines"
"""Hugging Face repo of the default-run mjlab checkpoints and eval videos, for envs whose
deterministic success reached at least 25% (see that repo's ``checkpoints/README.md``)."""


def download_baseline_checkpoint(
    env_id: str, repo_id: str = BASELINES_REPO_ID
) -> Path | None:
    """Download env_id's default checkpoint from the baselines Hugging Face repo.

    A thin wrapper the tutorials and :func:`myosuite.utils.checkpoint_utils.find_checkpoint`
    fall back to when no local checkpoint exists (e.g. a fresh clone, or a pip install with
    ``baselines/`` gitignored). huggingface_hub caches downloads locally, so repeated calls
    for the same env are free after the first.

    Args:
        env_id: Registered env id.
        repo_id: Hugging Face repo id to download from.

    Returns:
        The local ``checkpoints/<env_id>`` directory, or ``None`` when huggingface_hub is
        missing, there is no network, or the env has no checkpoint there (e.g. it never
        reached 25% deterministic success).
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return None
    try:
        snapshot_dir = Path(
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                allow_patterns=[f"checkpoints/{env_id}/*"],
            )
        )
    except Exception as err:  # noqa: BLE001  (network / auth / repo errors all mean "no baseline")
        print(f"Could not reach the {repo_id!r} Hugging Face repo: {err}")
        return None
    local = snapshot_dir / "checkpoints" / env_id
    if not local.is_dir() or not any(local.glob("model_*.pt")):
        return None
    return local


def default_musclemimic_cache_root() -> Path:
    """Return cache root used by MuscleMimic-compatible assets.

    Resolution order mirrors upstream MuscleMimic cache discovery:

    1. ``MUSCLEMIMIC_CONVERTED_AMASS_PATH``
    2. ``CONVERTED_AMASS_PATH``
    3. legacy ``~/.musclemimic/caches/AMASS``
    """
    for env_var in _MUSCLEMIMIC_CACHE_ENV_VARS:
        configured = os.environ.get(env_var)
        if configured:
            return Path(configured).expanduser()
    return Path.home() / ".musclemimic" / "caches" / "AMASS"


__all__ = [
    "BASELINES_REPO_ID",
    "HfRef",
    "default_musclemimic_cache_root",
    "download_baseline_checkpoint",
    "parse_hf_ref",
    "resolve_hf_snapshot",
]
