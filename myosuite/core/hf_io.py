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
    "HfRef",
    "default_musclemimic_cache_root",
    "parse_hf_ref",
    "resolve_hf_snapshot",
]
