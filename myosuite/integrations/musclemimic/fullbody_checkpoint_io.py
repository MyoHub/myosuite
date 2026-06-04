# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Checkpoint path resolution for native fullbody playback.

This module resolves local and ``hf://`` checkpoint references without relying
on upstream ``musclemimic`` imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from myosuite.core.hf_io import resolve_hf_snapshot


@dataclass(frozen=True)
class CheckpointRef:
    """Resolved checkpoint reference.

    Attributes:
        source: Original user argument.
        local_path: Resolved local directory for the checkpoint artifact.
    """

    source: str
    local_path: Path


def resolve_checkpoint_ref(path: str) -> CheckpointRef:
    """Resolve a checkpoint path to a local directory.

    Supports:
    - local directories (absolute or relative)
    - Hugging Face snapshot references, e.g. ``hf://owner/repo[/subpath]``
    """
    source = str(path)
    if source.startswith("hf://"):
        local = resolve_hf_snapshot(source, repo_type="model")
        return CheckpointRef(source=source, local_path=local)

    local = Path(source).expanduser().resolve()
    if not local.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {local}")
    return CheckpointRef(source=source, local_path=local)


def describe_checkpoint_dir(path: Path) -> dict[str, bool]:
    """Return basic checkpoint directory features for diagnostics."""
    p = Path(path)
    return {
        "exists": p.exists(),
        "is_dir": p.is_dir(),
        "has_orbax_metadata": (p / "_METADATA").exists(),
        "has_params_file": (p / "params").exists(),
        "has_config_yaml": (p / "config.yaml").exists(),
    }


__all__ = [
    "CheckpointRef",
    "describe_checkpoint_dir",
    "resolve_checkpoint_ref",
]
