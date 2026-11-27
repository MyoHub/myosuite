# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Pinned Hugging Face Hub helpers for DreamerV3 checkpoints and benchmark logs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

from myosuite_dreamerv3.compat import import_myosuite

MOVING_BRANCH_NAMES = frozenset({"main", "master", "dev", "HEAD", "latest"})
DREAMERV3_HF_DIR_ENV = "MYOSUITE_DREAMERV3_HF_DIR"
DEFAULT_HF_REPO_ID = "maco018/myosuite-dreamerv3"
DEFAULT_HF_REVISION = "f1d6f8634f2e5fc1208cad9eafec39f67b0aa843"
PINNED_HF_REVISIONS: dict[str, str] = {
    DEFAULT_HF_REPO_ID: DEFAULT_HF_REVISION,
}


def _validate_pinned_revision(repo_id: str, revision: str | None) -> str:
    """Require an explicit immutable commit SHA or release tag (never a moving branch)."""
    effective = revision or PINNED_HF_REVISIONS.get(repo_id)
    if not effective or not effective.strip():
        raise ValueError(
            f"Hugging Face reference for {repo_id!r} must specify an explicit pinned "
            "commit SHA or tag (e.g. 'hf://owner/repo@v0.1.0/path' or revision='<sha>'). "
            "Unpinned downloads from 'main' are disallowed."
        )
    rev = effective.strip()
    if rev in MOVING_BRANCH_NAMES:
        raise ValueError(
            f"Hugging Face revision {rev!r} for {repo_id!r} is a moving branch; "
            "pin to a specific commit SHA or immutable tag instead."
        )
    return rev


def resolve_dreamerv3_checkpoint(
    path_or_hf_uri: str | Path,
    *,
    revision: str | None = None,
    repo_type: str = "model",
) -> Path:
    """Resolve a local path or pinned ``hf://owner/repo@<revision>/subpath`` to disk.

    Respects ``$MYOSUITE_DREAMERV3_HF_DIR`` when set (for offline testing prior to
    uploading a staged bundle to Hugging Face).

    Args:
        path_or_hf_uri: Local filesystem path or ``hf://owner/repo@<rev>[/subpath]``.
        revision: Optional explicit commit SHA or release tag overriding ``@<rev>``.
        repo_type: Hugging Face Hub repository type (``"model"`` or ``"dataset"``).

    Returns:
        Resolved local ``Path`` to the checkpoint file or directory.
    """
    raw = str(path_or_hf_uri)
    if not raw.startswith("hf://"):
        local_path = Path(path_or_hf_uri).expanduser().resolve()
        if not local_path.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {local_path}")
        return local_path

    import_myosuite()
    from myosuite.core.hf_io import parse_hf_ref

    ref = parse_hf_ref(raw)
    pinned_rev = _validate_pinned_revision(ref.repo_id, revision or ref.revision)

    local_override = os.environ.get(DREAMERV3_HF_DIR_ENV)
    if local_override:
        candidate = Path(local_override).expanduser().resolve()
        if ref.subpath:
            candidate = candidate / ref.subpath
        if candidate.exists():
            return candidate

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "Downloading hf:// checkpoints requires huggingface_hub "
            "(`pip install huggingface_hub`)."
        ) from exc

    allow = (
        [ref.subpath, f"{ref.subpath}.manifest.json", f"{ref.subpath}/*"]
        if ref.subpath
        else None
    )
    snapshot_root = Path(
        snapshot_download(
            repo_id=ref.repo_id,
            repo_type=repo_type,
            revision=pinned_rev,
            allow_patterns=allow,
        )
    ).resolve()
    resolved = snapshot_root / ref.subpath if ref.subpath else snapshot_root
    if not resolved.exists():
        raise FileNotFoundError(
            f"Subpath {ref.subpath!r} not found in {ref.repo_id}@{pinned_rev}."
        )
    return resolved


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_hf_release_bundle(
    staging_dir: str | Path,
    *,
    checkpoints: dict[str, str | Path] | None = None,
    eval_logs: dict[str, str | Path] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Stage exported policies, checkpoints, and evaluation logs in MyoSuite HF layout.

    Layout matches MyoSuite's ``checkpoints/<env_id>/`` + ``evals/`` baseline convention
    with SHA-256 provenance recorded in ``manifest.json``.

    Args:
        staging_dir: Output directory to populate.
        checkpoints: Mapping ``env_id -> path`` (file or directory of exported `.onnx`/`.pt`
            or ``agent.pkl``).
        eval_logs: Mapping ``name -> json_log_path`` copied under ``evals/``.
        metadata: Additional provenance metadata written to ``manifest.json``.

    Returns:
        Path to ``manifest.json`` inside ``staging_dir``.
    """
    root = Path(staging_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, Any]] = {}

    for env_id, src_spec in (checkpoints or {}).items():
        dst_dir = root / "checkpoints" / env_id
        dst_dir.mkdir(parents=True, exist_ok=True)
        items = (
            [src_spec] if isinstance(src_spec, (str, Path)) else list(src_spec)  # type: ignore[arg-type]
        )
        for src_raw in items:
            src = Path(src_raw).expanduser().resolve()
            if src.is_file():
                for file_cand in (src, Path(f"{src}.manifest.json")):
                    if file_cand.is_file():
                        dst = dst_dir / file_cand.name
                        shutil.copy2(file_cand, dst)
                        rel = dst.relative_to(root).as_posix()
                        records[rel] = {
                            "sha256": _sha256_file(dst),
                            "bytes": dst.stat().st_size,
                        }
            elif src.is_dir():
                for item in sorted(src.rglob("*")):
                    if (
                        item.is_file()
                        and item.name != "replay.pkl"
                        and "replay" not in item.parts
                    ):
                        rel_in_src = item.relative_to(src)
                        dst = dst_dir / rel_in_src
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(item, dst)
                        rel = dst.relative_to(root).as_posix()
                        records[rel] = {
                            "sha256": _sha256_file(dst),
                            "bytes": dst.stat().st_size,
                        }

    for log_name, log_raw in (eval_logs or {}).items():
        src = Path(log_raw).expanduser().resolve()
        dst_dir = root / "evals"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / (log_name if log_name.endswith(".json") else f"{log_name}.json")
        shutil.copy2(src, dst)
        rel = dst.relative_to(root).as_posix()
        records[rel] = {"sha256": _sha256_file(dst), "bytes": dst.stat().st_size}

    manifest = {
        "package": "myosuite-dreamerv3",
        "files": records,
        **(metadata or {}),
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def upload_hf_release_bundle(
    staging_dir: str | Path,
    repo_id: str,
    *,
    tag: str | None = None,
    private: bool = False,
    commit_message: str = "Upload MyoSuite DreamerV3 checkpoints and benchmark logs",
) -> str:
    """Upload a staged bundle to Hugging Face Hub and return the immutable commit SHA.

    Requires authentication via ``hf auth login`` or ``HF_TOKEN``.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            "Uploading to Hugging Face requires huggingface_hub (`pip install huggingface_hub`)."
        ) from exc

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    commit_info = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(Path(staging_dir).expanduser().resolve()),
        commit_message=commit_message,
    )
    commit_oid = str(commit_info.oid)
    if tag:
        api.create_tag(
            repo_id=repo_id,
            repo_type="model",
            tag=tag,
            revision=commit_oid,
            exist_ok=True,
        )
    return commit_oid
