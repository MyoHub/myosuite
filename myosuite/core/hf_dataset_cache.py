# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Generic Hugging Face dataset cache helpers."""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def download_hf_dataset_file(
    *,
    repo_id: str,
    remote_path: str,
    local_path: Path,
) -> Path | None:
    """Download a dataset file, publishing it atomically after a complete copy."""
    from huggingface_hub import hf_hub_download

    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=remote_path,
            repo_type="dataset",
        )
        with tempfile.TemporaryDirectory(
            dir=local_path.parent, prefix=".myosuite-download-"
        ) as directory:
            staged = Path(directory) / local_path.name
            shutil.copy2(downloaded, staged)
            staged.replace(local_path)
        return local_path
    except Exception as err:
        logger.warning(
            "Could not download %s from %s: %s",
            remote_path,
            repo_id,
            err,
        )
        return None


def setup_named_demo_cache(
    *,
    env_name: str,
    motions: list[str],
    repo_id: str,
    cache_root: Path,
) -> list[str]:
    """Download a list of motion files into a local cache layout."""
    downloaded: list[str] = []
    for motion_path in motions:
        remote = f"{env_name}/{motion_path}"
        local = cache_root / env_name / motion_path
        file_path = download_hf_dataset_file(
            repo_id=repo_id,
            remote_path=remote,
            local_path=local,
        )
        if file_path is not None:
            downloaded.append(motion_path)
    return downloaded


__all__ = ["download_hf_dataset_file", "setup_named_demo_cache"]
