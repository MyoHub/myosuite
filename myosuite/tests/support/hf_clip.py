# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The default MuscleMimic walking clip, for tests that need one."""

from __future__ import annotations

import pathlib

CLIP_REPO_ID = "amathislab/musclemimic-retargeted"
CLIP_FILENAME = "MyoFullBody/gmr/KIT/167/walking_medium06_poses.npz"


def resolve_walk_clip() -> pathlib.Path | None:
    """Download (or reuse the cached) walking clip; ``None`` when unavailable.

    Called at import time to build a ``skipif``, so any failure (no
    ``huggingface_hub``, no network, a proxy refusing the host) means "skip",
    never an error that aborts collection of the whole suite.
    """
    try:
        from huggingface_hub import hf_hub_download

        path = pathlib.Path(
            hf_hub_download(
                repo_id=CLIP_REPO_ID, filename=CLIP_FILENAME, repo_type="dataset"
            )
        )
    except Exception:  # noqa: BLE001 -- any download failure means the clip is unavailable
        return None
    return path if path.is_file() else None
