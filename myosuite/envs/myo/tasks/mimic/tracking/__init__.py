# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Multi-clip full-body motion tracking (``myoMimic-v0``).

The task (what is to be done) is separate from the body model and from the learning
method:

* task: track MuscleMimic's retargeted motion clips (KIT split: 972 training, 108
  held-out clips) with a full body (free root, legs, torso, arms; 17 tracked bodies
  and 4 foot bodies). Default observations from a shared vocabulary
  (:mod:`myosuite.terms.mimic_track`), a proposed dense reward (MimicReward), a
  failure rule (1 m relative sites / 1 m root / 1 rad root orientation) and success
  metrics that do not depend on the control step (:mod:`.metrics`);
* model: a body adapter (:mod:`.models`; myo_sim's ``myofullbody`` so far), chosen with
  ``EnvConfig.model``;
* method: presets (:mod:`.config`) select a method's MDP choices (``"supermimic"``,
  the default); methods live in ``myosuite.methods``.

Build it with ``make_env(MIMIC_ENV_ID)`` on the CPU or ``backend="mjlab"``; the control
step is ``EnvConfig.ctrl_dt`` (default 0.02 s).
"""

from __future__ import annotations

import os
from importlib.resources import files
from pathlib import Path

MIMIC_ENV_ID = "myoMimic-v0"
"""Id of the task (rename here only)."""
KIT_DATASET = "amathislab/musclemimic-retargeted"
KIT_CLIP_SETS = {"kit_train": "train", "kit_test": "test"}
KIT_ROOT_ENV = "MYOSUITE_KIT_ROOT"
"""Environment variable naming a local copy of the KIT dataset (skips the download)."""


def kit_split(split: str) -> Path:
    """The packaged clip list of MuscleMimic's KIT split.

    Args:
        split: ``"train"`` (972 clips) or ``"test"`` (108 held-out clips).

    Returns:
        The ``.txt`` list; its paths are relative to the dataset root.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', not {split!r}")
    return Path(str(files(__name__) / "data" / f"kit_kinesis_{split}.txt"))


def download_kit_clips(local_dir: str | None = None) -> Path:
    """Download MuscleMimic's retargeted KIT clips (a gated Hugging Face dataset:
    accept its terms and log in first). Downloaded files are reused.

    Args:
        local_dir: Download into this directory instead of the Hugging Face cache.

    Returns:
        The dataset root (the root of the packaged lists).
    """
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    return Path(
        snapshot_download(
            KIT_DATASET,
            repo_type="dataset",
            allow_patterns="MyoFullBody/gmr/KIT/*",
            local_dir=local_dir,
        )
    )


def resolve_clips(clips: str, root: str = "") -> tuple[str, str]:
    """``(path, root)`` of a :class:`~.config.MotionCfg` clip set.

    The KIT names map to the packaged lists; their root defaults to
    ``$MYOSUITE_KIT_ROOT``, else to the dataset in the Hugging Face cache (downloaded
    on first use).
    """
    if clips in KIT_CLIP_SETS:
        root = root or os.environ.get(KIT_ROOT_ENV, "") or str(download_kit_clips())
        return str(kit_split(KIT_CLIP_SETS[clips])), root
    return clips, root
