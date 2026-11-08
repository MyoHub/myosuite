# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Multi-clip full-body tracking: MuscleMimic's tracking MDP on mjlab (``myoMimicFullbodyMulticlip-v0``).

MuscleMimic's multi-clip motion tracking (github.com/amathislab/musclemimic) on myo_sim's ``myofullbody`` (fingers
disabled, 31 independent joint DoF, 354 muscles): its observation, goal, MimicReward and deviation terminations,
over a clip list such as its KIT split (972 training and 108 held-out clips, packaged here; :func:`kit_split`,
:func:`download_kit_clips`). Every observation, reward and termination term is a function of the body state through
the differentiable torch kinematics, muscle geometry and activation dynamics of this package (:mod:`.heads`), so the
same terms also score states that are not simulated (a learned model's predictions). mjlab only, no CPU twin; a
different MDP from ``myoMimicFullbody-v0``, which is unchanged. Build it with ``make_env(..., backend="mjlab")``
and options (see :func:`.env_cfg.mimic_multiclip_env_cfg`).

Ported from muscle-supermimic (github.com/kaibohex/muscle-supermimic, Apache-2.0, commit
8a6244b0178eb2cf1b5a14a903d82ebba867b792), which includes parts of BlinkMimic / InstantTrack (MIT) and of
supermimic_vibed (MIT): see ``NOTICE-muscle-supermimic.txt`` in this package. Every ported file names its source.
:func:`register` is the package's one registration entry point.
"""

from importlib.resources import files
from pathlib import Path

ENV_ID = "myoMimicFullbodyMulticlip-v0"
"""Provisional id of the task."""
UPSTREAM_COMMIT = "8a6244b0178eb2cf1b5a14a903d82ebba867b792"
KIT_DATASET = "amathislab/musclemimic-retargeted"


def kit_split(split: str) -> Path:
    """The packaged clip list of MuscleMimic's KIT split.

    Args:
        split: ``"train"`` (972 clips) or ``"test"`` (108 held-out clips).

    Returns:
        The ``.txt`` list; its paths are relative to the dataset's root (pass it as ``motion_root``).
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', not {split!r}")
    return Path(str(files(__name__) / "data" / f"kit_kinesis_{split}.txt"))


def download_kit_clips(local_dir: str | None = None) -> Path:
    """Download MuscleMimic's retargeted KIT clips (a gated Hugging Face dataset: accept its terms
    and log in first). Already downloaded files are reused (the Hugging Face cache, or ``local_dir``).

    Args:
        local_dir: Download into this directory instead of the Hugging Face cache.

    Returns:
        The dataset root, the ``motion_root`` of the packaged lists.
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


def register() -> None:
    """Register the task with mjlab's registry and ``make_env`` (see :func:`.env_cfg.register`)."""
    from .env_cfg import register as register_task  # noqa: PLC0415

    register_task()
