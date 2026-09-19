# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Download MuscleMimic-style demo motions from Hugging Face.

Does not use the upstream ``musclemimic`` package. Mirrors
``musclemimic.utils.demo_cache`` layout under the cache root resolved by
:func:`myosuite.core.hf_io.default_musclemimic_cache_root` so downstream tools
find the same paths.
Motion lists should stay aligned with upstream when that package changes.

Requires: ``huggingface_hub`` (``pip install huggingface_hub`` or
``pip install 'MyoSuite[musclemimic]'``).
"""

from __future__ import annotations

import logging
from pathlib import Path

from myosuite.core.hf_dataset_cache import setup_named_demo_cache
from myosuite.core.hf_io import default_musclemimic_cache_root

logger = logging.getLogger(__name__)

_BIMANUAL_ENV_NAME = "MyoBimanualArm"
_REPO_ID = "amathislab/demo_dataset"


def download_demo_cache(
    env_name: str = _BIMANUAL_ENV_NAME,
    motion_path: str = "KIT/3/tennis_forehand_right04_poses.npz",
    repo_id: str = _REPO_ID,
    cache_dir: str | None = None,
) -> Path | None:
    """Download one demo motion from Hugging Face into the local AMASS cache.

    Args:
        env_name: Logical env name (e.g. ``MyoBimanualArm``, ``MyoFullBody``).
        motion_path: Path inside the env folder on the HF dataset.
        repo_id: Hugging Face dataset id.
        cache_dir: Override cache root; default
            upstream-compatible cache root from
            :func:`myosuite.core.hf_io.default_musclemimic_cache_root`.

    Returns:
        Path to the copied file, or ``None`` if download failed.
    """
    if cache_dir is None:
        cache_base = default_musclemimic_cache_root()
    else:
        cache_base = Path(cache_dir)

    normalized_env_name = env_name.removeprefix("Mjx")
    local_path = cache_base / normalized_env_name / motion_path
    downloaded = setup_named_demo_cache(
        env_name=normalized_env_name,
        motions=[motion_path],
        repo_id=repo_id,
        cache_root=cache_base,
    )
    if downloaded:
        logger.info(
            "Downloaded demo motion cache: %s -> %s",
            motion_path,
            local_path,
        )
        return local_path
    logger.warning("You may need to download AMASS and retarget manually.")
    return None


def get_demo_motions() -> dict[str, list[str]]:
    """Demo motion paths per env (match upstream ``demo_cache``)."""
    return {
        _BIMANUAL_ENV_NAME: [
            ("gmr/BioMotionLab_NTroje/rub039/" "0022_throwing_hard1_poses.npz"),
            ("gmr/BioMotionLab_NTroje/rub109/" "0021_catching_and_throwing_poses.npz"),
            "gmr/KIT/3/tennis_forehand_right04_poses.npz",
            "gmr/KIT/3/wave_left09_poses.npz",
            "gmr/KIT/572/throw_left03_poses.npz",
            "gmr/s9/banana_peel_1_stageii.npz",
            "gmr/s9/doorknob_use_2_stageii.npz",
        ],
        "MyoFullBody": [
            "gmr/KIT/314/walking_medium09_poses.npz",
            "gmr/KIT/348/turn_right03_poses.npz",
            "gmr/KIT/4/WalkInCounterClockwiseCircle04_poses.npz",
        ],
    }


def setup_demo(env_name: str) -> list[str]:
    """Download all demo motions listed for *env_name*."""
    all_demos = get_demo_motions()
    env_name = env_name.removeprefix("Mjx")
    if env_name not in all_demos:
        raise ValueError(
            f"Unknown env '{env_name}'. Available: {list(all_demos.keys())}"
        )

    motions = all_demos[env_name]
    logger.info("Downloading %d %s demo motions...", len(motions), env_name)

    downloaded: list[str] = []
    for motion_path in motions:
        file = download_demo_cache(env_name, motion_path)
        if file:
            downloaded.append(motion_path)

    logger.info("Downloaded %d/%d:", len(downloaded), len(motions))
    for m in downloaded:
        logger.info("  %s", m)

    return downloaded


def setup_demo_for_bimanual() -> list[str]:
    """Download all bimanual demo motions."""
    return setup_demo(_BIMANUAL_ENV_NAME)


def setup_demo_for_myo_fullbody() -> list[str]:
    """Download all Myo full-body demo motions."""
    return setup_demo("MyoFullBody")


__all__ = [
    "download_demo_cache",
    "get_demo_motions",
    "setup_demo",
    "setup_demo_for_bimanual",
    "setup_demo_for_myo_fullbody",
]
