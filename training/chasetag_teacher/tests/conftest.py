# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Fixtures: synthetic clips (no AMASS data), and the real clips / tracker checkpoint when they are available."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

CHECKPOINT_ENV = "CHASETAG_TEACHER_CHECKPOINT"
"""Environment variable naming a myoSuperMimic tracker checkpoint (``model_*.pt``) for the data tests."""


@pytest.fixture(scope="session", autouse=True)
def _library_cache(tmp_path_factory: Any) -> None:
    """Libraries built by the tests go to a temporary folder, not to the user's cache."""
    os.environ["CHASETAG_TEACHER_CACHE"] = str(tmp_path_factory.mktemp("library_cache"))


@pytest.fixture(scope="session", name="body")
def body_fixture() -> Any:
    from myosuite.envs.myo.tasks.mimic.tracking.models import compiled_body

    return compiled_body("myofullbody")


@pytest.fixture(scope="session", name="clips")
def clips_fixture(tmp_path_factory: Any, body: Any) -> Path:
    """Two walking clips (8 s, 6 s) and a list of them."""
    from myosuite.envs.myo.tasks.mimic.tracking.motion import write_synthetic_clip

    d = tmp_path_factory.mktemp("clips")
    for i, (sec, speed) in enumerate(((8.0, 1.0), (6.0, 1.2))):
        write_synthetic_clip(d / f"clip{i}.npz", body, sec, speed, 0.05, i)
    (d / "clips.txt").write_text("clip0.npz\nclip1.npz\n")
    return d


@pytest.fixture(name="fbvs_obs")
def fbvs_obs_fixture() -> Any:
    """64 random FBVs observations (float64) with unit root quaternions."""
    import torch

    rng = np.random.default_rng(0)
    obs = rng.normal(size=(64, 546))
    q = rng.normal(size=(64, 4))
    obs[:, 3:7] = q / np.linalg.norm(q, axis=1, keepdims=True)
    return torch.as_tensor(obs)


@pytest.fixture(scope="session", name="kit_root")
def kit_root_fixture() -> str:
    from myosuite.envs.myo.tasks.mimic.tracking import KIT_ROOT_ENV

    root = os.environ.get(KIT_ROOT_ENV, "")
    if not root or not Path(root).is_dir():
        pytest.skip(
            f"needs retargeted KIT clips (${KIT_ROOT_ENV}, see scripts/mimic/retarget_amass.py)"
        )
    return root


@pytest.fixture(scope="session", name="tracker_checkpoint")
def tracker_checkpoint_fixture() -> str:
    path = os.environ.get(CHECKPOINT_ENV, "")
    if not path or not Path(path).is_file():
        pytest.skip(f"needs a myoSuperMimic tracker checkpoint (${CHECKPOINT_ENV})")
    return path
