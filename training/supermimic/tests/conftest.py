# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Synthetic clips on MyoSuite's full body (MuscleMimic's KIT clips are a gated dataset)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(scope="session")
def body() -> Any:
    from myosuite.envs.myo.backends.mjlab.tasks.mimic_multiclip.body import (
        compiled_body,
    )

    return compiled_body()


@pytest.fixture(scope="session")
def clips(tmp_path_factory: Any, body: Any) -> Path:
    """Three clips of 8, 7 and 6 s and a list of them; the last rises at 1 m/s, so envs reset onto it fail."""
    from myosuite.envs.myo.backends.mjlab.tasks.mimic_multiclip.motion import (
        write_synthetic_clip,
    )

    d = tmp_path_factory.mktemp("clips")
    for i, (sec, speed, climb) in enumerate(
        ((8.0, 0.0, 0.0), (7.0, 0.5, 0.0), (6.0, 1.0, 1.0))
    ):
        write_synthetic_clip(d / f"clip{i}.npz", body, sec, speed, 0.05, i, climb)
    (d / "clips.txt").write_text("clip0.npz\nclip1.npz\nclip2.npz\n")
    return d
