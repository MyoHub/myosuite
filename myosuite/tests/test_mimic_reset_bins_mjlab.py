# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Normal and graph-safe mjlab reset draws obey the shared transition bins."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from myosuite.tests.test_mimic_reset_bins import _library

pytestmark = pytest.mark.tier2


@pytest.mark.parametrize("method", ["sample_frames", "sample_frames_masked"])
def test_mjlab_draws_have_a_successor(tmp_path: Path, method: str) -> None:
    """Execute the actual tensor samplers on CPU, with all bins equiprobable."""
    pytest.importorskip("mjlab")
    torch = pytest.importorskip("torch")
    from myosuite.envs.myo.backends.mjlab.tasks.mimic_tracking.commands import (
        MotionCommand,
    )

    lib = _library(tmp_path, (2, 101, 102, 201), 100).map(torch.as_tensor)
    n = 1000
    command = SimpleNamespace(
        motion=lib,
        device="cpu",
        num_envs=n,
        task=SimpleNamespace(reset=SimpleNamespace(mode="uniform")),
        reset_probs=lambda gate=None: torch.full((lib.num_bins,), 1.0 / lib.num_bins),
    )
    torch.manual_seed(0)
    arg = n if method == "sample_frames" else torch.tensor(True)
    frames, bins = getattr(MotionCommand, method)(command, arg)
    clips = lib.bin_clip[bins]
    assert torch.all(frames < lib.clip_end[clips])
    assert torch.all(lib.clip_id[frames + 1] == clips)
    np.testing.assert_array_equal(lib.frame_bin[frames].numpy(), bins.numpy())
