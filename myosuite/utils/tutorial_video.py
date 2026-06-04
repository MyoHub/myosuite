# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Backward-compatible wrappers around :mod:`myosuite.utils.video_io`."""

from __future__ import annotations

import numpy as np
from pathlib import Path

from myosuite.utils.video_io import show_video, write_video


def store_video(
    video_path: str | Path, video_frames: list | np.ndarray, fps=30
) -> None:
    """Store video data as MP4 on disk."""
    write_video(video_path=video_path, video_frames=video_frames, fps=fps)


__all__ = ["show_video", "store_video"]
