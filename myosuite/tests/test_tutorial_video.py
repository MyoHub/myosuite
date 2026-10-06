# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for :mod:`myosuite.utils.video_io`."""

from __future__ import annotations

from pathlib import Path

from myosuite.tests.support.optional_deps import require_ipython
from myosuite.utils.video_io import show_video


def test_show_video_missing_file_returns_empty_html() -> None:
    """Missing paths should not raise; return empty embed."""
    require_ipython()
    out = show_video(Path("/nonexistent/path/to/video_xyz123.mp4"))
    assert hasattr(out, "data")
    assert out.data == ""
