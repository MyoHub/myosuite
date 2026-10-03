# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""examine_env CLI: saving and plotting the same rollout."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.tier2


def test_save_and_plot_paths_together(tmp_path: Path) -> None:
    """--save_paths used to flatten the trace before --plot_paths read it (KeyError)."""
    pytest.importorskip("matplotlib")
    from click.testing import CliRunner

    from myosuite.utils.examine_env import main

    result = CliRunner().invoke(
        main,
        [
            "-e",
            "myoElbowPose1D6MRandom-v0",
            "-n",
            "1",
            "-r",
            "none",
            "-sp",
            "True",
            "-pp",
            "True",
            "-o",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"
    assert len(list(tmp_path.glob("random_policy*Trial0.pdf"))) == 1
    assert len(list(tmp_path.glob("random_policy*_trace.h5"))) == 1
