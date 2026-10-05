# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The mjlab backend hides the two harmless entity-attach warnings, and only those."""

import subprocess
import sys

_SCRIPT = """
import warnings
import myosuite.envs.myo.backends.mjlab  # installs the filters
warnings.warn("Entity 'robot' has non-default <option> fields (timestep) that will not be propagated", UserWarning)
warnings.warn("Attach conflict when attaching 'A' to 'mjlab scene', policy is 'warning'", UserWarning)
warnings.warn("Entity 'robot' has something else", UserWarning)
"""


def test_only_the_two_attach_warnings_are_hidden() -> None:
    result = subprocess.run(
        [sys.executable, "-W", "always", "-c", _SCRIPT],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "something else" in result.stderr
    assert "non-default <option>" not in result.stderr
    assert "Attach conflict" not in result.stderr
