# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from unittest.mock import patch

import pytest

from myosuite_init import fetch_simhive

my_file = Path("/path/to/file")

_SIMHIVE_SKELETON = Path("myosuite/simhive/myo_model/myoskeleton/myoskeleton.xml")


pytestmark = [pytest.mark.tier3, pytest.mark.legacy]


@pytest.mark.skipif(
    _SIMHIVE_SKELETON.exists(),
    reason="simhive already present; test_no_myoapi requires clean state",
)
@patch("builtins.input", side_effect=["no"])
def test_no_myoapi(mock_input):
    """When user answers 'no', fetch_simhive does not create simhive (run only when simhive absent)."""
    fetch_simhive()
    assert not _SIMHIVE_SKELETON.exists()


@patch("builtins.input", side_effect=["yes"])
def test_yes_myoapi(mock_input):
    """When user answers 'yes', fetch_simhive creates simhive (skipped if simhive already present)."""
    fetch_simhive()
    assert _SIMHIVE_SKELETON.exists()


# test_no_myoapi()
# test_yes_myoapi()
