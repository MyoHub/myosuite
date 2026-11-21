# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Regression tests for RunTrackEnv per-instance ACTUATOR_PARAM isolation."""

from __future__ import annotations

import pytest

from myosuite import make_env

pytestmark = pytest.mark.tier1


def test_runtrack_instance_isolation_of_actuator_params() -> None:
    env1 = make_env("myoChallengeOslRunFixed-v0").unwrapped
    env2 = make_env("myoChallengeOslRunFixed-v0").unwrapped
    try:
        assert env1.ACTUATOR_PARAM is not env2.ACTUATOR_PARAM
        assert env1.OSL_PARAM_LIST is not env2.OSL_PARAM_LIST
        env1.ACTUATOR_PARAM["custom"] = {"id": -1, "Fmax": 0.0}
        assert "custom" not in env2.ACTUATOR_PARAM
    finally:
        env1.close()
        env2.close()
