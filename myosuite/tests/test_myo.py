# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import myosuite
from myosuite.tests.test_envs import TestEnvs


pytestmark = [pytest.mark.tier3, pytest.mark.legacy]


class TestMyo(TestEnvs):
    def test_myosuite_envs(self):
        self.check_envs("MyoBase Suite", myosuite.myosuite_myobase_suite)

    def test_myochal_envs(self):
        self.check_envs("MyoChallenge Suite", myosuite.myosuite_myochal_suite)

    def test_myomimic_envs(self):
        # MuscleMimic envs sample random targets; they carry no reference motion
        # for examine_reference playback (that covered the removed MyoDM suite).
        self.check_envs("MyoMimic Suite", myosuite.myosuite_myomimic_suite)
