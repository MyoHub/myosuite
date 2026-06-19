# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import click
import click.testing
import pytest

import myosuite
from myosuite.tests.test_envs import TestEnvs


pytestmark = [pytest.mark.tier3, pytest.mark.legacy]

_ARM_REACH_XML = Path("myosuite/envs/myo/assets/arm/myoarm_reach.xml")


class TestMyo(TestEnvs):
    @pytest.mark.skipif(
        not _ARM_REACH_XML.exists(),
        reason=(
            "arm reach asset missing "
            "(run `uv run myoapi_init` to fetch sim assets before enabling this test)."
        ),
    )
    def test_myosuite_envs(self):
        myosuite.register_all_envs()
        self.check_envs("MyoBase Suite", myosuite.myosuite_myobase_suite)

    def test_myochal_envs(self):
        myosuite.register_all_envs()
        self.check_envs("MyoChallenge Suite", myosuite.myosuite_myochal_suite)

    def test_myomimic_envs(self):
        myosuite.register_all_envs()
        self.check_envs("MyoMimic Suite", myosuite.myosuite_myomimic_suite)

        # Check trajectory playback
        from myosuite.logger.examine_reference import examine_reference

        for env in myosuite.myosuite_myomimic_suite:
            print(f"Testing reference motion playback on: {env}")
            runner = click.testing.CliRunner()
            result = runner.invoke(
                examine_reference,
                [
                    "--env_name",
                    env,
                    "--horizon",
                    -1,
                    "--num_playback",
                    1,
                    "--render",
                    "none",
                ],
            )
            assert result.exception is None, result.exception

    def no_test_myomimic(self):
        env_names = [
            "MyoLegJump-v0",
            "MyoLegLunge-v0",
            "MyoLegSquat-v0",
            "MyoLegLand-v0",
            "MyoLegRun-v0",
            "MyoLegWalk-v0",
        ]
        # Check the envs
        self.check_envs("MyoMimic", env_names)

        # Check trajectory playback
        from myosuite.logger.examine_reference import examine_reference

        for env in env_names:
            print(f"Testing reference motion playback on: {env}")
            runner = click.testing.CliRunner()
            result = runner.invoke(
                examine_reference,
                [
                    "--env_name",
                    env,
                    "--horizon",
                    -1,
                    "--num_playback",
                    1,
                    "--render",
                    "none",
                ],
            )
            self.assertEqual(result.exception, None, result.exception)


#
