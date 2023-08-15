""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

import unittest
import click
import click.testing
from myosuite.tests.test_envs import TestEnvs
from myosuite import myosuite_myobase_suite
from myosuite import myosuite_myochal_suite
from myosuite import myosuite_myodex_suite

class TestMyo(TestEnvs):

    def test_myosuite_envs(self):
        self.check_envs('MyoBase Suite', myosuite_myobase_suite)


    def test_myochal_envs(self):
        self.check_envs('MyoChallenge Suite', myosuite_myochal_suite)


    def test_myodex_envs(self):
        self.check_envs('MyoDex Suite', myosuite_myodex_suite)

        # Check trajectory playback
        from myosuite.logger.examine_reference import examine_reference
        for env in myosuite_myodex_suite:
            print(f"Testing reference motion playback on: {env}")
            runner = click.testing.CliRunner()
            result = runner.invoke(examine_reference, ["--env_name", env, \
                                                        "--horizon", -1, \
                                                        "--num_playback", 1, \
                                                        "--render", "none"])
            self.assertEqual(result.exception, None, result.exception)

if __name__ == '__main__':
    unittest.main()