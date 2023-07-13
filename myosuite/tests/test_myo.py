""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

import unittest
from myosuite.tests.test_envs import TestEnvs
from myosuite import myo_suite_envs

class TestMyo(TestEnvs):
    def test_envs(self):
        self.check_envs('Myo Suite', myo_suite_envs)

if __name__ == '__main__':
    unittest.main()