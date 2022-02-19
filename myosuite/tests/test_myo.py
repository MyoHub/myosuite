""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

import unittest
from myosuite.tests.test_envs import TestEnvs

class TestMyo(TestEnvs):
    def test_myo(self):
        env_names = [
            'motorFingerReachFixed-v0', 'motorFingerReachRandom-v0',
            'myoFingerReachFixed-v0', 'myoFingerReachRandom-v0',
            'myoHandReachFixed-v0', 'myoHandReachRandom-v0',

            'motorFingerPoseFixed-v0', 'motorFingerPoseRandom-v0',
            'myoFingerPoseFixed-v0', 'myoFingerPoseRandom-v0',

            'myoElbowPose1D6MFixed-v0', 'myoElbowPose1D6MRandom-v0',
            'myoElbowPose1D6MExoRandom-v0', 'myoElbowPose1D6MExoRandom-v0',
            'myoHandPoseFixed-v0', 'myoHandPoseRandom-v0',

            'myoHandKeyTurnFixed-v0', 'myoHandKeyTurnRandom-v0',
            'myoHandObjHoldFixed-v0', 'myoHandObjHoldRandom-v0',
            'myoHandPenTwirlFixed-v0', 'myoHandPenTwirlRandom-v0',

            'myoHandBaodingFixed-v1', 'myoHandBaodingRandom-v1',
            'myoHandBaodingFixed4th-v1','myoHandBaodingFixed8th-v1',
        ]
        for k in range(10): env_names+=['myoHandPose'+str(k)+'Fixed-v0']

        self.check_envs('Myo', env_names)

if __name__ == '__main__':
    unittest.main()