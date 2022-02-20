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

        env_names+="myoSarFingerReachFixed-v0, myoFatFingerReachFixed-v0, myoSarFingerReachRandom-v0, myoFatFingerReachRandom-v0, myoSarElbowPose1D6MFixed-v0, myoFatElbowPose1D6MFixed-v0, myoSarElbowPose1D6MRandom-v0, myoFatElbowPose1D6MRandom-v0, myoSarElbowPose1D6MExoFixed-v0, myoFatElbowPose1D6MExoFixed-v0, myoSarElbowPose1D6MExoRandom-v0, myoFatElbowPose1D6MExoRandom-v0, myoSarFingerPoseFixed-v0, myoFatFingerPoseFixed-v0, myoSarFingerPoseRandom-v0, myoFatFingerPoseRandom-v0, myoSarHandPoseFixed-v0, myoFatHandPoseFixed-v0, myoTTHandPoseFixed-v0, myoSarHandPose0Fixed-v0, myoFatHandPose0Fixed-v0, myoTTHandPose0Fixed-v0, myoSarHandPose1Fixed-v0, myoFatHandPose1Fixed-v0, myoTTHandPose1Fixed-v0, myoSarHandPose2Fixed-v0, myoFatHandPose2Fixed-v0, myoTTHandPose2Fixed-v0, myoSarHandPose3Fixed-v0, myoFatHandPose3Fixed-v0, myoTTHandPose3Fixed-v0, myoSarHandPose4Fixed-v0, myoFatHandPose4Fixed-v0, myoTTHandPose4Fixed-v0, myoSarHandPose5Fixed-v0, myoFatHandPose5Fixed-v0, myoTTHandPose5Fixed-v0, myoSarHandPose6Fixed-v0, myoFatHandPose6Fixed-v0, myoTTHandPose6Fixed-v0, myoSarHandPose7Fixed-v0, myoFatHandPose7Fixed-v0, myoTTHandPose7Fixed-v0, myoSarHandPose8Fixed-v0, myoFatHandPose8Fixed-v0, myoTTHandPose8Fixed-v0, myoSarHandPose9Fixed-v0, myoFatHandPose9Fixed-v0, myoTTHandPose9Fixed-v0, myoSarHandPoseRandom-v0, myoFatHandPoseRandom-v0, myoTTHandPoseRandom-v0, myoSarHandReachFixed-v0, myoFatHandReachFixed-v0, myoTTHandReachFixed-v0, myoSarHandReachRandom-v0, myoFatHandReachRandom-v0, myoTTHandReachRandom-v0, myoSarHandKeyTurnFixed-v0, myoFatHandKeyTurnFixed-v0, myoTTHandKeyTurnFixed-v0, myoSarHandKeyTurnRandom-v0, myoFatHandKeyTurnRandom-v0, myoTTHandKeyTurnRandom-v0, myoSarHandObjHoldFixed-v0, myoFatHandObjHoldFixed-v0, myoTTHandObjHoldFixed-v0, myoSarHandObjHoldRandom-v0, myoFatHandObjHoldRandom-v0, myoTTHandObjHoldRandom-v0, myoSarHandPenTwirlFixed-v0, myoFatHandPenTwirlFixed-v0, myoTTHandPenTwirlFixed-v0, myoSarHandPenTwirlRandom-v0, myoFatHandPenTwirlRandom-v0, myoTTHandPenTwirlRandom-v0, myoSarHandBaodingFixed-v1, myoFatHandBaodingFixed-v1, myoTTHandBaodingFixed-v1, myoSarHandBaodingRandom-v1, myoFatHandBaodingRandom-v1, myoTTHandBaodingRandom-v1, myoSarHandBaodingFixed4th-v1, myoFatHandBaodingFixed4th-v1, myoTTHandBaodingFixed4th-v1, myoSarHandBaodingFixed8th-v1, myoFatHandBaodingFixed8th-v1, myoTTHandBaodingFixed8th-v1".split(", ")

        self.check_envs('Myo', env_names)

if __name__ == '__main__':
    unittest.main()