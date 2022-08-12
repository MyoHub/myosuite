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

        ]
        for k in range(10): env_names+=['myoHandPose'+str(k)+'Fixed-v0']

        env_names+="myoSarcFingerReachFixed-v0, myoFatiFingerReachFixed-v0, myoSarcFingerReachRandom-v0, myoFatiFingerReachRandom-v0, myoSarcElbowPose1D6MFixed-v0, myoFatiElbowPose1D6MFixed-v0, myoSarcElbowPose1D6MRandom-v0, myoFatiElbowPose1D6MRandom-v0, myoSarcElbowPose1D6MExoFixed-v0, myoFatiElbowPose1D6MExoFixed-v0, myoSarcElbowPose1D6MExoRandom-v0, myoFatiElbowPose1D6MExoRandom-v0, myoSarcFingerPoseFixed-v0, myoFatiFingerPoseFixed-v0, myoSarcFingerPoseRandom-v0, myoFatiFingerPoseRandom-v0, myoSarcHandPoseFixed-v0, myoFatiHandPoseFixed-v0, myoReafHandPoseFixed-v0, myoSarcHandPose0Fixed-v0, myoFatiHandPose0Fixed-v0, myoReafHandPose0Fixed-v0, myoSarcHandPose1Fixed-v0, myoFatiHandPose1Fixed-v0, myoReafHandPose1Fixed-v0, myoSarcHandPose2Fixed-v0, myoFatiHandPose2Fixed-v0, myoReafHandPose2Fixed-v0, myoSarcHandPose3Fixed-v0, myoFatiHandPose3Fixed-v0, myoReafHandPose3Fixed-v0, myoSarcHandPose4Fixed-v0, myoFatiHandPose4Fixed-v0, myoReafHandPose4Fixed-v0, myoSarcHandPose5Fixed-v0, myoFatiHandPose5Fixed-v0, myoReafHandPose5Fixed-v0, myoSarcHandPose6Fixed-v0, myoFatiHandPose6Fixed-v0, myoReafHandPose6Fixed-v0, myoSarcHandPose7Fixed-v0, myoFatiHandPose7Fixed-v0, myoReafHandPose7Fixed-v0, myoSarcHandPose8Fixed-v0, myoFatiHandPose8Fixed-v0, myoReafHandPose8Fixed-v0, myoSarcHandPose9Fixed-v0, myoFatiHandPose9Fixed-v0, myoReafHandPose9Fixed-v0, myoSarcHandPoseRandom-v0, myoFatiHandPoseRandom-v0, myoReafHandPoseRandom-v0, myoSarcHandReachFixed-v0, myoFatiHandReachFixed-v0, myoReafHandReachFixed-v0, myoSarcHandReachRandom-v0, myoFatiHandReachRandom-v0, myoReafHandReachRandom-v0, myoSarcHandKeyTurnFixed-v0, myoFatiHandKeyTurnFixed-v0, myoReafHandKeyTurnFixed-v0, myoSarcHandKeyTurnRandom-v0, myoFatiHandKeyTurnRandom-v0, myoReafHandKeyTurnRandom-v0, myoSarcHandObjHoldFixed-v0, myoFatiHandObjHoldFixed-v0, myoReafHandObjHoldFixed-v0, myoSarcHandObjHoldRandom-v0, myoFatiHandObjHoldRandom-v0, myoReafHandObjHoldRandom-v0, myoSarcHandPenTwirlFixed-v0, myoFatiHandPenTwirlFixed-v0, myoReafHandPenTwirlFixed-v0, myoSarcHandPenTwirlRandom-v0, myoFatiHandPenTwirlRandom-v0, myoReafHandPenTwirlRandom-v0".split(", ")

        self.check_envs('Myo', env_names)

    def test_myochallenge(self):
        env_names = [
            'myoChallengeDieReorientDemo-v0', 'myoChallengeDieReorientP1-v0', 'myoChallengeDieReorientP2-v0',
            'myoChallengeBaodingP1-v1', 'myoChallengeBaodingP2-v1'
        ]
        self.check_envs('Myo', env_names)

if __name__ == '__main__':
    unittest.main()