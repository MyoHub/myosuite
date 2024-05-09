""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

from myosuite.utils import gym
register=gym.register
from myosuite.envs.env_variants import register_env_variant

import os
import numpy as np

# utility to register envs with all muscle conditions
def register_env_with_variants(id, entry_point, max_episode_steps, kwargs):
    # register_env_with_variants base env
    register(
        id=id,
        entry_point=entry_point,
        max_episode_steps=max_episode_steps,
        kwargs=kwargs
    )
    #register variants env with sarcopenia
    if id[:3] == "myo":
        register_env_variant(
            env_id=id,
            variants={'muscle_condition':'sarcopenia'},
            variant_id=id[:3]+"Sarc"+id[3:],
            silent=True
        )
    #register variants with fatigue
    if id[:3] == "myo":
        register_env_variant(
            env_id=id,
            variants={'muscle_condition':'fatigue', 'fatigue_reset_random':True},
            variant_id=id[:3]+"Fati"+id[3:],
            silent=True
        )
        #register variants with fatigue reset at each episode to manually identified equilibrium fatigue state
        fatigue_reset_vec_dict = {
            "myoElbowPose1D6MRandom-v0": np.array([0.0804, 0.0466, 0.0373, 0.2758, 0.1293, 0.3045]),
            "myoHandPose05Cycle-v0": np.array([3.4449e-04, 2.3814e-03, 8.5076e-01, 1.9823e-04, 1.5574e-02,
                                3.2002e-04, 3.2150e-05, 3.1177e-05, 4.8601e-06, 2.5083e-03,
                                2.4133e-03, 5.8925e-10, 6.6611e-03, 3.6191e-03, 3.8226e-03,
                                2.4782e-03, 3.8353e-03, 1.4679e-02, 2.8178e-02, 1.4691e-02,
                                1.6764e-03, 5.7774e-04, 1.3082e-10, 2.4529e-03, 1.4881e-04,
                                6.4363e-03, 8.6249e-04, 5.7296e-03, 1.2569e-09, 7.6275e-03,
                                1.1764e-02, 1.7635e-05, 2.4913e-02, 1.9906e-04, 1.8719e-01,
                                4.9752e-03, 1.7388e-03, 2.7483e-09, 5.6728e-04]),
            "myoHandPenTwirlFixed-v0": np.array([8.6383e-05, 7.8825e-04, 8.6150e-01, 2.8245e-03, 1.3780e-01,
                                1.7651e-02, 1.4343e-03, 1.0818e-02, 3.1508e-05, 3.1988e-03,
                                1.7015e-04, 1.2109e-04, 2.6389e-03, 5.6151e-03, 4.4958e-04,
                                5.1500e-03, 9.7887e-03, 8.3209e-03, 4.6281e-03, 1.0558e-02,
                                7.0326e-01, 8.5350e-01, 8.4897e-01, 1.1064e-03, 6.4123e-04,
                                3.9897e-04, 7.8945e-04, 7.4652e-03, 8.2818e-04, 7.7591e-01,
                                5.8781e-01, 4.4565e-01, 2.6730e-03, 4.0706e-03, 1.3397e-02,
                                2.3874e-01, 1.0336e-02, 6.2254e-01, 1.8370e-02]),
            # "myoHandPenTwirlFixed-v0": np.array([0.4033, 0.0071, 0.1366, 0.0024, 0.0031, 0.0095, 0.0062, 0.0912,
            #                     0.0048, 0.0203, 0.1436, 0.003 , 0.0123, 0.0069, 0.0025, 0.0102,
            #                     0.0115, 0.0877, 0.1662, 0.2678, 0.4803, 0.5586, 0.1646, 0.0052,
            #                     0.0334, 0.0138, 0.0865, 0.0074, 0.0379, 0.6029, 0.2709, 0.3346,
            #                     0.3081, 0.0138, 0.0675, 0.0145, 0.021 , 0.4652, 0.0266]),
            "myoHandKeyTurnFixed-v0": np.array([4.6007e-02, 2.1294e-01, 1.9276e-02, 5.4516e-01, 6.6911e-01,
                                6.2423e-01, 3.1732e-01, 4.1236e-01, 1.2621e-01, 7.5696e-01,
                                7.3778e-01, 7.6790e-01, 4.2954e-01, 8.6821e-01, 8.1806e-01,
                                8.6900e-01, 1.4994e-01, 2.6219e-02, 1.0478e-02, 3.3517e-04,
                                6.3720e-02, 6.9702e-02, 8.7857e-02, 7.2428e-02, 8.5107e-01,
                                4.1436e-02, 6.4059e-01, 4.4154e-01, 4.5290e-03, 3.2601e-01,
                                6.0467e-01, 1.2205e-01, 3.7340e-01, 2.2250e-01, 3.6504e-01,
                                3.2788e-01, 3.6091e-01, 4.9714e-01, 7.1446e-03]),
            "myoHandObjHoldFixed-v0": np.array([8.6807e-01, 3.8260e-03, 1.9745e-05, 2.5193e-02, 5.9590e-04,
                                8.3116e-01, 1.2233e-02, 8.4629e-03, 1.2485e-02, 2.0755e-01,
                                6.6337e-04, 1.3505e-02, 2.4267e-01, 6.4649e-04, 1.1336e-03,
                                6.1975e-03, 9.4326e-04, 6.3968e-02, 7.1890e-02, 2.5620e-01,
                                2.3116e-01, 3.3667e-02, 1.7437e-01, 2.7189e-03, 4.3200e-02,
                                3.6952e-02, 8.6431e-01, 1.1083e-01, 4.0353e-01, 5.2389e-03,
                                3.6655e-03, 1.4145e-04, 4.9099e-02, 1.7231e-02, 1.0536e-02,
                                1.7525e-02, 1.2911e-01, 9.1333e-02, 3.2306e-01]),
            "myoHandReachRandom-v0": np.array([8.6294e-01, 8.6852e-01, 8.6902e-01, 4.0029e-05, 7.6834e-02,
                                1.9547e-04, 8.6901e-01, 4.8843e-02, 8.6902e-01, 1.6731e-01,
                                4.9793e-07, 3.7780e-04, 2.6074e-03, 3.5406e-02, 1.2404e-04,
                                9.4070e-03, 8.6473e-01, 8.5605e-01, 8.6902e-01, 3.5228e-07,
                                1.0631e-01, 1.0043e-01, 1.7756e-01, 1.6948e-04, 3.5678e-02,
                                8.6902e-01, 8.6902e-01, 2.2458e-03, 3.1866e-02, 8.6902e-01,
                                8.6657e-01, 7.9883e-03, 8.6288e-01, 2.1380e-03, 2.4776e-01,
                                8.6826e-01, 1.7121e-05, 8.6902e-01, 6.2568e-11]),
            "myoFingerReachFixed-v0": np.array([0.0611, 0.1361, 0.172, 0.8676, 0.0585]),
            "myoFingerPoseFixed-v0": np.array([0.8633, 0.1241, 0.1648, 0.013, 0.047]),
            "myoLegWalk-v0": np.array([8.69021830e-01, 8.69023762e-01, 3.23136858e-07, 1.99222159e-07,
                                1.51586939e-04, 8.64821287e-01, 1.41907768e-09, 8.65685817e-01,
                                8.68964644e-01, 8.66917977e-01, 8.68947136e-01, 6.76280765e-02,
                                4.84144590e-03, 6.02692972e-02, 7.09458197e-09, 1.03023089e-08,
                                4.48199475e-09, 8.74394273e-03, 7.52836879e-07, 8.08907454e-02,
                                8.68491519e-01, 8.25144919e-01, 8.68748884e-01, 8.67397234e-01,
                                2.06277250e-02, 3.57615264e-02, 5.85325778e-02, 5.65621547e-01,
                                3.31904047e-03, 8.69023762e-01, 8.68130447e-01, 4.57628624e-08,
                                8.34798696e-01, 8.69023762e-01, 8.67723903e-01, 8.69023762e-01,
                                1.66458609e-04, 3.62009561e-03, 7.76110063e-04, 2.73899536e-03,
                                3.94483772e-02, 1.82149964e-02, 1.70144998e-03, 1.57795838e-02,
                                3.12541234e-06, 8.67230219e-01, 1.09780744e-04, 7.38635549e-03,
                                9.53260503e-02, 3.97382941e-02, 6.91033643e-03, 7.01950407e-01,
                                1.81530536e-05, 3.20764310e-05, 1.28747178e-02, 2.42824403e-04,
                                1.51947551e-04, 8.61696114e-01, 4.39571224e-03, 1.48166631e-04,
                                5.74980644e-01, 3.47364606e-01, 1.65231069e-02, 8.69023761e-01,
                                5.39706144e-03, 2.34638678e-04, 8.68460155e-01, 1.54380939e-03,
                                2.73885084e-04, 2.05614990e-07, 7.50824646e-02, 2.57535266e-06,
                                4.30360478e-03, 1.15125818e-01, 3.03077188e-03, 8.60808309e-03,
                                8.22322334e-03, 8.06270705e-01, 3.29262800e-05, 8.12078660e-01])
            }
        if id in fatigue_reset_vec_dict.keys():
            register_env_variant(
                env_id=id,
                variants={'muscle_condition':'fatigue',
                          'fatigue_reset_random':False,
                          'fatigue_reset_vec':fatigue_reset_vec_dict[id]},
                variant_id=id[:3]+"FatiEQ"+id[3:],
                silent=True
            )
        register_env_variant(
            env_id=id,
            variants={'muscle_condition':'fatigue', 'fatigue_reset_random':False},
            variant_id=id[:3]+"FatiZero"+id[3:],
            silent=True
        )

    #register variants with tendon transfer
    if id[:7] == "myoHand":
        register_env_variant(
            env_id=id,
            variants={'muscle_condition':'reafferentation'},
            variant_id=id[:3]+"Reaf"+id[3:],
            silent=True
        )

curr_dir = os.path.dirname(os.path.abspath(__file__))

print("MyoSuite:> Registering Myo Envs")

# Finger-tip reaching ==============================
register_env_with_variants(id='motorFingerReachFixed-v0',
        entry_point='myosuite.envs.myo.myobase.reach_v0:ReachEnvV0',
        max_episode_steps=200,
        kwargs={
            'model_path': curr_dir+'/../../../simhive/myo_sim/finger/motorfinger_v0.xml',
            'target_reach_range': {'IFtip': ((0.2, 0.05, 0.20), (0.2, 0.05, 0.20)),},
            'normalize_act': True,
            'frame_skip': 5,
        }
    )
register_env_with_variants(id='motorFingerReachRandom-v0',
        entry_point='myosuite.envs.myo.myobase.reach_v0:ReachEnvV0',
        max_episode_steps=200,
        kwargs={
            'model_path': curr_dir+'/../../../simhive/myo_sim/finger/motorfinger_v0.xml',
            'target_reach_range': {'IFtip': ((.1, -.1, .1), (0.27, .1, .3)),},
            'normalize_act': True,
            'frame_skip': 5,
        }
    )
register_env_with_variants(id='myoFingerReachFixed-v0',
        entry_point='myosuite.envs.myo.myobase.reach_v0:ReachEnvV0',
        max_episode_steps=100,
        kwargs={
            'model_path': curr_dir+'/../../../simhive/myo_sim/finger/myofinger_v0.xml',
            'target_reach_range': {'IFtip': ((0.2, 0.05, 0.20), (0.2, 0.05, 0.20)),},
            'normalize_act': True,
        }
    )
register_env_with_variants(id='myoFingerReachRandom-v0',
        entry_point='myosuite.envs.myo.myobase.reach_v0:ReachEnvV0',
        max_episode_steps=100,
        kwargs={
            'model_path': curr_dir+'/../../../simhive/myo_sim/finger/myofinger_v0.xml',
            'target_reach_range': {'IFtip': ((.1, -.1, .1), (0.27, .1, .3)),},
            'normalize_act': True,
        }
    )

# Elbow posing ==============================
register_env_with_variants(id='myoElbowPose1D6MFixed-v0',
        entry_point='myosuite.envs.myo.myobase.pose_v0:PoseEnvV0',
        max_episode_steps=100,
        kwargs={
            'model_path': curr_dir+'/../assets/elbow/myoelbow_1dof6muscles.xml',
            'target_jnt_range': {'r_elbow_flex':(2, 2),},
            'viz_site_targets': ('wrist',),
            'normalize_act': True,
            'pose_thd': .175,
            'reset_type': 'random'
        }
    )
register_env_with_variants(id='myoElbowPose1D6MRandom-v0',
        entry_point='myosuite.envs.myo.myobase.pose_v0:PoseEnvV0',
        max_episode_steps=100,
        kwargs={
            'model_path': curr_dir+'/../assets/elbow/myoelbow_1dof6muscles.xml',
            'target_jnt_range': {'r_elbow_flex':(0, 2.27),},
            'viz_site_targets': ('wrist',),
            'normalize_act': True,
            'pose_thd': .175,
            'reset_type': 'random'
        }
    )


# Elbow Exo posing ==============================
register_env_with_variants(id='myoElbowPose1D6MExoFixed-v0',
        entry_point='myosuite.envs.myo.myobase.pose_v0:PoseEnvV0',
        max_episode_steps=100,
        kwargs={
            'model_path': curr_dir+'/../assets/elbow/myoelbow_1dof6muscles_1dofexo.xml',
            'target_jnt_range': {'r_elbow_flex':(2, 2),},
            'viz_site_targets': ('wrist',),
            'normalize_act': True,
            'pose_thd': .175,
            'reset_type': 'random',
            'weighted_reward_keys':{
                                "pose": 1.0,
                                "bonus": 4.0,
                                "act_reg": 5.0,
                                "penalty": 50,
            }
        }
    )
register_env_with_variants(id='myoElbowPose1D6MExoRandom-v0',
        entry_point='myosuite.envs.myo.myobase.pose_v0:PoseEnvV0',
        max_episode_steps=100,
        kwargs={
            'model_path': curr_dir+'/../assets/elbow/myoelbow_1dof6muscles_1dofexo.xml',
            'target_jnt_range': {'r_elbow_flex':(0, 2.27),},
            'viz_site_targets': ('wrist',),
            'normalize_act': True,
            'pose_thd': .175,
            'reset_type': 'random',
            'weight_bodyname':'carry_weight',
            'weight_range':(.1, 2),
            'weighted_reward_keys':{
                                "pose": 1.0,
                                "bonus": 4.0,
                                "act_reg": 5.0,
                                "penalty": 50,
            }
        }
    )


# Finger-Joint posing ==============================
register_env_with_variants(id='motorFingerPoseFixed-v0',
        entry_point='myosuite.envs.myo.myobase.pose_v0:PoseEnvV0',
        max_episode_steps=200,
        kwargs={
            'model_path': curr_dir+'/../../../simhive/myo_sim/finger/motorfinger_v0.xml',
            'target_jnt_range': {'IFadb':(0, 0),
                                'IFmcp':(0, 0),
                                'IFpip':(.75, .75),
                                'IFdip':(.75, .75)
                                },
            'viz_site_targets': ('IFtip',),
            'normalize_act': True,
            'frame_skip': 5,
        }
)
register_env_with_variants(id='motorFingerPoseRandom-v0',
        entry_point='myosuite.envs.myo.myobase.pose_v0:PoseEnvV0',
        max_episode_steps=200,
        kwargs={
            'model_path': curr_dir+'/../../../simhive/myo_sim/finger/motorfinger_v0.xml',
            'target_jnt_range': {'IFadb':(-.2, .2),
                                'IFmcp':(-.4, 1),
                                'IFpip':(.1, 1),
                                'IFdip':(.1, 1)
                                },
            'viz_site_targets': ('IFtip',),
            'normalize_act': True,
            'frame_skip': 5,
        }
    )
register_env_with_variants(id='myoFingerPoseFixed-v0',
        entry_point='myosuite.envs.myo.myobase.pose_v0:PoseEnvV0',
        max_episode_steps=100,
        kwargs={
            'model_path': curr_dir+'/../../../simhive/myo_sim/finger/myofinger_v0.xml',
            'target_jnt_range': {'IFadb':(0, 0),
                                'IFmcp':(0, 0),
                                'IFpip':(.75, .75),
                                'IFdip':(.75, .75)
                                },
            'viz_site_targets': ('IFtip',),
            'normalize_act': True,
        }
    )
register_env_with_variants(id='myoFingerPoseRandom-v0',
        entry_point='myosuite.envs.myo.myobase.pose_v0:PoseEnvV0',
        max_episode_steps=100,
        kwargs={
            'model_path': curr_dir+'/../../../simhive/myo_sim/finger/myofinger_v0.xml',
            'target_jnt_range': {'IFadb':(-.2, .2),
                                'IFmcp':(-.4, 1),
                                'IFpip':(.1, 1),
                                'IFdip':(.1, 1)
                                },
            'viz_site_targets': ('IFtip',),
            'normalize_act': True,
        }
    )

# Hand-Joint posing ==============================

# Remove this when the ASL envs stablizes
register_env_with_variants(id='myoHandPoseFixed-v0', # revisit
        entry_point='myosuite.envs.myo.myobase.pose_v0:PoseEnvV0',
        max_episode_steps=100,
        kwargs={
            'model_path': curr_dir+'/../assets/hand/myohand_pose.xml',
            'viz_site_targets': ('THtip','IFtip','MFtip','RFtip','LFtip'),
            'target_jnt_value': np.array([0, 0, 0, -0.0904, 0.0824475, -0.681555, -0.514888, 0, -0.013964, -0.0458132, 0, 0.67553, -0.020944, 0.76979, 0.65982, 0, 0, 0, 0, 0.479155, -0.099484, 0.95831, 0]),
            'normalize_act': True,
            'pose_thd': .7,
            'reset_type': "init",        # none, init, random
            'target_type': 'fixed',      # generate/ fixed
        }
    )

# Create ASL envs ==============================
jnt_namesHand=['pro_sup', 'deviation', 'flexion', 'cmc_abduction', 'cmc_flexion', 'mp_flexion', 'ip_flexion', 'mcp2_flexion', 'mcp2_abduction', 'pm2_flexion', 'md2_flexion', 'mcp3_flexion', 'mcp3_abduction', 'pm3_flexion', 'md3_flexion', 'mcp4_flexion', 'mcp4_abduction', 'pm4_flexion', 'md4_flexion', 'mcp5_flexion', 'mcp5_abduction', 'pm5_flexion', 'md5_flexion']

ASL_qpos={}
ASL_qpos[0]='0 0 0 0.5624 0.28272 -0.75573 -1.309 1.30045 -0.006982 1.45492 0.998897 1.26466 0 1.40604 0.227795 1.07614 -0.020944 1.46103 0.06284 0.83263 -0.14399 1.571 1.38248'.split(' ')
ASL_qpos[1]='0 0 0 0.0248 0.04536 -0.7854 -1.309 0.366605 0.010473 0.269258 0.111722 1.48459 0 1.45318 1.44532 1.44532 -0.204204 1.46103 1.44532 1.48459 -0.2618 1.47674 1.48459'.split(' ')
ASL_qpos[2]='0 0 0 0.0248 0.04536 -0.7854 -1.13447 0.514973 0.010473 0.128305 0.111722 0.510575 0 0.37704 0.117825 1.44532 -0.204204 1.46103 1.44532 1.48459 -0.2618 1.47674 1.48459'.split(' ')
ASL_qpos[3]='0 0 0 0.3384 0.25305 0.01569 -0.0262045 0.645885 0.010473 0.128305 0.111722 0.510575 0 0.37704 0.117825 1.571 -0.036652 1.52387 1.45318 1.40604 -0.068068 1.39033 1.571'.split(' ')
ASL_qpos[4]='0 0 0 0.6392 -0.147495 -0.7854 -1.309 0.637158 0.010473 0.128305 0.111722 0.510575 0 0.37704 0.117825 0.306345 -0.010472 0.400605 0.133535 0.21994 -0.068068 0.274925 0.01571'.split(' ')
ASL_qpos[5]='0 0 0 0.3384 0.25305 0.01569 -0.0262045 0.645885 0.010473 0.128305 0.111722 0.510575 0 0.37704 0.117825 0.306345 -0.010472 0.400605 0.133535 0.21994 -0.068068 0.274925 0.01571'.split(' ')
ASL_qpos[6]='0 0 0 0.6392 -0.147495 -0.7854 -1.309 0.637158 0.010473 0.128305 0.111722 0.510575 0 0.37704 0.117825 0.306345 -0.010472 0.400605 0.133535 1.1861 -0.2618 1.35891 1.48459'.split(' ')
ASL_qpos[7]='0 0 0 0.524 0.01569 -0.7854 -1.309 0.645885 -0.006982 0.128305 0.111722 0.510575 0 0.37704 0.117825 1.28036 -0.115192 1.52387 1.45318 0.432025 -0.068068 0.18852 0.149245'.split(' ')
ASL_qpos[8]='0 0 0 0.428 0.22338 -0.7854 -1.309 0.645885 -0.006982 0.128305 0.194636 1.39033 0 1.08399 0.573415 0.667675 -0.020944 0 0.06284 0.432025 -0.068068 0.18852 0.149245'.split(' ')
ASL_qpos[9]='0 0 0 0.5624 0.28272 -0.75573 -1.309 1.30045 -0.006982 1.45492 0.998897 0.39275 0 0.18852 0.227795 0.667675 -0.020944 0 0.06284 0.432025 -0.068068 0.18852 0.149245'.split(' ')

# ASl Eval envs for each numerals
for k in ASL_qpos.keys():
    register_env_with_variants(id='myoHandPose'+str(k)+'Fixed-v0',
            entry_point='myosuite.envs.myo.myobase.pose_v0:PoseEnvV0',
            max_episode_steps=100,
            kwargs={
                'model_path': curr_dir+'/../assets/hand/myohand_pose.xml',
                'viz_site_targets': ('THtip','IFtip','MFtip','RFtip','LFtip'),
                'target_jnt_value': np.array(ASL_qpos[k],'float'),
                'normalize_act': True,
                'pose_thd': .7,
                'reset_type': "init",        # none, init, random
                'target_type': 'fixed',      # generate/ fixed
            }
    )

# ASL Train Env
m = np.array([ASL_qpos[i] for i in range(10)]).astype(float)
Rpos = {}
for i_n, n  in enumerate(jnt_namesHand):
    Rpos[n]=(np.min(m[:,i_n]), np.max(m[:,i_n]))

register_env_with_variants(id='myoHandPoseRandom-v0',  #reconsider
        entry_point='myosuite.envs.myo.myobase.pose_v0:PoseEnvV0',
        max_episode_steps=100,
        kwargs={
            'model_path': curr_dir+'/../assets/hand/myohand_pose.xml',
            'viz_site_targets': ('THtip','IFtip','MFtip','RFtip','LFtip'),
            'target_jnt_range': Rpos,
            'normalize_act': True,
            'pose_thd': .7,
            'reset_type': "random",         # none, init, random
            'target_type': 'generate',      # generate/ fixed
        }
    )


# Gait Torso Reaching ==============================
from myosuite.physics.sim_scene import SimBackend
sim_backend = SimBackend.get_sim_backend()
leg_model='/../../../simhive/myo_sim/leg/myolegs.xml'
    

register_env_with_variants(id='myoLegStandRandom-v0',
        entry_point='myosuite.envs.myo.myobase.walk_v0:ReachEnvV0',
        max_episode_steps=150,
        kwargs={
            'model_path': curr_dir+leg_model,
            'joint_random_range': (-.2, 0.2), #range of joint randomization (jnt = init_qpos + random(range)
            'target_reach_range': {
                'pelvis': ((-.05, -.05, 0), (0.05, 0.05, 0)),
                },
            'normalize_act': True,
            'far_th': 0.44
        }
    )


# Gait Torso Walking ==============================
register_env_with_variants(id='myoLegWalk-v0',
        entry_point='myosuite.envs.myo.myobase.walk_v0:WalkEnvV0',
        max_episode_steps=1000,
        kwargs={
            'model_path': curr_dir + leg_model,
            'normalize_act': True,
            'min_height':0.8,    # minimum center of mass height before reset
            'max_rot':0.8,       # maximum rotation before reset
            'hip_period':100,    # desired periodic hip angle movement
            'reset_type':'init', # none, init, random
            'target_x_vel':0.0,  # desired x velocity in m/s
            'target_y_vel':1.2,  # desired y velocity in m/s
            'target_rot': None   # if None then the initial root pos will be taken, otherwise provide quat
        }
    )

# Rough Terrain Walking  ==============================
register_env_with_variants(id='myoLegRoughTerrainWalk-v0',
        entry_point='myosuite.envs.myo.myobase.walk_v0:TerrainEnvV0',
        max_episode_steps=1000,
        kwargs={
            'model_path': curr_dir + leg_model,
            'normalize_act': True,
            'min_height':0.8,    # minimum center of mass height before reset
            'max_rot':0.8,       # maximum rotation before reset
            'hip_period':100,    # desired periodic hip angle movement
            'reset_type':'init', # none, init, random
            'target_x_vel':0.0,  # desired x velocity in m/s
            'target_y_vel':1.2,  # desired y velocity in m/s
            'target_rot': None,   # if None then the initial root pos will be taken, otherwise provide quat
            'terrain':'rough',
            'variant': None
        }
    )

# Hilly Walking  ==============================
register_env_with_variants(id='myoLegHillyTerrainWalk-v0',
        entry_point='myosuite.envs.myo.myobase.walk_v0:TerrainEnvV0',
        max_episode_steps=1000,
        kwargs={
            'model_path': curr_dir + leg_model,
            'normalize_act': True,
            'min_height':0.8,    # minimum center of mass height before reset
            'max_rot':0.8,       # maximum rotation before reset
            'hip_period':100,    # desired periodic hip angle movement
            'reset_type':'init', # none, init, random
            'target_x_vel':0.0,  # desired x velocity in m/s
            'target_y_vel':1.2,  # desired y velocity in m/s
            'target_rot': None,   # if None then the initial root pos will be taken, otherwise provide quat
            'terrain':'hilly',
            'variant':'fixed'
        }
    )

# Stair Walking  ==============================
register_env_with_variants(id='myoLegStairTerrainWalk-v0',
        entry_point='myosuite.envs.myo.myobase.walk_v0:TerrainEnvV0',
        max_episode_steps=1000,
        kwargs={
            'model_path': curr_dir + leg_model,
            'normalize_act': True,
            'min_height':0.8,    # minimum center of mass height before reset
            'max_rot':0.8,       # maximum rotation before reset
            'hip_period':100,    # desired periodic hip angle movement
            'reset_type':'init', # none, init, random
            'target_x_vel':0.0,  # desired x velocity in m/s
            'target_y_vel':1.2,  # desired y velocity in m/s
            'target_rot': None,   # if None then the initial root pos will be taken, otherwise provide quat
            'terrain':'stairs',
            'variant':'fixed'
        }
    )



# Hand-Joint Reaching ==============================
register_env_with_variants(id='myoHandReachFixed-v0',
        entry_point='myosuite.envs.myo.myobase.reach_v0:ReachEnvV0',
        max_episode_steps=100,
        kwargs={
            'model_path': curr_dir+'/../assets/hand/myohand_pose.xml',
            'target_reach_range': {
                'THtip': ((-0.165, -0.537, 1.495), (-0.165, -0.537, 1.495)),
                'IFtip': ((-0.151, -0.547, 1.455), (-0.151, -0.547, 1.455)),
                'MFtip': ((-0.146, -0.547, 1.447), (-0.146, -0.547, 1.447)),
                'RFtip': ((-0.148, -0.543, 1.445), (-0.148, -0.543, 1.445)),
                'LFtip': ((-0.148, -0.528, 1.434), (-0.148, -0.528, 1.434)),
                },
            'normalize_act': True,
            'far_th': 0.044
        }
    )
register_env_with_variants(id='myoHandReachRandom-v0',
    entry_point='myosuite.envs.myo.myobase.reach_v0:ReachEnvV0',
    max_episode_steps=100,
    kwargs={
        'model_path': curr_dir+'/../assets/hand/myohand_pose.xml',
        'target_reach_range': {
            'THtip': ((-0.165-0.020, -0.537-0.040, 1.495-0.040), (-0.165+0.040, -0.537+0.020, 1.495+0.040)),
            'IFtip': ((-0.151-0.040, -0.547-0.020, 1.455-0.010), (-0.151+0.040, -0.547+0.020, 1.455+0.010)),
            'MFtip': ((-0.146-0.040, -0.547-0.020, 1.447-0.010), (-0.146+0.040, -0.547+0.020, 1.447+0.010)),
            'RFtip': ((-0.148-0.040, -0.543-0.020, 1.445-0.010), (-0.148+0.040, -0.543+0.020, 1.445+0.010)),
            'LFtip': ((-0.148-0.040, -0.528-0.020, 1.434-0.010), (-0.148+0.040, -0.528+0.020, 1.434+0.010)),
            },
        'normalize_act': True,
        'far_th': 0.034
    }
)

# Hand-Joint key turn ==============================
register_env_with_variants(id='myoHandKeyTurnFixed-v0',
        entry_point='myosuite.envs.myo.myobase.key_turn_v0:KeyTurnEnvV0',
        max_episode_steps=200,
        kwargs={
            'model_path': curr_dir+'/../assets/hand/myohand_keyturn.xml',
            'normalize_act': True
        }
    )
register_env_with_variants(id='myoHandKeyTurnRandom-v0',
        entry_point='myosuite.envs.myo.myobase.key_turn_v0:KeyTurnEnvV0',
        max_episode_steps=200,
        kwargs={
            'model_path': curr_dir+'/../assets/hand/myohand_keyturn.xml',
            'normalize_act': True,
            'key_init_range':(-np.pi/2, np.pi/2),
            'goal_th': 2*np.pi
        }
    )


# Hold objects ==============================
register_env_with_variants(id='myoHandObjHoldFixed-v0',
        entry_point='myosuite.envs.myo.myobase.obj_hold_v0:ObjHoldFixedEnvV0',
        max_episode_steps=75,
        kwargs={
            'model_path': curr_dir+'/../assets/hand/myohand_hold.xml',
            'normalize_act': True
        }
    )
register_env_with_variants(id='myoHandObjHoldRandom-v0', # revisit
        entry_point='myosuite.envs.myo.myobase.obj_hold_v0:ObjHoldRandomEnvV0',
        max_episode_steps=75,
        kwargs={
            'model_path': curr_dir+'/../assets/hand/myohand_hold.xml',
            'normalize_act': True
        }
    )

# Pen twirl ==============================
register_env_with_variants(id='myoHandPenTwirlFixed-v0',
            entry_point='myosuite.envs.myo.myobase.pen_v0:PenTwirlFixedEnvV0',
            max_episode_steps=50,
            kwargs={
                'model_path': curr_dir+'/../assets/hand/myohand_pen.xml',
                'normalize_act': True,
                'frame_skip': 5,
            }
    )
register_env_with_variants(id='myoHandPenTwirlRandom-v0',
        entry_point='myosuite.envs.myo.myobase.pen_v0:PenTwirlRandomEnvV0',
        max_episode_steps=50,
        kwargs={
            'model_path': curr_dir+'/../assets/hand/myohand_pen.xml',
            'normalize_act': True,
            'frame_skip': 5,
        }
    )

# SAR REORIENT: 8-object ==============================
register_env_with_variants(id='myoHandReorient8-v0',
            entry_point='myosuite.envs.myo.myobase.reorient_sar_v0:Geometries8EnvV0',
            max_episode_steps=50,
            kwargs={
                'model_path': curr_dir+'/../assets/hand/myohand_sar.xml',
                'normalize_act': True,
                'frame_skip': 5,
            }
    )

# SAR REORIENT: 100-object
register_env_with_variants(id='myoHandReorient100-v0',
            entry_point='myosuite.envs.myo.myobase.reorient_sar_v0:Geometries100EnvV0',
            max_episode_steps=50,
            kwargs={
                'model_path': curr_dir+'/../assets/hand/myohand_sar.xml',
                'normalize_act': True,
                'frame_skip': 5,
            }
    )

# SAR TEST ENVIRONMENT: in-distribution
register_env_with_variants(id='myoHandReorientID-v0',
            entry_point='myosuite.envs.myo.myobase.reorient_sar_v0:InDistribution',
            max_episode_steps=50,
            kwargs={
                'model_path': curr_dir+'/../assets/hand/myohand_sar.xml',
                'normalize_act': True,
                'frame_skip': 5,
            }
    )

# SAR TEST ENVIRONMENT: out of distribution
register_env_with_variants(id='myoHandReorientOOD-v0',
            entry_point='myosuite.envs.myo.myobase.reorient_sar_v0:OutofDistribution',
            max_episode_steps=50,
            kwargs={
                'model_path': curr_dir+'/../assets/hand/myohand_sar.xml',
                'normalize_act': True,
                'frame_skip': 5,
            }
    )