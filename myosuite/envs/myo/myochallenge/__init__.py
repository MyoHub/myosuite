from gym.envs.registration import register

import os
curr_dir = os.path.dirname(os.path.abspath(__file__))
import numpy as np

# MyoChallenge Die: Trial env
register(id='myoChallengeDieReorientDemo-v0',
        entry_point='myosuite.envs.myo.myochallenge.reorient_v0:ReorientEnvV0',
        max_episode_steps=150,
        kwargs={
            'model_path': curr_dir+'/../assets/hand/myo_hand_die.mjb',
            'normalize_act': True,
            'frame_skip': 5,
            'pos_th': np.inf,           # ignore position error threshold
            'goal_pos': (0, 0),         # 0 cm
            'goal_rot': (-.785, .785)   # +-45 degrees
        }
    )
# MyoChallenge Die: Phase1 env
register(id='myoChallengeDieReorientP1-v0',
        entry_point='myosuite.envs.myo.myochallenge.reorient_v0:ReorientEnvV0',
        max_episode_steps=150,
        kwargs={
            'model_path': curr_dir+'/../assets/hand/myo_hand_die.mjb',
            'normalize_act': True,
            'frame_skip': 5,
            'goal_pos': (-.010, .010),  # +- 1 cm
            'goal_rot': (-1.57, 1.57)   # +-90 degrees
        }
    )
# MyoChallenge Die: Phase2 env
register(id='myoChallengeDieReorientP2-v0',
        entry_point='myosuite.envs.myo.myochallenge.reorient_v0:ReorientEnvV0',
        max_episode_steps=150,
        kwargs={
            'model_path': curr_dir+'/../assets/hand/myo_hand_die.mjb',
            'normalize_act': True,
            'frame_skip': 5,
            # Randomization in goals
            'goal_pos': (-.020, .020),      # +- 2 cm
            'goal_rot': (-3.14, 3.14),      # +-180 degrees
            # Randomization in physical properties of the die
            'obj_size_change': 0.007,       # +-7mm delta change in object size
            'obj_mass_range': (0.050, 0.250),# 50gms to 250 gms
            'obj_friction_change': (0.2, 0.001, 0.00002) # nominal: 1.0, 0.005, 0.0001
        }
    )

# MyoChallenge Baoding: Phase1 env
register(id='myoChallengeBaodingP1-v1',
        entry_point='myosuite.envs.myo.myochallenge.baoding_v1:BaodingEnvV1',
        max_episode_steps=200,
        kwargs={
            'model_path': curr_dir+'/../assets/hand/myo_hand_baoding.mjb',
            'normalize_act': True,
            'goal_time_period': (5, 5),
            'goal_xrange': (0.025, 0.025),
            'goal_yrange': (0.028, 0.028),
        }
    )

# MyoChallenge Baoding: Phase1 env
register(id='myoChallengeBaodingP2-v1',
        entry_point='myosuite.envs.myo.myochallenge.baoding_v1:BaodingEnvV1',
        max_episode_steps=200,
        kwargs={
            'model_path': curr_dir+'/../assets/hand/myo_hand_baoding.mjb',
            'normalize_act': True,
            'goal_time_period': (4, 6),
            'goal_xrange': (0.020, 0.030),
            'goal_yrange': (0.022, 0.032),
            # Randomization in physical properties of the baoding balls
            'obj_size_range': (0.018, 0.024),       # Object size range. Nominal 0.022
            'obj_mass_range': (0.030, 0.300),       # Object weight range. Nominal 43 gms
            'obj_friction_change': (0.2, 0.001, 0.00002), # nominal: 1.0, 0.005, 0.0001
            'task_choice': 'random'
        }
    )