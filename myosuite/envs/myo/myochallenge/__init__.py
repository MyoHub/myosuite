from myosuite.utils import gym; register=gym.register
import os
curr_dir = os.path.dirname(os.path.abspath(__file__))
import numpy as np
from myosuite.envs.env_variants import register_env_variant

# utility to register envs with all muscle conditions
def register_env_with_variants(id, entry_point, max_episode_steps, kwargs):
    # register_env_with_variants base env
    register(
        id=id,
        entry_point=entry_point,
        max_episode_steps=max_episode_steps,
        kwargs=kwargs
    )
    #register variants env with sarcopenia  #TODO: needs to be tested with myochallenge models
    if id[:3] == "myo":
        register_env_variant(
            env_id=id,
            variants={'muscle_condition':'sarcopenia'},
            variant_id=id[:3]+"Sarc"+id[3:],
            silent=True
        )
    #register variants with fatigue  #TODO: needs to be tested with myochallenge models
    if id[:3] == "myo":
        register_env_variant(
            env_id=id,
            variants={'muscle_condition':'fatigue'},
            variant_id=id[:3]+"Fati"+id[3:],
            silent=True
        )

# ============================================== MyoChallenge 2025 envs ==============================================
## MyoChallenge Locomotion P1 (Soccer)
register_env_with_variants(id='myoChallengeSoccerP1-v0',
        entry_point='myosuite.envs.myo.myochallenge.soccer_v0:SoccerEnvV0',
        max_episode_steps=2000,
        kwargs={
            'model_path': curr_dir+'/../assets/leg_soccer/myolegs_soccer.xml',
            'normalize_act': True,
            'min_agent_spawn_distance': 1,
            'reset_type': 'random', # none, init, random
            'goalkeeper_probabilities': (1, 0),
        }
    )

register_env_with_variants(id='myoChallengePingpong-v0',
        entry_point='myosuite.envs.myo.myochallenge.pingpong_v0:PingPongEnvV0',
        max_episode_steps=1000,
        kwargs={
            'model_path': curr_dir + '/../assets/arm/myoarm_tabletennis.xml',
            'normalize_act': True,
            'frame_skip': 5,
        }
    )

register_env_with_variants(id='myoChallengeBimanual-v0',
        entry_point='myosuite.envs.myo.myochallenge.bimanual_v0:BimanualEnvV1',
        max_episode_steps=1000,
        kwargs={
            'model_path': curr_dir + '/../assets/arm/myoarm_bionic_bimanual.xml',
            'normalize_act': True,
            'frame_skip': 5,
            'obj_scale_change': [0.1, 0.05, 0.1],  # 10%, 5%, 10% scale variations in respective geom directions
            'obj_mass_change': (-0.050, 0.050),  # +-50gms
            'obj_friction_change': (0.1, 0.001, 0.00002)  # nominal: 1.0, 0.005, 0.0001
        }
    )

# MyoChallenge 2024 envs ==============================================
register_env_with_variants(id='myoChallengeOslRunFixed-v0',
        entry_point='myosuite.envs.myo.myochallenge.run_track_v0:RunTrack',
        max_episode_steps=1000,
        kwargs={
            'model_path': curr_dir+'/../assets/leg/myoosl_runtrack.xml',
            'normalize_act': True,
            'reset_type': 'random',  # none, init, random, osl_init
            'terrain': 'flat',  # flat, random, random_mixed
            'hills_difficulties': (0.0, 0.1, 0.0, 0.5, 0.0, 0.8, 0.0, 1.0),
            'rough_difficulties': (0.0, 0.1, 0.0, 0.15, 0.0, 0.2, 0.0, 0.3),
            'stairs_difficulties': (0.0, 0.05, 0.0, 0.1, 0.0, 0.2, 0.0, 0.3),
            'end_pos': -15,
            'frame_skip': 5,
            'start_pos': 14,
            'init_pose_path': curr_dir+'/../assets/leg/sample_gait_cycle.csv',
            'max_episode_steps': 1000
        }
    )


register_env_with_variants(id='myoChallengeOslRunRandom-v0',
        entry_point='myosuite.envs.myo.myochallenge.run_track_v0:RunTrack',
        max_episode_steps=60000,
        kwargs={
            'model_path': curr_dir+'/../assets/leg/myoosl_runtrack.xml',
            'normalize_act': True,
            'reset_type': 'random',  # none, init, random, osl_init
            'terrain': 'random',  # flat, random, random_mixed
            'hills_difficulties': (0.0, 0.0, 0.0, 0.0, 0.0, 0.03, 0.0, 0.06, 0.0, 0.09, 0.0, 0.12, 0.0, 0.15, 0.0, 0.18, 0.0, 0.21, 0.0, 0.24, 0.0, 0.0, 0.0, 0.0),
            'rough_difficulties': (0.0, 0.0, 0.0, 0.0, 0.0, 0.03, 0.0, 0.06, 0.0, 0.09, 0.0, 0.12, 0.0, 0.15, 0.0, 0.18, 0.0, 0.21, 0.0, 0.24, 0.0, 0.0, 0.0, 0.0),
            'stairs_difficulties': (0.0, 0.0, 0.0, 0.0, 0.0, 0.03, 0.0, 0.06, 0.0, 0.09, 0.0, 0.12, 0.0, 0.15, 0.0, 0.18, 0.0, 0.21, 0.0, 0.24, 0.0, 0.0, 0.0, 0.0),
            'end_pos': -45, # -35
            'frame_skip': 5, # 5
            'start_pos': 58, # 32
            'init_pose_path': curr_dir+'/../assets/leg/sample_gait_cycle.csv',
            'max_episode_steps':60000 # Duplicate this in the environment itself
        }
    )

# MyoChallenge 2023 envs ==============================================
# MyoChallenge Manipulation P1
register_env_with_variants(id='myoChallengeRelocateP1-v0',
        entry_point='myosuite.envs.myo.myochallenge.relocate_v0:RelocateEnvV0',
        max_episode_steps=150,
        kwargs={
            'model_path': curr_dir+'/../assets/arm/myoarm_relocate.xml',
            'normalize_act': True,
            'frame_skip': 5,
            'pos_th': 0.1,              # cover entire base of the receptacle
            'rot_th': np.inf,           # ignore rotation errors
            'target_xyz_range': {'high':[0.2, -.1, 0.9], 'low':[0.0, -.35, 0.9]},
            'target_rxryrz_range': {'high':[0.0, 0.0, 0.0], 'low':[0.0, 0.0, 0.0]}
        }
    )

# MyoChallenge Manipulation P2
register_env_with_variants(id='myoChallengeRelocateP2-v0',
        entry_point='myosuite.envs.myo.myochallenge.relocate_v0:RelocateEnvV0',
        max_episode_steps=150,
        kwargs={
            'model_path': curr_dir+'/../assets/arm/myoarm_relocate.xml',
            'normalize_act': True,
            'frame_skip': 5,
            'pos_th': 0.1,              # cover entire base of the receptacle
            'rot_th': np.inf,           # ignore rotation errors
            'qpos_noise_range':0.01,    # jnt initialization range
            'target_xyz_range': {'high':[0.3, -.1, 1.05], 'low':[0.0, -.45, 0.9]},
            'target_rxryrz_range': {'high':[0.2, 0.2, 0.2], 'low':[-.2, -.2, -.2]},
            'obj_xyz_range': {'high':[0.1, -.15, 1.0], 'low':[-0.1, -.35, 1.0]},
            'obj_geom_range': {'high':[.025, .025, .025], 'low':[.015, 0.015, 0.015]},
            'obj_mass_range': {'high':0.200, 'low':0.050},# 50gms to 200 gms
            'obj_friction_range': {'high':[1.2, 0.006, 0.00012], 'low':[0.8, 0.004, 0.00008]}
        }
    )

# Register MyoChallenge Manipulation P2 Evals
register_env_with_variants(id='myoChallengeRelocateP2eval-v0',
    entry_point='myosuite.envs.myo.myochallenge.relocate_v0:RelocateEnvV0',
    max_episode_steps=150,
    kwargs={
        'model_path': curr_dir + '/../assets/arm/myoarm_relocate.xml',
        'normalize_act': True,
        'frame_skip': 5,
        'pos_th': 0.1,              # cover entire base of the receptacle
        'rot_th': np.inf,           # ignore rotation errors
        'qpos_noise_range':0.015,    # jnt initialization range
        'target_xyz_range': {'high':[0.4, -.1, 1.1], 'low':[-.5, -.5, .9]},
        'target_rxryrz_range': {'high':[.3, .3, .3], 'low':[-.3, -.3, -.3]},
        'obj_xyz_range': {'high':[0.15, -.10, 1.0], 'low':[-0.20, -.40, 1.0]},
        'obj_geom_range': {'high':[.025, .025, .035], 'low':[.015, 0.015, 0.015]},
        'obj_mass_range': {'high':0.300, 'low':0.050},# 50gms to 250 gms
        'obj_friction_range': {'high':[1.2, 0.006, 0.00012], 'low':[0.8, 0.004, 0.00008]}
    }
)


## MyoChallenge Locomotion P1
register_env_with_variants(id='myoChallengeChaseTagP1-v0',
        entry_point='myosuite.envs.myo.myochallenge.chasetag_v0:ChaseTagEnvV0',
        max_episode_steps=2000,
        kwargs={
            'model_path': curr_dir+'/../assets/leg/myolegs_chasetag.xml',
            'normalize_act': True,
            'win_distance': 0.5,
            'min_spawn_distance': 2,
            'reset_type': 'init', # none, init, random
            'terrain': 'FLAT', # FLAT, random
            'task_choice': 'CHASE', # CHASE, EVADE, random
            'hills_range': (0.0, 0.0),
            'rough_range': (0.0, 0.0),
            'relief_range': (0.0, 0.0),
            'opponent_probabilities': (0.1, 0.45, 0.45),
        }
    )


# MyoChallenge Locomotion P2
register_env_with_variants(id='myoChallengeChaseTagP2-v0',
        entry_point='myosuite.envs.myo.myochallenge.chasetag_v0:ChaseTagEnvV0',
        max_episode_steps=2000,
        kwargs={
            'model_path': curr_dir+'/../assets/leg/myolegs_chasetag.xml',
            'normalize_act': True,
            'win_distance': 0.5,
            'min_spawn_distance': 2,
            'reset_type': 'random',  # none, init, random
            'terrain': 'random',  # FLAT, random
            'task_choice': 'random',  # CHASE, EVADE, random
            'hills_range': (0.03, 0.23),
            'rough_range': (0.05, 0.1),
            'relief_range': (0.1, 0.3),
            'repeller_opponent': False,
            'chase_vel_range': (1.0, 1.0),
            'random_vel_range': (-2, 2),
            'opponent_probabilities': (0.1, 0.45, 0.45),
        }
    )

# Register MyoChallenge Locomotion P2 Evals
register_env_with_variants(id='myoChallengeChaseTagP2eval-v0',
        entry_point='myosuite.envs.myo.myochallenge.chasetag_v0:ChaseTagEnvV0',
        max_episode_steps=2000,
        kwargs={
            'model_path': curr_dir+'/../assets/leg/myolegs_chasetag.xml',
            'normalize_act': True,
            'win_distance': 0.5,
            'min_spawn_distance': 2,
            'reset_type': 'random',  # none, init, random
            'terrain': 'random',  # FLAT, random
            'task_choice': 'random',  # CHASE, EVADE, random
            'hills_range': (0.03, 0.23),
            'rough_range': (0.05, 0.1),
            'relief_range': (0.1, 0.3),
            'repeller_opponent': True,
            'chase_vel_range': (1, 5),
            'random_vel_range': (-2, 2),
            'repeller_vel_range': (0.3, 1),
            'opponent_probabilities': (0.1, 0.35, 0.35, 0.2),
        }
    )

# MyoChallenge 2022 envs ==============================================
# MyoChallenge Die: Trial env
register_env_with_variants(id='myoChallengeDieReorientDemo-v0',
        entry_point='myosuite.envs.myo.myochallenge.reorient_v0:ReorientEnvV0',
        max_episode_steps=150,
        kwargs={
            'model_path': curr_dir+'/../assets/hand/myohand_die.xml',
            'normalize_act': True,
            'frame_skip': 5,
            'pos_th': np.inf,           # ignore position error threshold
            'goal_pos': (0, 0),         # 0 cm
            'goal_rot': (-.785, .785)   # +-45 degrees
        }
    )
# MyoChallenge Die: Phase1 env
register_env_with_variants(id='myoChallengeDieReorientP1-v0',
        entry_point='myosuite.envs.myo.myochallenge.reorient_v0:ReorientEnvV0',
        max_episode_steps=150,
        kwargs={
            'model_path': curr_dir+'/../assets/hand/myohand_die.xml',
            'normalize_act': True,
            'frame_skip': 5,
            'goal_pos': (-.010, .010),  # +- 1 cm
            'goal_rot': (-1.57, 1.57)   # +-90 degrees
        }
    )
# MyoChallenge Die: Phase2 env
register_env_with_variants(id='myoChallengeDieReorientP2-v0',
        entry_point='myosuite.envs.myo.myochallenge.reorient_v0:ReorientEnvV0',
        max_episode_steps=150,
        kwargs={
            'model_path': curr_dir+'/../assets/hand/myohand_die.xml',
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
register_env_with_variants(id='myoChallengeBaodingP1-v1',
        entry_point='myosuite.envs.myo.myochallenge.baoding_v1:BaodingEnvV1',
        max_episode_steps=200,
        kwargs={
            'model_path': curr_dir+'/../assets/hand/myohand_baoding.xml',
            'normalize_act': True,
            'goal_time_period': (5, 5),
            'goal_xrange': (0.025, 0.025),
            'goal_yrange': (0.028, 0.028),
        }
    )

# MyoChallenge Baoding: Phase1 env
register_env_with_variants(id='myoChallengeBaodingP2-v1',
        entry_point='myosuite.envs.myo.myochallenge.baoding_v1:BaodingEnvV1',
        max_episode_steps=200,
        kwargs={
            'model_path': curr_dir+'/../assets/hand/myohand_baoding.xml',
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
