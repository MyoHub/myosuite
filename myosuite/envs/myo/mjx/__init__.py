
from typing import Any, Dict, Optional, Union, Callable
from ml_collections import config_dict
import copy
from etils import epath
from jax import numpy as jp
from mujoco_playground import registry
import mujoco

from myosuite.envs.myo.mjx.playground_pose_v0 import MjxPoseEnvV0
from myosuite.envs.myo.mjx.playground_reach_v0 import MjxReachEnvV0

pose_env_config = config_dict.create(
        ctrl_dt=0.02, # not used
        sim_dt=0.002, # not used
        # episode_length=100,
        reward_config=config_dict.create(
            angle_reward_weight=1,
            ctrl_cost_weight=1,
            pose_thd=0.35,
            bonus_weight=4
        ),
        target_jnt_range=config_dict.ConfigDict(),
        far_th=4*jp.pi/2,
        model_path=epath.Path('/tmp/dummy.xml')
    )

reach_env_config = config_dict.create(
        ctrl_dt=0.02, # not used
        sim_dt=0.002, # not used
        # episode_length=100,
        reward_weights=config_dict.create(
            reach=1.,
            bonus=4.,
            penalty=50.,
        ),
        target_reach_range=config_dict.ConfigDict(),
        far_th=0.35,
        model_path=epath.Path('/tmp/dummy.xml')
    )

def config_callable(env_config) -> Callable[[], config_dict.ConfigDict]:
    fn = lambda : env_config
    return fn

# Elbow posing ==============================
elbow_pose_fixed_env_config = copy.deepcopy(pose_env_config)
model_path='envs/myo/assets/elbow/'
model_filename='myoelbow_1dof6muscles.xml'
elbow_pose_fixed_env_config['model_path'] = epath.Path(epath.resource_path('myosuite')) / model_path / model_filename
elbow_pose_fixed_env_config['target_jnt_range'] = config_dict.create(
            r_elbow_flex=jp.array(((2), (2)))
        )
registry.manipulation.register_environment("MjxElbowPoseFixed-v0",
                                           MjxPoseEnvV0,
                                           config_callable(elbow_pose_fixed_env_config))

elbow_pose_random_env_config = copy.deepcopy(elbow_pose_fixed_env_config)
elbow_pose_random_env_config['target_jnt_range'] = config_dict.create(
            r_elbow_flex=jp.array(((0), (2.27)))
        )
registry.manipulation.register_environment("MjxElbowPoseRandom-v0",
                                           MjxPoseEnvV0,
                                           config_callable(elbow_pose_random_env_config))

# Finger joint posing ==============================
finger_pose_fixed_env_config = copy.deepcopy(pose_env_config)
model_path='simhive/myo_sim/finger/'
model_filename='myofinger_v0.xml'
finger_pose_fixed_env_config['model_path'] = epath.Path(epath.resource_path('myosuite')) / model_path / model_filename
finger_pose_fixed_env_config['target_jnt_range'] = config_dict.create(
            IFadb=jp.array(((0), (0))),
            IFmcp=jp.array(((0), (0))),
            IFpip=jp.array(((0.75), (0.75))),
            IFdip=jp.array(((0.75), (0.75))),
        )
registry.manipulation.register_environment("MjxFingerPoseFixed-v0",
                                           MjxPoseEnvV0,
                                           config_callable(finger_pose_fixed_env_config))

finger_pose_random_env_config = copy.deepcopy(finger_pose_fixed_env_config)
finger_pose_random_env_config['target_jnt_range'] = config_dict.create(
            IFadb=jp.array(((-.2), (.2))),
            IFmcp=jp.array(((-.4), (1))),
            IFpip=jp.array(((.1), (1))),
            IFdip=jp.array(((.1), (1))),
        )
registry.manipulation.register_environment("MjxFingerPoseRandom-v0",
                                           MjxPoseEnvV0,
                                           config_callable(finger_pose_random_env_config))

# Hand tips reaching ==============================
# class MjxHandReach(MjxReachEnvV0):
#     def __init__(
#             self,
#             config: config_dict.ConfigDict,
#             config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
#     ) -> None:
#         model_path='envs/myo/assets/hand/'
#         model_filename='myohand_pose.xml'
#         path = epath.Path(epath.resource_path('myosuite')) / model_path
#         super().__init__(path/model_filename, config, config_overrides)

#     def preprocess_spec(self, spec:mujoco.MjSpec):
#         spec = super().preprocess_spec(spec)
#         for s in spec.sites:
#             if "_target" in s.name:
#                 print(f"Deleted target site \"{s.name}\"")
#                 s.delete()
#         for t in spec.tendons:
#             if "_err" in t.name:
#                 print(f"Deleted error tendon \"{t.name}\"")
#                 t.delete()
#         # TODO: Verify visual geoms impact performance
#         for g in spec.geoms:
#             if not g.name or "floor" in g.name:
#                 print(f"Deleted visual geom")
#                 g.delete()
#         return spec
hand_reach_fixed_env_config = copy.deepcopy(reach_env_config)
model_path='envs/myo/assets/hand/'
model_filename='myohand_pose.xml'
hand_reach_fixed_env_config['model_path'] = epath.Path(epath.resource_path('myosuite')) / model_path / model_filename
hand_reach_fixed_env_config['far_th'] = 0.044
hand_reach_fixed_env_config['target_reach_range'] = config_dict.create(
            THtip=jp.array(((-0.165, -0.537, 1.495), (-0.165, -0.537, 1.495))),
            IFtip=jp.array(((-0.151, -0.547, 1.455), (-0.151, -0.547, 1.455))),
            MFtip=jp.array(((-0.146, -0.547, 1.447), (-0.146, -0.547, 1.447))),
            RFtip=jp.array(((-0.148, -0.543, 1.445), (-0.148, -0.543, 1.445))),
            LFtip=jp.array(((-0.148, -0.528, 1.434), (-0.148, -0.528, 1.434))),
        )
registry.manipulation.register_environment("MjxHandReachFixed-v0",
                                           MjxReachEnvV0,
                                           config_callable(hand_reach_fixed_env_config))

hand_reach_random_env_config = copy.deepcopy(hand_reach_fixed_env_config)
hand_reach_random_env_config['far_th'] = 0.034
hand_reach_random_env_config['target_reach_range'] = config_dict.create(
            THtip=jp.array(((-0.165-0.020, -0.537-0.040, 1.495-0.040), (-0.165+0.040, -0.537+0.020, 1.495+0.040))),
            IFtip=jp.array(((-0.151-0.040, -0.547-0.020, 1.455-0.010), (-0.151+0.040, -0.547+0.020, 1.455+0.010))),
            MFtip=jp.array(((-0.146-0.040, -0.547-0.020, 1.447-0.010), (-0.146+0.040, -0.547+0.020, 1.447+0.010))),
            RFtip=jp.array(((-0.148-0.040, -0.543-0.020, 1.445-0.010), (-0.148+0.040, -0.543+0.020, 1.445+0.010))),
            LFtip=jp.array(((-0.148-0.040, -0.528-0.020, 1.434-0.010), (-0.148+0.040, -0.528+0.020, 1.434+0.010))),
        )
registry.manipulation.register_environment("MjxHandReachRandom-v0",
                                           MjxReachEnvV0,
                                           config_callable(hand_reach_random_env_config))

env_list = ["MjxElbowPoseFixed-v0",
            "MjxElbowPoseRandom-v0",
            "MjxFingerPoseFixed-v0",
            "MjxFingerPoseRandom-v0",
            "MjxHandReachRandom-v0",
            "MjxHandReachFixed-v0"]

for env_name in env_list:
    try:
        env = registry.load(env_name)
        print(f"{env_name} loaded successfully")
    except:
        print(f"{env_name} failed to load")