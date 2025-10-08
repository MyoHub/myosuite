
from typing import Any, Dict, Optional, Union, Callable
from ml_collections import config_dict
import copy
from etils import epath
from jax import numpy as jp
import myo_registry as registry
from mujoco_playground._src import mjx_env
import mujoco

from myosuite.envs.myo.mjx.playground_pose_v0 import MjxPoseEnvV0
from myosuite.envs.myo.mjx.playground_reach_v0 import MjxReachEnvV0

pose_env_config = config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        num_envs=4096,
        reward_config=config_dict.create(
            angle_reward_weight=1.,
            ctrl_cost_weight=1.,
            pose_thd=0.35,
            far_th=4*jp.pi/2,
            bonus_weight=4.
        ),
        target_jnt_range=config_dict.ConfigDict(),
        max_episode_steps=100,
        model_path=epath.Path('/tmp/dummy.xml')
    )

reach_env_config = config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        num_envs=4096,
        reward_weights=config_dict.create(
            reach=1.,
            bonus=4.,
            penalty=50.,
        ),
        target_reach_range=config_dict.ConfigDict(),
        far_th=0.35,
        max_episode_steps=100,
        model_path=epath.Path('/tmp/dummy.xml')
    )

ppo_config = config_dict.create(
        num_timesteps=40_000_000,
        num_evals=16,
        reward_scaling=0.1,
        num_eval_envs=128,
        clipping_epsilon=0.3,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=10,
        num_minibatches=32,
        num_updates_per_batch=8,
        num_resets_per_eval=1,
        discounting=0.97,
        learning_rate=3e-4,
        entropy_cost=0.001,
        batch_size=512,
        max_grad_norm=1.0,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(64, 64, 64),
            value_hidden_layer_sizes=(64, 64, 64),
            policy_obs_key="state",
            value_obs_key="state",
        )
    )

# Elbow posing ==============================
elbow_pose_env_config = copy.deepcopy(pose_env_config)
model_path='envs/myo/assets/elbow/'
model_filename='myoelbow_1dof6muscles.xml'
elbow_pose_env_config['model_path'] = epath.Path(epath.resource_path('myosuite')) / model_path / model_filename

# Finger joint posing ==============================
finger_pose_env_config = copy.deepcopy(pose_env_config)
model_path='simhive/myo_sim/finger/'
model_filename='myofinger_v0.xml'
finger_pose_env_config['model_path'] = epath.Path(epath.resource_path('myosuite')) / model_path / model_filename

# Hand tips reaching ==============================
hand_reach_env_config = copy.deepcopy(reach_env_config)
model_path='envs/myo/assets/hand/'
model_filename='myohand_pose.xml'
hand_reach_env_config['model_path'] = epath.Path(epath.resource_path('myosuite')) / model_path / model_filename


def config_callable(env_config) -> Callable[[], config_dict.ConfigDict]:
    fn = lambda : env_config
    return fn


def get_default_config(env_name) -> config_dict.ConfigDict:
  return registry.get_default_config(env_name)


def make(env_name: str) -> mjx_env.MjxEnv:

    if "MjxElbowPose" in env_name:

        if env_name == "MjxElbowPoseFixed-v0":
            elbow_pose_env_config['target_jnt_range'] = config_dict.create(
                    r_elbow_flex=jp.array(((2), (2)))
                )
        elif env_name == "MjxElbowPoseRandom-v0":
            elbow_pose_env_config['target_jnt_range'] = config_dict.create(
                    r_elbow_flex=jp.array(((0), (2.27)))
                )
        registry.register_environment(env_name,
                                      MjxPoseEnvV0,
                                      config_callable(elbow_pose_env_config))
        env = registry.load(env_name)

        return env
    
    if "MjxFingerPose" in env_name:

        if env_name == "MjxFingerPoseFixed-v0":
            finger_pose_env_config['target_jnt_range'] = config_dict.create(
                IFadb=jp.array(((0), (0))),
                IFmcp=jp.array(((0), (0))),
                IFpip=jp.array(((0.75), (0.75))),
                IFdip=jp.array(((0.75), (0.75))),
            )
        elif env_name == "MjxFingerPoseRandom-v0":
            finger_pose_env_config['target_jnt_range'] = config_dict.create(
                IFadb=jp.array(((-.2), (.2))),
                IFmcp=jp.array(((-.4), (1))),
                IFpip=jp.array(((.1), (1))),
                IFdip=jp.array(((.1), (1))),
            )
        registry.register_environment(env_name,
                                      MjxPoseEnvV0,
                                      config_callable(finger_pose_env_config))
        env = registry.load(env_name)

        return env

    if "MjxHandReach" in env_name:

        if env_name == "MjxHandReachFixed-v0":
            hand_reach_env_config['far_th'] = 0.044
            hand_reach_env_config['target_reach_range'] = config_dict.create(
                        THtip=jp.array(((-0.165, -0.537, 1.495), (-0.165, -0.537, 1.495))),
                        IFtip=jp.array(((-0.151, -0.547, 1.455), (-0.151, -0.547, 1.455))),
                        MFtip=jp.array(((-0.146, -0.547, 1.447), (-0.146, -0.547, 1.447))),
                        RFtip=jp.array(((-0.148, -0.543, 1.445), (-0.148, -0.543, 1.445))),
                        LFtip=jp.array(((-0.148, -0.528, 1.434), (-0.148, -0.528, 1.434))),
                    )
        elif env_name == "MjxHandReachRandom-v0":
            hand_reach_env_config['far_th'] = 0.034
            hand_reach_env_config['target_reach_range'] = config_dict.create(
                        THtip=jp.array(((-0.165-0.020, -0.537-0.040, 1.495-0.040), (-0.165+0.040, -0.537+0.020, 1.495+0.040))),
                        IFtip=jp.array(((-0.151-0.040, -0.547-0.020, 1.455-0.010), (-0.151+0.040, -0.547+0.020, 1.455+0.010))),
                        MFtip=jp.array(((-0.146-0.040, -0.547-0.020, 1.447-0.010), (-0.146+0.040, -0.547+0.020, 1.447+0.010))),
                        RFtip=jp.array(((-0.148-0.040, -0.543-0.020, 1.445-0.010), (-0.148+0.040, -0.543+0.020, 1.445+0.010))),
                        LFtip=jp.array(((-0.148-0.040, -0.528-0.020, 1.434-0.010), (-0.148+0.040, -0.528+0.020, 1.434+0.010))),
                    )
        registry.register_environment(env_name,
                                      MjxReachEnvV0,
                                      config_callable(hand_reach_env_config))
        env = registry.load(env_name)

        return env

env_names = ["MjxElbowPoseFixed-v0",
             "MjxElbowPoseRandom-v0",
             "MjxFingerPoseFixed-v0",
             "MjxFingerPoseRandom-v0",
             "MjxHandReachRandom-v0",
             "MjxHandReachFixed-v0"]
