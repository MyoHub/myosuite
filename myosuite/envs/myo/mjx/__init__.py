from typing import Callable
from ml_collections import config_dict
import copy
from etils import epath
from jax import numpy as jp
import myosuite.envs.myo.mjx.myo_registry as registry
from mujoco_playground._src import mjx_env

from myosuite.envs.myo.mjx.playground_pose_v0 import MjxPoseEnvV0
from myosuite.envs.myo.mjx.playground_reach_v0 import MjxReachEnvV0

base_config = config_dict.create(
    ctrl_dt=0.02,
    sim_dt=0.002,
    num_envs=4_096,
    max_episode_steps=100,
    model_path=epath.Path("/tmp/dummy.xml"),
    impl="jax",
    norm_actions=True,
)

pose_env_config = config_dict.ConfigDict({**base_config, **config_dict.create(
    reward_config=config_dict.create(
        angle_reward_weight=1.0,
        ctrl_cost_weight=1.0,
        pose_thd=0.35,
        far_th=4 * jp.pi / 2,
        bonus_weight=4.0,
    ),
    target_jnt_range=config_dict.ConfigDict(),
)})

reach_env_config = config_dict.ConfigDict({**base_config, **config_dict.create(
    reward_config=config_dict.create(
        reach_weight=1.0,
        bonus_scale=4.0,
        penalty_scale=50.0,
    ),
    target_reach_range=config_dict.ConfigDict(),
    far_th=0.35,
)})

ppo_config = config_dict.create(
    num_timesteps=50_000_000,
    learning_rate=3e-4,
    discounting=0.97,
    gae_lambda=0.95,
    entropy_cost=0.001,
    clipping_epsilon=0.3,
    max_grad_norm=1.0,
    action_repeat=1,
    num_minibatches=32,
    num_updates_per_batch=8,
    batch_size=256,
    unroll_length=10,
    reward_scaling=1.0,
    normalize_observations=True,
    num_evals=16,
    num_eval_envs=128,
    num_resets_per_eval=1,
    network_factory=config_dict.create(
        policy_hidden_layer_sizes=(64, 64, 64),
        value_hidden_layer_sizes=(64, 64, 64),
        policy_obs_key="state",
        value_obs_key="state",
    ),
)

# Elbow posing ==============================
elbow_pose_env_config = copy.deepcopy(pose_env_config)
model_path = "envs/myo/assets/elbow/"
model_filename = "myoelbow_1dof6muscles.xml"
elbow_pose_env_config["model_path"] = (
    epath.Path(epath.resource_path("myosuite")) / model_path / model_filename
)

# Finger joint posing ==============================
finger_pose_env_config = copy.deepcopy(pose_env_config)
model_path = "simhive/myo_sim/finger/"
model_filename = "myofinger_v0.xml"
finger_pose_env_config["model_path"] = (
    epath.Path(epath.resource_path("myosuite")) / model_path / model_filename
)

# Hand tips reaching ==============================
hand_reach_env_config = copy.deepcopy(reach_env_config)
model_path = "envs/myo/assets/hand/"
model_filename = "myohand_pose.xml"
hand_reach_env_config["model_path"] = (
    epath.Path(epath.resource_path("myosuite")) / model_path / model_filename
)


def wrap_class(wrapper_cls, wrapped_env_cls, wrapper_config=None):
    def _get_wrapped_class(*args, **kwargs):
        return wrapper_cls(wrapped_env_cls(*args, **kwargs), **(wrapper_config if wrapper_config is not None else {}))
    return _get_wrapped_class


def config_callable(env_config) -> Callable[[], config_dict.ConfigDict]:
    fn = lambda: env_config
    return fn


def get_default_config(env_name) -> config_dict.ConfigDict:
    return registry.get_default_config(env_name)

# TODO: is there a reason these are not registered on import?
def make(env_name: str, config_overrides=None) -> mjx_env.MjxEnv:

    env_name_base = registry.get_base_env_name(env_name)
    if "MjxElbowPose" in env_name_base:

        if env_name_base == "MjxElbowPoseFixed-v0":
            elbow_pose_env_config["target_jnt_range"] = config_dict.create(
                    r_elbow_flex=jp.array(((2), (2)))
                )
        elif env_name_base == "MjxElbowPoseRandom-v0":
            elbow_pose_env_config["target_jnt_range"] = config_dict.create(
                    r_elbow_flex=jp.array(((0), (2.27)))
                )
        registry.register_environment_with_variants(env_name_base,
                                      MjxPoseEnvV0,
                                      config_callable(elbow_pose_env_config))
        env = registry.load(env_name, config_overrides=config_overrides)

        return env

    if "MjxFingerPose" in env_name_base:

        if env_name_base == "MjxFingerPoseFixed-v0":
            finger_pose_env_config["target_jnt_range"] = config_dict.create(
                IFadb=jp.array(((0), (0))),
                IFmcp=jp.array(((0), (0))),
                IFpip=jp.array(((0.75), (0.75))),
                IFdip=jp.array(((0.75), (0.75))),
            )
        elif env_name_base == "MjxFingerPoseRandom-v0":
            finger_pose_env_config["target_jnt_range"] = config_dict.create(
                IFadb=jp.array(((-0.2), (0.2))),
                IFmcp=jp.array(((-0.4), (1))),
                IFpip=jp.array(((0.1), (1))),
                IFdip=jp.array(((0.1), (1))),
            )
        registry.register_environment_with_variants(env_name_base,
                                      MjxPoseEnvV0,
                                      config_callable(finger_pose_env_config))
        env = registry.load(env_name, config_overrides=config_overrides)

        return env

    if "MjxHandReach" in env_name_base:

        if env_name_base == "MjxHandReachFixed-v0":
            hand_reach_env_config["far_th"] = 0.044
            hand_reach_env_config["target_reach_range"] = config_dict.create(
                THtip=jp.array(((-0.165, -0.537, 1.495), (-0.165, -0.537, 1.495))),
                IFtip=jp.array(((-0.151, -0.547, 1.455), (-0.151, -0.547, 1.455))),
                MFtip=jp.array(((-0.146, -0.547, 1.447), (-0.146, -0.547, 1.447))),
                RFtip=jp.array(((-0.148, -0.543, 1.445), (-0.148, -0.543, 1.445))),
                LFtip=jp.array(((-0.148, -0.528, 1.434), (-0.148, -0.528, 1.434))),
            )
        elif env_name_base == "MjxHandReachRandom-v0":
            hand_reach_env_config["far_th"] = 0.034
            hand_reach_env_config["target_reach_range"] = config_dict.create(
                THtip=jp.array(
                    (
                        (-0.165 - 0.020, -0.537 - 0.040, 1.495 - 0.040),
                        (-0.165 + 0.040, -0.537 + 0.020, 1.495 + 0.040),
                    )
                ),
                IFtip=jp.array(
                    (
                        (-0.151 - 0.040, -0.547 - 0.020, 1.455 - 0.010),
                        (-0.151 + 0.040, -0.547 + 0.020, 1.455 + 0.010),
                    )
                ),
                MFtip=jp.array(
                    (
                        (-0.146 - 0.040, -0.547 - 0.020, 1.447 - 0.010),
                        (-0.146 + 0.040, -0.547 + 0.020, 1.447 + 0.010),
                    )
                ),
                RFtip=jp.array(
                    (
                        (-0.148 - 0.040, -0.543 - 0.020, 1.445 - 0.010),
                        (-0.148 + 0.040, -0.543 + 0.020, 1.445 + 0.010),
                    )
                ),
                LFtip=jp.array(
                    (
                        (-0.148 - 0.040, -0.528 - 0.020, 1.434 - 0.010),
                        (-0.148 + 0.040, -0.528 + 0.020, 1.434 + 0.010),
                    )
                ),
            )
        registry.register_environment(
            env_name, MjxReachEnvV0, config_callable(hand_reach_env_config)
        )
        env = registry.load(env_name, config_overrides=config_overrides)

        return env


env_names = [
    "MjxElbowPoseFixed-v0",
    "MjxElbowPoseRandom-v0",
    "MjxFingerPoseFixed-v0",
    "MjxFingerPoseRandom-v0",
    "MjxHandReachRandom-v0",
    "MjxHandReachFixed-v0",
]
