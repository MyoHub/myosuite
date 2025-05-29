from datetime import datetime
from typing import Any, Dict, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground import State

from mujoco_playground._src import mjx_env  # Several helper functions are only visible under _src

from myosuite.envs.myo.mjx.playground_pose_v0 import MjxPoseEnvV0


def default_config() -> config_dict.ConfigDict:
    env_config = config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=100,
        action_repeat=1,
        action_scale=0.5,
        history_len=1,
        healthy_angle_range=(0, 2.1),
        reward_config=config_dict.create(
            angle_reward_weight=1,
            ctrl_cost_weight=1,
            pose_thd=0.35,
            bonus_weight=4
        )
    )

    rl_config = config_dict.create(
        num_timesteps=40_000_000,
        num_evals=16,
        reward_scaling=0.1,
        episode_length=env_config.episode_length,
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
        num_envs=8192,
        batch_size=512,
        max_grad_norm=1.0,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(50, 50, 50),
            value_hidden_layer_sizes=(50, 50, 50),
            policy_obs_key="state",
            value_obs_key="state",
        )
    )
    env_config["ppo_config"] = rl_config
    return env_config


# TODO: Consider just registering environment variant if obs/action space is the same
class MjxElbow(MjxPoseEnvV0):
    def __init__(
            self,
            config: config_dict.ConfigDict = default_config(),
            config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        model_path='envs/myo/assets/elbow/'
        model_filename='myoelbow_1dof6muscles.xml'
        path = epath.Path(epath.resource_path('myosuite')) / model_path
        super().__init__(path/model_filename, config, config_overrides)
