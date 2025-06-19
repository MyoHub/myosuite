from typing import Any, Dict, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground import State

from mujoco_playground._src import mjx_env  # Several helper functions are only visible under _src

import numpy as np

def default_config() -> config_dict.ConfigDict:
    env_config = config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=100,
        action_repeat=1,
        action_scale=0.5,
        history_len=1,
        healthy_angle_range=(0, 2.1),
        noise_config=config_dict.create(
            reset_noise_scale=1e-1,
        ),
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


class MjxPoseEnvV0(mjx_env.MjxEnv):
    def __init__(
            self,
            model_path,
            config: config_dict.ConfigDict = default_config(),
            config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
            far_th=4*jp.pi/2
    ) -> None:
        super().__init__(config, config_overrides)

        spec = mujoco.MjSpec.from_file(model_path.as_posix())
        spec = self.preprocess_spec(spec)
        self._mj_model = spec.compile()

        self._mj_model.geom_margin = np.zeros(self._mj_model.geom_margin.shape)
        print(f"All margins set to 0")

        self._mj_model.opt.timestep = self.sim_dt
        self._mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        self._mj_model.opt.iterations = 6
        self._mj_model.opt.ls_iterations = 6
        self._mj_model.opt.disableflags = self._mj_model.opt.disableflags | mjx.DisableBit.EULERDAMP

        self._mjx_model = mjx.put_model(self._mj_model)
        self._xml_path = model_path.as_posix()

        self.far_th = far_th

    def preprocess_spec(self, spec:mujoco.MjSpec):
        for geom in spec.geoms:
            if geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                geom.conaffinity = 0
                geom.contype = 0
                print(f"Disabled contacts for cylinder geom named \"{geom.name}\"")
        return spec

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        qpos = jax.random.uniform(
            rng1, (self.mjx_model.nq,),
            minval=self.mjx_model.jnt_range[:,0],
            maxval=self.mjx_model.jnt_range[:,1]
        )
        # TODO: Velocity initialization
        qvel = jp.zeros(self.mjx_model.nv)
        target_angle = jax.random.uniform(
            rng2, (self.mjx_model.nq,),
            minval=self._config.healthy_angle_range[0],
            maxval=self._config.healthy_angle_range[1]
        )

        # We store the target angle in the info, can't store it as an instance variable,
        # as it has to be determined in a parallelized manner
        info = {'rng': rng, 'target_angle': target_angle}

        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=jp.zeros((self.mjx_model.nu,)))

        obs = self._get_obs(data, info)
        reward, done, zero = jp.zeros(3)
        metrics = {
            'pose_reward': zero,
            'act_reg_reward': zero,
            'bonus_reward': zero,
            'penalty_reward': zero,
        }
        return State(data, {"state": obs}, reward, done, metrics, info)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""

        norm_action = 1.0/(1.0+jp.exp(-5.0*(action-0.5))) 

        data = mjx_env.step(self.mjx_model, state.data, norm_action)

        pose_dist = jp.linalg.norm(state.info['target_angle'] - data.qpos, axis=-1)
        act_mag = jp.linalg.norm(data.act, axis=-1)

        far_th = 4*jp.pi/2

        pose = pose_dist * -self._config.reward_config.angle_reward_weight
        act_reg = act_mag * -self._config.reward_config.ctrl_cost_weight
        bonus = (jp.where(pose_dist<self._config.reward_config.pose_thd, 1, 0)
                 + jp.where(pose_dist<self._config.reward_config.pose_thd*1.5, 1, 0)) * self._config.reward_config.bonus_weight
        penalty = -1.*(pose_dist>far_th)

        obs = self._get_obs(data, state.info)
        reward = pose + act_reg + bonus + penalty
        done = pose_dist>far_th

        state.metrics.update(
            pose_reward=pose,
            act_reg_reward=act_reg,
            bonus_reward=bonus,
            penalty_reward=penalty,
        )

        return state.replace(
            data=data, obs={"state": obs}, reward=reward, done=done
        )

    def _get_obs(
            self, data: mjx.Data, info) -> jp.ndarray:
        """Observe qpos, qvel, act and qpos_err."""
        return jp.concatenate([
            data.qpos,
            data.qvel*self.mjx_model.opt.timestep,
            data.act,
            info['target_angle']-data.qpos
        ])

    # Accessors.
    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
