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


class PlaygroundElbow(mjx_env.MjxEnv):
    def __init__(
            self,
            config: config_dict.ConfigDict = default_config(),
            config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
            is_msk=True
    ) -> None:
        super().__init__(config, config_overrides)
        xml_path = rf"../assets/elbow/myoelbow_1dof{6 if is_msk else 0}muscles_mjx.xml"
        self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self._mj_model.opt.timestep = self.sim_dt

        self._mjx_model = mjx.put_model(self._mj_model)
        self._xml_path = xml_path

        self._mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        self._mj_model.opt.iterations = 6
        self._mj_model.opt.ls_iterations = 6
        self._mj_model.opt.disableflags = self._mj_model.opt.disableflags | mjx.DisableBit.EULERDAMP

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

        low, hi = -self._config.noise_config.reset_noise_scale, self._config.noise_config.reset_noise_scale
        qpos = self.mjx_model.qpos0 + jax.random.uniform(
            rng1, (self.mjx_model.nq,), minval=self.mjx_model.jnt_range[:,0], maxval=self.mjx_model.jnt_range[:,1]
        )
        qvel = jp.array([0.0])
        target_angle = jax.random.uniform(
            rng3, (1,), minval=self._config.healthy_angle_range[0], maxval=self._config.healthy_angle_range[1]
        )

        # We store the target angle in the info, can't store it as an instance variable,
        # as it has to be determined in a parallelized manner
        info = {'rng': rng, 'target_angle': target_angle}

        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=jp.zeros((self.mjx_model.nu,)))

        obs = self._get_obs(data, jp.zeros(self.mjx_model.nu), info)
        reward, done, zero = jp.zeros(3)
        metrics = {
            'angle_reward': zero,
            'reward_quadctrl': zero,
        }
        return State(data, {"state": obs}, reward, done, metrics, info)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        data0 = state.data
        data = mjx_env.step(self.mjx_model, data0, action)

        angle_error = jp.abs(state.info['target_angle'][0] - data.qpos[0])
        # Smooth fall-off on angle reward. Exp is too costly normally,
        # should replace it later on.
        angle_reward = -self._config.reward_config.angle_reward_weight * angle_error
        ctrl_cost = self._config.reward_config.ctrl_cost_weight * jp.sqrt(jp.sum(jp.square(action)))
        bonus = (jp.where(angle_error<self._config.reward_config.pose_thd, 1, 0)
                 + jp.where(angle_error<self._config.reward_config.pose_thd*1.5, 1, 0)) * self._config.reward_config.bonus_weight

        obs = self._get_obs(data, action, state.info)
        reward = angle_reward - ctrl_cost + bonus
        done = 0.0
        state.metrics.update(
            angle_reward=angle_reward,
            reward_quadctrl=-ctrl_cost,
        )

        return state.replace(
            data=data, obs={"state": obs}, reward=reward, done=done
        )

    def _get_obs(
            self, data: mjx.Data, action: jp.ndarray, info
    ) -> jp.ndarray:
        """Observes elbow angle, velocities, and last applied torque."""
        position = data.qpos

        # external_contact_forces are excluded
        return jp.concatenate([
            jp.array([data.time]),
            position,
            data.qvel*self.mjx_model.opt.timestep,
            data.act,
            info['target_angle']-position
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
