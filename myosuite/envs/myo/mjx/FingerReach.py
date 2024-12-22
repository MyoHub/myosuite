from datetime import datetime
from etils import epath
import functools
from typing import Any, Dict, Sequence, Tuple, Union

import jax
from jax import numpy as jp
import numpy as np
from matplotlib import pyplot as plt
from jax import nn

import mujoco
from mujoco import mjx

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.sac import train as sac
from brax.io import html, mjcf, model

# from jax import config
# config.update('jax_disable_jit', True)
# config.update("jax_debug_nans", True)

class FingerReachEnvV0(PipelineEnv):

  # https://github.com/MyoHub/myo_sim/blob/da2346336e1f2c5d2ee745fa4c908d1bea632b63/finger/finger_v0.xml

  def __init__(
      self,
      reset_noise_scale=1e-2,
      far_th=.35,
      model_path='simhive/myo_sim/finger',
      model_filename='myofinger_v0.xml',
      target_reach_range={'IFtip': ((0.2, 0.05, 0.2), (0.2, 0.05, 0.2)),},
      **kwargs,
  ):
  
    path = epath.Path(epath.resource_path('myosuite')) / (model_path)
    mj_model = mujoco.MjModel.from_xml_path((path / model_filename).as_posix())

    mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
    mj_model.opt.iterations = 6
    mj_model.opt.ls_iterations = 6

    sys = mjcf.load_model(mj_model)

    physics_steps_per_control_step = 10
    kwargs['n_frames'] = kwargs.get(
        'n_frames', physics_steps_per_control_step)
    kwargs['backend'] = 'mjx'

    super().__init__(sys, **kwargs)

    self.obs_keys = ['qpos', 'qvel', 'tip_pos', 'reach_err']
    self.rwd_keys_wt = {
        "reach": 1.0,
        "bonus": 4.0,
        "penalty": 50,
    }
  
    self._reset_noise_scale = reset_noise_scale
    self._far_th = far_th
    self._target_reach_range = target_reach_range
    self._tip_sids = []
    sites = target_reach_range.keys()
    if sites:
      for site in sites:
        self._tip_sids.append(mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, site))

  def reset(self, rng: jp.ndarray) -> State:
    """Resets the environment to an initial state."""

    rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

    info = {}
    for site, span in self._target_reach_range.items():
      target = jax.random.uniform(rng1, (3,), minval=jp.array(span[0]), maxval=jp.array(span[1]))
      info[site + '_target'] = target

    low, hi = -self._reset_noise_scale, self._reset_noise_scale
    qpos0 = self.sys.qpos0 + jax.random.uniform(
        rng2, (self.sys.nq,), minval=low, maxval=hi
    )
    qvel0 = jax.random.uniform(
        rng3, (self.sys.nv,), minval=low, maxval=hi
    )

    data = self.pipeline_init(qpos0, qvel0)

    obs_dict = self._get_obs_dict(data, info)
    obs = self._get_obs(obs_dict)

    reward, done, zero = jp.zeros(3)

    metrics = {
        'reach': zero,
        'bonus': zero,
        'penalty': zero,
        'solved': zero,
    }

    return State(data, obs, reward, done, metrics, info)

  def step(self, state: State, action: jp.ndarray) -> State:
    """Runs one timestep of the environment's dynamics."""
    data0 = state.pipeline_state

    norm_action = nn.sigmoid(5.0 * (action - 0.5))

    data = self.pipeline_step(data0, norm_action)

    obs_dict = self._get_obs_dict(data, state.info)
    obs = self._get_obs(obs_dict)
    rwd_dict = self._get_reward(obs_dict)

    state.metrics.update(
        reach=rwd_dict['reach'],
        bonus=rwd_dict['bonus'],
        penalty=-rwd_dict['penalty'],
        solved=rwd_dict['solved'],
    )

    return state.replace(
        pipeline_state=data, obs=obs, reward=rwd_dict['dense'], done=rwd_dict['done']
    )

  def _get_obs_dict(
      self, data: mjx.Data, info: Dict[str, jp.ndarray]
  ) -> jp.ndarray:

    obs_dict = {}
    obs_dict['time'] = jp.atleast_1d(data.time)
    obs_dict['qpos'] = data.qpos
    obs_dict['qvel'] = data.qvel*self.dt
    obs_dict['act'] = data.act

    # reach error
    obs_dict['tip_pos'] = jp.array([])
    obs_dict['target_pos'] = jp.array([])
    for isite, site in enumerate(self._target_reach_range.keys()):
        obs_dict['tip_pos'] = jp.append(obs_dict['tip_pos'], data.site_xpos[self._tip_sids[isite]].copy())
        obs_dict['target_pos'] = jp.append(obs_dict['target_pos'], info[site + '_target'].copy())
    obs_dict['reach_err'] = jp.array(obs_dict['target_pos'])-jp.array(obs_dict['tip_pos'])

    return obs_dict

  def _get_obs(
      self, obs_dict: Dict[str, jp.ndarray]
  ) -> jp.ndarray:

    obs_list = [jp.zeros(0)]
    for key in self.obs_keys:
        obs_list.append(obs_dict[key].ravel()) # ravel helps with images
    obsvec = jp.concatenate(obs_list)

    return obsvec

  def _get_reward(
      self, obs_dict: Dict[str, jp.ndarray],
  ) -> jp.ndarray:

    reach_dist = jp.linalg.norm(obs_dict['reach_err'], axis=-1)
    act_mag = jp.linalg.norm(obs_dict['act'], axis=-1)/self.sys.nu if self.sys.nu !=0 else 0
    far_th = jp.where(jp.squeeze(obs_dict['time'])>2*self.dt, self._far_th*len(self._tip_sids), jp.inf)
    near_th = len(self._tip_sids)*.0125

    rwd_dict =  {
            # Optional Keys
            'reach':   -1.*reach_dist,
            'bonus':   1.*(reach_dist<2*near_th) + 1.*(reach_dist<near_th),
            'act_reg': -1.*act_mag,
            'penalty': -1.*(reach_dist>far_th),
            # Must keys
            'sparse':  -1.*reach_dist,
            'solved':  1.*(reach_dist<near_th),
            'done':    1.*(reach_dist>far_th)}

    rwd_dict['dense'] = sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()])

    return rwd_dict

def main():

  env_name = 'myoFingerReachFixed-MJX-v0'
  envs.register_environment(env_name, FingerReachEnvV0)
  env = envs.get_environment(env_name)

  # # define the jit reset/step functions
  # jit_reset = jax.jit(env.reset)
  # jit_step = jax.jit(env.step)
  # # initialize the state
  # state = jit_reset(jax.random.PRNGKey(0))
  # rollout = [state.pipeline_state]

  # # grab a trajectory
  # for i in range(10):
  #   ctrl = -0.1 * jp.ones(env.sys.nu)
  #   state = jit_step(state, ctrl)
  #   rollout.append(state.pipeline_state)

  train_fn = functools.partial(
      ppo.train, num_timesteps=20_000_000, num_evals=5, reward_scaling=0.1,
      episode_length=1000, normalize_observations=True, action_repeat=1,
      unroll_length=10, num_minibatches=24, num_updates_per_batch=8,
      discounting=0.97, learning_rate=3e-4, entropy_cost=1e-3, num_envs=3072,
      batch_size=512, seed=0)

  x_data = []
  y_data = []
  ydataerr = []
  times = [datetime.now()]

  max_y, min_y = 13000, 0
  def progress(num_steps, metrics):

    print(f"num steps: {num_steps}, eval/episode_reward: {metrics['eval/episode_reward']}")

    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics['eval/episode_reward'])
    ydataerr.append(metrics['eval/episode_reward_std'])

    plt.xlim([0, train_fn.keywords['num_timesteps'] * 1.25])
    plt.ylim([min_y, max_y])

    plt.xlabel('# environment steps')
    plt.ylabel('reward per episode')
    plt.title(f'y={y_data[-1]:.3f}')

    plt.errorbar(
        x_data, y_data, yerr=ydataerr)
    plt.show()

  make_inference_fn, params, _= train_fn(environment=env, progress_fn=progress)

  print(f'time to jit: {times[1] - times[0]}')
  print(f'time to train: {times[-1] - times[1]}')

if __name__ == '__main__':
  main()