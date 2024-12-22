# #@title Import packages for plotting and creating graphics

# import time
# import itertools
# import numpy as np
# from typing import Callable, NamedTuple, Optional, Union, List

# # Graphics and plotting.
# print('Installing mediapy:')
# !command -v ffmpeg >/dev/null || (apt update && apt install -y ffmpeg)
# !pip install -q mediapy
# import mediapy as media
# import matplotlib.pyplot as plt

# # More legible printing from numpy.
# np.set_printoptions(precision=3, suppress=True, linewidth=100)

#################################################################
#################################################################
#################################################################

#@title Import MuJoCo, MJX, and Brax

from datetime import datetime
from etils import epath
import functools
from IPython.display import HTML
from typing import Any, Dict, Sequence, Tuple, Union
import os
from ml_collections import config_dict

import jax
from jax import numpy as jp
import numpy as np
from flax.training import orbax_utils
from flax import struct
from matplotlib import pyplot as plt
import mediapy as media
from orbax import checkpoint as ocp
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

#################################################################
#################################################################
#################################################################

# from jax import config
# config.update('jax_disable_jit', True)
# config.update("jax_debug_nans", True)

class ReachEnvV0(PipelineEnv):

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
    # self._target_sids = []
    sites = target_reach_range.keys()
    if sites:
      for site in sites:
        self._tip_sids.append(mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, site))
        # self._target_sids.append(mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, site + "_target"))

  def reset(self, rng: jp.ndarray) -> State:
    """Resets the environment to an initial state."""

    rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

    # generate target pose
    # site_xpos = data.site_xpos.copy()
    # site_pos = self.sys.site_pos.copy()
    # for isite, span in enumerate(self._target_reach_range.values()):
    #   target = jax.random.uniform(rng1, (3,), minval=jp.array(span[0]), maxval=jp.array(span[1]))
    #   site_pos = site_pos.at[self._target_sids[isite]].set(target)
    #   self.sys = self.sys.replace(site_pos=site_pos)
      # self.sys.mj_model.site_pos[self._target_sids[isite]] = np.array(target)
      # self.sys.site_pos[self._target_sids[isite]] = np.array(target)
      # site_xpos = site_xpos.at[self._target_sids[isite]].set(target)
      # data = data.replace(site_xpos=site_xpos)

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

    # norm_action = 1.0 / (1.0 + jp.exp(-5.0 * (action - 0.5)))
    norm_action = nn.sigmoid(5.0 * (action - 0.5))
    # norm_action = jp.clip(1.0 / (1.0 + jp.exp(-5.0 * (action - 0.5))), min=0., max=1.)

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
        # obs_dict['target_pos'] = jp.append(obs_dict['target_pos'], data.site_xpos[self._target_sids[isite]].copy())
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

# register_env_with_variants(id='motorFingerReachFixed-v0',
#         entry_point='myosuite.envs.myo.myobase.reach_v0:ReachEnvV0',
#         max_episode_steps=200,
#         kwargs={
#             'model_path': curr_dir+'/../../../simhive/myo_sim/finger/motorfinger_v0.xml',
#             'target_reach_range': {'IFtip': ((0.2, 0.05, 0.20), (0.2, 0.05, 0.20)),},
#             'normalize_act': True,
#             'frame_skip': 5,
#         }
#     )

env_name = 'myoFingerReachFixed-MJX-v0'
envs.register_environment(env_name, ReachEnvV0)

# instantiate the environment
# model_path = 'simhive/myo_sim/finger'
# model_filename = 'myofinger_v0.xml'
# target_reach_range = {'IFtip': ((0.2, 0.05, 0.20), (0.2, 0.05, 0.20)),}
# env = envs.get_environment(env_name, model_path=model_path, model_filename=model_filename, target_reach_range=target_reach_range)
env = envs.get_environment(env_name)

#################################################################
#################################################################
#################################################################

# define the jit reset/step functions
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

#################################################################
#################################################################
#################################################################

# grab a trajectory
for e in range(1000):
  # initialize the state
  state = jit_reset(jax.random.PRNGKey(e))
  rollout = [state.pipeline_state]
  print("episode",e)
  # qpos = np.zeros((4,11))
  # qvel = np.zeros((4,11))
  # qpos[:,0] = state.pipeline_state.qpos
  # qvel[:,0] = state.pipeline_state.qvel
  for i in range(30):
    print([np.max(state.pipeline_state.efc_D), np.mean(state.pipeline_state.efc_D)])
    ctrl = 0. * jp.ones(env.sys.nu)
    # ctrl = np.random.normal(1) * 0.1 * jp.ones(env.sys.nu)
    # if e == 0 and (i == 0 or i == 1):
      # breakpoint()
    state = jit_step(state, ctrl)
    # qpos[:,i+1] = state.pipeline_state.qpos
    # qvel[:,i+1] = state.pipeline_state.qvel
    rollout.append(state.pipeline_state)
    # print(state.obs)
    if e == 42 and i == 2:
      print([e,i,state.reward])
      breakpoint()
    if jp.isnan(state.reward):
      print([e,i,state.reward])
      breakpoint()

breakpoint()

# state.pipeline_state.site_xpos[mujoco.mj_name2id(env.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, 'IFtip')]

# # initialize the state
# state = jit_reset(jax.random.PRNGKey(0))
# rollout = [state.pipeline_state]

# # grab a trajectory
# for i in range(30):
#   ctrl = -0.1 * jp.ones(env.sys.nu)
#   state = jit_step(state, ctrl)
#   rollout.append(state.pipeline_state)
#   print([i,state.reward])

# breakpoint()

# media.show_video(env.render(rollout), fps=1.0 / env.dt)

#################################################################
#################################################################
#################################################################

train_fn = functools.partial(
    ppo.train, num_timesteps=10_000, num_evals=5, reward_scaling=0.1,
    episode_length=1000, normalize_observations=True, action_repeat=1,
    unroll_length=10, num_minibatches=24, num_updates_per_batch=8,
    discounting=0.97, learning_rate=3e-4, entropy_cost=1e-3, num_envs=3072,
    batch_size=512, seed=0)

# train_fn = functools.partial(
#     sac.train, num_timesteps=6_553_600, num_evals=20, reward_scaling=30, 
#     episode_length=1000, normalize_observations=True, action_repeat=1, 
#     discounting=0.997, learning_rate=6e-4, num_envs=128, batch_size=512, 
#     grad_updates_per_step=64, max_devices_per_host=1, max_replay_size=1048576, 
#     min_replay_size=8192, seed=1)

x_data = []
y_data = []
ydataerr = []
times = [datetime.now()]

max_y, min_y = 13000, 0
def progress(num_steps, metrics):
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

breakpoint()