from datetime import datetime
from etils import epath
import functools
from typing import Any, Dict, Sequence, Tuple, Union

import jax
from jax import numpy as jp
import numpy as np
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

from matplotlib import pyplot as plt
import mediapy as media

from jax import config
# config.update('jax_disable_jit', True)
# config.update("jax_debug_nans", True)
# config.update('jax_enable_x64', True)

class ArmReachEnvV0(PipelineEnv):

  # https://github.com/MyoHub/myo_sim/blob/da2346336e1f2c5d2ee745fa4c908d1bea632b63/finger/finger_v0.xml

  def __init__(
      self,
      reset_noise_scale=1e-2,
      far_th=1.,
      model_path='simhive/myo_sim/arm',
      model_filename='myoarm.xml',
      # model_path='envs/myo/assets/hand',
      # model_filename='myohand_pose.xml',
      target_reach_range={'wrist': ((-0.2, -0.2, 1.2), (-0.2, -0.2, 1.2)),}, # fixed
      # target_reach_range={'forearm_tip': ((-0.2-0.15, -0.2-0.15, 1.2-0.15), (-0.2+0.15, -0.2+0.15, 1.2+0.15)),}, # random
      episode_length=150,
      normalize_act=True,
      frame_skip=5, # aka physics_steps_per_control_step
      **kwargs,
  ):
  
    path = epath.Path(epath.resource_path('myosuite')) / (model_path)
    
    # full model
    # mj_model = mujoco.MjModel.from_xml_path((path / model_filename).as_posix())

    # remove contacts for efficiency
    # mj_model.geom_contype[:] = 0.
    # mj_model.geom_conaffinity[:] = 0

    # amputee
    spec = mujoco.MjSpec.from_file((path / model_filename).as_posix())

    b_lunate = spec.find_body('lunate')
    b_lunate_pos = b_lunate.pos.copy()

    # add site to the parent body
    b_radius = spec.find_body('radius')
    b_radius.add_site(
      name='wrist',
        pos=b_lunate_pos,
        group=3
    )

    # add a target site
    spec.find_body('world').add_site(name='wrist_target', type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0.02, 0.02], pos=[-0.2, -0.2, 1.2], rgba=[0, 1, 0, .3])

    # spec.detach_body(b_lunate)

    mj_model = spec.compile()

    # breakpoint()
    
    # spec.detach_body(spec.find_body('proximal_thumb'))
    # spec.detach_body(spec.find_body('proxph2'))
    # spec.detach_body(spec.find_body('proxph3'))
    # spec.detach_body(spec.find_body('proxph4'))
    # spec.detach_body(spec.find_body('proxph5'))
    # mj_model = spec.compile()

    # breakpoint()

    # settings compatible with differentiating through env.step, as per https://github.com/google-deepmind/mujoco/issues/1182
    # mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    # mj_model.opt.iterations = 1
    # mj_model.opt.ls_iterations = 1

    # these values are taken from myoChallengeBimanual-v0
    # env.unwrapped.sim.model.opt.timestep
    # env.unwrapped.sim.model.opt.iterations
    # env.unwrapped.sim.model.opt.ls_iterations
    # env.unwrapped.sim.model.opt.solver (2, Newton: https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjtsolver)
    mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    mj_model.opt.iterations = 100
    mj_model.opt.ls_iterations = 50
    # mj_model.opt.iterations = 10
    # mj_model.opt.ls_iterations = 5
    mj_model.opt.timestep = 0.002 # dt = mj_model.opt.timestep * frame_skip

    # mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
    # mj_model.opt.iterations = 6
    # mj_model.opt.ls_iterations = 6
    # mj_model.opt.timestep = 0.002 # dt = mj_model.opt.timestep * frame_skip

    sys = mjcf.load_model(mj_model)

    kwargs['n_frames'] = kwargs.get('n_frames', frame_skip)
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
    self._target_sids = []
    sites = target_reach_range.keys()
    if sites:
      for site in sites:
        self._tip_sids.append(mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, site))
        self._target_sids.append(mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, site + '_target'))
    self._episode_length = episode_length
    self._normalize_act = normalize_act

  def reset(self, rng: jp.ndarray) -> State:
    """Resets the environment to an initial state."""

    rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

    low, hi = -self._reset_noise_scale, self._reset_noise_scale
    qpos0 = self.sys.qpos0 + jax.random.uniform(
        rng2, (self.sys.nq,), minval=low, maxval=hi
    )
    qvel0 = jax.random.uniform(
        rng3, (self.sys.nv,), minval=low, maxval=hi
    )

    data = self.pipeline_init(qpos0, qvel0)

    info = {}
    for site, span in self._target_reach_range.items():
      target = jax.random.uniform(rng1, (3,), minval=jp.array(span[0]), maxval=jp.array(span[1]))
      info[site + '_target'] = target

    obs_dict = self._get_obs_dict(data, info)
    obs = self._get_obs(obs_dict)

    reward, done, zero = jp.zeros(3)

    metrics = {
        'reach': zero,
        'bonus': zero,
        'penalty': zero,
        'solved': zero,
        'truncated': zero,
    }

    return State(data, obs, reward, done, metrics, info)

  def step(self, state: State, action: jp.ndarray) -> State:
    """Runs one timestep of the environment's dynamics."""
    data0 = state.pipeline_state

    if self._normalize_act:
      action = nn.sigmoid(5.0 * (action - 0.5))

    data = self.pipeline_step(data0, action)

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
            'done':    1.*(reach_dist>far_th),}

    rwd_dict['dense'] = sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()])

    return rwd_dict

def main():

  env_name = 'myoArmReachFixed-MJX-v0'
  envs.register_environment(env_name, ArmReachEnvV0)
  env = envs.get_environment(env_name)

  def _render(rollouts, video_type='single', height=480, width=640, camera='front_view'):

    front_view_pos = env.sys.mj_model.camera('front_view').pos.copy()
    front_view_pos[1] = -2
    front_view_pos[2] = 0.65
    # env.sys.mj_model.camera('front_view').pos = front_view_pos
    # env.sys.mj_model.camera('front_view').pos0 = front_view_pos
    env.sys.mj_model.camera('front_view').poscom0 = front_view_pos

    videos = []
    for rollout in rollouts:
      
      # change the target position of the environment for rendering
      env.sys.mj_model.site_pos[env._target_sids] = rollout['wrist_target']

      if video_type == 'single':
        videos += env.render(rollout['states'], height=height, width=width, camera=camera)
      elif video_type == 'multiple':
        videos.append(env.render(rollout['states'], height=height, width=width, camera=camera))

    if video_type == 'single':
      media.write_video(cwd + '/ArmReach.mp4', videos, fps=1.0 / env.dt) 
    elif video_type == 'multiple':
      for i, video in enumerate(videos):
        media.write_video(cwd + '/ArmReach' + str(i) + '.mp4', video, fps=1.0 / env.dt) 

    return None

  train_fn = functools.partial(
      ppo.train, num_timesteps=20_000_000, num_evals=5, reward_scaling=0.1,
      episode_length=150, normalize_observations=True, action_repeat=1,
      unroll_length=10, num_minibatches=24, num_updates_per_batch=8,
      discounting=0.97, learning_rate=3e-4, entropy_cost=1e-3, num_envs=3072,
      batch_size=512, seed=0)

  # train_fn = functools.partial(
  #   sac.train, num_timesteps=7_864_320, num_evals=20, reward_scaling=5, episode_length=1000,
  #   normalize_observations=True, action_repeat=1, discounting=0.997, learning_rate=6e-4,
  #   num_envs=128, batch_size=128, grad_updates_per_step=32, max_devices_per_host=1,
  #   max_replay_size=1048576, min_replay_size=8192, seed=1)

  # seg fault even with these settings
  # train_fn = functools.partial(
  #   sac.train, num_timesteps=1_000_000, num_evals=5, reward_scaling=1, episode_length=env._episode_length,
  #   normalize_observations=True, action_repeat=1, discounting=0.997, learning_rate=6e-4,
  #   num_envs=32, batch_size=128, grad_updates_per_step=32, max_devices_per_host=1,
  #   max_replay_size=1048576, min_replay_size=8192, seed=1)

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

  import os
  cwd = os.path.dirname(os.path.abspath(__file__))

  model.save_params(cwd + '/ArmReachParams', params)
  params = model.load_params(cwd + '/ArmReachParams')
  inference_fn = make_inference_fn(params)

  backend = 'positional' # @param ['generalized', 'positional', 'spring']
  env = envs.create(env_name=env_name, backend=backend, episode_length=env._episode_length)

  # def get_CV(act, state):

  #   state = env.step(state, act)

  #   return state.obs[0]
  
  # grad_get_CV = jax.jit(jax.grad(get_CV))

  # rng = jax.random.PRNGKey(seed=1)
  # jit_env_reset = jax.jit(env.reset)
  # state = jit_env_reset(rng=rng)
  # dd = grad_get_CV(jp.zeros(5,), state)

  # breakpoint()

  times = [datetime.now()]

  jit_env_reset = jax.jit(env.reset)
  jit_env_step = jax.jit(env.step)
  jit_inference_fn = jax.jit(inference_fn)

  rng = jax.random.PRNGKey(seed=0)
  state = jit_env_reset(rng=rng)
  act = jp.zeros(env.sys.nu)
  state = jit_env_step(state, act)

  times.append(datetime.now())
  print(f'time to jit: {times[1] - times[0]}')

  rollouts = []
  for episode in range(10):
    rng = jax.random.PRNGKey(seed=episode)
    state = jit_env_reset(rng=rng)
    rollout = {}
    rollout['wrist_target'] = state.info['wrist_target']
    states = []
    while not (state.done or state.info['truncation']):
      states.append(state.pipeline_state)
      act_rng, rng = jax.random.split(rng)
      act, _ = jit_inference_fn(state.obs, act_rng)
      # act = jp.zeros(env.sys.nu)
      state = jit_env_step(state, act)

    times = [datetime.now()]

    rollout['states'] = states
    rollouts.append(rollout)

  _render(rollouts)
  
  breakpoint()

if __name__ == '__main__':
  main()