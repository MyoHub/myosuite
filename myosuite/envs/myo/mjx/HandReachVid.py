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
config.update('jax_disable_jit', True)
# config.update("jax_debug_nans", True)
# config.update('jax_enable_x64', True)
config.update('jax_platform_name', 'cpu')

class HandReachEnvV0(PipelineEnv):

  # https://github.com/MyoHub/myo_sim/blob/da2346336e1f2c5d2ee745fa4c908d1bea632b63/finger/finger_v0.xml

  def __init__(
      self,
      reset_noise_scale=1e-2,
      far_th=0.044,
      model_path='simhive/myo_sim/hand',   
      model_filename='myohand.xml',
      target_reach_range={
                'THtip': ((-0.165, -0.537, 1.495), (-0.165, -0.537, 1.495)),
                'IFtip': ((-0.151, -0.547, 1.455), (-0.151, -0.547, 1.455)),
                'MFtip': ((-0.146, -0.547, 1.447), (-0.146, -0.547, 1.447)),
                'RFtip': ((-0.148, -0.543, 1.445), (-0.148, -0.543, 1.445)),
                'LFtip': ((-0.148, -0.528, 1.434), (-0.148, -0.528, 1.434)),
                }, # fixed
      episode_length=100,
      normalize_act=True,
      frame_skip=5, # aka physics_steps_per_control_step
      **kwargs,
  ):
  
    path = epath.Path(epath.resource_path('myosuite')) / (model_path)
    # mj_model = mujoco.MjModel.from_xml_path((path / model_filename).as_posix())

    spec = mujoco.MjSpec.from_file((path / model_filename).as_posix())

    # add a target site
    spec.find_body('world').add_site(name='THtip_target', type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.005, 0.005, 0.005], pos=[0., 0., 0.002], rgba=[0.8, 0., 0., .8])
    spec.find_body('world').add_site(name='IFtip_target', type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.005, 0.005, 0.005], pos=[0., 0., 0.002], rgba=[0., 0.8, 0., .8])
    spec.find_body('world').add_site(name='MFtip_target', type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.005, 0.005, 0.005], pos=[0., 0., 0.002], rgba=[0.8, 0., 0.8, .8])
    spec.find_body('world').add_site(name='RFtip_target', type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.005, 0.005, 0.005], pos=[0., 0., 0.002], rgba=[0.8, 0.8, 0., .8])
    spec.find_body('world').add_site(name='LFtip_target', type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.005, 0.005, 0.005], pos=[0., 0., 0.002], rgba=[0.8, 0., 0.8, .8])

    mj_model = spec.compile()

    # settings compatible with differentiating through env.step, as per https://github.com/google-deepmind/mujoco/issues/1182
    # mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    # mj_model.opt.iterations = 1
    # mj_model.opt.ls_iterations = 1

    # these values are taken from myoHandReachFixed-v0
    # env.unwrapped.sim.model.opt.timestep
    # env.unwrapped.sim.model.opt.iterations
    # env.unwrapped.sim.model.opt.ls_iterations
    # env.unwrapped.sim.model.opt.solver # (2, Newton: https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjtsolver)
    mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    mj_model.opt.iterations = 100
    mj_model.opt.ls_iterations = 50

    mj_model.opt.timestep = 0.002 # dt =  mj_model.opt.timestep * frame_skip

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

  env_name = 'myoHandReachFixed-MJX-v0'
  envs.register_environment(env_name, HandReachEnvV0)
  env = envs.get_environment(env_name)

  def _render(rollouts, video_type='single', height=480, width=640):

    videos = []
    for rollout in rollouts:

      # camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
      # camera.trackbodyid = mujoco.mj_name2id(env.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'lunate')
      camera = mujoco.MjvCamera()
      camera.type = mujoco.mjtCamera.mjCAMERA_FREE
      camera.lookat = [-0.2, -0.55, 1.45]
      camera.distance = 0.35
      camera.azimuth = 150
      camera.elevation = -30
      
      # change the target position of the environment for rendering
      for i, key in env._target_reach_range.keys():
        env.sys.mj_model.site_pos[env._target_sids[i]] = rollout[key + '_target']

      if video_type == 'single':
        videos += env.render(rollout['states'], height=height, width=width, camera=camera)
      elif video_type == 'multiple':
        videos.append(env.render(rollout['states'], height=height, width=width, camera=camera))

    if video_type == 'single':
      media.write_video(cwd + '/HandReach.mp4', videos, fps=1.0 / env.dt) 
    elif video_type == 'multiple':
      for i, video in enumerate(videos):
        media.write_video(cwd + '/HandReach' + str(i) + '.mp4', video, fps=1.0 / env.dt) 

    return None

  train_fn = functools.partial(
      ppo.train, num_timesteps=20_000_000, num_evals=5, reward_scaling=0.1,
      episode_length=env._episode_length, normalize_observations=True, action_repeat=1,
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

  # make_inference_fn, params, _= train_fn(environment=env, progress_fn=progress)
  # make_inference_fn, _, _ = train_fn(environment=env, num_timesteps=0) # just to get the model

  # print(f'time to jit: {times[1] - times[0]}')
  # print(f'time to train: {times[-1] - times[1]}')



  def make_inference_fn(
    observation_size: int,
    action_size: int,
    normalize_observations: bool = True,
    network_factory_kwargs: Optional[Dict[str, Any]] = None,
  ):
    normalize = lambda x, y: x
    if normalize_observations:
      normalize = running_statistics.normalize
    ppo_network = brax_networks.make_ppo_networks(
        observation_size,
        action_size,
        preprocess_observations_fn=normalize,
        **(network_factory_kwargs or {}),
    )
    make_policy = brax_networks.make_inference_fn(ppo_network)
    return make_policy

  config_dict = load_config_dict(checkpoint_path)
  make_policy = make_inference_fn(
      config_dict['observation_size'],
      config_dict['action_size'],
      config_dict['normalize_observations'],
      network_factory_kwargs=config_dict['network_factory_kwargs'], # {"policy_hidden_layer_sizes": (128,) * 4}
  )
  params = model.load_params(checkpoint_file)
  jit_inference_fn = jax.jit(make_policy(params, deterministic=True))



  import os
  cwd = os.path.dirname(os.path.abspath(__file__))

  # model.save_params(cwd + '/HandReachParams', params)
  params = model.load_params(cwd + '/HandReachParams')
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

  # jit_env_reset = jax.jit(env.reset)
  # jit_env_step = jax.jit(env.step)
  # jit_inference_fn = jax.jit(inference_fn)

  rollouts = []
  breakpoint()
  for episode in range(10):
    rng = jax.random.PRNGKey(seed=episode)
    # state = jit_env_reset(rng=rng)
    state = env.reset(rng=rng)
    rollout = {}
    for key in env._target_reach_range.keys():
        rollout[key + '_target'] = state.info[key + '_target']
    states = []
    while not (state.done or state.info['truncation']):
      states.append(state.pipeline_state)
      act_rng, rng = jax.random.split(rng)
      # act, _ = jit_inference_fn(state.obs, act_rng)
      act, _ = inference_fn(state.obs, act_rng)
      # act = jp.zeros(env.action_size)
      # state = jit_env_step(state, act)
      state = env.step(state, act)

    rollout['states'] = states
    rollouts.append(rollout)

  _render(rollouts)

if __name__ == '__main__':
  main()