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

# define
# init inputs
# relevant mjspec edits
# obs and obs keys
# rewards reward keys

class ReorientEnvV0(PipelineEnv):

  # https://github.com/MyoHub/myo_sim/blob/da2346336e1f2c5d2ee745fa4c908d1bea632b63/finger/finger_v0.xml

  def __init__(
      self,
      reset_noise_scale=1e-2,
      model_path='envs/myo/assets/hand/',
      model_filename='myohand_die.xml',
      # model_path='envs/myo/assets/hand',
      # model_filename='myohand_pose.xml',
      # goal_pos=(-.010, .010), # +- 1 cm
      # goal_rot=(-1.57, 1.57), # +-90 degrees
      goal_pos = (0.0, 0.0),          # goal position range (relative to initial pos)
      goal_rot = (.785, .785),        # goal rotation range (relative to initial rot)
      obj_size_change = 0,            # object size change (relative to initial size)
      obj_mass_range = (.108,.108),   # object size change (relative to initial size)
      obj_friction_change = (0,0,0),  # object friction change (relative to initial size)
      pos_th = .025,                  # position error threshold
      rot_th = 0.262,                 # rotation error threshold
      drop_th = .200,                 # drop height threshold
      episode_length=150,
      normalize_act=True,
      frame_skip=5, # aka physics_steps_per_control_step
      **kwargs,
  ):
  
    path = epath.Path(epath.resource_path('myosuite')) / (model_path)
    
    # full model
    # mj_model = mujoco.MjModel.from_xml_path((path / model_filename).as_posix())

    spec = mujoco.MjSpec.from_file((path / model_filename).as_posix())

    # for body in spec.bodies:
    #   for geom in body.geoms:
    #     if geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER and geom.name == '':
    #       breakpoint()

    # the first cylinder has contype conaffinity 1 1 for some reason
    for geom in spec.geoms:
      if geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        # breakpoint()
        if geom.contype != 0:
          geom.contype = 0
        if geom.conaffinity != 0:
          geom.conaffinity = 0

    for geom in spec.geoms:
      if geom.type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE

    mj_model = spec.compile()

    # remove contacts for efficiency
    # mj_model.geom_contype[:] = 0.
    # mj_model.geom_conaffinity[:] = 0

    # # add a target site
    # spec.find_body('world').add_site(name='wrist_target', type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0.02, 0.02], pos=[-0.2, -0.2, 1.2], rgba=[0, 1, 0, .3])

    # spec.detach_body(b_lunate)

    # mj_model = spec.compile()

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

    # self.obs_keys = ['hand_qpos_noMD5', 'hand_qvel', 'obj_pos', 'goal_pos', 'pos_err', 'obj_rot', 'goal_rot', 'rot_err']
    self.obs_keys = ['hand_qpos', 'hand_qvel', 'obj_pos', 'goal_pos', 'pos_err', 'obj_rot', 'goal_rot', 'rot_err']
    self.rwd_keys_wt = {
        'pos_dist': 100.0,
        'rot_dist': 1.0,
        'bonus': 0.0, # 4.0,
        'act_reg': 0.0, # 1,
        'penalty': 0.0 # 10,
    }

    self.object_sid = mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, 'object_o')
    self.goal_sid = mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, 'target_o')
    self.success_indicator_sid = mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, 'target_ball')
    self.goal_bid = mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'target')
    
    # self.goal_init_pos = self.sim.data.site_xpos[self.goal_sid].copy()
    # self.goal_obj_offset = self.sim.data.site_xpos[self.goal_sid]-self.sim.data.site_xpos[self.object_sid] # visualization offset between target and object
    
    self.goal_pos = goal_pos
    self.goal_rot = goal_rot
    self.pos_th = pos_th
    self.rot_th = rot_th
    self.drop_th = drop_th

    # setup for object randomization
    self.target_gid = mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_GEOM.value, 'target_dice')
    self.target_default_size = self.sys.mj_model.geom_size[self.target_gid].copy()

    self.object_bid = mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'Object')
    self.object_gid0 = self.sys.mj_model.body_geomadr[self.object_bid]
    self.object_gidn = self.object_gid0 + self.sys.mj_model.body_geomnum[self.object_bid]
    self.object_default_size = self.sys.mj_model.geom_size[self.object_gid0:self.object_gidn].copy()
    self.object_default_pos = self.sys.mj_model.geom_pos[self.object_gid0:self.object_gidn].copy()

    self.obj_mass_range = {'minval': obj_mass_range[0], 'maxval': obj_mass_range[1]}
    self.obj_size_range = {'minval': -obj_size_change, 'maxval': obj_size_change}
    self.obj_friction_range = {'minval': self.sys.mj_model.geom_friction[self.object_gid0:self.object_gidn] - obj_friction_change,
                               'maxval': self.sys.mj_model.geom_friction[self.object_gid0:self.object_gidn] + obj_friction_change}

    # self.init_qpos[:-7] *= 0 # Use fully open as init pos
    # self.init_qpos[0] = -1.5 # Palm up

    self._reset_noise_scale = reset_noise_scale
    # self._far_th = far_th
    # self._target_reach_range = target_reach_range
    # self._tip_sids = []
    # self._target_sids = []
    # sites = target_reach_range.keys()
    # if sites:
    #   for site in sites:
    #     self._tip_sids.append(mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, site))
    #     self._target_sids.append(mujoco.mj_name2id(self.sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, site + '_target'))
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
    # for site, span in self._target_reach_range.items():
      # target = jax.random.uniform(rng1, (3,), minval=jp.array(span[0]), maxval=jp.array(span[1]))
      # info[site + '_target'] = target

    obs_dict = self._get_obs_dict(data, info)
    obs = self._get_obs(obs_dict)

    reward, done, zero = jp.zeros(3)

    metrics = {
        'pos_dist': zero,
        'rot_dist': zero,
        'bonus': zero,
        'act_reg': zero,
        'penalty': zero,
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
        # reach=rwd_dict['reach'],
        # bonus=rwd_dict['bonus'],
        # penalty=-rwd_dict['penalty'],
        solved=rwd_dict['solved'],
    )

    return state.replace(
        pipeline_state=data, obs=obs, reward=rwd_dict['dense'], done=rwd_dict['done']
    )
  
  def mat2euler(self, mat):
    """ Convert Rotation Matrix to Euler Angles """
    mat = jp.asarray(mat, dtype=jp.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    _FLOAT_EPS = jp.finfo(jp.float64).eps
    _EPS4 = _FLOAT_EPS * 4.0

    cy = jp.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > _EPS4
    euler = jp.empty(mat.shape[:-1], dtype=jp.float64)
    euler = euler.at[..., 2].set(jp.where(condition,
                            -jp.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
                            -jp.arctan2(-mat[..., 1, 0], mat[..., 1, 1])))
    euler = euler.at[..., 1].set(jp.where(condition,
                            -jp.arctan2(-mat[..., 0, 2], cy),
                            -jp.arctan2(-mat[..., 0, 2], cy)))
    euler = euler.at[..., 0].set(jp.where(condition,
                            -jp.arctan2(mat[..., 1, 2], mat[..., 2, 2]),
                            0.0))
    return euler

  def _get_obs_dict(
      self, data: mjx.Data, info: Dict[str, jp.ndarray]
  ) -> jp.ndarray:

    obs_dict = {}
    obs_dict['time'] = jp.atleast_1d(data.time)
    obs_dict['hand_qpos_noMD5'] = data.qpos[:-7].copy() # ??? This is a bug. This needs to be qpos[:-6]. This bug omits the distal joint of the little finger from the observation. A fix to this will break all the submitted policies. A fix to this will be pushed after the myochallenge23
    obs_dict['hand_qpos'] = data.qpos[:-6].copy() # V1 of the env will use this corrected key by default
    obs_dict['hand_qvel'] = data.qvel[:-6].copy()*self.dt
    obs_dict['obj_pos'] = data.site_xpos[self.object_sid]
    obs_dict['goal_pos'] = data.site_xpos[self.goal_sid]
    obs_dict['pos_err'] = np.zeros(3)
    obs_dict['obj_rot'] = np.zeros(3)
    obs_dict['goal_rot'] = np.zeros(3)
    # obs_dict['pos_err'] = obs_dict['goal_pos'] - obs_dict['obj_pos'] - self.goal_obj_offset # correct for visualization offset between target and object
    # self.mat2euler(jp.reshape(data.site_xmat[self.object_sid],(3,3)))
    # self.mat2euler(jp.reshape(data.site_xmat[self.goal_sid],(3,3)))
    obs_dict['rot_err'] = obs_dict['goal_rot'] - obs_dict['obj_rot']

    if self.sys.nu:
        obs_dict['act'] = data.act[:].copy()

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

      pos_dist = jp.abs(jp.linalg.norm(obs_dict['pos_err'], axis=-1))
      rot_dist = jp.abs(jp.linalg.norm(obs_dict['rot_err'], axis=-1))
      act_mag = jp.linalg.norm(obs_dict['act'], axis=-1)/self.sys.nu if self.sys.nu !=0 else 0
      drop = pos_dist > self.drop_th

      rwd_dict = {
          # Perform reward tuning here --
          # Update Optional Keys section below
          # Update reward keys (DEFAULT_RWD_KEYS_AND_WEIGHTS) accordingly to update final rewards
          # Examples: Env comes pre-packaged with two keys pos_dist and rot_dist

          # Optional Keys
          'pos_dist': -1.*pos_dist,
          'rot_dist': -1.*rot_dist,
          'bonus': 1.*(pos_dist<2*self.pos_th) + 1.*(pos_dist<self.pos_th),
          'act_reg': -1.*act_mag,
          'penalty': -1.*drop,
          # Must keys
          'sparse': -rot_dist-10.0*pos_dist,
          'solved': (pos_dist<self.pos_th) and (rot_dist<self.rot_th) and (not drop),
          'done': drop,}
      rwd_dict['dense'] = sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()])

      # Sucess Indicator
      # self.sim.model.site_rgba[self.success_indicator_sid, :2] = np.array([0, 2]) if rwd_dict['solved'] else np.array([2, 0])

      return rwd_dict

def main():

    # self.sys.body_mass
    # self.sys.geom_size
    # self.sys.geom_pos

    # # Die mass changes
    # self.sim.model.body_mass[self.object_bid] = self.np_random.uniform(**self.obj_mass_range) # call to mj_setConst(m,d) is being ignored. Derive quantities wont be updated. Die is simple shape. So this is reasonable approximation.

    # # Die and Target size changes
    # del_size = self.np_random.uniform(**self.obj_size_range)
    # # adjust size of target
    # self.sim.model.geom_size[self.target_gid] = self.target_default_size + del_size
    # # adjust size of die
    # self.sim.model.geom_size[self.object_gid0:self.object_gidn-3][:,1] = self.object_default_size[:-3][:,1] + del_size
    # self.sim.model.geom_size[self.object_gidn-3:self.object_gidn] = self.object_default_size[-3:] + del_size
    # # adjust boundary of die
    # object_gpos = self.sim.model.geom_pos[self.object_gid0:self.object_gidn]
    # self.sim.model.geom_pos[self.object_gid0:self.object_gidn] = object_gpos/abs(object_gpos+1e-16) * (abs(self.object_default_pos) + del_size)


  def domain_randomize(env, sys, rng):
    """Randomizes the mjx.Model."""
    @jax.vmap
    def rand(rng):
      # body_pos
      key, subkey = jax.random.split(rng, 2)
      body_pos = env.goal_init_pos + jax.random.uniform(subkey, (3,), minval=env.goal_pos[0], maxval=env.goal_pos[1])
      body_pos = sys.body_pos.at[env.goal_bid].set(body_pos)
      # body_quat
      key, subkey = jax.random.split(rng, 2)
      body_quat = euler2quat(jax.random.uniform(subkey, (3,), minval=env.goal_rot[0], maxval=env.goal_rot[1]))
      body_quat = sys.body_quat.at[env.goal_bid].set(body_quat)
      # geom_friction
      key, subkey = jax.random.split(rng, 2)
      geom_friction = jax.random.uniform(subkey, (1,), **self.obj_friction_range)
      geom_friction = sys.geom_friction.at[env.object_gid0:env.object_gidn].set(geom_friction)


      # actuator
      _, key = jax.random.split(key, 2)
      gain_range = (-5, 5)
      param = jax.random.uniform(
          key, (1,), minval=gain_range[0], maxval=gain_range[1]
      ) + sys.actuator_gainprm[:, 0]
      gain = sys.actuator_gainprm.at[:, 0].set(param)
      bias = sys.actuator_biasprm.at[:, 1].set(-param)
      return friction, gain, bias

    friction, gain, bias = rand(rng)

    in_axes = jax.tree_util.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace({
        'geom_friction': 0,
        'actuator_gainprm': 0,
        'actuator_biasprm': 0,
    })

    sys = sys.tree_replace({
        'geom_friction': friction,
        'actuator_gainprm': gain,
        'actuator_biasprm': bias,
    })

    return sys, in_axes

  env_name = 'myoChallengeDieReorientP1-v0'
  envs.register_environment(env_name, ReorientEnvV0)
  env = envs.get_environment(env_name)

  breakpoint()

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