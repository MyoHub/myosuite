"""=================================================
# Copyright (c) MyoSuite Authors
Authors  :: Cheryl Wang (cheryl.wang.huiyi@gmail.com), Balint Hodossy (bkh16@ic.ac.uk),
            Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com), Guillaume Durandau (guillaume.durandau@mcgill.ca)
================================================="""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from loop_rate_limiters import RateLimiter
from myosuite.envs.obs_vec_dict import ObsVecDict
from myosuite.utils import gym
from myosuite.envs.myo.myochallenge.tabletennis_v0 import TableTennisEnvV0

MAX_TIME = 3.0


class TableTennisEnvRotatedV0(TableTennisEnvV0):
  def _preprocess_spec(self, spec, remove_body_collisions=True, add_left_arm=True):
    spec: mujoco.MjSpec = super()._preprocess_spec(spec,
                                                   remove_body_collisions=remove_body_collisions,
                                                   add_left_arm=add_left_arm)
    [spec.delete(k) for k in spec.keys]
    spec.compile()
    spec_base = spec.copy()

    spec.body("full_body").pos[:2] *= -1
    spec.body("full_body").alt.euler[2] -= np.pi
    spec.body("paddle").alt.euler[2] += np.pi
    spec.body("paddle").alt.euler[1] *= -1
    spec.body("paddle").alt.euler[0] *= -1
    spec.body("paddle").pos[:2] *= -1
    spec.to_xml()

    for b in spec.worldbody.bodies:
      if "paddle" not in b.name and "full_body" not in b.name:
        print(b.name)
        spec.delete(b)
    [spec.delete(g) for g in spec.worldbody.geoms]
    [spec.delete(l) for l in spec.worldbody.lights]
    [spec.delete(c) for c in spec.worldbody.cameras]
    spec.compile()

    self.spec = spec

    att = spec_base.worldbody.add_frame()
    [spec.delete(k) for k in spec.keys]

    paddle_spec = spec.copy()
    [[paddle_spec.delete(e) for e in typ] for typ in [paddle_spec.meshes, paddle_spec.textures,
                                                      paddle_spec.materials, paddle_spec.tendons,
                                                      paddle_spec.equalities, paddle_spec.actuators]]

    att.attach_body(spec.worldbody.bodies[0], 'opponent_')
    att.attach_body(paddle_spec.worldbody.bodies[1], 'opponent_')
    return spec_base



  def _setup(
        self,
        frame_skip: int = 10,
        qpos_noise_range=None,  # Noise in joint space for initialization
        obs_keys: list = TableTennisEnvV0.DEFAULT_OBS_KEYS,
        ball_xyz_range=None,
        ball_qvel=None,
        ball_friction_range=None,
        paddle_mass_range=None,
        rally_count=1,
        weighted_reward_keys: list = TableTennisEnvV0.DEFAULT_RWD_KEYS_AND_WEIGHTS,
        **kwargs,
    ):
    pass


class TableTennisEnv2PV0(gym.Env, gym.utils.EzPickle, ObsVecDict):

  def __init__(self, base_path="../assets/arm/myoarm_tabletennis.xml"):
    super().__init__()
    self.visu_env = TableTennisEnvRotatedV0(base_path)
    self.env1 = TableTennisEnvV0(base_path)
    self.env2 = TableTennisEnvV0(base_path)
    self.rot180 = R.from_euler('z', 180, degrees=True)
    self.mj_model = self.visu_env.mj_model
    self.mj_data = self.visu_env.mj_data
    self.ball_side = 1

  def step(self, action1, action2):
    ret1 = self.env1.step(action1)
    ret2 = self.env2.step(action2)
    self._sync_state()
    return ret1, ret2

  def reset(self):
    self.env1.reset()
    self.env2.reset()
    self._sync_state()

  def _sync_state(self):
    self.visu_env.mj_data.qpos[:self.env1.mj_model.nq] = self.env1.mj_data.qpos
    self.visu_env.mj_data.qpos[self.env1.mj_model.nq:] = self.env2.mj_data.qpos[:-7]
    self._rotate_paddle_qpos()
    mujoco.mj_forward(self.visu_env.mj_model, self.visu_env.mj_data)

  def _rotate_paddle_qpos(self):
    pqpos = self.env2.mj_data.joint("paddle_freejoint").qpos

    self.visu_env.mj_data.joint("opponent_paddle_freejoint").qpos = self._rotate_freejoint(pqpos)

  def _update_ball(self):
    donor_data = self.env1.mj_data if self.ball_side==1 else self.env2.mj_data
    receiver_data = self.env2.mj_data if self.ball_side==2 else self.env1.mj_data
    receiver_data.joint("pingpong_freejoint").qpos = self._rotate_freejoint(donor_data.joint("pingpong_freejoint").qpos)
    receiver_data.joint("pingpong_freejoint").qvel = self.rot180.apply(donor_data.joint("pingpong_freejoint").qvel)
    if donor_data.body("pingpong").xpos[0] <= 0:
      self.ball_side = 1 if self.ball_side == 2 else 2

  def _rotate_freejoint(self, pos_quat):
    return [*self.rot180.apply(pos_quat[:3]),
            *(self.rot180*R.from_quat(pos_quat[3:7], scalar_first=True)).as_quat(scalar_first=True)]

if __name__ == "__main__":
  env = TableTennisEnv2PV0("../assets/arm/myoarm_tabletennis.xml")
  from mujoco import viewer
  base_env = env.env1
  obs = env.reset()

  rate = RateLimiter(frequency=1 / base_env.mj_model.opt.timestep // 10, warn=False)
  with mujoco.viewer.launch_passive(env.mj_model, env.mj_data) as viewer:
    while viewer.is_running():
      action1 = np.zeros(base_env.mj_model.nu)
      action2 = np.zeros(base_env.mj_model.nu)
      ret1, ret2 = env.step(action1, action2)  # obs, reward, done, info
      viewer.sync()

      if ret1[2] or ret2[2]:
        env.reset()
      rate.sleep()
  pass
