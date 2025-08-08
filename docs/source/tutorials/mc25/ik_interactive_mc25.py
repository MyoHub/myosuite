"""=================================================
Copyright (C) 2025
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================="""

### Adapted from: https://github.com/kevinzakka/mink/examples
# REQUIRES:
# Python 3.9
# MINK -- pip install "myosuite[examples]"

import os

import mink
import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
from myosuite.envs.myo.myochallenge.tabletennis_v0 import TableTennisEnvV0
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d

class IKTableTennisEnv(TableTennisEnvV0):
    def _preprocess_spec(self,
                         spec: mujoco.MjSpec,
                         remove_body_collisions=True,
                         add_left_arm=True):
      tar = spec.worldbody.add_body(name="target", pos=[0, 0, 0], quat=[0, 1, 0, 0], mocap=True)
      tar.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[.15, .15, .15], contype=0, conaffinity=0, rgba=[.6, .3, .3, .0])
      tar_paddle = spec.worldbody.add_body(name="ping_pong_paddle_target", pos=[1.5, 0.3, 1.13], quat=[0.69, -0.153, .701, -0.0923], mocap=True)
      tar_paddle.quat = (R.from_quat(tar_paddle.quat, scalar_first=True)*R.from_euler("xyz", [180, 0, 0], degrees=True)).as_quat(scalar_first=True)
      offset = -spec.body("paddle").sites[0].pos
      for g in spec.body("paddle").geoms:
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
          continue
        tar_paddle.add_geom(type=g.type, size=g.size, rgba=[.6, .3, .3, .3], pos=g.pos+offset, quat=g.quat, euler=g.alt.euler)

      orig_qpos = np.array(spec.keys[0].qpos)
      spec_copy = spec.copy()
      [k.delete() for k in spec_copy.keys]
      [t.delete() for t in spec_copy.textures]
      [m.delete() for m in spec_copy.materials]
      [t.delete() for t in spec_copy.tendons]
      [a.delete() for a in spec_copy.actuators]
      [e.delete() for e in spec_copy.equalities]
      [s.delete() for s in spec_copy.sensors if "paddle" not in s.name]
      [a.delete() for a in spec_copy.assets]
      [m.delete() for m in spec_copy.meshes]
      [c.delete() for c in spec_copy.cameras]

      paddle = spec_copy.body("paddle")
      paddle.joints[0].delete()
      paddle.name="ppp"
      paddle.pos = [0, 0, 0]
      spec.detach_body(spec.body("paddle"))
      fr = spec.site("S_grasp").parent.add_frame(quat =R.from_euler("yxz", [90, 0, -30], degrees=True).as_quat(scalar_first=True), pos=spec.site("S_grasp").pos+np.array([0.05, 0, 0]))
      fr.attach_body(paddle)
      #
      #
      # for k in spec.keys:
      #     k.delete()
      spec = super()._preprocess_spec(spec, remove_body_collisions, add_left_arm=False)
      spec.keys[0].delete()
      spec.actuators[0].name="interp"

      for a in spec.actuators[1:]:
        a.delete()
      return spec

    def _setup(self,
               frame_skip: int = 10,
               qpos_noise_range=None,  # Noise in joint space for initialization
               obs_keys: list = TableTennisEnvV0.DEFAULT_OBS_KEYS,
               ball_xyz_range=None,
               weighted_reward_keys: list = TableTennisEnvV0.DEFAULT_RWD_KEYS_AND_WEIGHTS,
               **kwargs,
               ):
      super(TableTennisEnvV0, self)._setup(obs_keys=obs_keys,
                     weighted_reward_keys=weighted_reward_keys,
                     frame_skip=frame_skip,
                     **kwargs,
                     )
      keyFrame_id = 0
      self.ball_xyz_range=None
      self.qpos_noise_range=None
      self.start_vel=np.array([[5.6, 1.6, 0.1] ])
      self.ball_dofadr = 0
      self.init_qpos[:] = self.sim.model.key_qpos[keyFrame_id].copy()

    def get_obs_dict(self, sim):
      return {k: np.zeros(1) for k in TableTennisEnvV0.DEFAULT_OBS_KEYS+["time", "act"]}

    def get_reward_dict(self, obs_dict):
      return {"sparse": 0, "dense": 0, "solved": 0, "done": 0}


env = IKTableTennisEnv(r"..\..\..\..\myosuite\envs\myo\assets\arm\myoarm_tabletennis.xml")

env.reset()

model = env.sim.model._model
data = env.sim.data._data

## =================== ##
## Setup IK.
## =================== ##

configuration = mink.Configuration(model)

tasks = [
    end_effector_task := mink.FrameTask(
        frame_name="paddle",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    ),
    posture_task := mink.PostureTask(model=model, cost=1e-2),
]

## =================== ##

# IK settings.
solver = "quadprog"
pos_threshold = 3e-4
ori_threshold = 1e-4
max_iters = 20

with mujoco.viewer.launch_passive(
    model=model, data=data, show_left_ui=False, show_right_ui=False
) as viewer:
    mujoco.mjv_defaultFreeCamera(model, viewer.cam)

    configuration.update(data.qpos)
    posture_task.set_target_from_configuration(configuration)
    mujoco.mj_forward(model, data)

    # Initialize the mocap target at the end-effector site.
    mink.move_mocap_to_frame(model, data, "target", "paddle", "site")
    slerp = Slerp([0, 1], R.from_quat(data.mocap_quat, scalar_first=True))
    lerp = interp1d([0, 1], data.mocap_pos, axis=0)
    rate = RateLimiter(frequency=500.0, warn=False)
    while viewer.is_running():
        # Update task target.
        T_wt = mink.SE3.from_mocap_name(model, data, "target")
        end_effector_task.set_target(T_wt)
        data.mocap_pos[0] = lerp(data.ctrl[0])
        data.mocap_quat[0] = slerp(data.ctrl[0]).as_quat(scalar_first=True)

        # Compute velocity and integrate into the next configuration.
        for i in range(max_iters):
            vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-3)
            configuration.integrate_inplace(vel, rate.dt)
            err = end_effector_task.compute_error(configuration)
            pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
            ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold
            if pos_achieved and ori_achieved:
                break

        data.qpos[:] = configuration.q
        mujoco.mj_forward(model, data)

        # Visualize at fixed FPS.
        viewer.sync()
        rate.sleep()
