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
import h5py
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

      temp_model = spec.compile()
      temp_data = mujoco.MjData(temp_model)
      spec.body("paddle").quat = R.from_euler('xyz', spec.body("paddle").alt.euler).as_quat(scalar_first=True)
      mujoco.mj_resetDataKeyframe(temp_model, temp_data, 0)
      mujoco.mj_forward(temp_model, temp_data)
      body_B = temp_data.body(spec.site("S_grasp").parent.name)
      body_A = temp_data.body("paddle")
      rel_pos, rel_quat = reparent_to(body_A.xpos, body_A.xquat, body_B.xpos, body_B.xquat)

      tar = spec.worldbody.add_body(name="target", pos=[0, 0, 0], quat=[0, 1, 0, 0], mocap=True)
      tar.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[.15, .15, .15], contype=0, conaffinity=0, rgba=[.6, .3, .3, .0])
      tar_paddle = spec.worldbody.add_body(name="ping_pong_paddle_target", pos=[1.5, 0.3, 1.13], quat=[0.69, -0.153, .701, -0.0923], mocap=True)
      offset = -spec.body("paddle").sites[0].pos
      for g in spec.body("paddle").geoms:
        if g.type == mujoco.mjtGeom.mjGEOM_MESH:
          continue
        tar_paddle.add_geom(type=g.type, size=g.size, rgba=[.6, .3, .3, .3], pos=g.pos+offset, quat=g.quat, euler=g.alt.euler)

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
      paddle.pos *= 0
      paddle.alt.euler *= 0
      paddle.quat = [1, 0, 0, 0]
      spec.detach_body(spec.body("paddle"))
      fr = spec.site("S_grasp").parent.add_frame(quat=rel_quat, pos=rel_pos)
      fr.attach_body(paddle)

      temp_model2 = spec.compile()
      temp_data2 = mujoco.MjData(temp_model2)
      mujoco.mj_resetDataKeyframe(temp_model2, temp_data2, 0)
      mujoco.mj_forward(temp_model2, temp_data2)

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


def reparent_to(xpos_A, xquat_A, xpos_B, xquat_B):
  qA_scipy = R.from_quat(xquat_A, scalar_first=True)
  qB_scipy = R.from_quat(xquat_B, scalar_first=True)

  rel_pos = qA_scipy.inv().apply(xpos_B - xpos_A)
  rel_quat = (qA_scipy.inv() * qB_scipy).inv().as_quat(scalar_first=True)
  return rel_pos, rel_quat



env = IKTableTennisEnv(r"..\..\..\..\myosuite\envs\myo\assets\arm\myoarm_tabletennis.xml")
diff_env = TableTennisEnvV0(r"..\..\..\..\myosuite\envs\myo\assets\arm\myoarm_tabletennis.xml")

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

    def insert_paddle_pos(qpos):
      return np.insert(qpos, qpos.shape[0]-7, np.concatenate([data.body("paddle").xpos, data.body("paddle").xquat]))

    rollout = [{"qpos": insert_paddle_pos(data.qpos), "qvel": data.qvel}]
    T_movement = 0.5  # movement in seconds
    for t in np.linspace(0, 1, int(T_movement//model.opt.timestep)):
        # Update task target.
        T_wt = mink.SE3.from_mocap_name(model, data, "target")
        end_effector_task.set_target(T_wt)
        data.mocap_pos[0] = lerp(t)
        data.mocap_quat[0] = slerp(t).as_quat(scalar_first=True)

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
        qvel = np.zeros(model.nv+6)
        mujoco.mj_differentiatePos(diff_env.sim.model._model, qvel, model.opt.timestep,
                                   rollout[-1]['qpos'], insert_paddle_pos(data.qpos))
        rollout.append({"qpos": insert_paddle_pos(data.qpos), "qvel": qvel})

        # Visualize at fixed FPS.
        viewer.sync()
        rate.sleep()
    rollout[0]['qvel'] = rollout[1]['qvel']
    with h5py.File('traj.h5', 'w') as h5f:
        h5f.create_dataset('qpos', data=[s["qpos"] for s in rollout])
        h5f.create_dataset('qvel', data=[s["qvel"] for s in rollout])
        h5f.close()
