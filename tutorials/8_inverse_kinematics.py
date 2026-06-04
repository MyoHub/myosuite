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

os.chdir("../../../myosuite/simhive/myo_sim/arm")
_XML_ARM_Model = "myoarm.xml"

xml_string = f"""
        <mujoco model="MyoArm with Mocap">
            <include file="{_XML_ARM_Model}"/>
            <worldbody>
                <body name="target" pos="0 0 0" quat="0 1 0 0" mocap="true">
                    <geom type="box" size=".15 .15 .15" contype="0" conaffinity="0" rgba=".6 .3 .3 .2"/>
                </body>
            </worldbody>
        </mujoco>
        """

model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)

## =================== ##
## Setup IK.
## =================== ##

configuration = mink.Configuration(model)

tasks = [
    end_effector_task := mink.FrameTask(
        frame_name="S_grasp",
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
pos_threshold = 1e-4
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
    mink.move_mocap_to_frame(model, data, "target", "S_grasp", "site")

    rate = RateLimiter(frequency=500.0, warn=False)
    while viewer.is_running():
        # Update task target.
        T_wt = mink.SE3.from_mocap_name(model, data, "target")
        end_effector_task.set_target(T_wt)

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
        mujoco.mj_step(model, data)

        # Visualize at fixed FPS.
        viewer.sync()
        rate.sleep()
