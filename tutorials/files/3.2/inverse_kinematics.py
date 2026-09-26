"""=================================================
Copyright (C) 2025
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================="""

### Adapted from: https://github.com/kevinzakka/mink/examples
# REQUIRES:
# Python 3.9
# MINK -- pip install "myosuite[examples]"
#
# Interactive viewer needs mjpython on macOS:
#   mjpython tutorials/files/3.2/inverse_kinematics.py
# Headless smoke test:
#   python tutorials/files/3.2/inverse_kinematics.py --no-viewer

from __future__ import annotations

import argparse

import mink
import mujoco
import numpy as np

from myosuite.envs.myo.assets._resolve import resolve_arm_xml
from myosuite.utils.asset_path_resolver import resolve_model_xml_path

_XML_ARM_Model = str(resolve_model_xml_path(resolve_arm_xml("myoarm.xml")))

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

solver = "quadprog"
pos_threshold = 1e-4
ori_threshold = 1e-4
max_iters = 20


def _solve_once() -> None:
    """One IK iteration so students can check mink + the arm XML without a GUI."""
    configuration.update(data.qpos)
    posture_task.set_target_from_configuration(configuration)
    mujoco.mj_forward(model, data)
    mink.move_mocap_to_frame(model, data, "target", "S_grasp", "site")
    T_wt = mink.SE3.from_mocap_name(model, data, "target")
    end_effector_task.set_target(T_wt)
    vel = mink.solve_ik(configuration, tasks, 0.002, solver, 1e-3)
    configuration.integrate_inplace(vel, 0.002)
    err = end_effector_task.compute_error(configuration)
    print(
        "IK ok: nq=",
        model.nq,
        "pos_err=",
        float(np.linalg.norm(err[:3])),
        "ori_err=",
        float(np.linalg.norm(err[3:])),
    )


def _run_viewer() -> None:
    import mujoco.viewer
    from loop_rate_limiters import RateLimiter

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)
        mujoco.mj_forward(model, data)

        mink.move_mocap_to_frame(model, data, "target", "S_grasp", "site")

        rate = RateLimiter(frequency=500.0, warn=False)
        while viewer.is_running():
            T_wt = mink.SE3.from_mocap_name(model, data, "target")
            end_effector_task.set_target(T_wt)

            for _ in range(max_iters):
                vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-3)
                configuration.integrate_inplace(vel, rate.dt)
                err = end_effector_task.compute_error(configuration)
                pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
                ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold
                if pos_achieved and ori_achieved:
                    break

            data.qpos[:] = configuration.q
            mujoco.mj_step(model, data)

            viewer.sync()
            rate.sleep()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mink IK on MyoArm. Use mjpython for the interactive viewer."
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Solve one IK step and exit (no GUI).",
    )
    args = parser.parse_args()
    if args.no_viewer:
        _solve_once()
    else:
        _run_viewer()
