"""=================================================
Copyright (C) 2025
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================="""

# MyoArm with a mocap target body, for tutorial 3.2.
#
# Importing this module builds `model` and `data`; posing a site needs only
# MyoSuite's solve_ik. The interactive viewer drives the arm with mink
# (adapted from https://github.com/kevinzakka/mink/examples), which is optional:
#   pip install "myosuite[examples]"
#
# Headless check (solve_ik):
#   python tutorials/files/3.2/inverse_kinematics.py --no-viewer
# Interactive mink viewer (mjpython on macOS):
#   mjpython tutorials/files/3.2/inverse_kinematics.py

from __future__ import annotations

import argparse

import mujoco
import numpy as np

from myosuite.envs.myo.assets._resolve import resolve_arm_xml
from myosuite.physics.inverse_kinematics import solve_ik
from myosuite.utils.asset_path_resolver import resolve_model_xml_path

_XML_ARM_Model = str(resolve_model_xml_path(resolve_arm_xml("myoarm.xml")))

xml_string = f"""
        <mujoco model="MyoArm with Mocap">
            <include file="{_XML_ARM_Model}"/>
            <worldbody>
                <body name="target" pos="0 0 0" quat="0 1 0 0" mocap="true">
                    <geom type="box" size=".03 .03 .03" contype="0" conaffinity="0" rgba=".8 .2 .2 .4"/>
                </body>
            </worldbody>
        </mujoco>
        """

model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)


def mink_tasks():
    """mink configuration, hand task and posture task for realtime IK (needs mink)."""
    import mink

    configuration = mink.Configuration(model)
    end_effector_task = mink.FrameTask(
        frame_name="S_grasp",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    posture_task = mink.PostureTask(model=model, cost=1e-2)
    return configuration, end_effector_task, posture_task


def _solve_once() -> None:
    """Raise the hand by 10 cm with solve_ik, without a GUI."""
    mujoco.mj_forward(model, data)
    target = data.site("S_grasp").xpos + np.array([0.0, 0.0, 0.1])
    data.qpos[:] = solve_ik(model, {"S_grasp": target})
    mujoco.mj_forward(model, data)
    error = np.linalg.norm(data.site("S_grasp").xpos - target)
    print(f"solve_ik ok: nq={model.nq} position error={error:.2e} m")


def _run_viewer() -> None:
    import mink
    import mujoco.viewer
    from loop_rate_limiters import RateLimiter

    configuration, end_effector_task, posture_task = mink_tasks()
    tasks = [end_effector_task, posture_task]
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

            for _ in range(20):
                vel = mink.solve_ik(configuration, tasks, rate.dt, "quadprog", 1e-3)
                configuration.integrate_inplace(vel, rate.dt)
                err = end_effector_task.compute_error(configuration)
                if np.linalg.norm(err[:3]) <= 1e-4 and np.linalg.norm(err[3:]) <= 1e-4:
                    break

            data.qpos[:] = configuration.q
            mujoco.mj_step(model, data)

            viewer.sync()
            rate.sleep()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="IK on MyoArm: solve_ik headless, or the mink viewer (mjpython)."
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Pose the hand once with solve_ik and exit (no GUI, no mink).",
    )
    args = parser.parse_args()
    if args.no_viewer:
        _solve_once()
    else:
        _run_viewer()
