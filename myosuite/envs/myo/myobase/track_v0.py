"""=================================================
Copyright (C) 2026 Vittorio Caggiano
Author  :: Vittorio Caggiano (caggiano@gmail.com)
Source  :: https://github.com/MyoHub/myosuite
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================="""

import collections
from typing import Optional

import mujoco
import numpy as np

from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.logger.reference_motion import ReferenceMotion
from myosuite.utils import gym
from myosuite.utils.quat_math import quat2euler
import collections

class TrackEnv(BaseV0):

    DEFAULT_OBS_KEYS = ["qp", "qv", "hand_qpos_err", "hand_qvel_err", "obj_com_err"]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pose": 1.0,  # 1.0,
        "velocity": 1.0,
        "penalty": -2,
    }

    def __init__(
        self, model_path, obsd_model_path=None, seed=None, edit_fn=None, **kwargs
    ):

        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(
            self, model_path, obsd_model_path, seed, edit_fn=edit_fn, **kwargs
        )

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(
            model_path=model_path,
            obsd_model_path=obsd_model_path,
            seed=seed,
            env_credits=self.DEFAULT_CREDIT,
        )
        self._setup(**kwargs)

    def _setup(
        self,
        reference,  # reference target/motion for behaviors
        motion_start_time: float = 0,  # useful to skip initial motion
        motion_extrapolation: bool = True,  # Hold the last frame if motion is over
        obs_keys=None,
        weighted_reward_keys=None,
        reference_state_init=True,
        target_name: Optional = "target",
        **kwargs,
    ):
        obs_keys = obs_keys if obs_keys is not None else self.DEFAULT_OBS_KEYS
        weighted_reward_keys = weighted_reward_keys \
          if weighted_reward_keys is not None else self.DEFAULT_RWD_KEYS_AND_WEIGHTS
        # prep reference
        self.ref = ReferenceMotion(
            reference_data=reference,
            motion_extrapolation=motion_extrapolation,
            random_generator=self.np_random,
        )
        self.motion_start_time = motion_start_time
        self.target_sid = self.mj_model.site(target_name).id if target_name is not None else None

        ### DEEPMIMIC
        self.qpos_reward_weight = 0.35
        self.qpos_err_scale = 5.0

        self.qvel_reward_weight = 0.05
        self.qvel_err_scale = 0.1
        self.use_rsi = reference_state_init

        super()._setup(
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            frame_skip=10,
            **kwargs,
        )

    def update_reference_insim(self, curr_ref):
        if curr_ref.object is not None:
            self.mj_model.site_pos[self.target_sid][:] = curr_ref.object[:3]
            self.obsd_mj_model.site_pos[self.target_sid][:] = curr_ref.object[:3]
            mujoco.mj_forward(self.mj_model, self.mj_data)

    def norm2(self, x):
        return np.sum(np.square(x))

    def get_obs_dict(self, mj_model, mj_data):
        obs_dict = {}

        # get reference for current time (returns a named tuple)
        curr_ref = self.ref.get_reference(mj_data.time + self.motion_start_time)

        self.update_reference_insim(curr_ref)

        obs_dict["time"] = np.array([self.mj_data.time])
        obs_dict["qp"] = mj_data.qpos.copy()
        obs_dict["qv"] = mj_data.qvel.copy()
        obs_dict["robot_err"] = obs_dict["qp"].copy() - curr_ref.robot
        obs_dict["robot_err_vel"] = obs_dict["qv"].copy() - curr_ref.robot_vel

        if mj_model.na > 0:
            obs_dict["act"] = mj_data.act[:].copy()
        # self.mj_model.body_names --> body names
        return obs_dict

    def get_reward_dict(self, obs_dict):
        # calculate reward terms
        qpos_reward = np.exp(-self.qpos_err_scale * self.norm2(obs_dict["robot_err"]))
        qvel_reward = (
            np.array([0])
            if obs_dict["robot_err_vel"] is None
            else np.exp(-self.qvel_err_scale * self.norm2(obs_dict["robot_err_vel"]))
        )

        # weight and sum individual reward terms
        pose_reward = self.qpos_reward_weight * qpos_reward
        vel_reward = self.qvel_reward_weight * qvel_reward

        rwd_dict = collections.OrderedDict(
            (
                # Optional Keys
                ("pose", float(pose_reward)),
                ("velocity", float(vel_reward)),
                ("penalty", float(self.check_termination(obs_dict))),
                # Must keys
                ("sparse", 0),
                ("solved", 0),
                ("done", self.initialized_pos and self.check_termination(obs_dict)),
            )
        )
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict.get(key, 0) for key, wt in self.rwd_keys_wt.items()], axis=0
        )

        # print(rwd_dict['dense'], obj_com_err,rwd_dict['done'],rwd_dict['sparse'])
        return rwd_dict

    def qpos_from_robot_object(self, qpos, robot, object):
        qpos[: len(robot)] = robot
        qpos[len(robot) : len(robot) + 3] = object[:3]
        qpos[len(robot) + 3 :] = quat2euler(object[3:])

    def playback(self):
        idxs = self.ref.find_timeslot_in_reference(self.time + self.motion_start_time)
        # print(f"Time {self.time} {idxs} {self.ref.horizon}")
        ref_mot = self.ref.get_reference(self.time + self.motion_start_time)
        self.qpos_from_robot_object(self.mj_data.qpos, ref_mot.robot, ref_mot.object)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.mj_data.time = self.mj_data.time + 0.02  # self.env.env.dt
        return idxs[0] < self.ref.horizon - 1

    def reset(self, **kwargs):
        # print("Reset")
        self.ref.reset()
        if self.use_rsi:
          new_ref = self.ref.get_reference(np.random.choice(self.ref.reference["time"]))
          self.ref.index_cache = self.ref.find_timeslot_in_reference(new_ref.time)[0]

        # TODO: Prewarming muscle activation
        obs = super().reset(
            reset_qpos=new_ref.robot, reset_qvel=new_ref.robot_vel, **kwargs
        )

        return obs

    def check_termination(self, obs_dict):

        qpos_term = False

        if self.TermPose:  # termination on posture
            qpos_term = (
                True
                if self.norm2(obs_dict["robot_err"]) >= self.qpos_fail_thresh
                else False
            )

        return qpos_term


class ElbowTrackEnv(TrackEnv):
    """TrackEnv subclass for elbow that handles missing target site when object is None"""

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
      "pose": 1.0,  # 1.0,
      "velocity": 1.0,
      "bonus": 1.0,
      "penalty": -2,
    }

    DEFAULT_OBS_KEYS = TrackEnv.DEFAULT_OBS_KEYS

    def _setup(self,
        reference,  # reference target/motion for behaviors
        motion_start_time: float = 0,  # useful to skip initial motion
        motion_extrapolation: bool = True,  # Hold the last frame if motion is over
        obs_keys=None,
        weighted_reward_keys=None,
        reference_state_init=True,
        **kwargs,):
        """Setup with required attributes, handling missing target site"""
        from myosuite.logger.reference_motion import ReferenceMotion

        # Set initialized_pos before calling super to avoid errors
        self.initialized_pos = False
        obs_keys = obs_keys if obs_keys is not None else self.DEFAULT_OBS_KEYS
        weighted_reward_keys = weighted_reward_keys \
          if weighted_reward_keys is not None else self.DEFAULT_RWD_KEYS_AND_WEIGHTS



        # Call parent _setup (skip TrackEnv._setup, go to BaseV0._setup)
        super()._setup(
            reference=reference,
            motion_start_time=motion_start_time,
            motion_extrapolation=motion_extrapolation,
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            target_name=None,
            **kwargs
        )

        # Only get target_sid if object exists and site is available
        # For elbow, object is None so we don't need target site
        self.target_sid = None

        if self.ref.object_dim > 0:
            try:
                self.target_sid = self.mj_model.site("target").id
            except Exception:
                # Target site doesn't exist, but object is specified
                raise ValueError("Reference has object but model lacks 'target' site")
        # If object_dim is 0 (object is None), target_sid stays None

        # Set to True after setup completes (like myodm_v0 does)
        self.initialized_pos = True

    def update_reference_insim(self, curr_ref):
        """Override to handle case where object is None (no target site needed)"""
        if curr_ref.object is not None and self.target_sid is not None:
            super().update_reference_insim(curr_ref)
        # If object is None, we don't need to update any site

    def qpos_from_robot_object(self, qpos, robot, object):
        """Override to handle elbow model with only robot DOF"""
        # Elbow model only has robot joint, no object in qpos
        qpos[: len(robot)] = robot
        # Don't try to set object if qpos doesn't have space
        if object is not None and len(qpos) > len(robot):
            try:
                qpos[len(robot) : len(robot) + 3] = object[:3]
                if len(object) > 3 and len(qpos) > len(robot) + 3:
                    from myosuite.utils.quat_math import quat2euler

                    qpos[len(robot) + 3 :] = quat2euler(object[3:])
            except (ValueError, IndexError):
                # qpos doesn't have space for object, skip it
                pass

    def get_reward_dict(self, obs_dict):
        base_reward_dict = super().get_reward_dict(obs_dict)

        base_reward_dict["bonus"] = 0  # TODO: decide if this concept is meaningful to keep around for TrackEnv
        base_reward_dict["dense"] = np.sum(
          [wt * base_reward_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0
        )
        return base_reward_dict

    def check_termination(self, obs_dict):
        """Override to handle missing TermPose attribute"""
        qpos_term = False
        if hasattr(self, "TermPose") and self.TermPose:
            qpos_fail_thresh = getattr(self, "qpos_fail_thresh", 0.75)
            qpos_term = (
                True if self.norm2(obs_dict["robot_err"]) >= qpos_fail_thresh else False
            )
        return qpos_term


def main():
    import os
    import time

    """Example: Render sinusoidal elbow motion on screen"""
    # Parameters for sinusoidal motion applied to elbow angle
    duration = 5.0  # seconds
    dt = 0.02  # time step
    frequency = 0.5  # Hz
    amplitude = 1.0  # radians (peak deviation)
    offset = 1.0  # radians (center position)

    # Generate discrete time points
    num_frames = int(duration / dt)
    time_array = np.arange(0, duration, dt)[:num_frames]

    # Sinusoidal trajectory for elbow joint (angle)
    # The sinusoid is directly applied to the elbow angle
    elbow_angle = offset + amplitude * np.sin(2 * np.pi * frequency * time_array)
    elbow_angle = np.clip(elbow_angle, 0, 2.27)  # Enforce physical joint limits
    elbow_velocity = np.gradient(elbow_angle, dt)

    # For elbow, the target is the joint angle itself, not a 3D site position
    # So we set object to None - we're only tracking the robot joint
    reference = {
        "time": time_array,
        "robot": elbow_angle.reshape(-1, 1),  # Nx1, elbow angle (sinusoid)
        "robot_vel": elbow_velocity.reshape(-1, 1),  # Nx1, elbow velocity
        "robot_init": np.array([elbow_angle[0]]),  # initial angle
    }

    # Get the elbow model path
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(curr_dir, "../assets/elbow/myoelbow_1dof6muscles.xml")
    model_path = os.path.abspath(model_path)

    # Create the ElbowTrackEnv with the sinusoidal reference
    obs_keys = ["qp", "qv", "robot_err", "robot_err_vel", "act"]
    env = ElbowTrackEnv(
        model_path=model_path,
        reference=reference,
        motion_extrapolation=True,
        obs_keys=obs_keys,
    )

    print("Starting sinusoidal elbow motion playback...")
    print("Press ESC or close the window to exit")
    print(f"Motion duration: {duration} seconds")
    print(f"Frequency: {frequency} Hz")
    print(f"Amplitude: {amplitude} radians")

    # Playback the sinusoidal elbow trajectory with rendering
    env.reset()

    try:
        while True:
            has_more = env.playback()
            env.mj_render()  # Render on screen
            time.sleep(env.dt)  # Control playback speed

            if not has_more:
                # Loop the motion
                env.reset()
                print("Motion loop completed, restarting...")

    except KeyboardInterrupt:
        print("\nPlayback interrupted by user")
    finally:
        env.close()
        print("Environment closed")


if __name__ == "__main__":
    main()
