
import collections
import numpy as np
import mujoco
import os
from myosuite.envs.myo.myobase.track_v0 import TrackEnv
from myosuite.logger.grouped_datasets import Trace

class MyoSkeletonTrackEnv(TrackEnv):

    DEFAULT_OBS_KEYS = ["qp", "qv", "robot_err", "robot_err_vel"]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pose": 2.0,     # INCREASED from 1.0
        "velocity": 1.0,
        "alive": 1.0,
        "penalty": -10,
    }

    def __init__(self, model_path, reference_path, **kwargs):
        self.reference_path = reference_path
        # We don't pass reference_path to super, only model_path and kwargs
        super().__init__(model_path=model_path, **kwargs)

    def _setup(self, **kwargs):
        self.initialized_pos = False

        # Identify "clean" joint indices: Hinges + Root pos
        # We do this FIRST because super()._setup() calls self.step() internally
        self.clean_indices = []
        for i in range(self.mj_model.njnt):
            jnt_type = self.mj_model.jnt_type[i]
            q_adr = self.mj_model.jnt_qposadr[i]
            if jnt_type == 3: # Hinge joint
                self.clean_indices.append(int(q_adr))
            elif jnt_type == 0: # Free joint
                # Keep only Root Height (index 2)
                # Exclude Root X (0) and Y (1) to permit horizontal drift
                self.clean_indices.append(int(q_adr + 2))
        self.clean_indices = np.array(self.clean_indices)

        # mj_model is available now because BaseV0.__init__ (called by TrackEnv.__init__)
        # has already loaded the model.
        reference = self._load_reference_motion(self.reference_path)

        # Call TrackEnv._setup with parsed reference
        print(f"DEBUG: Loaded reference with {len(reference['time'])} frames.")
        super()._setup(reference=reference, target_name=None, **kwargs)

        # Enable extrapolation so we can stand forever after the 55 frames end
        self.motion_extrapolation = True

        # RELAX the error scale
        self.qpos_err_scale = 0.5
        self.qpos_reward_weight = 1.0
        self.initialized_pos = True

    def _load_reference_motion(self, reference_path):
        try:
             traj = Trace.load(reference_path)
        except Exception as e:
             raise e

        # Extract generic motion name (first key)
        motion_name = list(traj.trace.keys())[0]
        motion_data = traj.trace[motion_name]

        time = np.array(motion_data['time'])
        dt = time[1] - time[0]

        # Filter qpos to match model
        joint_names = [self.mj_model.joint(i).name for i in range(self.mj_model.njnt)]
        ref_qpos = np.zeros((len(time), self.mj_model.nq))

        root_key = 'myoskeleton_root'
        if root_key in motion_data['qpos']:
            ref_qpos[:, :7] = np.array(motion_data['qpos'][root_key])

        for i, name in enumerate(joint_names):
            if name == 'myoskeleton_root': continue
            if name in motion_data['qpos']:
                 qpos_adr = int(self.mj_model.joint(i).qposadr)
                 ref_qpos[:, qpos_adr] = np.array(motion_data['qpos'][name]).flatten()

        # Calculate qvel
        ref_qvel = np.zeros((len(time), self.mj_model.nv))
        qpos_diff = (ref_qpos[1:] - ref_qpos[:-1]) / dt

        # Linear velocity (0:3)
        ref_qvel[:-1, 0:3] = qpos_diff[:, 0:3]

        # Joints (6:)
        # Assuming free joint is first 7 in qpos and first 6 in qvel
        # And we want to map qpos[7:] to qvel[6:]
        if self.mj_model.nv == 117: # Specific case for a known model
             ref_qvel[:-1, 6:] = qpos_diff[:, 7:]
        else:
             # Fallback for other models, assuming qpos[7:] maps to qvel[6:]
             # This assumes a 7-dim root qpos and 6-dim root qvel
             ref_qvel[:-1, 6:] = qpos_diff[:, 7:]

        ref_qvel[-1] = ref_qvel[-2] # Repeat last velocity for consistency

        return {
            "time": time,
            "robot": ref_qpos,
            "robot_vel": ref_qvel
        }

    def get_reward_dict(self, obs_dict):
        # We override the base tracking reward to use a more stable scale.
        # TrackEnv uses exp(-5.0 * norm2(error)), which vanishes too quickly.

        # Calculate MAE only on CLEAN joints (hinges + h)
        robot_err = obs_dict.get('robot_err', np.zeros(self.mj_model.nq))
        if np.size(robot_err) >= self.mj_model.nq:
            err_vec = robot_err.flatten()
            clean_err = err_vec[self.clean_indices]
            mae = np.mean(np.abs(clean_err))
        else:
            mae = 0.0

        # New Pose Reward: exp(-scale * MAE)
        pose_reward = self.DEFAULT_RWD_KEYS_AND_WEIGHTS['pose'] * np.exp(-self.qpos_err_scale * mae)

        # Standard velocity reward
        robot_err_vel = obs_dict.get('robot_err_vel', np.zeros(1))
        vel_reward = np.exp(-0.1 * np.mean(np.abs(robot_err_vel)))

        # Weights from class definition
        rwd_dict = collections.OrderedDict((
            ("pose", float(pose_reward)),
            # ("velocity", float(vel_reward)),
        ))

        # Add alive bonus
        if not self.check_termination(obs_dict):
            rwd_dict['alive'] = self.DEFAULT_RWD_KEYS_AND_WEIGHTS['alive']
        else:
            rwd_dict['alive'] = 0.0

        rwd_dict["penalty"] = float(self.check_termination(obs_dict))
        rwd_dict["sparse"] = 0
        rwd_dict["solved"] = 0
        rwd_dict["done"] = self.initialized_pos and self.check_termination(obs_dict)

        # Recalculate dense reward
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict.get(key, 0) for key, wt in self.rwd_keys_wt.items()], axis=0
        )

        return rwd_dict

    def check_termination(self, obs_dict):
        fall_threshold = 0.3

        # 1. Flexible Root Height Check
        try:
            qp = obs_dict.get('qp', [])
            if hasattr(qp, 'ndim') and qp.ndim == 3: # 3D shape (e.g. MJX)
                root_z = qp[0, 0, 2]
            elif hasattr(qp, 'ndim') and qp.ndim == 1 and len(qp) > 2: # 1D shape (Standard)
                root_z = qp[2]
            elif isinstance(qp, list) and len(qp) > 2:
                root_z = qp[2]
            else:
                root_z = -1
            is_fallen = (root_z < fall_threshold) if root_z != -1 else False
        except Exception:
            is_fallen = False

        # 2. Check for deviations
        is_deviating = False
        try:
            robot_err = obs_dict.get('robot_err', [])
            if np.size(robot_err) >= self.mj_model.nq and hasattr(self, 'clean_indices'):
                err_vec = robot_err.flatten()
                clean_err_norm = np.linalg.norm(err_vec[self.clean_indices])
                is_deviating = clean_err_norm > 10000
        except Exception:
            is_deviating = False

        return is_fallen or is_deviating

    def reset(self, **kwargs):
        # Reset reference motion cache
        self.ref.reset()

        # Constrain random start time to ensure we don't exceed motion duration
        # Assume max episode length of 5 seconds (conservative estimate)
        max_episode_duration = 5.0  # seconds
        max_start_time = max(0.0, self.ref.reference['time'][-1] - max_episode_duration)

        # Pick a random start time within valid range
        if self.use_rsi and max_start_time > 0:
            # Random start within valid range
            valid_times = self.ref.reference['time'][self.ref.reference['time'] <= max_start_time]
            if len(valid_times) > 0:
                start_time = np.random.choice(valid_times)
            else:
                start_time = self.ref.reference['time'][0]
        else:
            # Use first frame
            start_time = self.ref.reference['time'][0]

        new_ref = self.ref.get_reference(start_time)
        self.ref.index_cache = self.ref.find_timeslot_in_reference(new_ref.time)[0]

        # ADD NOISE TO INITIAL STATE to create a tracking challenge
        # This is CRITICAL - otherwise the agent starts at the target with zero error!
        init_qpos = new_ref.robot.copy()
        init_qvel = new_ref.robot_vel.copy() if new_ref.robot_vel is not None else np.zeros(self.mj_model.nv)

        # Add noise to joint positions (skip root orientation for stability)
        # Root position: indices 0-3
        # Root orientation (quaternion): indices 3-7 (keep stable)
        # Joints: indices 7+
        # INCREASED noise - if noise is too low, the agent never leaves the
        # "peak" of the reward function and doesn't discover the gradient.
        joint_noise_std = 0.05
        root_pos_noise_std = 0.02

        # Add noise to root position (0:3)
        init_qpos[0:3] += self.np_random.normal(0, root_pos_noise_std, 3)

        # Add noise to joint positions (7:)
        if len(init_qpos) > 7:
            init_qpos[7:] += self.np_random.normal(0, joint_noise_std, len(init_qpos) - 7)

        # Add small noise to velocities
        vel_noise_std = 0.05
        init_qvel += self.np_random.normal(0, vel_noise_std, len(init_qvel))

        # Reset with noisy initial state
        obs = super(TrackEnv, self).reset(
            reset_qpos=init_qpos, reset_qvel=init_qvel, **kwargs
        )

        return obs

