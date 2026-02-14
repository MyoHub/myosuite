import collections
import numpy as np
from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.utils.quat_math import quat2euler

class MyoSkeletonStandEnv(BaseV0):
    DEFAULT_OBS_KEYS = ["qp", "qv", "stand_err"]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pose": 5.0,      # Similarity to standing qpos
        "root_pos": 2.0,  # Keep root at (0,0,0.95)
        "alive": 10.0,    # SIGNIFICANTLY INCREASED Survival bonus
        "upright": 5.0,   # INCREASED torso vertical reward
        "quat_sim": 5.0,  # Match reference root orientation
        "vel_penalty": 1.0, # Discourage spazzing
    }

    def __init__(self, model_path, reference_path=None, **kwargs):
        self.reference_path = reference_path
        super().__init__(model_path=model_path)
        self._setup(**kwargs)

    def _setup(self, **kwargs):
        self.initialized_pos = False
        self.muscle_condition = kwargs.get('muscle_condition', "")
        self.fatigue_reset_vec = kwargs.get('fatigue_reset_vec', None)
        self.fatigue_reset_random = kwargs.get('fatigue_reset_random', False)

        # Load reference if path provided
        if self.reference_path:
            from myosuite.logger.grouped_datasets import Trace
            from myosuite.utils.quat_math import mulQuat, negQuat
            traj = Trace.load(self.reference_path)
            motion_name = list(traj.trace.keys())[0]
            motion_data = traj.trace[motion_name]

            # Construct target_qpos from reference (first frame)
            self.target_qpos = np.zeros(self.mj_model.nq)

            # Mapping logic similar to track_v0
            joint_names = [self.mj_model.joint(i).name for i in range(self.mj_model.njnt)]
            root_key = 'myoskeleton_root'
            if root_key in motion_data['qpos']:
                root_target = np.array(motion_data['qpos'][root_key])[0]
                # COMPENSATE for pelvis offset in XML
                # Joint = Goal * inv(Pelvis_Offset)
                pelvis_id = self.mj_model.body('pelvis').id
                pelvis_offset = self.mj_model.body_quat[pelvis_id]

                # root_target[:3] is pos, [3:7] is quat
                self.target_qpos[3:7] = mulQuat(root_target[3:7], negQuat(pelvis_offset))
                self.target_qpos[:3] = root_target[:3]

                # Re-center: set x, y to 0 to avoid starting 'done' due to distance
                self.target_qpos[0] = 0.0
                self.target_qpos[1] = 0.0

            # Store target pelvis orientation for reward
            from myosuite.utils.quat_math import quat2euler
            # Since we compensated, the target global pelvis orientation IS the reference root quat
            self.target_pelvis_quat = root_target[3:7].copy()

            for i, name in enumerate(joint_names):
                if name == root_key: continue
                if name in motion_data['qpos']:
                    q_adr = int(self.mj_model.joint(i).qposadr)
                    self.target_qpos[q_adr] = np.array(motion_data['qpos'][name])[0]
        else:
            self.target_qpos = self.mj_model.key_qpos[0].copy() if self.mj_model.nkey > 0 else self.mj_model.qpos0.copy()
            self.target_pelvis_quat = np.array([1, 0, 0, 0])

        self.target_root_pos = self.target_qpos[:3].copy()

        self.clean_indices = []
        for i in range(self.mj_model.njnt):
            jnt_type = self.mj_model.jnt_type[i]
            q_adr = self.mj_model.jnt_qposadr[i]
            if jnt_type == 3: # Hinge
                self.clean_indices.append(int(q_adr))
        self.clean_indices = np.array(self.clean_indices)

        # Call BaseV0._setup which calls MujocoEnv._setup
        super()._setup(
            obs_keys=kwargs.pop('obs_keys', self.DEFAULT_OBS_KEYS),
            weighted_reward_keys=kwargs.pop('weighted_reward_keys', self.DEFAULT_RWD_KEYS_AND_WEIGHTS),
            **kwargs
        )
        self.initialized_pos = True

    def reset(self, **kwargs):
        # Always start at target_qpos for standing task
        obs = super().reset(reset_qpos=self.target_qpos, **kwargs)
        return obs

    def get_obs_dict(self, mj_model, mj_data):
        obs_dict = {}
        obs_dict['time'] = np.array([mj_data.time])
        obs_dict['qp'] = mj_data.qpos.copy()
        obs_dict['qv'] = mj_data.qvel.copy()
        obs_dict['stand_err'] = obs_dict['qp'] - self.target_qpos

        # Add global pelvis orientation to obs_dict
        pelvis_id = mj_model.body('pelvis').id
        obs_dict['pelvis_quat'] = mj_data.xquat[pelvis_id].copy()
        return obs_dict

    def get_reward_dict(self, obs_dict):
        # Use flattened versions to handle (1,1,N) batched shapes from MJX/wrappers
        qp = obs_dict['qp'].flatten()
        qv = obs_dict['qv'].flatten()
        stand_err = obs_dict['stand_err'].flatten()
        pelvis_quat = obs_dict['pelvis_quat'].flatten()

        # 1. Pose Reward (Hinge joints only)
        clean_err = stand_err[self.clean_indices]
        mae_pose = np.mean(np.abs(clean_err))
        pose_reward = np.exp(-2.0 * mae_pose)

        # 2. Root Position Reward (X, Y, Z)
        root_now = qp[:3]
        root_err = np.linalg.norm(root_now - self.target_root_pos)
        root_reward = np.exp(-4.0 * root_err)

        # 3. Upright Reward (Relative to target orientation)
        euler = quat2euler(pelvis_quat)
        target_euler = quat2euler(self.target_pelvis_quat)
        upright_reward = np.exp(-5.0 * (np.abs(euler[0] - target_euler[0]) + np.abs(euler[1] - target_euler[1])))

        # 4. Root Orientation Reward (Direct quat similarity)
        dot_product = np.abs(np.sum(pelvis_quat * self.target_pelvis_quat))
        dot_product = np.clip(dot_product, 0, 1.0)
        quat_reward = dot_product**2

        # 5. Velocity Penalty (to discourage spazzing)
        vel_penalty = np.mean(np.square(qv))

        # 6. Alive Bonus
        alive_bonus = 1.0

        rwd_dict = collections.OrderedDict((
            ("pose", pose_reward),
            ("root_pos", root_reward),
            ("upright", upright_reward),
            ("quat_sim", quat_reward),
            ("vel_penalty", -0.01 * vel_penalty),
            ("alive", alive_bonus),
            ("sparse", 0),
            ("solved", 0),
            ("done", self.check_termination(obs_dict)),
        ))

        # Update weights to include new keys
        weights = self.DEFAULT_RWD_KEYS_AND_WEIGHTS.copy()
        if "quat_sim" not in weights: weights["quat_sim"] = 5.0
        if "vel_penalty" not in weights: weights["vel_penalty"] = 1.0

        rwd_dict['dense'] = np.sum([weights.get(key, 0) * rwd_dict[key] for key in rwd_dict.keys()], axis=0)
        return rwd_dict

    def check_termination(self, obs_dict):
        if not self.initialized_pos:
            return False

        # Use flattened qpos to handle (1,1,N) shapes
        qp = obs_dict['qp'].flatten()
        pelvis_quat = obs_dict['pelvis_quat'].flatten()

        # Terminate if root Z drops too low
        root_z = qp[2]
        if root_z < 0.6: # Slightly higher threshold
            # print(f"DEBUG: Terminated due to low Z: {root_z}")
            return True
        # Terminate if root moves too far horizontally (escaped)
        root_dist = np.linalg.norm(qp[:2] - self.target_root_pos[:2])
        if root_dist > 0.5: # Harder constraint
            # print(f"DEBUG: Terminated due to root dist: {root_dist}")
            return True

        # Terminate if pelvis tilts too much (relative to target posture)
        euler = quat2euler(pelvis_quat)
        target_euler = quat2euler(self.target_pelvis_quat)
        diff = np.abs(euler - target_euler)
        if diff[0] > 1.0 or diff[1] > 1.0:
            # print(f"DEBUG: Terminated due to tilt: {euler} vs {target_euler}")
            return True

        return False
