""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

import collections
import gym
import numpy as np
import mujoco_py

from myosuite.envs.myo.base_v0 import BaseV0
from scipy.spatial.transform import Rotation as R



class ReachEnvV0(BaseV0):

    DEFAULT_OBS_KEYS = ['qpos', 'qvel', 'tip_pos', 'reach_err']
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "reach": 1.0,
        "bonus": 4.0,
        "penalty": 50,
        "act_reg": 1
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):

        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed)

        self._setup(**kwargs)


    def _setup(self,
            target_reach_range:dict,
            far_th = .35,
            obs_keys:list = DEFAULT_OBS_KEYS,
            weighted_reward_keys:dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
            **kwargs,
        ):
        self.far_th = far_th
        self.target_reach_range = target_reach_range
        super()._setup(obs_keys=obs_keys,
                weighted_reward_keys=weighted_reward_keys,
                sites=self.target_reach_range.keys(),
                **kwargs,
                )
        self.init_qpos[:] = self.sim.model.key_qpos[0]
        self.update_camera(distance=3.0)

    def get_obs_vec(self):
        self.obs_dict['time'] = np.array([self.sim.data.time])
        self.obs_dict['qpos'] = self.sim.data.qpos[:].copy()
        self.obs_dict['qvel'] = self.sim.data.qvel[:].copy()*self.dt
        if self.sim.model.na>0:
            self.obs_dict['act'] = self.sim.data.act[:].copy()

        # reach error
        self.obs_dict['tip_pos'] = np.array([])
        self.obs_dict['target_pos'] = np.array([])
        for isite in range(len(self.tip_sids)):
            self.obs_dict['tip_pos'] = np.append(self.obs_dict['tip_pos'], self.sim.data.site_xpos[self.tip_sids[isite]].copy())
            self.obs_dict['target_pos'] = np.append(self.obs_dict['target_pos'], self.sim.data.site_xpos[self.target_sids[isite]].copy())
        self.obs_dict['reach_err'] = np.array(self.obs_dict['target_pos'])-np.array(self.obs_dict['tip_pos'])

        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['t'] = np.array([sim.data.time])
        obs_dict['time'] = np.array([sim.data.time])
        obs_dict['qpos'] = sim.data.qpos[:].copy()
        obs_dict['qvel'] = sim.data.qvel[:].copy()*self.dt
        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()

        # reach error
        obs_dict['tip_pos'] = np.array([])
        obs_dict['target_pos'] = np.array([])
        for isite in range(len(self.tip_sids)):
            obs_dict['tip_pos'] = np.append(obs_dict['tip_pos'], sim.data.site_xpos[self.tip_sids[isite]].copy())
            obs_dict['target_pos'] = np.append(obs_dict['target_pos'], sim.data.site_xpos[self.target_sids[isite]].copy())
        obs_dict['reach_err'] = np.array(obs_dict['target_pos'])-np.array(obs_dict['tip_pos'])
        return obs_dict

    def get_reward_dict(self, obs_dict):
        reach_dist = np.linalg.norm(obs_dict['reach_err'], axis=-1)
        vel_dist = np.linalg.norm(obs_dict['qvel'], axis=-1)
        act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0
        far_th = self.far_th*len(self.tip_sids) if np.squeeze(obs_dict['time'])>2*self.dt else np.inf
        # near_th = len(self.tip_sids)*.0125
        near_th = len(self.tip_sids)*.050
        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('reach',   -1.*reach_dist -10.*vel_dist),
            ('bonus',   1.*(reach_dist<2*near_th) + 1.*(reach_dist<near_th)),
            ('act_reg', -100.*act_mag),
            ('penalty', -1.*(reach_dist>far_th)),
            # Must keys
            ('sparse',  -1.*reach_dist),
            ('solved',  reach_dist<near_th),
            ('done',    reach_dist > far_th),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    # generate a valid target
    def generate_target_pose(self):
        for site, span in self.target_reach_range.items():
            sid =  self.sim.model.site_name2id(site+'_target')
            self.sim.model.site_pos[sid] = self.np_random.uniform(low=span[0], high=span[1])
        self.sim.forward()


    def reset(self):
        self.generate_target_pose()
        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super().reset()
        return obs


    def viewer_setup(self):
        self.viewer.cam.azimuth = 90
        self.viewer.cam.elevation = -15
        self.viewer.cam.distance = 5.0
        self.viewer.vopt.flags[3] = 1 # render actuators
        self.sim.forward()


class WalkStraightEnvV0(ReachEnvV0):

    DEFAULT_OBS_KEYS = [
        'qpos_without_xy',
        'qvel',
        'com_vel',
        'torso_angle',
        'feet_heights',
        'height',
        'feet_rel_positions',
        'phase_var',
        'muscle_length',
        'muscle_velocity',
        'muscle_force'
    ]

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "forward": 5.0,
        "self_contact": -0.2,
        "done": -100,
        "cyclic_hip": -10,
        "ref_rot": 10.0,
        "hip_rotation": 5.0
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):

        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super(ReachEnvV0, self).__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed)
        self._prepare_initial_settings(kwargs)
        self._setup(**kwargs)

    def _setup(self,
               obs_keys: list = DEFAULT_OBS_KEYS,
               weighted_reward_keys: dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
               **kwargs,
               ):
        super(ReachEnvV0, self)._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       **kwargs
                       )
        self.init_qpos[:] = self.sim.model.key_qpos[0]
        self.init_qvel[:] = 0.0
        self.update_camera(distance=3.0)

    def _prepare_initial_settings(self, kwargs):
        self._steps = 0
        self._min_height = kwargs.pop('min_height')
        self._max_rot = kwargs.pop('max_rot')
        self._hip_period = kwargs.pop('hip_period')
        # initial root position and orientation
        self._init_pos_root = np.array([
            0.0000e+00,
            0.0000e+00,
            9.2000e-01,
            7.0739e-01,
            0.0000e+00,
            0.0000e+00,
            -7.0683e-01])

    def get_obs_vec(self):
        self.obs_dict['time'] = np.array([self.sim.data.time])
        self.obs_dict['qpos_without_xy'] = self.sim.data.qpos[2:].copy()
        self.obs_dict['qvel'] = self.sim.data.qvel[:].copy() * self.dt
        self.obs_dict['com_vel'] = np.array([self._get_com_velocity().copy()])
        self.obs_dict['torso_angle'] = np.array([self._get_torso_angle().copy()])
        self.obs_dict['feet_heights'] = self._get_feet_heights().copy()
        self.obs_dict['height'] = np.array([self._get_height()]).copy()
        self.obs_dict['feet_rel_positions'] = self._get_feet_relative_position().copy()
        self.obs_dict['phase_var'] = np.array([(self._steps/self._hip_period) % 1]).copy()
        self.obs_dict['muscle_length'] = self.muscle_lengths()
        self.obs_dict['muscle_velocity'] = self.muscle_velocities()
        self.obs_dict['muscle_force'] = self.muscle_forces()

        if self.sim.model.na>0:
            self.obs_dict['act'] = self.sim.data.act[:].copy()

        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['t'] = np.array([sim.data.time])
        obs_dict['time'] = np.array([sim.data.time])
        obs_dict['qpos_without_xy'] = sim.data.qpos[2:].copy()
        obs_dict['qvel'] = sim.data.qvel[:].copy() * self.dt
        obs_dict['com_vel'] = np.array([self._get_com_velocity().copy()])
        obs_dict['torso_angle'] = np.array([self._get_torso_angle().copy()])
        obs_dict['feet_heights'] = self._get_feet_heights().copy()
        obs_dict['height'] = np.array([self._get_height()]).copy()
        obs_dict['feet_rel_positions'] = self._get_feet_relative_position().copy()
        obs_dict['phase_var'] = np.array([(self._steps/self._hip_period) % 1]).copy()
        obs_dict['muscle_length'] = self.muscle_lengths()
        obs_dict['muscle_velocity'] = self.muscle_velocities()
        obs_dict['muscle_force'] = self.muscle_forces()

        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()

        return obs_dict

    def get_reward_dict(self, obs_dict):
        forward = self._get_forward_reward()
        self_contact = self._get_contact_force()
        cyclic_hip = self._get_cyclic_rew()
        ref_rot = self._get_ref_deviation_cost()
        hip_rotation = self._get_hip_rot_cost()
        act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('forward', forward),
            ('self_contact', self_contact),
            ('cyclic_hip',  cyclic_hip),
            ('ref_rot',  ref_rot),
            ('hip_rotation', hip_rotation),
            ('act_mag', act_mag),
            # Must keys
            ('sparse',  forward),
            ('solved',    forward >= 1.0),
            ('done',  self._get_done()),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    def get_randomized_initial_state(self):
        qpos, qvel = self._get_initial_state()
        if np.random.uniform() < 0.5:
            # randomly start with flexed left or right knee
            qpos, qvel = self._switch_legs(qpos, qvel)
        rot_state = qpos[3:7]
        height = qpos[2]
        qpos[:] = qpos[:] + np.random.normal(0, 0.02, size=qpos.shape)
        qpos[3:7] = rot_state
        qpos[2] = height
        return qpos, qvel

    def step(self, *args, **kwargs):
        obs, reward, done, info = super().step(*args, **kwargs)
        self._steps += 1
        return obs, reward, done, info

    def reset(self):
        self._steps = 0
        qpos, qvel = self.get_randomized_initial_state()
        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super(ReachEnvV0, self).reset(reset_qpos=qpos, reset_qvel=qvel)
        return obs

    def muscle_lengths(self):
        return self.sim.data.actuator_length

    def muscle_forces(self):
        return np.clip(self.sim.data.actuator_force / 1000, -100, 100)

    def muscle_velocities(self):
        return np.clip(self.sim.data.actuator_velocity, -100, 100)

    def _get_done(self):
        height = self._get_height()
        if height < self._min_height:
            return 1
        if self._get_rot_condition():
            return 1
        return 0

    def _get_hip_rot_cost(self):
        """
        Get a cost proportional to the rotation of the hip.
        """
        joint_names = ['hip_rotation_l', 'hip_rotation_r']
        mag = 0
        hip_rot_angles = self._get_angle(joint_names)
        mag = np.sum(np.clip(np.abs(hip_rot_angles) - 0.2, 0, 1000))
        return np.exp(- 5 * mag)

    def _get_contact_force(self):
        """
        Get non-ground related contact forces. (i.e. self-contact from leg-on-leg)
        """
        norm = 0
        # mujoco_py function needs to write into a c-style array
        c_array = np.zeros(6, dtype=np.float64)
        for i in range(self.sim.data.ncon):
            if self.sim.data.contact[i].geom1 == 0 or self.sim.data.contact[i].geom2 == 0:
                continue
            mujoco_py.functions.mj_contactForce(self.sim.model, self.sim.data, i, c_array)
            # in non-ground body collisions, only the first element is non-zero atm.
            norm += c_array[0]
        return np.clip(norm, -100, 100) / 100

    def _get_feet_heights(self):
        """
        Get the height of both feet.
        """
        foot_id_l = self.sim.model.body_name2id('talus_l')
        foot_id_r = self.sim.model.body_name2id('talus_r')
        return np.array([self.sim.data.body_xpos[foot_id_l][2], self.sim.data.body_xpos[foot_id_r][2]])

    def _get_feet_relative_position(self):
        """
        Get the feet positions relative to the pelvis.
        """
        foot_id_l = self.sim.model.body_name2id('talus_l')
        foot_id_r = self.sim.model.body_name2id('talus_r')
        pelvis = self.sim.model.body_name2id('pelvis')
        return np.array([self.sim.data.body_xpos[foot_id_l]-self.sim.data.body_xpos[pelvis], self.sim.data.body_xpos[foot_id_r]-self.sim.data.body_xpos[pelvis]])

    def _get_forward_reward(self):
        """
        Gaussian that incentivizes a walking velocity. Going
        over only achieves flat rewards.
        """
        vel = self._get_com_velocity()
        if vel > 1.2:
            return 1
        else:
            return np.exp(-np.square(1.2 - vel))

    def _get_cyclic_rew(self):
        """
        Cyclic extension of hip angles is rewarded to incentivize a walking gait.
        """
        phase_var = (self._steps/self._hip_period) % 1
        des_angles = np.array([0.8 * np.cos(phase_var * 2 * np.pi + np.pi), 0.8 * np.cos(phase_var * 2 * np.pi)], dtype=np.float32)
        angles = self._get_angle(['hip_flexion_l', 'hip_flexion_r'])
        return np.linalg.norm(des_angles - angles)

    def _get_ref_deviation_cost(self):
        """
        Incentivize staying close to the initial reference orientation up to a certain threshold.
        """
        return np.exp(-max(0.0, np.linalg.norm(5.0 * (self.sim.data.qpos[3:7] - self._init_pos_root[3:7])) - 0.1))

    def _get_torso_angle(self):
        body_id = self.sim.model.body_name2id('torso')
        return self.sim.data.body_xquat[body_id]

    def _get_com_velocity(self):
        """
        Compute the center of mass velocity of the model.
        """
        mass = np.expand_dims(self.sim.model.body_mass, -1)
        cvel = - self.sim.data.cvel
        return (np.sum(mass * cvel, 0) / np.sum(mass))[4]

    def _get_height(self):
        """
        Get center-of-mass height.
        """
        return self._get_com()[2]

    def _get_rot_condition(self):
        """
        MuJoCo specifies the orientation as a quaternion representing the rotation
        from the [1,0,0] vector to the orientation vector. To check if
        a body is facing in the right direction, we can check if the
        quaternion when applied to the vector [1,0,0] as a rotation
        yields a vector with a strong x component.
        """
        # quaternion of root
        quat = np.roll(self.sim.data.qpos[3:7], -1)
        return [1 if np.abs(R.from_quat(quat).apply([1, 0, 0])[0]) > self._max_rot else 0][0]

    def _get_com(self):
        """
        Compute the center of mass of the robot.
        """
        mass = np.expand_dims(self.sim.model.body_mass, -1)
        com =  self.sim.data.xipos
        return (np.sum(mass * com, 0) / np.sum(mass))

    def _get_angle(self, names):
        """
        Get the angles of a list of named joints.
        """
        return np.array([self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id(name)]] for name in names])

    def _get_initial_state(self):
        """
        Set the body to the scone state. The initial file in scone might be a good
        starting state for mujoco.
        The values for the initial joint angles and velocities were copied and slightly
        modified from the SCONE simulator, which is written by:
        Thomas Geijtenbeek <thomas@goatstream.com>
        """
        qpos = self.init_qpos.copy()
        # positions
        qpos[0] = 0  # pelvis x
        qpos[1] = 0  # pelvis y
        qpos[2] = 1.0  # pelvis z
        qpos[7] = -0.1652  # hip_flexion_r
        qpos[12] = +0.0888  # knee_angle_r
        qpos[15] = -0.019  # ankle_angle_r
        qpos[21] = -0.2326  # hip_flexion_l
        qpos[26] = +1.227  # knee_angle_l
        qpos[29] = +0.1672  # ankle_angle_l

        # velocities
        qvel = self.init_qvel.copy()
        qvel[0] = 0  # pelvis x
        qvel[1] = -1.5  # pelvis y
        qvel[2] = 0.0  # pelvis z
        qvel[6] = -0.576  # hip_flexion_r
        qvel[11] = +0.175  # knee_angle_r
        qvel[14] = +0.988  # ankle_angle_r
        qvel[20] = +4.9066  # hip_flexion_l
        qvel[25] = -3.59786  # knee_angle_l
        qvel[28] = +0.633  # ankle_angle_l
        return qpos, qvel

    def _switch_legs(self, qpos, qvel):
        """
        Switches the joint angle and velocities symmetrically
        between both legs, given as qpos and qvel arrays.
        """
        qpos[7], qpos[21] = qpos[21], qpos[7]   # hip_flexion
        qpos[12], qpos[26] = qpos[26], qpos[12] # knee_angle
        qpos[15],qpos[29] = qpos[29], qpos[15]  # ankle_angle

        qvel[7], qvel[21] = qvel[21], qvel[7]   # hip_flexion
        qvel[12], qvel[26] = qvel[26], qvel[12] # knee_angle
        qvel[15], qvel[29] = qvel[29], qvel[15]  # ankle_angle
        return qpos, qvel
