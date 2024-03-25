""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com), Pierre Schumacher (schumacherpier@gmail.com), Cameron Berg (cam.h.berg@gmail.com)
================================================= """

import collections
from myosuite.utils import gym
import numpy as np
from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.utils.quat_math import quat2mat


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
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)

        self._setup(**kwargs)


    def _setup(self,
            target_reach_range:dict,
            joint_random_range:tuple=(0.0,0.0),
            far_th = .35,
            obs_keys:list = DEFAULT_OBS_KEYS,
            weighted_reward_keys:dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
            **kwargs,
        ):
        self.far_th = far_th
        self.target_reach_range = target_reach_range
        self.joint_random_range = joint_random_range
        super()._setup(obs_keys=obs_keys,
                weighted_reward_keys=weighted_reward_keys,
                sites=self.target_reach_range.keys(),
                **kwargs,
                )
        self.init_qpos[:] = self.sim.model.key_qpos[0]
        self.init_qvel[:] = self.sim.model.key_qvel[0]
        # find geometries with ID == 1 which indicates the skins
        geom_1_indices = np.where(self.sim.model.geom_group == 1)
        # Change the alpha value to make it transparent
        self.sim.model.geom_rgba[geom_1_indices, 3] = 0

        # move heightfield down if not used
        self.sim.model.geom_rgba[self.sim.model.geom_name2id('terrain')][-1] = 0.0
        self.sim.model.geom_pos[self.sim.model.geom_name2id('terrain')] = np.array([0, 0, -10])


    def get_obs_dict(self, sim):
        obs_dict = {}
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
            ('reach',   10.-1.*reach_dist -10.*vel_dist),
            ('bonus',   1.*(reach_dist<2*near_th) + 1.*(reach_dist<near_th)),
            ('act_reg', -100.*act_mag),
            ('penalty', -1.*(reach_dist>far_th)),
            # Must keys
            ('sparse',  -1.*reach_dist),
            ('solved',  reach_dist<near_th),
            ('done',    reach_dist > far_th),
        ))
        # print(f"reach_dist:{reach_dist}, far_th:{far_th}")
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict


    # generate a valid target
    def generate_targets(self):
        for site, span in self.target_reach_range.items():
            sid = self.sim.model.site_name2id(site)
            sid_target = self.sim.model.site_name2id(site+'_target')
            self.sim.model.site_pos[sid_target] = self.sim.data.site_xpos[sid].copy() + self.np_random.uniform(low=span[0], high=span[1])
        self.sim.forward()


    # generate random qpos for targets (only at linear joints)
    def generate_qpos(self):
        qpos_rand = self.np_random.uniform(low= self.joint_random_range[0], high= self.joint_random_range[1], size=self.init_qpos.shape)
        qpos_new = self.init_qpos.copy()
        qpos_new[self.sim.model.jnt_qposadr] += qpos_rand[self.sim.model.jnt_qposadr] # only linear joints
        qpos_new[self.sim.model.jnt_qposadr] = np.clip(qpos_new[self.sim.model.jnt_qposadr], self.sim.model.jnt_range[:,0], self.sim.model.jnt_range[:,1])
        return qpos_new


    def reset(self, **kwargs):
        # generate random targets
        if np.ptp(self.joint_random_range)>0:
            self.sim.data.qpos = self.generate_qpos()
        self.sim.forward()
        self.generate_targets()

        # sync targets to sim_obsd
        self.robot.sync_sims(self.sim, self.sim_obsd)

        # generate resets
        if np.ptp(self.joint_random_range)>0:
            obs = super().reset(reset_qpos= self.generate_qpos(), **kwargs)
        else:
            obs = super().reset(**kwargs)
        return obs

class WalkEnvV0(BaseV0):

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
        "vel_reward": 5.0,
        "done": -100,
        "cyclic_hip": -10,
        "ref_rot": 10.0,
        "joint_angle_rew": 5.0
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
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)
        self._setup(**kwargs)

    def _setup(self,
               obs_keys: list = DEFAULT_OBS_KEYS,
               weighted_reward_keys: dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
               min_height = 0.8,
               max_rot = 0.8,
               hip_period = 100,
               reset_type='init',
               target_x_vel=0.0,
               target_y_vel=1.2,
               target_rot = None,
               **kwargs,
               ):
        self.min_height = min_height
        self.max_rot = max_rot
        self.hip_period = hip_period
        self.reset_type = reset_type
        self.target_x_vel = target_x_vel
        self.target_y_vel = target_y_vel
        self.target_rot = target_rot
        self.steps = 0
        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       **kwargs
                       )
        self.init_qpos[:] = self.sim.model.key_qpos[0]
        self.init_qvel[:] = 0.0

        # move heightfield down if not used
        self.sim.model.geom_rgba[self.sim.model.geom_name2id('terrain')][-1] = 0.0
        self.sim.model.geom_pos[self.sim.model.geom_name2id('terrain')] = np.array([0, 0, -10])

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
        obs_dict['phase_var'] = np.array([(self.steps/self.hip_period) % 1]).copy()
        obs_dict['muscle_length'] = self.muscle_lengths()
        obs_dict['muscle_velocity'] = self.muscle_velocities()
        obs_dict['muscle_force'] = self.muscle_forces()

        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()

        return obs_dict

    def get_reward_dict(self, obs_dict):
        vel_reward = self._get_vel_reward()
        cyclic_hip = self._get_cyclic_rew()
        ref_rot = self._get_ref_rotation_rew()
        joint_angle_rew = self._get_joint_angle_rew(['hip_adduction_l', 'hip_adduction_r', 'hip_rotation_l',
                                                       'hip_rotation_r'])
        act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('vel_reward', vel_reward),
            ('cyclic_hip',  cyclic_hip),
            ('ref_rot',  ref_rot),
            ('joint_angle_rew', joint_angle_rew),
            ('act_mag', act_mag),
            # Must keys
            ('sparse',  vel_reward),
            ('solved',    vel_reward >= 1.0),
            ('done',  self._get_done()),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    def get_randomized_initial_state(self):
        # randomly start with flexed left or right knee
        if  self.np_random.uniform() < 0.5:
            qpos = self.sim.model.key_qpos[2].copy()
            qvel = self.sim.model.key_qvel[2].copy()
        else:
            qpos = self.sim.model.key_qpos[3].copy()
            qvel = self.sim.model.key_qvel[3].copy()

        # randomize qpos coordinates
        # but dont change height or rot state
        rot_state = qpos[3:7]
        height = qpos[2]
        qpos[:] = qpos[:] + self.np_random.normal(0, 0.02, size=qpos.shape)
        qpos[3:7] = rot_state
        qpos[2] = height
        return qpos, qvel

    def step(self, *args, **kwargs):
        results = super().step(*args, **kwargs)
        self.steps += 1
        return results

    def reset(self, **kwargs):
        self.steps = 0
        if self.reset_type == 'random':
            qpos, qvel = self.get_randomized_initial_state()
        elif self.reset_type == 'init':
                qpos, qvel = self.sim.model.key_qpos[2], self.sim.model.key_qvel[2]
        else:
            qpos, qvel = self.sim.model.key_qpos[0], self.sim.model.key_qvel[0]
        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super().reset(reset_qpos=qpos, reset_qvel=qvel, **kwargs)
        return obs

    def muscle_lengths(self):
        return self.sim.data.actuator_length

    def muscle_forces(self):
        return np.clip(self.sim.data.actuator_force / 1000, -100, 100)

    def muscle_velocities(self):
        return np.clip(self.sim.data.actuator_velocity, -100, 100)

    def _get_done(self):
        height = self._get_height()
        if height < self.min_height:
            return 1
        if self._get_rot_condition():
            return 1
        return 0

    def _get_joint_angle_rew(self, joint_names):
        """
        Get a reward proportional to the specified joint angles.
        """
        mag = 0
        joint_angles = self._get_angle(joint_names)
        mag = np.mean(np.abs(joint_angles))
        return np.exp(-5 * mag)

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

    def _get_vel_reward(self):
        """
        Gaussian that incentivizes a walking velocity. Going
        over only achieves flat rewards.
        """
        vel = self._get_com_velocity()
        return np.exp(-np.square(self.target_y_vel - vel[1])) + np.exp(-np.square(self.target_x_vel - vel[0]))

    def _get_cyclic_rew(self):
        """
        Cyclic extension of hip angles is rewarded to incentivize a walking gait.
        """
        phase_var = (self.steps/self.hip_period) % 1
        des_angles = np.array([0.8 * np.cos(phase_var * 2 * np.pi + np.pi), 0.8 * np.cos(phase_var * 2 * np.pi)], dtype=np.float32)
        angles = self._get_angle(['hip_flexion_l', 'hip_flexion_r'])
        return np.linalg.norm(des_angles - angles)

    def _get_ref_rotation_rew(self):
        """
        Incentivize staying close to the initial reference orientation up to a certain threshold.
        """
        target_rot = [self.target_rot if self.target_rot is not None else self.init_qpos[3:7]][0]
        return np.exp(-np.linalg.norm(5.0 * (self.sim.data.qpos[3:7] - target_rot)))

    def _get_torso_angle(self):
        body_id = self.sim.model.body_name2id('torso')
        return self.sim.data.body_xquat[body_id]

    def _get_com_velocity(self):
        """
        Compute the center of mass velocity of the model.
        """
        mass = np.expand_dims(self.sim.model.body_mass, -1)
        cvel = - self.sim.data.cvel
        return (np.sum(mass * cvel, 0) / np.sum(mass))[3:5]

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
        quat = self.sim.data.qpos[3:7].copy()
        return [1 if np.abs((quat2mat(quat) @ [1, 0, 0])[0]) > self.max_rot else 0][0]

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

class TerrainEnvV0(WalkEnvV0):

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
        "vel_reward": 5.0,
        "done": -100,
        "cyclic_hip": -10,
        "ref_rot": 10.0,
        "joint_angle_rew": 5.0
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
        BaseV0.__init__(self, model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)
        self._setup(**kwargs)

    def _setup(self,
               obs_keys: list = DEFAULT_OBS_KEYS,
               weighted_reward_keys: dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
               min_height = 0.8,
               max_rot = 0.8,
               hip_period = 100,
               reset_type='init',
               target_x_vel=0.0,
               target_y_vel=1.2,
               target_rot = None,
               terrain = 'rough',
               variant = None,
               **kwargs,
               ):
        self.min_height = min_height
        self.max_rot = max_rot
        self.hip_period = hip_period
        self.reset_type = reset_type
        self.target_x_vel = target_x_vel
        self.target_y_vel = target_y_vel
        self.target_rot = target_rot
        self.terrain = terrain
        self.variant = variant
        self.steps = 0

        BaseV0._setup(self, obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       **kwargs
                       )
        self.init_qpos[:] = self.sim.model.key_qpos[0]
        self.init_qvel[:] = 0.0

    def reset(self, **kwargs):
        self.steps = 0
        if self.terrain == 'rough':
            rough = self.np_random.uniform(low=-.5, high=.5, size=(10000,))
            normalized_data = (rough - np.min(rough)) / (np.max(rough) - np.min(rough))
            scalar, offset = .08, .02
            self.sim.model.hfield_data[:] = normalized_data*scalar - offset

        elif self.terrain == 'hilly':
            flat_length, frequency = 3000, 3
            scalar = 0.63 if self.variant == 'fixed' else self.np_random.uniform(low=0.53, high=0.73)

            combined_data = np.concatenate((-2 * np.ones(flat_length), -2 + 0.5 * (np.sin(np.linspace(0, frequency*np.pi, int(1e4-flat_length)) + np.pi/2) - 1)))
            normalized_data = (combined_data - combined_data.min()) / (combined_data.max() - combined_data.min())

            self.sim.model.hfield_data[:] = np.flip(normalized_data.reshape(100,100)*scalar, [0,1]).reshape(10000,)

        elif self.terrain == 'stairs':
            num_stairs = 12
            stair_height = .1
            flat = 5200 - (1e4 - 5200) % num_stairs
            stairs_width = (1e4 - flat) // num_stairs
            scalar = 2.5 if self.variant == 'fixed' else self.np_random.uniform(low=1.5, high=3.5)

            stair_parts = [np.full((int(stairs_width//100), 100), -2 + stair_height * j) for j in range(num_stairs)]
            new_terrain_data = np.concatenate([np.full((int(flat // 100), 100), -2)] + stair_parts, axis=0)

            normalized_data = (new_terrain_data + 2) / (2 + stair_height * num_stairs)
            self.sim.model.hfield_data[:] = np.flip(normalized_data.reshape(100,100)*scalar, [0,1]).reshape(10000,)

        self.sim.model.geom_rgba[self.sim.model.geom_name2id('terrain')][-1] = 1.0
        self.sim.model.geom_pos[self.sim.model.geom_name2id('terrain')] = np.array([0,0,0])
        self.sim.model.geom_contype[self.sim.model.geom_name2id('terrain')] = 1
        self.sim.model.geom_conaffinity[self.sim.model.geom_name2id('terrain')] = 1

        if self.reset_type == 'random':
            qpos, qvel = self.get_randomized_initial_state()
        elif self.reset_type == 'init':
                qpos, qvel = self.sim.model.key_qpos[2], self.sim.model.key_qvel[2]
        else:
            qpos, qvel = self.sim.model.key_qpos[0], self.sim.model.key_qvel[0]
        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = BaseV0.reset(self, reset_qpos=qpos, reset_qvel=qvel, **kwargs)
        return obs

    def _get_done(self):
        height = self._get_height()
        if height < self.min_height:
            return 1
        if self._get_rot_condition():
            return 1
        if self._get_knee_condition():
            return 1
        return 0

    def _get_knee_condition(self):
        """
        Checks if the agent is on its knees by comparing the distance between the center of mass and the feet.
        """
        feet_heights = self._get_feet_heights()
        com_height = self._get_height()
        if com_height - np.mean(feet_heights) < .61:
            return 1
        else:
            return 0
