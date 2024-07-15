""" =================================================
# Copyright (c) MyoSuite Authors
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com), Pierre Schumacher (schumacherpier@gmail.com), Chun Kwang Tan (cktan.neumove@gmail.com)
================================================= """

import collections
from myosuite.utils import gym
import numpy as np
import os
from enum import Enum
from typing import Optional, Tuple
import copy
import csv

from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.envs.myo.myobase.walk_v0 import WalkEnvV0
from myosuite.utils.quat_math import quat2euler, euler2mat, euler2quat
from myosuite.envs.heightfields import TrackField
from myosuite.envs.myo.assets.leg.myoosl_control import MyoOSLController


class TrackTypes(Enum):
    FLAT = 0
    HILLY = 1
    ROUGH = 2
    STAIRS = 3


class RunTrack(WalkEnvV0):

    DEFAULT_OBS_KEYS = [
        'internal_qpos',
        'internal_qvel',
        'grf',
        'torso_angle',
        'model_root_pos',
        'model_root_vel',
        'muscle_length',
        'muscle_velocity',
        'muscle_force',
    ]

    # You can change reward weights here
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "sparse": 1,
        "solved": +10,
    }

    # OSL-related paramters
    ACTUATOR_PARAM = {}
    OSL_PARAM_LIST = []
    OSL_PARAM_SELECT = 0

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        # This flag needs to be here to prevent the simulation from starting in a done state
        # Before setting the key_frames, the model and opponent will be in the cartesian position,
        # causing the step() function to evaluate the initialization as "done".

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
               reset_type='init',
               terrain='FLAT',
               hills_difficulties=(0,0),
               rough_difficulties=(0,0),
               stairs_difficulties=(0,0),
               real_length=15,
               real_width=1,
               distance_thr = 10,
               init_pose_path=None,
               **kwargs,
               ):

        self.startFlag = False
        # Terrain type
        self.terrain_type = TrackTypes.FLAT.value

        # Env initialization with data
        if init_pose_path is not None:
            file_path = os.path.join(init_pose_path)
            self.INIT_DATA = np.loadtxt(file_path, skiprows=1, delimiter=',')
            self.init_lookup = self.generate_init_lookup(keys=np.arange(48), value='e_swing')
            self.init_lookup = self.generate_init_lookup(keys=np.arange(48, 99), value='l_swing', existing_dict=self.init_lookup)
            self.init_lookup = self.generate_init_lookup(keys=np.arange(99, 183), value='e_stance', existing_dict=self.init_lookup)
            self.init_lookup = self.generate_init_lookup(keys=np.arange(183, 247), value='l_stance', existing_dict=self.init_lookup)
            with open(file_path) as csv_file:
                csv_reader = csv.DictReader(csv_file)
                temp = dict(list(csv_reader)[0])
                headers = list(temp.keys())
            self.gait_cycle_headers = dict(zip(headers, range(len(headers))))

        # OSL specific init
        self.OSL_CTRL = MyoOSLController(np.sum(self.sim.model.body_mass), init_state='e_stance')
        self.OSL_CTRL.start()

        self.muscle_space = self.sim.model.na # muscles only
        self.full_ctrl_space = self.sim.model.nu # Muscles + actuators
        self._get_actuator_params()

        self._setup_convenience_vars()
        self.distance_thr = distance_thr
        self.trackfield = TrackField(
            sim=self.sim,
            rng=self.np_random,
            rough_difficulties=rough_difficulties,
            hills_difficulties=hills_difficulties,
            stairs_difficulties=stairs_difficulties,
        )
        self.real_width = real_width
        self.real_length = real_length
        self.reset_type = reset_type
        self.terrain = terrain
        self.grf_sensor_names = ['l_foot', 'l_toes']
        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       reset_type=reset_type,
                       **kwargs
                       )
        self.init_qpos[:] = self.sim.model.keyframe('stand').qpos.copy()
        self.init_qvel[:] = 0.0
        self.startFlag = True

    def get_obs_dict(self, sim):
        obs_dict = {}

        # Time
        obs_dict['time'] = np.array([sim.data.time])
        obs_dict['terrain'] = np.array([self.terrain_type])

        # proprioception
        obs_dict['internal_qpos'] = self.get_internal_qpos() #sim.data.qpos[7:].copy()
        obs_dict['internal_qvel'] = self.get_internal_qvel() #sim.data.qvel[6:].copy() * self.dt
        obs_dict['grf'] = self._get_grf().copy()
        obs_dict['socket_force'] = self._get_socket_force().copy()
        obs_dict['torso_angle'] = self.sim.data.body('pelvis').xquat.copy()

        obs_dict['muscle_length'] = self.muscle_lengths()
        obs_dict['muscle_velocity'] = self.muscle_velocities()
        obs_dict['muscle_force'] = self.muscle_forces()

        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()

        # exteroception
        obs_dict['model_root_pos'] = sim.data.qpos[:2].copy()
        obs_dict['model_root_vel'] = sim.data.qvel[:2].copy()

        # active task
        # trackfield view of 10x10 grid of points around agent. Reshape to (10, 10) for visual inspection
        if not self.trackfield is None:
            obs_dict['hfield'] = self.trackfield.get_heightmap_obs()

        return obs_dict

    def get_reward_dict(self, obs_dict):
        """
        Rewards are computed from here, using the <self.weighted_reward_keys>.
        These weights can either be set in this file in the
        DEFAULT_RWD_KEYS_AND_WEIGHTS dict, or when registering the environment
        with gym.register in myochallenge/__init__.py
        """
        act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0

        # The task is entirely defined by these 3 lines
        score = self.get_score()
        win_cdt = self._win_condition()
        rwd_dict = collections.OrderedDict((
            # Perform reward tuning here --
            # Update Optional Keys section below
            # Update reward keys (DEFAULT_RWD_KEYS_AND_WEIGHTS) accordingly to update final rewards
            # Example: simple distance function

                # Optional Keys
                ('act_reg', act_mag.squeeze()),
                # Must keys
                ('sparse',  score),
                ('solved',  win_cdt),
                ('done',  self._get_done()),
            ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    def get_metrics(self, paths):
        """
        Evaluate paths and report metrics
        """
        # average sucess over entire env horizon
        score = np.mean([np.sum(p['env_infos']['rwd_dict']['sparse']) for p in paths])

        metrics = {
            'score': score,
            }
        return metrics

    def step(self, *args, **kwargs):
        out_act = self._prepareActions(*args)
        results = super().step(out_act, **kwargs)

        return results

    def reset(self, **kwargs):
        # randomized terrain types
        self._maybe_sample_terrain()
        self.terrain_type = self.trackfield.terrain_type.value
        # randomized initial state
        qpos, qvel = self._get_reset_state()
        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super(WalkEnvV0, self).reset(reset_qpos=qpos, reset_qvel=qvel, **kwargs)
        self.sim.forward()
        self.OSL_CTRL.reset('e_stance')

        # Sync the states again as the randomization might cause the part of the model to be inside the ground
        if self.reset_type != 'init':
            new_qpos, new_qvel = self.adjust_model_height()
            self.robot.sync_sims(self.sim, self.sim_obsd)
            obs = super(WalkEnvV0, self).reset(reset_qpos=new_qpos, reset_qvel=new_qvel, **kwargs)
            self.sim.forward()
        self.OSL_CTRL.start()

        return obs

    def _maybe_flatten_agent_patch(self, qpos):
        """
        Ensure that initial state patch is flat.
        """
        if self.trackfield is not None:
            self.trackfield.flatten_agent_patch(qpos)
            if hasattr(self.sim, 'renderer') and not self.sim.renderer._window is None:
                self.sim.renderer._window.update_hfield(0)

    def _maybe_sample_terrain(self):
        """
        Sample a new terrain if the terrain type asks for it.
        """
        if not self.trackfield is None:
            self.trackfield.sample(self.np_random)
            self.sim.model.geom_rgba[self.sim.model.geom_name2id('terrain')][-1] = 1.0
            self.sim.model.geom_pos[self.sim.model.geom_name2id('terrain')] = np.array([0, 0, 0.005])
        else:
            # move trackfield down if not used
            self.sim.model.geom_rgba[self.sim.model.geom_name2id('terrain')][-1] = 0.0
            self.sim.model.geom_pos[self.sim.model.geom_name2id('terrain')] = np.array([0, 0, -10])

    def _randomize_position_orientation(self, qpos, qvel):
        orientation = self.np_random.uniform(0, 2 * np.pi)
        euler_angle = quat2euler(qpos[3:7])
        euler_angle[-1] = orientation
        qpos[3:7] = euler2quat(euler_angle)
        # rotate original velocity with unit direction vector
        qvel[:2] = np.array([np.cos(orientation), np.sin(orientation)]) * np.linalg.norm(qvel[:2])
        return qpos, qvel

    def _get_reset_state(self):
        if self.reset_type == 'random':
            qpos, qvel = self._get_randomized_initial_state()
            return self._randomize_position_orientation(qpos, qvel)
        elif self.reset_type == 'init':
            return self.sim.model.key_qpos[2], self.sim.model.key_qvel[2]
        elif self.reset_type == 'osl_init':
            self.initializeFromData()
            return self.init_qpos.copy(), self.init_qvel.copy()
        else:
            return self.sim.model.key_qpos[0], self.sim.model.key_qvel[0]

    def _maybe_adjust_height(self, qpos, qvel):
        """
        Currently not used.
        """
        if self.trackfield is not None:
                map_i, map_j = self.trackfield.cart2map(qpos[:2])
                hfield_val = self.trackfield.hfield.data[map_i, map_j]
                if hfield_val > 0.05:
                    qpos[2] += hfield_val
        return qpos, qvel

    def viewer_setup(self, *args, **kwargs):
       """
       Setup the default camera
       """
       distance = 5.0
       azimuth = 90
       elevation = -15
       lookat = None
       self.sim.renderer.set_free_camera_settings(
               distance=distance,
               azimuth=azimuth,
               elevation=elevation,
               lookat=lookat
       )
       render_tendon = True
       render_actuator = True
       self.sim.renderer.set_viewer_settings(
           render_actuator=render_actuator,
           render_tendon=render_tendon
       )

    def _get_randomized_initial_state(self):
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
        qpos[0] = self.np_random.uniform(-self.real_width * 0.8, self.real_width * 0.8, size=1).squeeze()

        qpos[3:7] = rot_state
        qpos[2] = height
        qpos[1] = 14
        return qpos, qvel

    def _setup_convenience_vars(self):
        """
        Convenience functions for easy access. Important: There will be no access during the challenge evaluation to this,
        but the values can be hard-coded, as they do not change over time.
        """
        self.actuator_names = np.array(self._get_actuator_names())
        self.joint_names = np.array(self._get_joint_names())
        self.muscle_fmax = np.array(self._get_muscle_fmax())
        self.muscle_lengthrange = np.array(self._get_muscle_lengthRange())
        self.tendon_len = np.array(self._get_tendon_lengthspring())
        self.musc_operating_len = np.array(self._get_muscle_operating_length())

    def _get_done(self):
        if self._lose_condition():
            return self._get_fallen_condition()
        if self._win_condition():
            return 1
        return 0

    def _win_condition(self):
        y_pos = self.obs_dict['model_root_pos'].squeeze()[1]
        if np.abs(y_pos) > self.distance_thr:
            return 1
        return 0

    def _lose_condition(self):
        x_pos = self.obs_dict['model_root_pos'].squeeze()[0]
        y_pos = self.obs_dict['model_root_pos'].squeeze()[1]
        if x_pos > self.real_width or x_pos < - self.real_width:
            return 1
        if y_pos > 15.0:
            return 1
        if self._get_fallen_condition():
            return 1
        return 0

    def _get_fallen_condition(self):
        """
        Checks if the agent has fallen by comparing the head site height with the
        average foot height.
        """
        head = self.sim.data.site('head').xpos
        foot_l = self.sim.data.body('talus_l').xpos
        foot_r = self.sim.data.body('osl_foot_assembly').xpos
        mean = (foot_l + foot_r) / 2
        if head[2] - mean[2] < 0.2:
            return 1
        else:
            return 0

    # Helper functions
    def _get_body_mass(self):
        """
        Get total body mass of the biomechanical model.
        :return: the weight
        :rtype: float
        """
        return self.sim.model.body('root').subtreemass

    def get_score(self):
        # initial environment needs to be setup for self.horizon to work
        if not self.startFlag:
            return -1
        y_pos = self.obs_dict['model_root_pos'].squeeze()[1]
        score = (y_pos - 15) / (- 15 - 15)
        return score.squeeze()

    def _get_muscle_lengthRange(self):
        return self.sim.model.actuator_lengthrange.copy()

    def _get_tendon_lengthspring(self):
        return self.sim.model.tendon_lengthspring.copy()

    def _get_muscle_operating_length(self):
        return self.sim.model.actuator_gainprm[:,0:2].copy()

    def _get_muscle_fmax(self):
        return self.sim.model.actuator_gainprm[:, 2].copy()

    def _get_grf(self):
        grf = np.array([self.sim.data.sensor(sens_name).data[0] for sens_name in self.grf_sensor_names]).copy()
        return grf

    def _get_socket_force(self):
        return self.sim.data.sensor('r_socket_load').data.copy()

    def _get_pelvis_angle(self):
        return self.sim.data.body('pelvis').xquat.copy()

    def _get_joint_names(self):
        '''
        Return a list of joint names according to the index ID of the joint angles
        '''
        return [self.sim.model.joint(jnt_id).name for jnt_id in range(1, self.sim.model.njnt)]

    def _get_actuator_names(self):
        '''
        Return a list of actuator names according to the index ID of the actuators
        '''
        return [self.sim.model.actuator(act_id).name for act_id in range(1, self.sim.model.na)]


    def _get_fallen_condition(self):
        """
        Checks if the agent has fallen by comparing the head site height with the
        average foot height.
        """
        if self.terrain == 'FLAT':
            if self.sim.data.body('pelvis').xpos[2] < 0.5:
                return 1
            return 0
        else:
            head = self.sim.data.site('head').xpos
            foot_l = self.sim.data.body('talus_l').xpos
            foot_r = self.sim.data.body('osl_foot_assembly').xpos
            mean = (foot_l + foot_r) / 2
            if head[2] - mean[2] < 0.2:
                return 1
            else:
                return 0

    def get_internal_qpos(self):
        temp_qpos = self.sim.data.qpos.copy()
        to_remove = [self.sim.model.joint('osl_knee_angle_r').qposadr[0].copy(), self.sim.model.joint('osl_ankle_angle_r').qposadr[0].copy()]

        temp_qpos[to_remove] = 100
        temp_qpos[temp_qpos != 100]
        return temp_qpos[7:]

    def get_internal_qvel(self):
        temp_qvel = self.sim.data.qvel.copy()
        to_remove = [self.sim.model.joint('osl_knee_angle_r').qposadr[0].copy() -1, self.sim.model.joint('osl_ankle_angle_r').qposadr[0].copy() -1]

        temp_qvel[to_remove] = 100
        temp_qvel[temp_qvel != 100]
        return temp_qvel[6:] * self.dt

    def muscle_lengths(self):
        temp_len = self.sim.data.actuator_length.copy()
        to_remove = [self.sim.data.actuator('osl_knee_torque_actuator').id, self.sim.data.actuator('osl_ankle_torque_actuator').id]

        temp_len[to_remove] = 100
        temp_len[temp_len != 100]
        return temp_len

    def muscle_forces(self):
        temp_frc = self.sim.data.actuator_force.copy()
        to_remove = [self.sim.data.actuator('osl_knee_torque_actuator').id, self.sim.data.actuator('osl_ankle_torque_actuator').id]

        temp_frc[to_remove] = 100
        temp_frc[temp_frc != 100]

        return np.clip(temp_frc / 1000, -100, 100)

    def muscle_velocities(self):
        temp_vel = self.sim.data.actuator_velocity.copy()
        to_remove = [self.sim.data.actuator('osl_knee_torque_actuator').id, self.sim.data.actuator('osl_ankle_torque_actuator').id]

        temp_vel[to_remove] = 100
        temp_vel[temp_vel != 100]

        return np.clip(temp_vel, -100, 100)

    def _get_actuator_params(self):
        actuators = ['osl_knee_torque_actuator', 'osl_ankle_torque_actuator', ]

        for actu in actuators:
            self.ACTUATOR_PARAM[actu] = {}
            self.ACTUATOR_PARAM[actu]['id'] = self.sim.data.actuator(actu).id
            self.ACTUATOR_PARAM[actu]['Fmax'] = np.max(self.sim.model.actuator(actu).ctrlrange) * self.sim.model.actuator(actu).gear[0]

    """
    Math func: Intrinsic Euler angles, for euler in body coordinate frame
    """
    def get_intrinsic_EulerXYZ(self, q):
        w, x, y, z = q

        # Compute sin and cos values
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)

        # Roll (X-axis rotation)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Compute sin and cos values
        sinp = 2 * (w * y - z * x)

        # Pitch (Y-axis rotation)
        if abs(sinp) >= 1:
            # Use 90 degrees if out of range
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Compute sin and cos values
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)

        # Yaw (Z-axis rotation)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])

    def intrinsic_EulerXYZ_toQuat(self, roll, pitch, yaw):
        # Half angles
        half_roll = roll * 0.5
        half_pitch = pitch * 0.5
        half_yaw = yaw * 0.5

        # Compute sin and cos values for half angles
        sin_roll = np.sin(half_roll)
        cos_roll = np.cos(half_roll)
        sin_pitch = np.sin(half_pitch)
        cos_pitch = np.cos(half_pitch)
        sin_yaw = np.sin(half_yaw)
        cos_yaw = np.cos(half_yaw)

        # Compute quaternion
        w = cos_roll * cos_pitch * cos_yaw + sin_roll * sin_pitch * sin_yaw
        x = sin_roll * cos_pitch * cos_yaw - cos_roll * sin_pitch * sin_yaw
        y = cos_roll * sin_pitch * cos_yaw + sin_roll * cos_pitch * sin_yaw
        z = cos_roll * cos_pitch * sin_yaw - sin_roll * sin_pitch * cos_yaw

        return np.array([w, x, y, z])

    def rotate_frame(self, x, y, theta):
        #print(theta)
        x_rot = np.cos(theta)*x - np.sin(theta)*y
        y_rot = np.sin(theta)*x + np.cos(theta)*y
        return x_rot, y_rot

    """
    Environment initialization functions
    """
    def initializeFromData(self):

        if self._operation_mode == 'eval':
            start_idx = 0
        else:
            start_idx = self.np_random.integers(low=0, high=self.INIT_DATA.shape[0])

        for joint in self.gait_cycle_headers.keys():
            if joint not in ['pelvis_euler_roll', 'pelvis_euler_pitch', 'pelvis_euler_yaw',
                                'l_foot_relative_X', 'l_foot_relative_Y', 'l_foot_relative_Z',
                                'r_foot_relative_X', 'r_foot_relative_Y', 'r_foot_relative_Z',
                                'pelvis_vel_X', 'pelvis_vel_Y', 'pelvis_vel_Z']:
                self.init_qpos[self.sim.model.joint(joint).qposadr[0]] = self.INIT_DATA[start_idx,self.gait_cycle_headers[joint]]

        # Get the Yaw from the init pose
        default_quat = self.init_qpos[3:7].copy() # Get the default facing direction first

        init_quat = self.intrinsic_EulerXYZ_toQuat(self.INIT_DATA[start_idx, self.gait_cycle_headers['pelvis_euler_roll']],
                                                   self.INIT_DATA[start_idx, self.gait_cycle_headers['pelvis_euler_pitch']],
                                                   self.INIT_DATA[start_idx, self.gait_cycle_headers['pelvis_euler_yaw']])
        self.init_qpos[3:7] = init_quat

        # Use the default facing direction to set the world frame velocity
        temp_euler = self.get_intrinsic_EulerXYZ(default_quat)
        world_vel_X, world_vel_Y = self.rotate_frame(self.INIT_DATA[start_idx, self.gait_cycle_headers['pelvis_vel_X']],
                                                     self.INIT_DATA[start_idx, self.gait_cycle_headers['pelvis_vel_Y']],
                                                     temp_euler[2])
        self.init_qvel[:] = 0
        self.init_qvel[0] = world_vel_X
        self.init_qvel[1] = world_vel_Y
        self.init_qvel[2] = self.INIT_DATA[start_idx, self.gait_cycle_headers['pelvis_vel_Z']]

        state = self.init_lookup[start_idx]
        # Override previous OSL controller
        self.OSL_CTRL.reset(state)

    def adjust_model_height(self):

        curr_qpos = self.sim.data.qpos.copy()
        curr_qvel = self.sim.data.qvel.copy()

        temp_sens_height = 100
        for sens_site in ['r_heel_btm', 'r_toe_btm', 'l_heel_btm', 'l_toe_btm']:
            if temp_sens_height > self.sim.data.site(sens_site).xpos[2]:
                temp_sens_height = self.sim.data.site(sens_site).xpos[2].copy()

        diff_height = 0.0 - temp_sens_height
        curr_qpos[2] = curr_qpos[2] + diff_height

        return curr_qpos, curr_qvel

    def generate_init_lookup(self, keys, value, existing_dict=None):
        # Initialize an empty dictionary if existing_dict is None
        if existing_dict is None:
            result_dict = {}
        else:
            # Use the existing dictionary if provided
            result_dict = copy.deepcopy(existing_dict)

        # Iterate through the list of integers (keys)
        for key in keys:
            # Assign each key in the dictionary with the value "early_stance"
            result_dict[key] = value

        return result_dict

    """
    OSL leg interaction functions
    """
    def _prepareActions(self, mus_actions):
        """
        Combines OSL torques with the muscle activations
        Only considers the 54 muscles of the OSLMyoleg model
        """

        full_actions = np.zeros(self.sim.model.nu,)
        full_actions[0:self.sim.model.na] = mus_actions[0:self.sim.model.na].copy()

        self.OSL_CTRL.update(self.get_osl_sens())

        osl_torque = self.OSL_CTRL.get_osl_torque()

        for jnt in ['knee', 'ankle']:
            osl_id = self.sim.model.actuator(f"osl_{jnt}_torque_actuator").id
            full_actions[osl_id] = np.clip(osl_torque[jnt] / self.sim.model.actuator(f"osl_{jnt}_torque_actuator").gear[0],
                    self.sim.model.actuator(f"osl_{jnt}_torque_actuator").ctrlrange[0],
                    self.sim.model.actuator(f"osl_{jnt}_torque_actuator").ctrlrange[1])

        return full_actions

    def get_osl_sens(self):

        osl_sens_data = {}
        osl_sens_data['knee_angle'] = self.sim.data.joint('osl_knee_angle_r').qpos[0].copy()
        osl_sens_data['knee_vel'] = self.sim.data.joint('osl_knee_angle_r').qvel[0].copy()
        osl_sens_data['ankle_angle'] = self.sim.data.joint('osl_ankle_angle_r').qpos[0].copy()
        osl_sens_data['ankle_vel'] = self.sim.data.joint('osl_ankle_angle_r').qvel[0].copy()
        osl_sens_data['load'] = self.sim.data.sensor('r_socket_load').data[1].copy() # Only vertical

        return osl_sens_data
