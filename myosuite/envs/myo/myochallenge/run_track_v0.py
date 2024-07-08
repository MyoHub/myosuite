""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com), Pierre Schumacher (schumacherpier@gmail.com), Chun Kwang Tan (cktan.neumove@gmail.com)
================================================= """

import collections
from myosuite.utils import gym
import numpy as np
import pink
import os
from enum import Enum
from typing import Optional, Tuple
import copy
import csv

from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.envs.myo.myobase.walk_v0 import WalkEnvV0
from myosuite.utils.quat_math import quat2euler, euler2mat, euler2quat
from myosuite.utils.heightfields import TrackField

from opensourceleg.control.state_machine import Event, State, StateMachine
from opensourceleg.osl import OpenSourceLeg

from gymnasium import spaces

class TerrainTypes(Enum):
    FLAT = 0
    HILLY = 1
    ROUGH = 2


class SpecialTerrains(Enum):
    RELIEF = 0



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
        "solved": -10,
    }

    # OSL-related paramters
    ACTUATOR_PARAM = {}
    OSL_PARAM_LIST = []
    OSL_PARAM_SELECT = 0

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        # This flag needs to be here to prevent the simulation from starting in a done state
        # Before setting the key_frames, the model and opponent will be in the cartesian position,
        # causing the step() function to evaluate the initialization as "done".
        self.startFlag = False

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
               reset_type='none',
               terrain='FLAT',
               hills_difficulties=(0,0),
               rough_difficulties=(0,0),
               real_length=20,
               real_width=1,
               run_mode='train',
               **kwargs,
               ):

        # Env initialization with data
        self._operation_mode = run_mode

        file_path = os.path.join(os.getcwd(), '..','assets', 'leg', 'init_data_withVel.csv')

        self.INIT_DATA = np.loadtxt(file_path, skiprows=1, delimiter=',')
        self.init_lookup = self.generate_init_lookup(keys=np.arange(48), value='e_swing')
        self.init_lookup = self.generate_init_lookup(keys=np.arange(48, 99), value='l_swing', existing_dict=self.init_lookup)
        self.init_lookup = self.generate_init_lookup(keys=np.arange(99, 183), value='e_stance', existing_dict=self.init_lookup)
        self.init_lookup = self.generate_init_lookup(keys=np.arange(183, 247), value='l_stance', existing_dict=self.init_lookup)
        with open(file_path) as csv_file:
            csv_reader = csv.DictReader(csv_file)
            temp = dict(list(csv_reader)[0])
            headers = list(temp.keys())
        self.imitation_lookup = dict(zip(headers, range(len(headers))))


        # OSL specific init
        self.OSL = OpenSourceLeg(frequency=200)
        self.OSL.add_joint(name="knee", gear_ratio=49.4, offline_mode=True)
        self.OSL.add_joint(name="ankle", gear_ratio=58.4, offline_mode=True)
        self._init_default_OSL_param()
        self.OSL_FSM = self.build_4_state_FSM(self.OSL, 'e_stance')

        self.muscle_space = self.sim.model.na # muscles only
        self.full_ctrl_space = self.sim.model.nu # Muscles + actuators
        self._get_actuator_params()

        self._setup_convenience_vars()
        self.trackfield = TrackField(
            sim=self.sim,
            rng=self.np_random,
            rough_difficulties=rough_difficulties,
            hills_difficulties=hills_difficulties,
        )
        self.real_width = real_width
        self.real_length = real_length
        self.reset_type = reset_type
        self.terrain = terrain
        self.grf_sensor_names = ['r_prosth', 'l_foot', 'l_toes']
        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       reset_type=reset_type,
                       **kwargs
                       )
        self.init_qpos[:] = self.sim.model.keyframe('osl_forward').qpos.copy()
        self.init_qvel[:] = 0.0
        self.assert_settings()


    def assert_settings(self):
        pass

    def get_obs_dict(self, sim):
        obs_dict = {}

        # Time
        obs_dict['time'] = np.array([sim.data.time])

        # proprioception
        obs_dict['internal_qpos'] = self.get_internal_qpos() #sim.data.qpos[7:].copy()
        obs_dict['internal_qvel'] = self.get_internal_qvel() #sim.data.qvel[6:].copy() * self.dt
        obs_dict['grf'] = self._get_grf().copy()
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

        score = 1
        win_cdt = 0
        rwd_dict = collections.OrderedDict((
            # Perform reward tuning here --
            # Update Optional Keys section below
            # Update reward keys (DEFAULT_RWD_KEYS_AND_WEIGHTS) accordingly to update final rewards
            # Example: simple distance function

                # Optional Keys
                ('act_reg', act_mag),
                # Must keys
                ('sparse',  score),
                ('solved',  win_cdt),
                ('done',  self._get_done()),
            ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)

        # Success Indicator
        # self.sim.model.site_rgba[self.success_indicator_sid, :] = np.array([0, 2, 0, 0.2]) if rwd_dict['solved'] else np.array([2, 0, 0, 0])
        return rwd_dict

    def get_metrics(self, paths):
        """
        Evaluate paths and report metrics
        """
        # average sucess over entire env horizon
        score = np.mean([np.sum(p['env_infos']['rwd_dict']['sparse']) for p in paths])
        # average activations over entire trajectory (can be shorter than horizon, if done) realized

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
        # randomized initial state
        qpos, qvel = self._get_reset_state()
        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super(WalkEnvV0, self).reset(reset_qpos=qpos, reset_qvel=qvel, **kwargs)
        self.sim.forward()
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
            self.sim.model.geom_pos[self.sim.model.geom_name2id('terrain')] = np.array([0, -10, 0])
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
        qpos[0] = self.np_random.uniform(-self.real_width * 0.8, self.real_width * 0.8, size=1)
        
        qpos[3:7] = rot_state
        qpos[2] = height
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
            return 1
        if self._win_condition():
            return 1
        return 0

    def _win_condition(self):
        return 0

    def _lose_condition(self):
        x_pos = self.obs_dict['model_root_pos'].squeeze()[0]
        if x_pos > self.real_width or x_pos < - self.real_width:
            return 1
        return 0

    # Helper functions
    def _get_body_mass(self):
        """
        Get total body mass of the biomechanical model.
        :return: the weight
        :rtype: float
        """
        return self.sim.model.body('root').subtreemass

    def _get_score(self, time):
        return 0

    def _get_muscle_lengthRange(self):
        return self.sim.model.actuator_lengthrange.copy()

    def _get_tendon_lengthspring(self):
        return self.sim.model.tendon_lengthspring.copy()

    def _get_muscle_operating_length(self):
        return self.sim.model.actuator_gainprm[:,0:2].copy()

    def _get_muscle_fmax(self):
        return self.sim.model.actuator_gainprm[:, 2].copy()

    def _get_grf(self):
        return np.array([self.sim.data.sensor(sens_name).data[0] for sens_name in self.grf_sensor_names]).copy()

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
            foot_r = self.sim.data.body('talus_r').xpos
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

    """
    Environment initialization functions
    """
    def initializeFromData(self):

        if self._operation_mode == 'eval':
            start_idx = 0
        else:
            start_idx = self.np_random.integers(low=0, high=self.INIT_DATA.shape[0])
        
        for joint in self.imitation_lookup.keys():
            if joint not in ['pelvis_euler_roll', 'pelvis_euler_pitch', 'pelvis_euler_yaw', 
                                'l_foot_relative_X', 'l_foot_relative_Y', 'l_foot_relative_Z', 
                                'r_foot_relative_X', 'r_foot_relative_Y', 'r_foot_relative_Z',
                                'pelvis_vel_X', 'pelvis_vel_Y', 'pelvis_vel_Z']:
                self.sim.init_qpos[self.sim.model.joint(joint).qposadr[0]] = self.INIT_DATA[start_idx,self.imitation_lookup[joint]]

        # Get the Yaw from the init pose
        default_quat = self.init_qpos[3:7].copy() # Get the default facing direction first

        init_quat = self.intrinsic_EulerXYZ_toQuat(self.INIT_DATA[start_idx, self.imitation_lookup['pelvis_euler_roll']], 
                                                   self.INIT_DATA[start_idx, self.imitation_lookup['pelvis_euler_pitch']], 
                                                   self.INIT_DATA[start_idx, self.imitation_lookup['pelvis_euler_yaw']])
        self.init_qpos[3:7] = init_quat
        
        # Use the default facing direction to set the world frame velocity
        temp_euler = self.get_intrinsic_EulerXYZ(default_quat)
        world_vel_X, world_vel_Y = self.rotate_frame(self.INIT_DATA[start_idx, self.imitation_lookup['pelvis_vel_X']], 
                                                     self.INIT_DATA[start_idx, self.imitation_lookup['pelvis_vel_Y']],
                                                     temp_euler[2])
        self.init_qvel[:] = 0
        self.init_qvel[0] = world_vel_X
        self.init_qvel[1] = world_vel_Y
        self.init_qvel[2] = self.INIT_DATA[start_idx, self.imitation_lookup['pelvis_vel_Z']]

        state = self.init_lookup[start_idx]
        # Override previous OSL controller
        self.OSL_FSM = self.build_4_state_FSM(self.OSL, state)

    def adjust_model_height(self):
        temp_sens_height = 100
        for sens_site in ['r_prosth_touch', 'l_foot_touch', 'l_toes_touch']:
            if temp_sens_height > self.sim.data.site(sens_site).xpos[2]:
                temp_sens_height = self.sim.data.site(sens_site).xpos[2].copy()

        diff_height = 0.01 - temp_sens_height
        self.sim.data.qpos[2] = self.sim.data.qpos[2] + diff_height
        self.forward()

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
    OSL related functions
    """
    def _prepareActions(self, mus_actions):
        assert len(mus_actions) == self.sim.model.na, "Should be 54 inputs" # Should be 54

        full_actions = np.zeros(self.sim.model.nu,)
        full_actions[0:len(mus_actions)] = mus_actions.copy()

        osl_knee_id = self.sim.model.actuator('osl_knee_torque_actuator').id
        osl_knee_act = self.get_osl_action('knee')
        osl_ankle_id = self.sim.model.actuator('osl_ankle_torque_actuator').id
        osl_ankle_act = self.get_osl_action('ankle')

        full_actions[osl_knee_id] = osl_knee_act
        full_actions[osl_ankle_id] = osl_ankle_act

        return full_actions
    
    def get_osl_action(self, joint):
        if joint not in ['knee', 'ankle']:
            print(f"Non-existant joint. Can only be either 'knee' or 'ankle'")
            raise Exception

        K = eval(f"self.OSL_FSM.current_state.{joint}_stiffness")
        B = eval(f"self.OSL_FSM.current_state.{joint}_damping")
        theta = (eval(f"self.OSL_FSM.current_state.{joint}_theta"))
        peak_torque = self.ACTUATOR_PARAM[f"osl_{joint}_torque_actuator"]['Fmax']

        temp_shorten = self.sim.data.joint(f"osl_{joint}_angle_r")

        T = np.clip( K*(theta - temp_shorten.qpos[0].copy()) - B*(temp_shorten.qvel[0].copy()) , -1*peak_torque, peak_torque)

        return np.clip(T / self.sim.model.actuator(f"osl_{joint}_torque_actuator").gear[0], 
                       self.sim.model.actuator(f"osl_{joint}_torque_actuator").ctrlrange[0], 
                       self.sim.model.actuator(f"osl_{joint}_torque_actuator").ctrlrange[1])

    # State machine.
    def build_4_state_FSM(self, osl: OpenSourceLeg, init_state: str) -> StateMachine:
        """
        This method builds a state machine with 4 states.
        The states are early stance, late stance, early swing, and late swing.
        It uses the impedance parameters and transition criteria above.

        Inputs:
            OSL instance
        Returns:
            FSM object
        
        NOTE: The OSL variable are ignored here, and values are all from the Myosuite environment
        CALL THIS FUNCTION AFTER CREATING THE ENVIRONMENT
        """

        # ------------- TUNABLE FSM PARAMETERS ---------------- #
        # NOTE: Ankle angles : (+) Dorsiflexion (-) Plantarflexion

        BODY_WEIGHT = np.sum(self.sim.model.body_mass) * 9.81
        # ---------------------------------------------------- #

        early_stance = State(name="e_stance")
        late_stance = State(name="l_stance")
        early_swing = State(name="e_swing")
        late_swing = State(name="l_swing")
        self.OSL_STATE_LIST = [early_stance, late_stance, early_swing, late_swing]

        early_stance.set_knee_impedance_paramters(
            theta=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['e_stance']['knee']['target_angle'], 
            k=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['e_stance']['knee']['stiffness'], 
            b=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['e_stance']['knee']['damping']
        )
        early_stance.make_knee_active()
        early_stance.set_ankle_impedance_paramters(
            theta=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['e_stance']['ankle']['target_angle'], 
            k=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['e_stance']['ankle']['stiffness'], 
            b=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['e_stance']['ankle']['damping']
        )
        early_stance.make_ankle_active()

        late_stance.set_knee_impedance_paramters(
            theta=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['l_stance']['knee']['target_angle'], 
            k=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['l_stance']['knee']['stiffness'], 
            b=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['l_stance']['knee']['damping']
        )
        late_stance.make_knee_active()
        late_stance.set_ankle_impedance_paramters(
            theta=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['l_stance']['ankle']['target_angle'], 
            k=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['l_stance']['ankle']['stiffness'], 
            b=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['l_stance']['ankle']['damping']
        )
        late_stance.make_ankle_active()

        early_swing.set_knee_impedance_paramters(
            theta=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['e_swing']['knee']['target_angle'], 
            k=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['e_swing']['knee']['stiffness'], 
            b=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['e_swing']['knee']['damping']
        )
        early_swing.make_knee_active()
        early_swing.set_ankle_impedance_paramters(
            theta=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['e_swing']['ankle']['target_angle'], 
            k=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['e_swing']['ankle']['stiffness'], 
            b=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['e_swing']['ankle']['damping']
        )
        early_swing.make_ankle_active()

        late_swing.set_knee_impedance_paramters(
            theta=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['l_swing']['knee']['target_angle'], 
            k=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['l_swing']['knee']['stiffness'], 
            b=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['l_swing']['knee']['damping']
        )
        late_swing.make_knee_active()
        late_swing.set_ankle_impedance_paramters(
            theta=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['l_swing']['ankle']['target_angle'], 
            k=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['l_swing']['ankle']['stiffness'], 
            b=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['l_swing']['ankle']['damping']
        )
        late_swing.make_ankle_active()

        def estance_to_lstance(osl: OpenSourceLeg) -> bool:
            """
            Transition from early stance to late stance when the loadcell
            reads a force greater than a threshold.
            """
            foot_load = self.sim.data.sensor('r_prosth').data[0].copy()
            ankle_pos = self.sim.data.joint('osl_ankle_angle_r').qpos[0].copy()

            return bool(
                foot_load > BODY_WEIGHT * self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['e_stance']['threshold']['load']
                and ankle_pos > self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['e_stance']['threshold']['ankle_angle']
            )

        def lstance_to_eswing(osl: OpenSourceLeg) -> bool:
            """
            Transition from late stance to early swing when the loadcell
            reads a force less than a threshold.
            """
            foot_load = self.sim.data.sensor('r_prosth').data[0].copy()

            return bool(foot_load < BODY_WEIGHT * self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['l_stance']['threshold']['load'])

        def eswing_to_lswing(osl: OpenSourceLeg) -> bool:
            """
            Transition from early swing to late swing when the knee angle
            is greater than a threshold and the knee velocity is less than
            a threshold.
            """
            
            knee_pos = self.sim.data.joint('osl_knee_angle_r').qpos[0].copy()
            knee_vel = self.sim.data.joint('osl_knee_angle_r').qvel[0].copy()

            return bool(
                knee_pos > self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['e_swing']['threshold']['knee_angle']
                and knee_vel < self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['e_swing']['threshold']['knee_vel']
            )

        def lswing_to_estance(osl: OpenSourceLeg) -> bool:
            """
            Transition from late swing to early stance when the loadcell
            reads a force greater than a threshold or the knee angle is
            less than a threshold.
            """

            knee_pos = self.sim.data.joint('osl_knee_angle_r').qpos[0].copy()
            foot_load = self.sim.data.sensor('r_prosth').data[0].copy()

            return bool(
                foot_load > BODY_WEIGHT * self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['l_swing']['threshold']['load']
                or knee_pos < self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['l_swing']['threshold']['knee_angle']
            )

        foot_flat = Event(name="foot_flat")
        heel_off = Event(name="heel_off")
        toe_off = Event(name="toe_off")
        pre_heel_strike = Event(name="pre_heel_strike")
        heel_strike = Event(name="heel_strike")

        fsm = StateMachine(osl=osl, spoof=False)

        for item in self.OSL_STATE_LIST:
            if item.name == init_state:
                fsm.add_state(state=item, initial_state=True)
            else:
                fsm.add_state(state=item)

        fsm.add_event(event=foot_flat)
        fsm.add_event(event=heel_off)
        fsm.add_event(event=toe_off)
        fsm.add_event(event=pre_heel_strike)
        fsm.add_event(event=heel_strike)

        fsm.add_transition(
            source=early_stance,
            destination=late_stance,
            event=foot_flat,
            callback=estance_to_lstance,
        )
        fsm.add_transition(
            source=late_stance,
            destination=early_swing,
            event=heel_off,
            callback=lstance_to_eswing,
        )
        fsm.add_transition(
            source=early_swing,
            destination=late_swing,
            event=toe_off,
            callback=eswing_to_lswing,
        )
        fsm.add_transition(
            source=late_swing,
            destination=early_stance,
            event=heel_strike,
            callback=lswing_to_estance,
        )
        return fsm
    
    """
    OSL parameter loading helper functions
    """
    def set_osl_params_batch(self, params, mode=0):

        assert len(params) == 31, "Should have 31 params"

        phase_list = ['e_stance', 'l_stance', 'e_swing', 'l_swing']
        joint_list = ['knee', 'ankle', 'threshold']
        idx = 0

        if isinstance(params, np.ndarray):
            for phase in phase_list:
                for jnt_arg in joint_list:
                    for key in self.OSL_PARAM_LIST[mode][phase][jnt_arg].keys():
                        self.OSL_PARAM_LIST[mode][phase][jnt_arg][key] = params[idx]
                        idx += 1

        elif isinstance(params, dict):
            self.OSL_PARAM_LIST[mode] = copy.deepcopy(params)

    def set_osl_param(self, phase_name, type, item, value, mode=0):

        assert phase_name in ['e_stance', 'l_stance', 'e_swing', 'l_swing'], f"Phase should be : {['e_stance', 'l_stance', 'e_swing', 'l_swing']}"
        assert type in ['knee', 'ankle', 'threshold'], f"Type should be : {['knee', 'ankle', 'threshold']}"
        assert item in ['stiffness', 'damping', 'load', 'knee_angle', 'knee_vel', 'ankle_angle'], f"Type should be : {['stiffness', 'damping', 'load', 'knee_angle', 'knee_vel', 'ankle_angle']}"

        self.OSL_PARAM_LIST[mode][phase_name][type][item] = value

    def set_osl_mode(self, mode=0):
        self.OSL_PARAM_SELECT = np.clip(mode, 0, 2)
        self._update_param_to_osl()

    def _update_param_to_osl(self):
        """
        Updates the currently selected OSL parameter to the OSL leg state machine
        """
        for item in self.OSL_STATE_LIST:
            item.set_knee_impedance_paramters(
                theta = self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT][item.name]['knee']['target_angle'],
                k = self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT][item.name]['knee']['stiffness'],
                b = self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT][item.name]['knee']['damping'],
            )
            item.set_ankle_impedance_paramters(
                theta=self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT][item.name]['ankle']['target_angle'],
                k = self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT][item.name]['ankle']['stiffness'],
                b = self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT][item.name]['ankle']['damping'],
            )

    def _init_default_OSL_param(self):
        temp_dict = {}
        temp_dict['e_stance'] = {}
        temp_dict['e_stance']['knee'] = {}
        temp_dict['e_stance']['ankle'] = {}
        temp_dict['e_stance']['threshold'] = {}
        temp_dict['e_stance']['knee']['stiffness'] = 99.372
        temp_dict['e_stance']['knee']['damping'] = 3.180
        temp_dict['e_stance']['knee']['target_angle'] = np.deg2rad(5)
        temp_dict['e_stance']['ankle']['stiffness'] = 19.874
        temp_dict['e_stance']['ankle']['damping'] = 0
        temp_dict['e_stance']['ankle']['target_angle'] = np.deg2rad(-2)
        temp_dict['e_stance']['threshold']['load'] = 0.25
        temp_dict['e_stance']['threshold']['ankle_angle'] = np.deg2rad(6)

        temp_dict['l_stance'] = {}
        temp_dict['l_stance']['knee'] = {}
        temp_dict['l_stance']['ankle'] = {}
        temp_dict['l_stance']['threshold'] = {}
        temp_dict['l_stance']['knee']['stiffness'] = 99.372
        temp_dict['l_stance']['knee']['damping'] = 1.272
        temp_dict['l_stance']['knee']['target_angle'] = np.deg2rad(8)
        temp_dict['l_stance']['ankle']['stiffness'] = 79.498
        temp_dict['l_stance']['ankle']['damping'] = 0.063
        temp_dict['l_stance']['ankle']['target_angle'] = np.deg2rad(-20)
        temp_dict['l_stance']['threshold']['load'] = 0.15

        temp_dict['e_swing'] = {}
        temp_dict['e_swing']['knee'] = {}
        temp_dict['e_swing']['ankle'] = {}
        temp_dict['e_swing']['threshold'] = {}
        temp_dict['e_swing']['knee']['stiffness'] = 39.749
        temp_dict['e_swing']['knee']['damping'] = 0.063
        temp_dict['e_swing']['knee']['target_angle'] = np.deg2rad(60)
        temp_dict['e_swing']['ankle']['stiffness'] = 7.949
        temp_dict['e_swing']['ankle']['damping'] = 0
        temp_dict['e_swing']['ankle']['target_angle'] = np.deg2rad(25)
        temp_dict['e_swing']['threshold']['knee_angle'] = np.deg2rad(50)
        temp_dict['e_swing']['threshold']['knee_vel'] = np.deg2rad(3)

        temp_dict['l_swing'] = {}
        temp_dict['l_swing']['knee'] = {}
        temp_dict['l_swing']['ankle'] = {}
        temp_dict['l_swing']['threshold'] = {}
        temp_dict['l_swing']['knee']['stiffness'] = 15.899
        temp_dict['l_swing']['knee']['damping'] = 3.816
        temp_dict['l_swing']['knee']['target_angle'] = np.deg2rad(5)
        temp_dict['l_swing']['ankle']['stiffness'] = 7.949
        temp_dict['l_swing']['ankle']['damping'] = 0
        temp_dict['l_swing']['ankle']['target_angle'] = np.deg2rad(15)
        temp_dict['l_swing']['threshold']['load'] = 0.4
        temp_dict['l_swing']['threshold']['knee_angle'] = np.deg2rad(30)

        for idx in np.arange(3):
            self.OSL_PARAM_LIST.append(copy.deepcopy(temp_dict))