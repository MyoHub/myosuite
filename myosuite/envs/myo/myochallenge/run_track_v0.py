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
from myosuite.utils.quat_math import quat2euler, euler2mat, euler2quat, quat2euler_intrinsic, intrinsic_euler2quat
from myosuite.envs.heightfields import TrackField
from myosuite.envs.myo.assets.leg.myoosl_control import MyoOSLController


class TrackTypes(Enum):
    FLAT = 0
    HILLY = 1
    ROUGH = 2
    STAIRS = 3
    MIXED = 4


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

    # Joint dict
    pain_jnt = ['hip_adduction_l', 'hip_adduction_r', 'hip_flexion_l', 'hip_flexion_r', 'hip_rotation_l', 'hip_rotation_r',
                'knee_angle_l', 'knee_angle_l_rotation2', 'knee_angle_l_rotation3',
                'mtp_angle_l', 'ankle_angle_l', 'subtalar_angle_l']

    biological_jnt = ['hip_adduction_l', 'hip_flexion_l', 'hip_rotation_l',
                      'hip_adduction_r', 'hip_flexion_r', 'hip_rotation_r',
                      'knee_angle_l', 'knee_angle_l_beta_rotation1',
                      'knee_angle_l_beta_translation1', 'knee_angle_l_beta_translation2',
                      'knee_angle_l_rotation2', 'knee_angle_l_rotation3', 'knee_angle_l_translation1',
                      'knee_angle_l_translation2', 'mtp_angle_l', 'ankle_angle_l',
                      'subtalar_angle_l']
    biological_act = ['addbrev_l', 'addbrev_r', 'addlong_l', 'addlong_r', 'addmagDist_l', 'addmagIsch_l', 'addmagMid_l',
                      'addmagProx_l', 'bflh_l', 'bfsh_l', 'edl_l', 'ehl_l', 'fdl_l', 'fhl_l', 'gaslat_l', 'gasmed_l',
                      'glmax1_l', 'glmax1_r', 'glmax2_l', 'glmax2_r', 'glmax3_l', 'glmax3_r', 'glmed1_l', 'glmed1_r',
                      'glmed2_l', 'glmed2_r', 'glmed3_l', 'glmed3_r', 'glmin1_l', 'glmin1_r', 'glmin2_l', 'glmin2_r',
                      'glmin3_l', 'glmin3_r', 'grac_l', 'iliacus_l', 'iliacus_r',
                      'perbrev_l', 'perlong_l', 'piri_l', 'piri_r', 'psoas_l', 'psoas_r', 'recfem_l', 'sart_l',
                      'semimem_l', 'semiten_l', 'soleus_l', 'tfl_l', 'tibant_l', 'tibpost_l', 'vasint_l',
                      'vaslat_l', 'vasmed_l']

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
               terrain='random',
               hills_difficulties=(0,0),
               rough_difficulties=(0,0),
               stairs_difficulties=(0,0),
               real_width=1,
               real_length=120,
               end_pos = -15,
               start_pos = 14,
               init_pose_path=None,
               osl_param_set=4,
               max_episode_steps=36000,
               **kwargs,
               ):

        self.startFlag = False
        self.start_pos = start_pos
        # Terrain type
        self.terrain_type = TrackTypes.FLAT.value

        self.osl_param_set = osl_param_set
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
        self.OSL_CTRL = MyoOSLController(np.sum(self.sim.model.body_mass), init_state='e_stance', n_sets=self.osl_param_set)
        self.OSL_CTRL.start()

        self.muscle_space = self.sim.model.na # muscles only
        self.full_ctrl_space = self.sim.model.nu # Muscles + actuators
        self._get_actuator_params()

        self._setup_convenience_vars()
        self.end_pos = end_pos
        self.trackfield = TrackField(
            sim=self.sim,
            rng=self.np_random,
            rough_difficulties=rough_difficulties,
            hills_difficulties=hills_difficulties,
            stairs_difficulties=stairs_difficulties,
            real_length=real_length,
            real_width=real_width,
            reset_type=terrain,
            view_distance=5,
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

        # OSL controls will be automatically infered from the OSL controller. It will not be exposed as action space. Let's fix the action space.
        act_low = -np.ones(self.sim.model.na) if self.normalize_act else self.sim.model.actuator_ctrlrange[:,0].copy()
        act_high = np.ones(self.sim.model.na) if self.normalize_act else self.sim.model.actuator_ctrlrange[:,1].copy()
        self.action_space = gym.spaces.Box(act_low, act_high, dtype=np.float32)

        # Lets fix initial pose
        self.init_qpos[:] = self.sim.model.keyframe('stand').qpos.copy()
        self.init_qvel[:] = 0.0
        self.startFlag = True

        # Max time for time metric
        self.maxTime = self.dt * max_episode_steps

    def get_obs_dict(self, sim):
        obs_dict = {}

        # Time
        obs_dict['time'] = np.array([sim.data.time])
        obs_dict['terrain'] = np.array([self.terrain_type])

        # proprioception
        obs_dict['internal_qpos'] = self.get_internal_qpos()
        obs_dict['internal_qvel'] = self.get_internal_qvel()
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
        # act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0
        act_mag = np.mean( np.square(self.obs_dict['act']) ) if self.sim.model.na !=0 else 0
        pain = self.get_pain()

        # The task is entirely defined by these 3 lines
        score = self.get_score()
        win_cdt = self._win_condition()
        lose_cdt = self._lose_condition()

        self.obs_dict['time'] = self.maxTime if lose_cdt else self.obs_dict['time']

        rwd_dict = collections.OrderedDict((
            # Perform reward tuning here --
            # Update Optional Keys section below
            # Update reward keys (DEFAULT_RWD_KEYS_AND_WEIGHTS) accordingly to update final rewards
            # Example: simple distance function

                # Optional Keys
                ('act_reg', act_mag.squeeze()),
                ('pain', pain),
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
        times = np.mean([np.round(p['env_infos']['obs_dict']['time'][-1], 5) for p in paths])
        # use best achieved position over trajectory as score
        score = np.mean([np.min(p['env_infos']['obs_dict']['model_root_pos'][..., 1]) for p in paths])
        effort = np.mean([np.sum(p['env_infos']['rwd_dict']['act_reg']) for p in paths])
        pain = np.mean([np.sum(p['env_infos']['rwd_dict']['pain']) for p in paths])

        # normalize score to be between 0 and 1
        if self.start_pos > self.end_pos:
            score = (self.start_pos - score) / (self.start_pos - self.end_pos)
        else:
            score = (score - self.end_pos) / (self.start_pos - self.end_pos)

        metrics = {
            'score': np.clip(score, 0, 1),
            'time': times,
            'effort': effort,
            'pain': pain,
            }
        return metrics

    # build full action by combining user actions and OSL actions.
    def step(self, a, **kwargs):
        myoosl_a = self._append_osl_actions(mus_actions=a, is_normalized=self.normalize_act)
        results = super().step(myoosl_a, **kwargs)
        return results

    def reset(self, OSL_params=None, **kwargs):

        if OSL_params is not None:
            self.upload_osl_param(OSL_params)

        # randomized terrain types
        self._maybe_sample_terrain()
        self.terrain_type = self.trackfield.terrain_type.value
        # randomized initial state
        qpos, qvel = self._get_reset_state()
        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super(WalkEnvV0, self).reset(reset_qpos=qpos, reset_qvel=qvel, **kwargs)
        self.sim.forward()

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
        orientation = self.np_random.uniform(np.deg2rad(-125), np.deg2rad(-60))

        euler_angle = quat2euler_intrinsic(qpos[3:7]) # Roll, Pitch, Yaw format
        euler_angle[2] = orientation
        qpos[3:7] = intrinsic_euler2quat([euler_angle[0], euler_angle[1], euler_angle[2]])

        # rotate original velocity with unit direction vector
        qvel[:2] = np.array([np.cos(orientation), np.sin(orientation)]) * np.linalg.norm(qvel[:2])
        return qpos, qvel

    def _get_reset_state(self):
        if self.reset_type == 'random':
            qpos, qvel = self._get_randomized_initial_state()
            return self._randomize_position_orientation(qpos, qvel)
        elif self.reset_type == 'init':
            self.OSL_CTRL.reset('e_stance')
            return self.sim.model.key_qpos[0], self.sim.model.key_qvel[0]
        elif self.reset_type == 'osl_init':
            self.initializeFromData()
            return self.init_qpos.copy(), self.init_qvel.copy()
        else:
            self.OSL_CTRL.reset('e_stance')
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
        rndInt = self.np_random.integers(low=0, high=3) # high exclusive
        qpos = self.sim.model.key_qpos[rndInt].copy()
        qvel = self.sim.model.key_qvel[rndInt].copy()

        # Set OSL leg initial state for based on initial key pose
        if rndInt == 0 or rndInt == 2:
            self.OSL_CTRL.reset('e_stance')
        else:
            self.OSL_CTRL.reset('e_swing')

        # randomize qpos coordinates
        # but dont change height or rot state
        rot_state = qpos[3:7]
        height = qpos[2]
        qpos[0] = self.np_random.uniform(-self.real_width * 0.8, self.real_width * 0.8, size=1).squeeze()

        qpos[3:7] = rot_state
        qpos[2] = height
        qpos[1] = self.start_pos + 1
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
        y_pos = self.obs_dict['model_root_pos'].squeeze()[1]
        if  y_pos < self.end_pos:
            return 1
        return 0

    def _lose_condition(self):
        x_pos = self.obs_dict['model_root_pos'].squeeze()[0]
        y_pos = self.obs_dict['model_root_pos'].squeeze()[1]
        if x_pos > self.real_width or x_pos < - self.real_width:
            return 1
        if y_pos > self.start_pos + 2:
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
        # head is below the mean foot height
        if head[2] - mean[2] < 0.2:
            return 1
        # head is too close to the ground
        if head[2] < 0.2:
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
        """
        Score is the negative velocity in the y direction, which makes the humanoid run forward.
        """
        # initial environment needs to be setup for self.horizon to work
        if not self.startFlag:
            return -1
        vel = self.obs_dict['model_root_vel'].squeeze()[1]

        return  - vel.squeeze()

    def get_pain(self):
        """
        Pain is the sum of the joint limit violation forces.
        """
        if not self.startFlag:
            return -1

        pain_score = 0
        for joint in self.pain_jnt:
            pain_score += np.clip(np.abs(self.get_limitfrc(joint).squeeze()), -1000, 1000) / 1000
        return pain_score / len(self.pain_jnt)

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

    def get_limitfrc(self, joint_name):
        """
        Get the joint limit force for a given joint.
        """
        non_joint_limit_efc_idxs = np.where(self.sim.data.efc_type != self.sim.lib.mjtConstraint.mjCNSTR_LIMIT_JOINT)[0]
        only_jnt_lim_efc_force = self.sim.data.efc_force.copy()
        only_jnt_lim_efc_force[non_joint_limit_efc_idxs] = 0.0
        joint_force = np.zeros((self.sim.model.nv,))
        self.sim.lib.mj_mulJacTVec(self.sim.model._model, self.sim.data._data, joint_force, only_jnt_lim_efc_force)
        return joint_force[self.sim.model.joint(joint_name).dofadr]

    def get_internal_qpos(self):
        """
        Get the internal joint positions without the osl leg joints.
        """
        temp_qpos = np.zeros(len(self.biological_jnt),)
        counter = 0
        for jnt in self.biological_jnt:
            temp_qpos[counter] = self.sim.data.joint(jnt).qpos[0].copy()
            counter+=1
        return temp_qpos

    def get_internal_qvel(self):
        """
        Get the internal joint velocities without the osl leg joints.
        """
        temp_qvel = np.zeros(len(self.biological_jnt),)
        counter = 0
        for jnt in self.biological_jnt:
            temp_qvel[counter] = self.sim.data.joint(jnt).qvel[0].copy()
            counter+=1
        return temp_qvel * self.dt

    def muscle_lengths(self):
        """
        Get the muscle lengths. Remove the osl leg actuators from the data.
        """
        temp_len = np.zeros(len(self.biological_act),)
        counter = 0
        for jnt in self.biological_act:
            temp_len[counter] = self.sim.data.actuator(jnt).length[0].copy()
            counter+=1
        return temp_len

    def muscle_forces(self):
        """
        Get the muscle forces. Remove the osl leg actuators from the data.
        """
        temp_frc = np.zeros(len(self.biological_act),)
        counter = 0
        for jnt in self.biological_act:
            temp_frc[counter] = self.sim.data.actuator(jnt).force[0].copy()
            counter+=1

        return np.clip(temp_frc / 1000, -100, 100)

    def muscle_velocities(self):
        """
        Get the muscle velocities. Remove the osl leg actuators from the data.
        """
        temp_vel = np.zeros(len(self.biological_act),)
        counter = 0
        for jnt in self.biological_act:
            temp_vel[counter] = self.sim.data.actuator(jnt).velocity[0].copy()
            counter+=1

        return np.clip(temp_vel, -100, 100)

    def _get_actuator_params(self):
        """
        Get the actuator parameters. Remove the osl leg actuators from the data.
        """
        actuators = ['osl_knee_torque_actuator', 'osl_ankle_torque_actuator', ]

        for actu in actuators:
            self.ACTUATOR_PARAM[actu] = {}
            self.ACTUATOR_PARAM[actu]['id'] = self.sim.data.actuator(actu).id
            self.ACTUATOR_PARAM[actu]['Fmax'] = np.max(self.sim.model.actuator(actu).ctrlrange) * self.sim.model.actuator(actu).gear[0]


    def rotate_frame(self, x, y, theta):
        #print(theta)
        x_rot = np.cos(theta)*x - np.sin(theta)*y
        y_rot = np.sin(theta)*x + np.cos(theta)*y
        return x_rot, y_rot

    """
    Environment initialization functions
    """
    def initializeFromData(self):

        start_idx = self.np_random.integers(low=0, high=self.INIT_DATA.shape[0])

        for joint in self.gait_cycle_headers.keys():
            if joint not in ['pelvis_euler_roll', 'pelvis_euler_pitch', 'pelvis_euler_yaw',
                                'l_foot_relative_X', 'l_foot_relative_Y', 'l_foot_relative_Z',
                                'r_foot_relative_X', 'r_foot_relative_Y', 'r_foot_relative_Z',
                                'pelvis_vel_X', 'pelvis_vel_Y', 'pelvis_vel_Z']:
                self.init_qpos[self.sim.model.joint(joint).qposadr[0]] = self.INIT_DATA[start_idx,self.gait_cycle_headers[joint]]

        # Get the Yaw from the init pose
        default_quat = self.init_qpos[3:7].copy() # Get the default facing direction first

        init_quat = intrinsic_euler2quat([
            self.INIT_DATA[start_idx, self.gait_cycle_headers['pelvis_euler_roll']],
            self.INIT_DATA[start_idx, self.gait_cycle_headers['pelvis_euler_pitch']],
            self.INIT_DATA[start_idx, self.gait_cycle_headers['pelvis_euler_yaw']]
        ])
        self.init_qpos[3:7] = init_quat

        # Use the default facing direction to set the world frame velocity
        temp_euler = quat2euler_intrinsic(default_quat)
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

        if not self.trackfield is None:
            diff_height = 0.005 - temp_sens_height
        else:
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
    def _append_osl_actions(self, mus_actions, is_normalized):
        """
        Combines OSL torques with the muscle activations
        Only considers the 54 muscles of the OSLMyoleg model
        """

        # copy over the muscle activations
        full_actions = np.zeros(self.sim.model.nu,)
        full_actions[0:self.sim.model.na] = mus_actions[0:self.sim.model.na].copy()

        # append the osl torques
        self.OSL_CTRL.update(self.get_osl_sens())
        osl_torque = self.OSL_CTRL.get_osl_torque()

        for jnt in ['knee', 'ankle']:
            osl_id = self.sim.model.actuator(f"osl_{jnt}_torque_actuator").id
            osl_ctrl = osl_torque[jnt] / self.sim.model.actuator(f"osl_{jnt}_torque_actuator").gear[0]

            # clip for control limits
            min_ctrl = self.sim.model.actuator(f"osl_{jnt}_torque_actuator").ctrlrange[0]
            max_ctrl = self.sim.model.actuator(f"osl_{jnt}_torque_actuator").ctrlrange[1]
            osl_ctrl = np.clip(osl_ctrl, min_ctrl, max_ctrl)

            # normalize if needed
            if is_normalized:
                ctrl_mean = (min_ctrl + max_ctrl) / 2.0
                ctrl_rng = (max_ctrl - min_ctrl) / 2.0
                osl_ctrl = (osl_ctrl-ctrl_mean)/ctrl_rng

            full_actions[osl_id] = osl_ctrl

        return full_actions

    def get_osl_sens(self):

        osl_sens_data = {}
        osl_sens_data['knee_angle'] = self.sim.data.joint('osl_knee_angle_r').qpos[0].copy()
        osl_sens_data['knee_vel'] = self.sim.data.joint('osl_knee_angle_r').qvel[0].copy()
        osl_sens_data['ankle_angle'] = self.sim.data.joint('osl_ankle_angle_r').qpos[0].copy()
        osl_sens_data['ankle_vel'] = self.sim.data.joint('osl_ankle_angle_r').qvel[0].copy()
        osl_sens_data['load'] = -1*self.sim.data.sensor('r_osl_load').data[1].copy() # Only vertical

        return osl_sens_data

    def upload_osl_param(self, dict_of_dict):
        """
        Accessor function to upload full set of paramters to OSL leg
        """
        assert len(dict_of_dict.keys()) <= 4
        for idx in dict_of_dict.keys():
            self.OSL_CTRL.set_osl_param_batch(dict_of_dict[idx], mode=idx)

    def change_osl_mode(self, mode=0):
        """
        Accessor function to activte a set of state machine variables
        """
        assert mode < 4
        self.OSL_CTRL.change_osl_mode(mode)