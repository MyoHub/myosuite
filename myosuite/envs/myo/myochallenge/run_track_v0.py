""" ================================================= # Copyright (c) Facebook, Inc. and its affiliates Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com), Pierre Schumacher (schumacherpier@gmail.com), Chun Kwang Tan (cktan.neumove@gmail.com) =================================================
"""
import collections
from myosuite.utils import gym
import numpy as np
import pink
import os
from enum import Enum
from typing import Optional, Tuple
import copy

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
        self.startFlag = True

    def _setup(self,
               obs_keys: list = DEFAULT_OBS_KEYS,
               weighted_reward_keys: dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
               reset_type='none',
               terrain='FLAT',
               hills_difficulties=(0,0),
               rough_difficulties=(0,0),
               stairs_difficulties=(0,0),
               real_length=15,
               real_width=1,
               distance_thr = 10,
               **kwargs,
               ):

        # OSL specific init
        self.OSL = OpenSourceLeg(frequency=200)
        self.OSL.add_joint(name="knee", gear_ratio=49.4, offline_mode=True)
        self.OSL.add_joint(name="ankle", gear_ratio=58.4, offline_mode=True)
        self._init_default_OSL_param()
        self.OSL_FSM = self.build_4_state_FSM(self.OSL)

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
            # self.sim.model.geom_pos[self.sim.model.geom_name2id('terrain')] = np.array([0, -10, 0.005])
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
        qpos[1] = 15
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
        y_pos = self.obs_dict['model_root_pos'].squeeze()[1] if self.startFlag else -1
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
    OSL related functions
    """
    def _prepareActions(self, mus_actions):
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
        P = eval(f"self.OSL_FSM.current_state.{joint}_damping")
        theta = np.deg2rad(eval(f"self.OSL_FSM.current_state.{joint}_theta"))
        peak_torque = self.ACTUATOR_PARAM[f"osl_{joint}_torque_actuator"]['Fmax']

        temp_shorten = self.sim.data.joint(f"osl_{joint}_angle_r")

        T = np.clip( K*(theta - temp_shorten.qpos[0].copy()) - P*(temp_shorten.qvel[0].copy()) , -1*peak_torque, peak_torque)

        return np.clip(T / self.sim.model.actuator(f"osl_{joint}_torque_actuator").gear[0],
                       self.sim.model.actuator(f"osl_{joint}_torque_actuator").ctrlrange[0],
                       self.sim.model.actuator(f"osl_{joint}_torque_actuator").ctrlrange[1])

    def set_osl_mode(self, mode=0):
        self.OSL_PARAM_SELECT = mode


    # State machine.
    def build_4_state_FSM(self, osl: OpenSourceLeg) -> StateMachine:
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

        # STATE 1: EARLY STANCE
        KNEE_K_ESTANCE = self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['early_stance']['knee']['stiffness']
        KNEE_B_ESTANCE = self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['early_stance']['knee']['damping']
        KNEE_THETA_ESTANCE = self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['early_stance']['knee']['target_angle']
        ANKLE_K_ESTANCE = self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['early_stance']['ankle']['stiffness']
        ANKLE_B_ESTANCE = self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['early_stance']['ankle']['damping']
        ANKLE_THETA_ESTANCE = self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['early_stance']['ankle']['target_angle']
        LOAD_LSTANCE: float = BODY_WEIGHT * self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['early_stance']['threshold']['load'] # -1.0 * BODY_WEIGHT * 0.25
        ANKLE_THETA_ESTANCE_TO_LSTANCE = self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['early_stance']['threshold']['ankle_angle']

        # STATE 2: LATE STANCE
        KNEE_K_LSTANCE = self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['late_stance']['knee']['stiffness']
        KNEE_B_LSTANCE = self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['late_stance']['knee']['damping']
        KNEE_THETA_LSTANCE = self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['late_stance']['knee']['target_angle']
        ANKLE_K_LSTANCE = self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['late_stance']['ankle']['stiffness']
        ANKLE_B_LSTANCE = self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['late_stance']['ankle']['damping']
        ANKLE_THETA_LSTANCE = self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['late_stance']['ankle']['target_angle']
        LOAD_ESWING: float = BODY_WEIGHT * self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['late_stance']['threshold']['load'] # -1.0 * BODY_WEIGHT * 0.15

        # STATE 3: EARLY SWING
        KNEE_K_ESWING = self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['early_swing']['knee']['stiffness']
        KNEE_B_ESWING = self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['early_swing']['knee']['damping']
        KNEE_THETA_ESWING = self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['early_swing']['knee']['target_angle']
        ANKLE_K_ESWING = self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['early_swing']['ankle']['stiffness']
        ANKLE_B_ESWING = self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['early_swing']['ankle']['damping']
        ANKLE_THETA_ESWING = self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['early_swing']['ankle']['target_angle']
        KNEE_THETA_ESWING_TO_LSWING = self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['early_swing']['threshold']['knee_angle']
        KNEE_DTHETA_ESWING_TO_LSWING = self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['early_swing']['threshold']['knee_vel']

        # STATE 4: LATE SWING
        KNEE_K_LSWING = self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['late_swing']['knee']['stiffness']
        KNEE_B_LSWING = self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['late_swing']['knee']['damping']
        KNEE_THETA_LSWING = self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['late_swing']['knee']['target_angle']
        ANKLE_K_LSWING = self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['late_swing']['ankle']['stiffness']
        ANKLE_B_LSWING = self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['late_swing']['ankle']['damping']
        ANKLE_THETA_LSWING = self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['late_swing']['ankle']['target_angle']
        LOAD_ESTANCE: float = BODY_WEIGHT * self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['late_swing']['threshold']['load'] # -1.0 * BODY_WEIGHT * 0.4
        KNEE_THETA_LSWING_TO_ESTANCE = self.OSL_PARAM_LIST[self.OSL_PARAM_SELECT]['late_swing']['threshold']['knee_angle']
        # ---------------------------------------------------- #


        early_stance = State(name="e_stance")
        late_stance = State(name="l_stance")
        early_swing = State(name="e_swing")
        late_swing = State(name="l_swing")

        early_stance.set_knee_impedance_paramters(
            theta=KNEE_THETA_ESTANCE, k=KNEE_K_ESTANCE, b=KNEE_B_ESTANCE
        )
        early_stance.make_knee_active()
        early_stance.set_ankle_impedance_paramters(
            theta=ANKLE_THETA_ESTANCE, k=ANKLE_K_ESTANCE, b=ANKLE_B_ESTANCE
        )
        early_stance.make_ankle_active()

        late_stance.set_knee_impedance_paramters(
            theta=KNEE_THETA_LSTANCE, k=KNEE_K_LSTANCE, b=KNEE_B_LSTANCE
        )
        late_stance.make_knee_active()
        late_stance.set_ankle_impedance_paramters(
            theta=ANKLE_THETA_LSTANCE, k=ANKLE_K_LSTANCE, b=ANKLE_B_LSTANCE
        )
        late_stance.make_ankle_active()

        early_swing.set_knee_impedance_paramters(
            theta=KNEE_THETA_ESWING, k=KNEE_K_ESWING, b=KNEE_B_ESWING
        )
        early_swing.make_knee_active()
        early_swing.set_ankle_impedance_paramters(
            theta=ANKLE_THETA_ESWING, k=ANKLE_K_ESWING, b=ANKLE_B_ESWING
        )
        early_swing.make_ankle_active()

        late_swing.set_knee_impedance_paramters(
            theta=KNEE_THETA_LSWING, k=KNEE_K_LSWING, b=KNEE_B_LSWING
        )
        late_swing.make_knee_active()
        late_swing.set_ankle_impedance_paramters(
            theta=ANKLE_THETA_LSWING, k=ANKLE_K_LSWING, b=ANKLE_B_LSWING
        )
        late_swing.make_ankle_active()

        def estance_to_lstance(osl: OpenSourceLeg) -> bool:
            """
            Transition from early stance to late stance when the loadcell
            reads a force greater than a threshold.
            """
            #assert osl.loadcell is not None
            foot_load = self.sim.data.sensor('r_prosth').data[0].copy()
            ankle_pos = self.sim.data.joint('osl_ankle_angle_r').qpos[0].copy()

            return bool(
                foot_load < LOAD_LSTANCE
                and ankle_pos > ANKLE_THETA_ESTANCE_TO_LSTANCE
            )

        def lstance_to_eswing(osl: OpenSourceLeg) -> bool:
            """
            Transition from late stance to early swing when the loadcell
            reads a force less than a threshold.
            """
            #assert osl.loadcell is not None
            foot_load = self.sim.data.sensor('r_prosth').data[0].copy()

            return bool(foot_load > LOAD_ESWING)

        def eswing_to_lswing(osl: OpenSourceLeg) -> bool:
            """
            Transition from early swing to late swing when the knee angle
            is greater than a threshold and the knee velocity is less than
            a threshold.
            """
            #assert osl.knee is not None

            knee_pos = self.sim.data.joint('osl_knee_angle_r').qpos[0].copy()
            knee_vel = self.sim.data.joint('osl_knee_angle_r').qvel[0].copy()

            return bool(
                knee_pos > KNEE_THETA_ESWING_TO_LSWING
                and knee_vel < KNEE_DTHETA_ESWING_TO_LSWING
            )

        def lswing_to_estance(osl: OpenSourceLeg) -> bool:
            """
            Transition from late swing to early stance when the loadcell
            reads a force greater than a threshold or the knee angle is
            less than a threshold.
            """
            #assert osl.knee is not None and osl.loadcell is not None

            knee_pos = self.sim.data.joint('osl_knee_angle_r').qpos[0].copy()
            foot_load = self.sim.data.sensor('r_prosth').data[0].copy()

            return bool(
                foot_load < LOAD_ESTANCE
                or np.rad2deg(knee_pos) < KNEE_THETA_LSWING_TO_ESTANCE
            )

        foot_flat = Event(name="foot_flat")
        heel_off = Event(name="heel_off")
        toe_off = Event(name="toe_off")
        pre_heel_strike = Event(name="pre_heel_strike")
        heel_strike = Event(name="heel_strike")

        fsm = StateMachine(osl=osl, spoof=False)
        fsm.add_state(state=early_stance, initial_state=True)
        fsm.add_state(state=late_stance)
        fsm.add_state(state=early_swing)
        fsm.add_state(state=late_swing)

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
    def set_osl_params(self, params, mode=0):

        phase_list = ['early_stance', 'late_stance', 'early_swing', 'late_swing']
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


    def _init_default_OSL_param(self):
        temp_dict = {}
        temp_dict['early_stance'] = {}
        temp_dict['early_stance']['knee'] = {}
        temp_dict['early_stance']['ankle'] = {}
        temp_dict['early_stance']['threshold'] = {}
        temp_dict['early_stance']['knee']['stiffness'] = 99.372
        temp_dict['early_stance']['knee']['damping'] = 3.180
        temp_dict['early_stance']['knee']['target_angle'] = np.deg2rad(5)
        temp_dict['early_stance']['ankle']['stiffness'] = 19.874
        temp_dict['early_stance']['ankle']['damping'] = 0
        temp_dict['early_stance']['ankle']['target_angle'] = np.deg2rad(-2)
        temp_dict['early_stance']['threshold']['load'] = 0.25
        temp_dict['early_stance']['threshold']['ankle_angle'] = np.deg2rad(6)

        temp_dict['late_stance'] = {}
        temp_dict['late_stance']['knee'] = {}
        temp_dict['late_stance']['ankle'] = {}
        temp_dict['late_stance']['threshold'] = {}
        temp_dict['late_stance']['knee']['stiffness'] = 99.372
        temp_dict['late_stance']['knee']['damping'] = 1.272
        temp_dict['late_stance']['knee']['target_angle'] = np.deg2rad(8)
        temp_dict['late_stance']['ankle']['stiffness'] = 79.498
        temp_dict['late_stance']['ankle']['damping'] = 0.063
        temp_dict['late_stance']['ankle']['target_angle'] = np.deg2rad(-20)
        temp_dict['late_stance']['threshold']['load'] = 0.15

        temp_dict['early_swing'] = {}
        temp_dict['early_swing']['knee'] = {}
        temp_dict['early_swing']['ankle'] = {}
        temp_dict['early_swing']['threshold'] = {}
        temp_dict['early_swing']['knee']['stiffness'] = 39.749
        temp_dict['early_swing']['knee']['damping'] = 0.063
        temp_dict['early_swing']['knee']['target_angle'] = np.deg2rad(60)
        temp_dict['early_swing']['ankle']['stiffness'] = 7.949
        temp_dict['early_swing']['ankle']['damping'] = 0
        temp_dict['early_swing']['ankle']['target_angle'] = np.deg2rad(25)
        temp_dict['early_swing']['threshold']['knee_angle'] = np.deg2rad(50)
        temp_dict['early_swing']['threshold']['knee_vel'] = np.deg2rad(3)

        temp_dict['late_swing'] = {}
        temp_dict['late_swing']['knee'] = {}
        temp_dict['late_swing']['ankle'] = {}
        temp_dict['late_swing']['threshold'] = {}
        temp_dict['late_swing']['knee']['stiffness'] = 15.899
        temp_dict['late_swing']['knee']['damping'] = 3.816
        temp_dict['late_swing']['knee']['target_angle'] = np.deg2rad(5)
        temp_dict['late_swing']['ankle']['stiffness'] = 7.949
        temp_dict['late_swing']['ankle']['damping'] = 0
        temp_dict['late_swing']['ankle']['target_angle'] = np.deg2rad(15)
        temp_dict['late_swing']['threshold']['load'] = 0.4
        temp_dict['late_swing']['threshold']['knee_angle'] = np.deg2rad(30)

        for idx in np.arange(3):
            self.OSL_PARAM_LIST.append(copy.deepcopy(temp_dict))