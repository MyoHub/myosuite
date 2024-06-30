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

from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.envs.myo.myobase.walk_v0 import WalkEnvV0
from myosuite.utils.quat_math import quat2euler, euler2mat, euler2quat
from myosuite.utils.heightfields import TrackField


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
               **kwargs,
               ):


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
        self.grf_sensor_names = ['r_foot', 'r_toes', 'l_foot', 'l_toes']
        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       reset_type=reset_type,
                       **kwargs
                       )
        self.init_qpos[:] = self.sim.model.key_qpos[0]
        self.init_qvel[:] = 0.0
        self.assert_settings()



    def assert_settings(self):
        pass

    def get_obs_dict(self, sim):
        obs_dict = {}

        # Time
        obs_dict['time'] = np.array([sim.data.time])

        # proprioception
        obs_dict['internal_qpos'] = sim.data.qpos[7:].copy()
        obs_dict['internal_qvel'] = sim.data.qvel[6:].copy() * self.dt
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
        results = super().step(*args, **kwargs)
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
