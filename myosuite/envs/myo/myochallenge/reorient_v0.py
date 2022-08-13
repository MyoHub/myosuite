""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

import collections
from traceback import print_tb
import numpy as np
import gym

from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.utils.quat_math import mat2euler, euler2quat

class ReorientEnvV0(BaseV0):

    DEFAULT_OBS_KEYS = ['hand_qpos', 'hand_qvel', 'obj_pos', 'goal_pos', 'pos_err', 'obj_rot', 'goal_rot', 'rot_err']
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pos_dist": 100.0,
        "rot_dist": 1.0,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        # Two step construction (init+setup) is required for pickling to work correctly.
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed)
        self._setup(**kwargs)

    def _setup(self,
            obs_keys:list = DEFAULT_OBS_KEYS,
            weighted_reward_keys:list = DEFAULT_RWD_KEYS_AND_WEIGHTS,
            goal_pos = (0.0, 0.0),      # goal position range (relative to initial pos)
            goal_rot = (.785, .785),    # goal rotation range (relative to initial rot)
            pos_th = .025,              # position error threshold
            rot_th = 0.262,             # rotation error threshold
            drop_th = .200,             # drop height threshold
            **kwargs,
        ):
        self.object_sid = self.sim.model.site_name2id("object_o")
        self.goal_sid = self.sim.model.site_name2id("target_o")
        self.success_indicator_sid = self.sim.model.site_name2id("target_ball")
        self.goal_bid = self.sim.model.body_name2id("target")
        self.goal_init_pos = self.sim.data.site_xpos[self.goal_sid].copy()
        self.goal_obj_offset = self.sim.data.site_xpos[self.goal_sid]-self.sim.data.site_xpos[self.object_sid] # visualization offset between target and object
        self.goal_pos = goal_pos
        self.goal_rot = goal_rot
        self.pos_th = pos_th
        self.rot_th = rot_th
        self.drop_th = drop_th

        super()._setup(obs_keys=obs_keys,
                    weighted_reward_keys=weighted_reward_keys,
                    **kwargs,
        )
        self.init_qpos[:-7] *= 0 # Use fully open as init pos
        self.init_qpos[0] = -1.5 # Palm up

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['t'] = np.array([sim.data.time])
        obs_dict['hand_qpos'] = sim.data.qpos[:-7].copy()
        obs_dict['hand_qvel'] = sim.data.qvel[:-6].copy()*self.dt
        obs_dict['obj_pos'] = sim.data.site_xpos[self.object_sid]
        obs_dict['goal_pos'] = sim.data.site_xpos[self.goal_sid]
        obs_dict['pos_err'] = obs_dict['goal_pos'] - obs_dict['obj_pos'] - self.goal_obj_offset # correct for visualization offset between target and object
        obs_dict['obj_rot'] = mat2euler(np.reshape(sim.data.site_xmat[self.object_sid],(3,3)))
        obs_dict['goal_rot'] = mat2euler(np.reshape(sim.data.site_xmat[self.goal_sid],(3,3)))
        obs_dict['rot_err'] = obs_dict['goal_rot'] - obs_dict['obj_rot']

        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()
        return obs_dict

    def get_reward_dict(self, obs_dict):
        pos_dist = np.abs(np.linalg.norm(self.obs_dict['pos_err'], axis=-1))
        rot_dist = np.abs(np.linalg.norm(self.obs_dict['rot_err'], axis=-1))
        act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0
        drop = pos_dist > self.drop_th

        rwd_dict = collections.OrderedDict((
            # Perform reward tuning here --
            # Update Optional Keys section below
            # Update reward keys (DEFAULT_RWD_KEYS_AND_WEIGHTS) accordingly to update final rewards
            # Examples: Env comes pre-packaged with two keys pos_dist and rot_dist

            # Optional Keys
            ('pos_dist', -1.*pos_dist),
            ('rot_dist', -1.*rot_dist),
            # Must keys
            ('act_reg', -1.*act_mag),
            ('sparse', -rot_dist-10.0*pos_dist),
            ('solved', (pos_dist<self.pos_th) and (rot_dist<self.rot_th) and (not drop) ),
            ('done', drop),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)

        # Sucess Indicator
        self.sim.model.site_rgba[self.success_indicator_sid, :2] = np.array([0, 2]) if rwd_dict['solved'] else np.array([2, 0])
        return rwd_dict


    def get_metrics(self, paths, successful_steps=5):
        """
        Evaluate paths and report metrics
        """
        num_success = 0
        num_paths = len(paths)

        # average sucess over entire env horizon
        for path in paths:
            # record success if solved for provided successful_steps
            if np.sum(path['env_infos']['rwd_dict']['solved'] * 1.0) > successful_steps:
                num_success += 1
        score = num_success/num_paths

        # average activations over entire trajectory (can be shorter than horizon, if done) realized
        effort = -1.0*np.mean([np.mean(p['env_infos']['rwd_dict']['act_reg']) for p in paths])

        metrics = {
            'score': score,
            'effort':effort,
            }
        return metrics

    def reset(self):
        self.sim.model.body_pos[self.goal_bid] = self.goal_init_pos + \
            self.np_random.uniform( high=self.goal_pos[1], low=self.goal_pos[0], size=3)

        self.sim.model.body_quat[self.goal_bid] = \
            euler2quat(self.np_random.uniform( high=self.goal_rot[1], low=self.goal_rot[0], size=3))

        obs = super().reset()
        return obs