""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

import collections
import numpy as np
from myosuite.utils import gym

from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.utils.quat_math import mat2euler, euler2quat

class ReorientEnvV0(BaseV0):

    DEFAULT_OBS_KEYS = ['hand_qpos_noMD5', 'hand_qvel', 'obj_pos', 'goal_pos', 'pos_err', 'obj_rot', 'goal_rot', 'rot_err']
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pos_dist": 100.0,
        "rot_dist": 1.0,
        "bonus": 0.0, #4.0,
        "act_reg": 0.0, #1,
        "penalty": 0.0 # 10,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        # Two step construction (init+setup) is required for pickling to work correctly.
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)
        self._setup(**kwargs)

    def _setup(self,
            obs_keys:list = DEFAULT_OBS_KEYS,
            weighted_reward_keys:list = DEFAULT_RWD_KEYS_AND_WEIGHTS,
            goal_pos = (0.0, 0.0),          # goal position range (relative to initial pos)
            goal_rot = (.785, .785),        # goal rotation range (relative to initial rot)
            obj_size_change = 0,            # object size change (relative to initial size)
            obj_mass_range = (.108,.108),   # object size change (relative to initial size)
            obj_friction_change = (0,0,0),  # object friction change (relative to initial size)
            pos_th = .025,                  # position error threshold
            rot_th = 0.262,                 # rotation error threshold
            drop_th = .200,                 # drop height threshold
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

        # setup for object randomization
        self.target_gid = self.sim.model.geom_name2id('target_dice')
        self.target_default_size = self.sim.model.geom_size[self.target_gid].copy()

        self.object_bid = self.sim.model.body_name2id('Object')
        self.object_gid0 = self.sim.model.body_geomadr[self.object_bid]
        self.object_gidn = self.object_gid0 + self.sim.model.body_geomnum[self.object_bid]
        self.object_default_size = self.sim.model.geom_size[self.object_gid0:self.object_gidn].copy()
        self.object_default_pos = self.sim.model.geom_pos[self.object_gid0:self.object_gidn].copy()

        self.obj_mass_range = {'low':obj_mass_range[0], 'high':obj_mass_range[1]}
        self.obj_size_range = {'low':-obj_size_change, 'high':obj_size_change}
        self.obj_friction_range = {'low':self.sim.model.geom_friction[self.object_gid0:self.object_gidn] - obj_friction_change,
                                    'high':self.sim.model.geom_friction[self.object_gid0:self.object_gidn] + obj_friction_change}

        super()._setup(obs_keys=obs_keys,
                    weighted_reward_keys=weighted_reward_keys,
                    **kwargs,
        )
        self.init_qpos[:-7] *= 0 # Use fully open as init pos
        self.init_qpos[0] = -1.5 # Palm up

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time])
        obs_dict['hand_qpos_noMD5'] = sim.data.qpos[:-7].copy() # ??? This is a bug. This needs to be qpos[:-6]. This bug omits the distal joint of the little finger from the observation. A fix to this will break all the submitted policies. A fix to this will be pushed after the myochallenge23
        obs_dict['hand_qpos'] = sim.data.qpos[:-6].copy() # V1 of the env will use this corrected key by default
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
            ('bonus', 1.*(pos_dist<2*self.pos_th) + 1.*(pos_dist<self.pos_th)),
            ('act_reg', -1.*act_mag),
            ('penalty', -1.*drop),
            # Must keys
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

    def reset(self, reset_qpos=None, reset_qvel=None, **kwargs):
        self.sim.model.body_pos[self.goal_bid] = self.goal_init_pos + \
            self.np_random.uniform( high=self.goal_pos[1], low=self.goal_pos[0], size=3)

        self.sim.model.body_quat[self.goal_bid] = \
            euler2quat(self.np_random.uniform(high=self.goal_rot[1], low=self.goal_rot[0], size=3))

        # Die friction changes
        self.sim.model.geom_friction[self.object_gid0:self.object_gidn] = self.np_random.uniform(**self.obj_friction_range)
        # Die mass changes
        self.sim.model.body_mass[self.object_bid] = self.np_random.uniform(**self.obj_mass_range) # call to mj_setConst(m,d) is being ignored. Derive quantities wont be updated. Die is simple shape. So this is reasonable approximation.

        # Die and Target size changes
        del_size = self.np_random.uniform(**self.obj_size_range)
        # adjust size of target
        self.sim.model.geom_size[self.target_gid] = self.target_default_size + del_size
        # adjust size of die
        self.sim.model.geom_size[self.object_gid0:self.object_gidn-3][:,1] = self.object_default_size[:-3][:,1] + del_size
        self.sim.model.geom_size[self.object_gidn-3:self.object_gidn] = self.object_default_size[-3:] + del_size
        # adjust boundary of die
        object_gpos = self.sim.model.geom_pos[self.object_gid0:self.object_gidn]
        self.sim.model.geom_pos[self.object_gid0:self.object_gidn] = object_gpos/abs(object_gpos+1e-16) * (abs(self.object_default_pos) + del_size)

        obs = super().reset(reset_qpos, reset_qvel, **kwargs)
        return obs