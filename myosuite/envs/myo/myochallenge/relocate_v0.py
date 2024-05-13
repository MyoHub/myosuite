""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

import collections
import numpy as np
from myosuite.utils import gym

from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.utils.quat_math import mat2euler, euler2quat

class RelocateEnvV0(BaseV0):

    DEFAULT_OBS_KEYS = ['hand_qpos', 'hand_qvel', 'obj_pos', 'goal_pos', 'pos_err', 'obj_rot', 'goal_rot', 'rot_err']
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pos_dist": 100.0,
        "rot_dist": 1.0,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        # Two step construction (init+setup) is required for pickling to work correctly.
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)
        self._setup(**kwargs)


    def _setup(self,
            target_xyz_range,        # target position range (relative to initial pos)
            target_rxryrz_range,     # target rotation range (relative to initial rot)
            obj_xyz_range = None,    # object position range (relative to initial pos)
            obj_geom_range = None,   # randomization sizes for object geoms
            obj_mass_range = None,   # object size range
            obj_friction_range = None,# object friction range
            qpos_noise_range = None, # Noise in joint space for initialization
            obs_keys:list = DEFAULT_OBS_KEYS,
            weighted_reward_keys:list = DEFAULT_RWD_KEYS_AND_WEIGHTS,
            pos_th = .025,          # position error threshold
            rot_th = 0.262,         # rotation error threshold
            drop_th = 0.50,         # drop height threshold
            **kwargs,
        ):
        self.palm_sid = self.sim.model.site_name2id("S_grasp")
        self.object_sid = self.sim.model.site_name2id("object_o")
        self.object_bid = self.sim.model.body_name2id("Object")
        self.goal_sid = self.sim.model.site_name2id("target_o")
        self.success_indicator_sid = self.sim.model.site_name2id("target_ball")
        self.goal_bid = self.sim.model.body_name2id("target")
        self.target_xyz_range = target_xyz_range
        self.target_rxryrz_range = target_rxryrz_range
        self.obj_geom_range = obj_geom_range
        self.obj_mass_range = obj_mass_range
        self.obj_friction_range = obj_friction_range
        self.obj_xyz_range = obj_xyz_range
        self.qpos_noise_range = qpos_noise_range
        self.pos_th = pos_th
        self.rot_th = rot_th
        self.drop_th = drop_th

        super()._setup(obs_keys=obs_keys,
                    weighted_reward_keys=weighted_reward_keys,
                    **kwargs,
        )
        keyFrame_id = 0 if self.obj_xyz_range is None else 1
        self.init_qpos[:] = self.sim.model.key_qpos[keyFrame_id].copy()


    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time])
        obs_dict['hand_qpos'] = sim.data.qpos[:-7].copy()
        obs_dict['hand_qpos_corrected'] = sim.data.qpos[:-6].copy()
        obs_dict['hand_qvel'] = sim.data.qvel[:-6].copy()*self.dt
        obs_dict['obj_pos'] = sim.data.site_xpos[self.object_sid]
        obs_dict['goal_pos'] = sim.data.site_xpos[self.goal_sid]
        obs_dict['palm_pos'] = sim.data.site_xpos[self.palm_sid]
        obs_dict['pos_err'] = obs_dict['goal_pos'] - obs_dict['obj_pos']
        obs_dict['reach_err'] = obs_dict['palm_pos'] - obs_dict['obj_pos']
        obs_dict['obj_rot'] = mat2euler(np.reshape(sim.data.site_xmat[self.object_sid],(3,3)))
        obs_dict['goal_rot'] = mat2euler(np.reshape(sim.data.site_xmat[self.goal_sid],(3,3)))
        obs_dict['rot_err'] = obs_dict['goal_rot'] - obs_dict['obj_rot']

        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()
        return obs_dict


    def get_reward_dict(self, obs_dict):
        reach_dist = np.abs(np.linalg.norm(self.obs_dict['reach_err'], axis=-1))
        pos_dist = np.abs(np.linalg.norm(self.obs_dict['pos_err'], axis=-1))
        rot_dist = np.abs(np.linalg.norm(self.obs_dict['rot_err'], axis=-1))
        act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0
        drop = reach_dist > self.drop_th
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

        # Success Indicator
        self.sim.model.site_rgba[self.success_indicator_sid, :2] = np.array([0, 2]) if rwd_dict['solved'] else np.array([2, 0])
        self.sim.model.site_size[self.success_indicator_sid, :] = np.array([.25,]) if rwd_dict['solved'] else np.array([0.1,])
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
        self.sim.model.body_pos[self.goal_bid] = self.np_random.uniform(**self.target_xyz_range)
        self.sim.model.body_quat[self.goal_bid] = euler2quat(self.np_random.uniform(**self.target_rxryrz_range))


        if self.obj_xyz_range is not None:
            self.sim.model.body_pos[self.object_bid] = self.np_random.uniform(**self.obj_xyz_range)


        if self.obj_geom_range is not None:
            for body in ["Object", ]:
                # object shapes and locations
                bid = self.sim.model.body_name2id(body)
                for gid in range(self.sim.model.body_geomnum[bid]):
                    gid+=self.sim.model.body_geomadr[bid] # get geom ids
                    # update type, size, and collision bounds
                    self.sim.model.geom_type[gid]=self.np_random.choice([2,3,4,5,6]) # random shape
                    self.sim.model.geom_size[gid]=self.np_random.uniform(low=self.obj_geom_range['low'], high=self.obj_geom_range['high']) # random size
                    self.sim.model.geom_aabb[gid][3:]= self.obj_geom_range['high'] # bounding box, (center, size)
                    self.sim.model.geom_rbound[gid] = 2.0*max(self.obj_geom_range['high']) # radius of bounding sphere

                    self.sim.model.geom_pos[gid]=self.np_random.uniform(low=-1.0*self.sim.model.geom_size[gid], high=self.sim.model.geom_size[gid]) # random pos
                    self.sim.model.geom_quat[gid]=euler2quat(self.np_random.uniform(low=(-np.pi/2, -np.pi/2, -np.pi/2), high=(np.pi/2, np.pi/2, np.pi/2)) ) # random quat
                    self.sim.model.geom_rgba[gid]=self.np_random.uniform(low=[.2, .2, .2, 1], high=[.9, .9, .9, 1]) # random color

                    # friction changes
                    if self.obj_friction_range is not None:
                        self.sim.model.geom_friction[gid] = self.np_random.uniform(**self.obj_friction_range)

                # mass changes
                if self.obj_mass_range is not None:
                    self.sim.model.body_mass[self.object_bid] = self.np_random.uniform(**self.obj_mass_range)
                    # ??? Derive quantities wont be updated.

                self.sim.forward()

        # randomize init arms pose
        if self.qpos_noise_range is not None:
            reset_qpos_local = self.init_qpos + self.qpos_noise_range*(self.sim.model.jnt_range[:,1]-self.sim.model.jnt_range[:,0])
            reset_qpos_local[-6:] = self.init_qpos[-6:]
        else:
            reset_qpos_local = reset_qpos

        obs = super().reset(reset_qpos=reset_qpos_local, reset_qvel=reset_qvel,**kwargs)
        if self.sim.data.ncon>0:
            self.reset(reset_qpos=reset_qpos, reset_qvel=reset_qvel,**kwargs)

        return obs