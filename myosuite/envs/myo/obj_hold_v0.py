""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

import collections
import numpy as np
import gym

from myosuite.envs.myo.base_v0 import BaseV0


class ObjHoldFixedEnvV0(BaseV0):

    DEFAULT_OBS_KEYS = ['hand_qpos', 'hand_qvel', 'obj_pos', 'obj_err']
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "goal_dist": 100.0,
        "bonus": 4.0,
        "penalty": 10,
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
            obs_keys:list = DEFAULT_OBS_KEYS,
            weighted_reward_keys:list = DEFAULT_RWD_KEYS_AND_WEIGHTS,
            **kwargs,
        ):
        self.object_sid = self.sim.model.site_name2id("object")
        self.goal_sid = self.sim.model.site_name2id("goal")
        self.object_init_pos = self.sim.data.site_xpos[self.object_sid].copy()

        super()._setup(obs_keys=obs_keys,
                    weighted_reward_keys=weighted_reward_keys,
                    **kwargs,
        )
        self.init_qpos[:-7] *= 0 # Use fully open as init pos
        self.init_qpos[0] = -1.5 # place palm up


    def get_obs_vec(self):
        self.obs_dict['t'] = np.array([self.sim.data.time])
        self.obs_dict['hand_qpos'] = self.sim.data.qpos[:-7].copy()
        self.obs_dict['hand_qvel'] = self.sim.data.qvel[:-6].copy()*self.dt
        self.obs_dict['obj_pos'] = self.sim.data.site_xpos[self.object_sid]
        self.obs_dict['obj_err'] = self.sim.data.site_xpos[self.goal_sid] - self.sim.data.site_xpos[self.object_sid]
        if self.sim.model.na>0:
            self.obs_dict['act'] = self.sim.data.act[:].copy()

        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['t'] = np.array([sim.data.time])
        obs_dict['hand_qpos'] = sim.data.qpos[:-7].copy()
        obs_dict['hand_qvel'] = sim.data.qvel[:-6].copy()*self.dt
        obs_dict['obj_pos'] = sim.data.site_xpos[self.object_sid]
        obs_dict['obj_err'] = sim.data.site_xpos[self.goal_sid] - sim.data.site_xpos[self.object_sid]
        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()
        return obs_dict

    def get_reward_dict(self, obs_dict):
        goal_dist = np.abs(np.linalg.norm(self.obs_dict['obj_err'], axis=-1)) #-0.040)
        act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0
        gaol_th = .010
        drop = goal_dist > 0.300

        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('goal_dist', -1.*goal_dist),
            ('bonus', 1.*(goal_dist<2*gaol_th) + 1.*(goal_dist<gaol_th)),
            ('act_reg', -1.*act_mag),
            ('penalty', -1.*drop),
            # Must keys
            ('sparse', -goal_dist),
            ('solved', goal_dist<gaol_th),
            ('done', drop),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict


class ObjHoldRandomEnvV0(ObjHoldFixedEnvV0):

    def reset(self):
        # randomize target pos
        self.sim.model.site_pos[self.goal_sid] = self.object_init_pos + self.np_random.uniform(high=np.array([0.030, 0.030, 0.030]), low=np.array([-.030, -.030, -.030]))
        # randomize object
        size = self.np_random.uniform(high=np.array([0.030, 0.030, 0.030]), low=np.array([.020, .020, .020]))
        self.sim.model.geom_size[-1] = size
        self.sim.model.site_size[self.goal_sid] = size
        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super().reset()
        return obs