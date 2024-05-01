""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

import collections
from myosuite.utils import gym
import numpy as np

from myosuite.envs.myo.base_v0 import BaseV0



class ReachEnvV0(BaseV0):

    DEFAULT_OBS_KEYS = ['qpos', 'qvel', 'tip_pos', 'reach_err']
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "reach": 1.0,
        "bonus": 4.0,
        "penalty": 50,
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
            far_th = .35,
            obs_keys:list = DEFAULT_OBS_KEYS,
            weighted_reward_keys:dict = DEFAULT_RWD_KEYS_AND_WEIGHTS,
            **kwargs,
        ):
        self.far_th = far_th
        self.target_reach_range = target_reach_range
        super()._setup(obs_keys=obs_keys,
                weighted_reward_keys=weighted_reward_keys,
                sites=self.target_reach_range.keys(),
                **kwargs,
                )

    def get_obs_vec(self):
        self.obs_dict['time'] = np.array([self.sim.data.time])
        self.obs_dict['qpos'] = self.sim.data.qpos[:].copy()
        self.obs_dict['qvel'] = self.sim.data.qvel[:].copy()*self.dt
        if self.sim.model.na>0:
            self.obs_dict['act'] = self.sim.data.act[:].copy()

        # reach error
        self.obs_dict['tip_pos'] = np.array([])
        self.obs_dict['target_pos'] = np.array([])
        for isite in range(len(self.tip_sids)):
            self.obs_dict['tip_pos'] = np.append(self.obs_dict['tip_pos'], self.sim.data.site_xpos[self.tip_sids[isite]].copy())
            self.obs_dict['target_pos'] = np.append(self.obs_dict['target_pos'], self.sim.data.site_xpos[self.target_sids[isite]].copy())
        self.obs_dict['reach_err'] = np.array(self.obs_dict['target_pos'])-np.array(self.obs_dict['tip_pos'])

        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs

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
        act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0
        far_th = self.far_th*len(self.tip_sids) if np.squeeze(obs_dict['time'])>2*self.dt else np.inf
        near_th = len(self.tip_sids)*.0125
        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('reach',   -1.*reach_dist),
            ('bonus',   1.*(reach_dist<2*near_th) + 1.*(reach_dist<near_th)),
            ('act_reg', -1.*act_mag),
            ('penalty', -1.*(reach_dist>far_th)),
            # Must keys
            ('sparse',  -1.*reach_dist),
            ('solved',  reach_dist<near_th),
            ('done',    reach_dist > far_th),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    # generate a valid target
    def generate_target_pose(self):
        for site, span in self.target_reach_range.items():
            sid =  self.sim.model.site_name2id(site+'_target')
            self.sim.model.site_pos[sid] = self.np_random.uniform(low=span[0], high=span[1])
        self.sim.forward()


    def reset(self, **kwargs):
        self.generate_target_pose()
        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super().reset(**kwargs)
        return obs