""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

import collections
import numpy as np
import gym

from myosuite.envs.myo.base_v0 import BaseV0


class KeyTurnEnvV0(BaseV0):

    DEFAULT_OBS_KEYS = ['hand_qpos', 'hand_qvel', 'key_qpos', 'key_qvel', 'IFtip_approach', 'THtip_approach']
    DEFAULT_RWD_KEYS_AND_WEIGHTS= {
        'key_turn':1.0,
        'IFtip_approach':10.0,
        'THtip_approach':10.0,
        'act_reg':1.0,
        'bonus':4.0,
        'penalty':25.0
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
            goal_th:float=3.14,
            obs_keys:list = DEFAULT_OBS_KEYS,
            weighted_reward_keys:list = DEFAULT_RWD_KEYS_AND_WEIGHTS,
            key_init_range:tuple=(0,0),
            **kwargs,
        ):
        self.goal_th = goal_th
        self.keyhead_sid = self.sim.model.site_name2id("keyhead")
        self.IF_sid = self.sim.model.site_name2id("IFtip")
        self.TH_sid = self.sim.model.site_name2id("THtip")
        self.key_init_range = key_init_range
        self.key_init_pos = self.sim.data.site_xpos[self.keyhead_sid].copy()

        super()._setup(obs_keys=obs_keys,
                    weighted_reward_keys=weighted_reward_keys,
                    **kwargs,
        )
        self.init_qpos[:-1] *= 0 # Use fully open as init pos

    def get_obs_vec(self):
        self.obs_dict['t'] = np.array([self.sim.data.time])
        self.obs_dict['hand_qpos'] = self.sim.data.qpos[:-1].copy()
        self.obs_dict['hand_qvel'] = self.sim.data.qvel[:-1].copy()*self.dt
        self.obs_dict['key_qpos'] = np.array([self.sim.data.qpos[-1]])
        self.obs_dict['key_qvel'] = np.array([self.sim.data.qvel[-1]])*self.dt
        self.obs_dict['IFtip_approach'] = self.sim.data.site_xpos[self.keyhead_sid]-self.sim.data.site_xpos[self.IF_sid]
        self.obs_dict['THtip_approach'] = self.sim.data.site_xpos[self.keyhead_sid]-self.sim.data.site_xpos[self.TH_sid]

        if self.sim.model.na>0:
            self.obs_dict['act'] = self.sim.data.act[:].copy()

        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['t'] = np.array([sim.data.time])
        obs_dict['hand_qpos'] = sim.data.qpos[:-1].copy()
        obs_dict['hand_qvel'] = sim.data.qvel[:-1].copy()*self.dt
        obs_dict['key_qpos'] = np.array([sim.data.qpos[-1]])
        obs_dict['key_qvel'] = np.array([sim.data.qvel[-1]])*self.dt
        obs_dict['IFtip_approach'] = sim.data.site_xpos[self.keyhead_sid]-sim.data.site_xpos[self.IF_sid]
        obs_dict['THtip_approach'] = sim.data.site_xpos[self.keyhead_sid]-sim.data.site_xpos[self.TH_sid]
        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()
        return obs_dict

    def get_reward_dict(self, obs_dict):
        IF_approach_dist = np.abs(np.linalg.norm(self.obs_dict['IFtip_approach'], axis=-1)-0.030)
        TH_approach_dist = np.abs(np.linalg.norm(self.obs_dict['THtip_approach'], axis=-1)-0.030)
        key_pos = obs_dict['key_qpos'][:,:,0] if obs_dict['key_qpos'].ndim==3 else obs_dict['key_qpos'][0]
        act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0
        far_th = 0.1
        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('key_turn', key_pos),
            ('IFtip_approach', -1.*IF_approach_dist),
            ('THtip_approach', -1.*TH_approach_dist),
            ('act_reg', -1.*act_mag),
            ('bonus', 1.*(key_pos>np.pi/2) + 1.*(key_pos>np.pi)),
            ('penalty', -1.*(IF_approach_dist>far_th/2)-1.*(TH_approach_dist>far_th/2) ),
            # Must keys
            ('sparse', key_pos),
            ('solved', obs_dict['key_qpos']>self.goal_th),
            ('done', (IF_approach_dist>far_th) or (TH_approach_dist>far_th)),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    def reset(self, reset_qpos=None, reset_qvel=None):
        qpos = self.init_qpos.copy() if reset_qpos is None else reset_qpos
        qvel = self.init_qvel.copy() if reset_qvel is None else reset_qvel
        qpos[-1] = self.np_random.uniform(low=self.key_init_range[0], high=self.key_init_range[1])
        if self.key_init_range[0]!=self.key_init_range[1]: # randomEnv
            self.sim.model.body_pos[-1] = self.key_init_pos+self.np_random.uniform(low=np.array([-0.01, -0.01, -.01]), high=np.array([0.01, 0.01, 0.01]))
        self.robot.reset(qpos, qvel)
        return self.get_obs()