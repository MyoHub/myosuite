""" =================================================
# Copyright (c) MyoSuite Authors
Authors  :: Cheryl Wang (cheryl.wang.huiyi@gmail.com), Balint Hodossy (bkh16@ic.ac.uk), 
            Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com), Guillaume Durandau (guillaume.durandau@mcgill.ca)
================================================= """

import collections
import numpy as np
from myosuite.utils import gym
import mujoco

from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.utils.quat_math import mat2euler, euler2quat

MAX_TIME = 5.0

class PingPongEnvV0(BaseV0):

    DEFAULT_OBS_KEYS = ['body_qpos', 'body_qvel', 'ball_pos', 'ball_vel', 'paddle_pos', "paddle_vel", 'reach_err']
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "reach_dist": -1,
        "act": 1,
        "sparse": 1,
        "solved": 1,
        'done': -10
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        # Two step construction (init+setup) is required for pickling to work correctly.
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)
        self._setup(**kwargs)


    def _setup(self,
            qpos_noise_range = None, # Noise in joint space for initialization
            obs_keys:list = DEFAULT_OBS_KEYS,
            ball_xyz_range = None,
            weighted_reward_keys:list = DEFAULT_RWD_KEYS_AND_WEIGHTS,
            **kwargs,
        ):
        self.paddle_sid = self.sim.model.site_name2id("paddle")
        self.ball_sid = self.sim.model.site_name2id("pingpong")
        self.ball_bid = self.sim.model.body_name2id("pingpong")
        self.ball_xyz_range = ball_xyz_range
        self.qpos_noise_range = qpos_noise_range

        self.id_info = IdInfo(self.sim.model)

        super()._setup(obs_keys=obs_keys,
                    weighted_reward_keys=weighted_reward_keys,
                    **kwargs,
        )
        keyFrame_id = 0
        self.init_qpos[:] = self.sim.model.key_qpos[keyFrame_id].copy()
        self.init_qvel[:] = 0


    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time])

        obs_dict['body_qpos'] = sim.data.qpos[self.id_info.myo_joint_range].copy()
        obs_dict['body_qvel'] = sim.data.qvel[self.id_info.myo_dof_range].copy()

        obs_dict["ball_pos"] = sim.data.site_xpos[self.ball_sid]
        obs_dict["ball_vel"] = self.get_sensor_by_name(sim.model, sim.data, "pingpong_vel_sensor")

        obs_dict["paddle_pos"] = sim.data.site_xpos[self.paddle_sid]
        obs_dict["paddle_vel"] = self.get_sensor_by_name(sim.model, sim.data, "paddle_vel_sensor")

        obs_dict['reach_err'] = obs_dict['paddle_pos'] - obs_dict['ball_pos']

        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()
        return obs_dict


    def get_reward_dict(self, obs_dict):
        reach_dist = np.abs(np.linalg.norm(self.obs_dict['reach_err'], axis=-1))
        act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0
        ball_pos = obs_dict["ball_pos"][0][0] if obs_dict['ball_pos'].ndim == 3 else obs_dict['ball_pos']
        rwd_dict = collections.OrderedDict((
            # Perform reward tuning here --
            # Update Optional Keys section below
            # Update reward keys (DEFAULT_RWD_KEYS_AND_WEIGHTS) accordingly to update final rewards
            # Examples: Env comes pre-packaged with two keys pos_dist and rot_dist
            # Optional Keys
            ('reach_dist', -1.*reach_dist),
            # Must keys
            ('act', -1.*act_mag),
            ('sparse', np.array([[ball_pos[0] < 0]])), #for reaching the other side of the table.
            ('solved', np.array([[0]])), ### FILL IN THE TOUCH CONDITION FOR THE OTHER END OF TABLE
            ('done', np.array([[self._get_done(ball_pos[-1])]])),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)

        return rwd_dict

    def _get_done(self, z):
        if self.obs_dict['time'] > MAX_TIME:
            return 1  
        elif z < 0.3:
            self.obs_dict['time'] = MAX_TIME
            return 1
        elif self.rwd_dict and self.rwd_dict['solved']:
            return 1
        return 0


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
    
    def get_sensor_by_name(self, model, data, name):
        sensor_id = model.sensor_name2id(name)
        start = model.sensor_adr[sensor_id]
        dim = model.sensor_dim[sensor_id]
        return data.sensordata[start:start+dim]


    def reset(self, reset_qpos=None, reset_qvel=None, **kwargs):
        #self.sim.model.body_pos[self.object_bid] = self.np_random.uniform(**self.target_xyz_range)
        #self.sim.model.body_quat[self.object_bid] = euler2quat(self.np_random.uniform(**self.target_rxryrz_range))

        if self.ball_xyz_range is not None:
            self.sim.model.body_pos[self.ball_bid] = self.np_random.uniform(**self.ball_xyz_range)

        # randomize init arms pose
        if self.qpos_noise_range is not None:
            reset_qpos_local = self.init_qpos + self.qpos_noise_range*(self.sim.model.jnt_range[:,1]-self.sim.model.jnt_range[:,0])
            reset_qpos_local[-6:] = self.init_qpos[-6:]
        else:
            reset_qpos_local = reset_qpos
            
        self.init_qpos[:] = self.sim.model.key_qpos[0].copy()
        obs = super().reset(reset_qpos=reset_qpos_local, reset_qvel=reset_qvel,**kwargs)

        return obs
    
class IdInfo:
    def __init__(self, model: mujoco.MjModel):
        self.ball_body_id = model.body("pingpong").id

        myo_bodies = [model.body(i).id for i in range(model.nbody)
                    if not model.body(i).name.startswith("ping")
                    and not model.body(i).name in ["pingpong"]]
        self.myo_body_range = (min(myo_bodies), max(myo_bodies))

        self.myo_joint_range = np.concatenate([model.joint(i).qposadr for i in range(model.njnt)
                                            if not model.joint(i).name.startswith("ping")
                                            and not model.joint(i).name == "pingpong_freejoint"])

        self.myo_dof_range = np.concatenate([model.joint(i).dofadr for i in range(model.njnt)
                                            if not model.joint(i).name.startswith("ping")
                                            and not model.joint(i).name == "pingpong_freejoint"])