""" =================================================
# Copyright (c) MyoSuite Authors
Authors  :: Cheryl Wang (cheryl.wang.huiyi@gmail.com), Balint Hodossy (bkh16@ic.ac.uk), 
            Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com), Guillaume Durandau (guillaume.durandau@mcgill.ca)
================================================= """

import collections
from typing import List
import enum
import mujoco
import numpy as np
from myosuite.utils import gym
import mujoco

from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.utils.quat_math import mat2euler, euler2quat

MAX_TIME = 5.0

class PingPongEnvV0(BaseV0):

    DEFAULT_OBS_KEYS = ['pelvis_pos', 'body_qpos', 'body_qvel', 'ball_pos', 'ball_vel', 'paddle_pos', "paddle_vel", 'reach_err', "touching_info"]
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
        self.contact_trajectory = []

        self.id_info = IdInfo(self.sim.model)
        self.ball_dofadr = self.sim.model.body_dofadr[self.id_info.ball_body_id]

        super()._setup(obs_keys=obs_keys,
                    weighted_reward_keys=weighted_reward_keys,
                    **kwargs,
        )
        keyFrame_id = 0
        self.init_qpos[:] = self.sim.model.key_qpos[keyFrame_id].copy()
        self.start_vel = np.array([[5.5, 1, -2.8] ])
        self.init_qvel[self.ball_dofadr : self.ball_dofadr + 3] = self.start_vel


    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time])

        obs_dict['pelvis_pos'] = sim.data.site_xpos[self.sim.model.site_name2id("pelvis")]

        obs_dict['body_qpos'] = sim.data.qpos[self.id_info.myo_joint_range].copy()
        obs_dict['body_qvel'] = sim.data.qvel[self.id_info.myo_dof_range].copy()

        obs_dict["ball_pos"] = sim.data.site_xpos[self.ball_sid]
        obs_dict["ball_vel"] = self.get_sensor_by_name(sim.model, sim.data, "pingpong_vel_sensor")

        obs_dict["paddle_pos"] = sim.data.site_xpos[self.paddle_sid]
        obs_dict["paddle_vel"] = self.get_sensor_by_name(sim.model, sim.data, "paddle_vel_sensor")

        obs_dict['reach_err'] = obs_dict['paddle_pos'] - obs_dict['ball_pos']

        this_model = sim.model
        this_data = sim.data

        touching_objects = set(get_touching_objects(this_model, this_data, self.id_info))
        self.contact_trajectory.append(touching_objects)

        obs_vec = self._ball_label_to_obs(touching_objects)
        obs_dict["touching_info"] = obs_vec


        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()
        return obs_dict


    def get_reward_dict(self, obs_dict):
        reach_dist = np.abs(np.linalg.norm(self.obs_dict['reach_err'], axis=-1))
        act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0
        ball_pos = obs_dict["ball_pos"][0][0] if obs_dict['ball_pos'].ndim == 3 else obs_dict['ball_pos']
        solved = evaluate_pingpong_trajectory(self.contact_trajectory) == None
        
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
            ('solved', np.array([[solved]])),
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
        elif evaluate_pingpong_trajectory(self.contact_trajectory) in [0, 2, 3]:
            return 1
        return 0

    def _ball_label_to_obs(self, touching_body):
        # Function to convert touching body set to a binary observation vector
        # order follows the definition in enum class
        obs_vec = np.array([0, 0, 0, 0, 0, 0])

        for i in touching_body:
            if i == PingpongContactLabels.PADDLE:
                obs_vec[0] += 1
            elif i == PingpongContactLabels.OWN:
                obs_vec[1] += 1
            elif i == PingpongContactLabels.OPPONENT:
                obs_vec[2] += 1
            elif i == PingpongContactLabels.NET:
                obs_vec[3] += 1
            elif i == PingpongContactLabels.GROUND:
                obs_vec[4] += 1
            else: 
                obs_vec[5] += 1
        return obs_vec


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
        self.init_qvel[self.ball_dofadr : self.ball_dofadr + 3] = self.start_vel
        obs = super().reset(reset_qpos=self.init_qpos, reset_qvel=self.init_qvel,**kwargs)

        return obs

    def step(self, a, **kwargs):
        # We unnormalize robotic actuators of the "locomotion", muscle ones are handled in the parent implementation
        processed_controls = a.copy()
        if self.normalize_act:
            robotic_act_ind = self.sim.model.actuator_dyntype != mujoco.mjtDyn.mjDYN_MUSCLE
            processed_controls[robotic_act_ind] = (np.mean(self.sim.model.actuator_ctrlrange[robotic_act_ind], axis=-1)
                                                   + processed_controls[robotic_act_ind]
                                                   * (self.sim.model.actuator_ctrlrange[robotic_act_ind, 1]
                                                      - self.sim.model.actuator_ctrlrange[robotic_act_ind, 0]) / 2.0)
        return super().step(processed_controls, **kwargs)


class IdInfo:
    def __init__(self, model: mujoco.MjModel):
        self.ball_body_id = model.body("pingpong").id
        self.own_half_id = model.geom("coll_own_half").id
        self.own_half_id = model.geom("coll_own_half").id
        self.paddle_id = model.geom("ping_pong_paddle").id
        self.opponent_half_id = model.geom("coll_opponent_half").id
        self.ground_id = model.geom("ground").id
        self.net_id = model.geom("coll_net").id

        myo_bodies = [model.body(i).id for i in range(model.nbody)
                    if not model.body(i).name.startswith("ping")
                    and not model.body(i).name in ["pingpong"]]
        self.myo_body_range = (min(myo_bodies), max(myo_bodies))

        # TODO add locomotion joint ids

        self.myo_joint_range = np.concatenate([model.joint(i).qposadr for i in range(model.njnt)
                                            if not model.joint(i).name.startswith("ping")
                                            and not model.joint(i).name == "pingpong_freejoint"])

        self.myo_dof_range = np.concatenate([model.joint(i).dofadr for i in range(model.njnt)
                                            if not model.joint(i).name.startswith("ping")
                                            and not model.joint(i).name == "pingpong_freejoint"])


class PingpongContactLabels(enum.Enum):
    PADDLE = 0 # TODO: Remove collisions with myo
    OWN = 1
    OPPONENT = 2
    GROUND = 3
    NET = 4
    ENV = 5


class ContactTrajIssue(enum.Enum):
    OWN_HALF = 0
    MISS = 1
    NO_PADDLE = 2
    DOUBLE_TOUCH = 3


def get_touching_objects(model: mujoco.MjModel, data: mujoco.MjData, id_info: IdInfo):
    for con in data.contact:
        if model.geom(con.geom1).bodyid == id_info.ball_body_id:
            yield geom_id_to_label(con.geom2, id_info)
        elif model.geom(con.geom2).bodyid == id_info.ball_body_id:
            yield geom_id_to_label(con.geom1, id_info)


def geom_id_to_label(body_id, id_info: IdInfo):
    if body_id == id_info.paddle_id:
        return PingpongContactLabels.PADDLE
    elif body_id == id_info.own_half_id:
        return PingpongContactLabels.OWN
    elif body_id == id_info.opponent_half_id:
        return PingpongContactLabels.OPPONENT
    elif body_id == id_info.net_id:
        return PingpongContactLabels.NET
    elif body_id == id_info.ground_id:
        return PingpongContactLabels.GROUND
    else:
        return PingpongContactLabels.ENV


def evaluate_pingpong_trajectory(contact_trajectory: List[set]):

    has_hit_paddle = False
    has_bounced_from_paddle = False
    for s in contact_trajectory:
        if PingpongContactLabels.PADDLE not in s and has_hit_paddle:
            has_bounced_from_paddle = True
        if PingpongContactLabels.PADDLE in s and has_bounced_from_paddle:
            return ContactTrajIssue.DOUBLE_TOUCH
        if PingpongContactLabels.PADDLE in s:
            has_hit_paddle = True
        if PingpongContactLabels.OWN in s:
            return ContactTrajIssue.OWN_HALF
        if PingpongContactLabels.OPPONENT in s:
            if has_hit_paddle:
                return None
            else:
                return ContactTrajIssue.NO_PADDLE

    return ContactTrajIssue.MISS


