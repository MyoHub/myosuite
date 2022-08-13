""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

import collections
import enum
import gym
import numpy as np

from myosuite.envs.myo.base_v0 import BaseV0

# Define the task enum
class Task(enum.Enum):
    BAODING_CW = 1
    BAODING_CCW = 2
# Choose task
WHICH_TASK = Task.BAODING_CCW

class BaodingEnvV1(BaseV0):

    DEFAULT_OBS_KEYS = [
        'hand_pos',
        'object1_pos', 'object1_velp',
        'object2_pos', 'object2_velp',
        'target1_pos', 'target2_pos',
        'target1_err', 'target2_err',
        ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
       'pos_dist_1':5.0,
       'pos_dist_2':5.0,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        # Two step construction (init+setup) is required for pickling to work correctly.
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed)
        self._setup(**kwargs)

    def _setup(self,
            frame_skip:int=10,
            drop_th = 1.25,             # drop height threshold
            proximity_th = 0.015,       # object-target proximity threshold
            goal_time_period = (5, 5),  # target rotation time period
            goal_xrange = (0.025, 0.025),  # target rotation: x radius (0.03)
            goal_yrange = (0.028, 0.028),  # target rotation: x radius (0.02 * 1.5 * 1.2)
            obs_keys:list = DEFAULT_OBS_KEYS,
            weighted_reward_keys:list = DEFAULT_RWD_KEYS_AND_WEIGHTS,
            **kwargs,
        ):

        # user parameters
        self.which_task = Task(WHICH_TASK)
        self.drop_th = drop_th
        self.proximity_th = proximity_th
        self.goal_time_period = goal_time_period
        self.goal_xrange = goal_xrange
        self.goal_yrange = goal_yrange

        # balls start at these angles
        #   1= yellow = top right
        #   2= pink = bottom left
        self.ball_1_starting_angle = 3.*np.pi/4.0
        self.ball_2_starting_angle = -1.*np.pi/4.0

        # init desired trajectory, for rotations
        self.center_pos = [-.0125, -.07] # [-.0020, -.0522]
        self.x_radius=self.np_random.uniform(low=self.goal_xrange[0], high=self.goal_xrange[1])
        self.y_radius=self.np_random.uniform(low=self.goal_yrange[0], high=self.goal_yrange[1])

        self.counter=0
        self.goal = self.create_goal_trajectory(time_step=frame_skip*self.sim.model.opt.timestep, time_period=6)

        # init target and body sites
        self.object1_sid = self.sim.model.site_name2id('ball1_site')
        self.object2_sid = self.sim.model.site_name2id('ball2_site')
        self.object1_gid = self.sim.model.geom_name2id('ball1')
        self.object2_gid = self.sim.model.geom_name2id('ball2')
        self.target1_sid = self.sim.model.site_name2id('target1_site')
        self.target2_sid = self.sim.model.site_name2id('target2_site')
        self.sim.model.site_group[self.target1_sid] = 2
        self.sim.model.site_group[self.target2_sid] = 2

        super()._setup(obs_keys=obs_keys,
                    weighted_reward_keys=weighted_reward_keys,
                    frame_skip=frame_skip,
                    **kwargs,
                )

        # reset position
        self.init_qpos[:-14] *= 0 # Use fully open as init pos
        self.init_qpos[0] = -1.57 # Palm up

    def step(self, a):
        if self.which_task in [Task.BAODING_CW, Task.BAODING_CCW]:
            desired_angle_wrt_palm = self.goal[self.counter].copy()
            desired_angle_wrt_palm[0] = desired_angle_wrt_palm[0] + self.ball_1_starting_angle
            desired_angle_wrt_palm[1] = desired_angle_wrt_palm[1] + self.ball_2_starting_angle

            desired_positions_wrt_palm = [0,0,0,0]
            desired_positions_wrt_palm[0] = self.x_radius*np.cos(desired_angle_wrt_palm[0]) + self.center_pos[0]
            desired_positions_wrt_palm[1] = self.y_radius*np.sin(desired_angle_wrt_palm[0]) + self.center_pos[1]
            desired_positions_wrt_palm[2] = self.x_radius*np.cos(desired_angle_wrt_palm[1]) + self.center_pos[0]
            desired_positions_wrt_palm[3] = self.y_radius*np.sin(desired_angle_wrt_palm[1]) + self.center_pos[1]

            # update both sims with desired targets
            for sim in [self.sim, self.sim_obsd]:
                sim.model.site_pos[self.target1_sid, 0] = desired_positions_wrt_palm[0]
                sim.model.site_pos[self.target1_sid, 1] = desired_positions_wrt_palm[1]
                sim.model.site_pos[self.target2_sid, 0] = desired_positions_wrt_palm[2]
                sim.model.site_pos[self.target2_sid, 1] = desired_positions_wrt_palm[3]
                # move upward, to be seen
                # sim.model.site_pos[self.target1_sid, 2] = -0.037
                # sim.model.site_pos[self.target2_sid, 2] = -0.037
        self.counter +=1
        return super().step(a)

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['t'] = np.array([sim.data.time])
        obs_dict['hand_pos'] = sim.data.qpos[:-14].copy() # 7*2 for ball's free joint, rest should be hand

        # object positions
        obs_dict['object1_pos'] = sim.data.site_xpos[self.object1_sid].copy()
        obs_dict['object2_pos'] = sim.data.site_xpos[self.object2_sid].copy()

        # object translational velocities
        obs_dict['object1_velp'] = sim.data.qvel[-12:-9].copy()*self.dt
        obs_dict['object2_velp'] = sim.data.qvel[-6:-3].copy()*self.dt

        # site locations in world frame, populated after the step/forward call
        obs_dict['target1_pos'] = sim.data.site_xpos[self.target1_sid].copy()
        obs_dict['target2_pos'] = sim.data.site_xpos[self.target2_sid].copy()

        # object position error
        obs_dict['target1_err'] = obs_dict['target1_pos'] - obs_dict['object1_pos']
        obs_dict['target2_err'] = obs_dict['target2_pos'] - obs_dict['object2_pos']

        # muscle activations
        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()

        return obs_dict

    def get_reward_dict(self, obs_dict):
        # tracking error
        target1_dist = np.linalg.norm(obs_dict['target1_err'], axis=-1)
        target2_dist = np.linalg.norm(obs_dict['target2_err'], axis=-1)
        target_dist = target1_dist+target2_dist
        act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0

        # detect fall
        object1_pos = obs_dict['object1_pos'][:,:,2] if obs_dict['object1_pos'].ndim==3 else obs_dict['object1_pos'][2]
        object2_pos = obs_dict['object2_pos'][:,:,2] if obs_dict['object2_pos'].ndim==3 else obs_dict['object2_pos'][2]
        is_fall_1 = object1_pos < self.drop_th
        is_fall_2 = object2_pos < self.drop_th
        is_fall = np.logical_or(is_fall_1, is_fall_2) # keep both balls up

        rwd_dict = collections.OrderedDict((
            # Perform reward tuning here --
            # Update Optional Keys section below
            # Update reward keys (DEFAULT_RWD_KEYS_AND_WEIGHTS) accordingly to update final rewards
            # Examples: Env comes pre-packaged with two keys pos_dist_1 and pos_dist_2

            # Optional Keys
            ('pos_dist_1',      -1.*target1_dist),
            ('pos_dist_2',      -1.*target2_dist),
            # Must keys
            ('act_reg',         -1.*act_mag),
            ('sparse',          -target_dist),
            ('solved',          (target1_dist < self.proximity_th)*(target2_dist < self.proximity_th)*(~is_fall)),
            ('done',            is_fall),
        ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)

        # Sucess Indicator
        self.sim.model.geom_rgba[self.object1_gid, :2] = np.array([1, 1]) if target1_dist < self.proximity_th else np.array([0.5, 0.5])
        self.sim.model.geom_rgba[self.object2_gid, :2] = np.array([0.9, .7]) if target1_dist < self.proximity_th else np.array([0.5, 0.5])

        return rwd_dict


    def get_metrics(self, paths):
        """
        Evaluate paths and report metrics
        """
        # average sucess over entire env horizon
        score = np.mean([np.sum(p['env_infos']['rwd_dict']['solved'])/self.horizon for p in paths])
        # average activations over entire trajectory (can be shorter than horizon, if done) realized
        effort = -1.0*np.mean([np.mean(p['env_infos']['rwd_dict']['act_reg']) for p in paths])

        metrics = {
            'score': score,
            'effort': effort,
            }
        return metrics


    def reset(self, reset_pose=None, reset_vel=None, reset_goal=None, time_period=None):
        # reset counters
        self.counter=0
        self.x_radius=self.np_random.uniform(low=self.goal_xrange[0], high=self.goal_xrange[1])
        self.y_radius=self.np_random.uniform(low=self.goal_yrange[0], high=self.goal_yrange[1])

        # reset goal
        if time_period == None:
            time_period = self.np_random.uniform(low=self.goal_time_period[0], high=self.goal_time_period[1])
        self.goal = self.create_goal_trajectory(time_step=self.dt, time_period=time_period) if reset_goal is None else reset_goal.copy()

        # reset scene
        obs = super().reset(reset_qpos=reset_pose, reset_qvel=reset_vel)
        return obs

    def create_goal_trajectory(self, time_step=.1, time_period=6):
        len_of_goals = 1000 # assumes that its greator than env horizon

        goal_traj = []
        if self.which_task==Task.BAODING_CW:
            sign = -1
        if self.which_task==Task.BAODING_CCW:
            sign = 1

        # Target updates in continuous rotation
        t = 0
        while t < len_of_goals:
            angle_before_shift = sign * 2 * np.pi * (t * time_step / time_period)
            goal_traj.append(np.array([angle_before_shift, angle_before_shift]))
            t += 1

        goal_traj = np.array(goal_traj)
        return goal_traj