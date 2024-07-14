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
from myosuite.envs.heightfields import ChaseTagField


class Task(Enum):
    CHASE = 0
    EVADE = 1


class ChallengeOpponent:
    """
    Training Opponent for the Locomotion Track of the MyoChallenge 2023.
    Contains several different policies. For the final evaluation, an additional
    non-disclosed policy will be used.
    """
    def __init__(self,
                 sim,
                 rng,
                 probabilities: Tuple[float],
                 min_spawn_distance: float,
                 chase_vel_range: Tuple[float],
                 random_vel_range: Tuple[float],
                 dt=0.01,
        ):
        """
        Initialize the opponent class.
        :param sim: Mujoco sim object.
        :param rng: np_random generator.
        :param probabilities: Probabilities for the different policies, (static_stationary, stationary, random).
        :param min_spawn_distance: Minimum distance for opponent to spawn from the model.
        :param chase_vel_range: Range of velocities for the chase policy. Randomly drawn.
        :param random_vel_range: Range of velocities for the random policy. Clipped.
        :param dt: Simulation timestep.
        """
        self.dt = dt
        self.sim = sim
        self.opponent_probabilities = probabilities
        self.min_spawn_distance = min_spawn_distance
        self.chase_vel_range = chase_vel_range
        self.random_vel_range = random_vel_range
        self.reset_opponent(rng=rng)

    def reset_noise_process(self):
        self.noise_process = pink.ColoredNoiseProcess(beta=2, size=(2, 2000), scale=10, rng=self.rng)

    def get_opponent_pose(self):
        """
        Get opponent Pose
        :return: The  pose.
        :rtype: list -> [x, y, angle]
        """
        angle = quat2euler(self.sim.data.mocap_quat[0, :])[-1]
        return np.concatenate([self.sim.data.mocap_pos[0, :2], [angle]])

    def set_opponent_pose(self, pose: list):
        """
        Set opponent pose directly.
        :param pose: Pose of the opponent.
        :type pose: list -> [x, y, angle]
        """
        self.sim.data.mocap_pos[0, :2] = pose[:2]
        self.sim.data.mocap_quat[0, :] = euler2quat([0, 0, pose[-1]])

    def move_opponent(self, vel: list):
        """
        This is the main function that moves the opponent and should always be used if you want to physically move
        it by giving it a velocity. If you want to teleport it to a new position, use `set_opponent_pose`.
        :param vel: Linear and rotational velocities in [-1, 1]. Moves opponent
                  forwards or backwards and turns it. vel[0] is assumed to be linear vel and
                  vel[1] is assumed to be rotational vel
        :type vel: list -> [lin_vel, rot_vel].
        """
        self.opponent_vel = vel
        assert len(vel) == 2
        vel[0] = np.abs(vel[0])
        vel = np.clip(vel, -2, 2)
        pose = self.get_opponent_pose()
        x_vel = vel[0] * np.cos(pose[-1]+0.5*np.pi)
        y_vel = vel[0] * np.sin(pose[-1] +0.5*np.pi)
        pose[0] -= self.dt * x_vel
        pose[1] -= self.dt * y_vel
        pose[2] += self.dt * vel[1]
        pose[:2] = np.clip(pose[:2], -5.5, 5.5)
        self.set_opponent_pose(pose)

    def random_movement(self):
        """
        This moves the opponent randomly in a correlated
        pattern.
        """
        return np.clip(self.noise_process.sample(), self.random_vel_range[0], self.random_vel_range[1])

    def sample_opponent_policy(self):
        """
        Takes in three probabilities and returns the policies with the given frequency.
        """
        rand_num = self.rng.uniform()
        if rand_num < self.opponent_probabilities[0]:
            self.opponent_policy = 'static_stationary'
        elif rand_num < self.opponent_probabilities[0] + self.opponent_probabilities[1]:
            self.opponent_policy = 'stationary'
        elif rand_num < self.opponent_probabilities[0] + self.opponent_probabilities[1] + self.opponent_probabilities[2]:
            self.opponent_policy = 'random'

    def update_opponent_state(self):
        """
        This function executes an opponent step with
        one of the control policies.
        """
        if self.opponent_policy == 'stationary' or self.opponent_policy == 'static_stationary':
            opponent_vel = np.zeros(2,)

        elif self.opponent_policy == 'random':
            opponent_vel = self.random_movement()

        elif self.opponent_policy == 'chase_player':
            opponent_vel = self.chase_player()
        else:
            raise NotImplementedError(f"This opponent policy doesn't exist. Chose: static_stationary, stationary or random. Policy was: {self.opponent_policy}")
        self.move_opponent(opponent_vel)

    def reset_opponent(self, player_task='CHASE', rng=None):
        """
        This function should initially place the opponent on a random position with a
        random orientation with a minimum radius to the model.
        :task: Task for the PLAYER, I.e. 'CHASE' means that the player has to chase and the opponent has to evade.
        :rng: np_random generator
        """
        if rng is not None:
            self.rng = rng
            self.reset_noise_process()

        self.opponent_vel = np.zeros((2,))
        if player_task == 'CHASE':
            self.sample_opponent_policy()
        elif player_task == 'EVADE':
            self.opponent_policy = 'chase_player'
        else:
            raise NotImplementedError

        dist = 0
        while dist < self.min_spawn_distance:
            pose = [self.rng.uniform(-5, 5), self.rng.uniform(-5, 5), self.rng.uniform(- 2 * np.pi, 2 * np.pi)]
            dist = np.linalg.norm(pose[:2] - self.sim.data.body('root').xpos[:2])
        if self.opponent_policy == "static_stationary":
            pose[:] = [0, -5, 0]
        self.set_opponent_pose(pose)
        self.opponent_vel[:] = 0.0

        # Randomize opponent forward velocity
        self.chase_velocity = self.rng.uniform(self.chase_vel_range[0], self.chase_vel_range[1])

    def chase_player(self):
        """
        This moves the opponent randomly in a correlated
        pattern.
        """
        pose = self.get_opponent_pose()
        vec = pose[:2]
        pel = self.sim.data.body('pelvis').xpos[:2]
        theta = pose[-1]
        new_vec = np.array([np.cos(theta), np.sin(theta)])
        new_vec2 = pel - vec
        vel = np.dot(new_vec, new_vec2)
        return np.array([self.chase_velocity, vel])


class RepellerChallengeOpponent(ChallengeOpponent):
    # Repeller parameters
    DIST_INFLUENCE = 3.5 # Distance of influence by the repeller
    ETA = 20.0 # Scaling factor
    MIN_SPAWN_DIST = 1.5
    BOUND_RESOLUTIONS = [-8.7, 8.7, 25]

    def __init__(self,
                 sim,
                 rng,
                 probabilities: Tuple[float],
                 min_spawn_distance: float,
                 chase_vel_range: Tuple[float],
                 random_vel_range: Tuple[float],
                 repeller_vel_range: Tuple[float],
                 dt=0.01,
        ):
        """
        Initialize the opponent class. This class additionally contains a repeller policy which always runs away from the
        agent.
        :param sim: Mujoco sim object.
        :param rng: np_random generator.
        :param probabilities: Probabilities for the different policies, (static_stationary, stationary, random, repeller).
        :param min_spawn_distance: Minimum distance for opponent to spawn from the model.
        :param chase_vel_range: Range of velocities for the chase policy. Randomly drawn.
        :param random_vel_range: Range of velocities for the random policy. Clipped.
        :param dt: Simulation timestep.
        """
        self.dt = dt
        self.sim = sim
        self.rng = rng
        self.opponent_probabilities = probabilities

        self.min_spawn_distance = min_spawn_distance
        self.noise_process = pink.ColoredNoiseProcess(beta=2, size=(2, 2000), scale=10, rng=rng)
        self.chase_vel_range = chase_vel_range
        self.random_vel_range = random_vel_range
        self.repeller_vel_range = repeller_vel_range
        self.reset_opponent()

    def get_agent_pos(self):
        """
        Get agent Pose
        :param pose: Pose of the agent, measured from the pelvis.
        :type pose: array -> [x, y]
        """
        return self.sim.data.body('pelvis').xpos[:2]

    def get_wall_pos(self):
        """
        Get location of quad boundaries.
        :param pose: Pose of points along quad boundaries.
        :type pose: array -> [x, y]
        """
        bound_resolution = np.linspace(self.BOUND_RESOLUTIONS[0], self.BOUND_RESOLUTIONS[1], self.BOUND_RESOLUTIONS[2])
        right_left_bounds = np.vstack( (np.array([[8.7,x] for x in bound_resolution]),
                                        np.array([[-8.7,x] for x in bound_resolution])) )
        all_bounds = np.vstack( (right_left_bounds, right_left_bounds[:,[1,0]]) )

        return all_bounds

    def get_repellers(self):
        """
        Get location of all repellers.
        :param pose: Pose of all repellers
        :type pose: array -> [x, y]
        """
        agent_pos = self.get_agent_pos()
        wall_pos = self.get_wall_pos()

        obstacle_list = np.vstack( (agent_pos, wall_pos) )
        return obstacle_list

    def repeller_stochastic(self):
        """
        Returns the linear velocity for the opponent
        :param pose: Pose of points of all repellers
        :type pose: array -> [x, y, rotation]
        """
        obstacle_pos = self.get_repellers()
        opponent_pos = self.get_opponent_pose().copy()

        # Calculate over all the workspace
        distance = np.array([np.linalg.norm(diff) for diff in (obstacle_pos - opponent_pos[0:2])])

        # Check if any obstacles are around
        dist_idx = np.where(distance < self.DIST_INFLUENCE)[0]

        # Take a random step if no repellers are close by, making it a non-stationary target
        if len(dist_idx) == 0:
            lin, rot = self.noise_process.sample()
            escape_linear = np.clip(lin, self.repeller_vel_range[0], self.repeller_vel_range[1])
            escape_ang_rot = self._calc_angular_vel(opponent_pos[2], rot)
            return np.hstack((escape_linear, escape_ang_rot))

        repel_COM = np.mean(obstacle_pos[dist_idx,:], axis=0)
        # Use repeller force as linear velocity to escape
        repel_force = 0.5 * self.ETA * ( 1/np.maximum(distance[dist_idx], 0.00001) - 1/self.DIST_INFLUENCE )**2
        escape_linear = np.clip(np.mean(repel_force), self.repeller_vel_range[0], self.repeller_vel_range[1])
        escape_xpos = opponent_pos[0:2] - repel_COM

        equil_idx = np.where(np.abs(escape_xpos) <= 0.1 )[0]
        if len(equil_idx) != 0:
            for idx in equil_idx:
                escape_xpos[idx] = -1*np.sign(escape_xpos[idx]) * self.rng.uniform(low=0.3, high=0.9)

        escape_direction = np.arctan2(escape_xpos[1], escape_xpos[0]) # Direction
        escape_direction = escape_direction + 1.57 # Account for rotation in world frame

        # Determines turning direction
        escape_ang_rot = self._calc_angular_vel(opponent_pos[2], escape_direction)

        return np.hstack((escape_linear, escape_ang_rot))

    def _calc_angular_vel(self, current_pos, desired_pos):
        # Checking for sign of the current position and escape position to prevent inefficient turning
        # E.g. 3.14 and -3.14 are pointing in the same direction, so a simple substraction of facing direction will make the opponent turn a lot

        # Bring the current pos and desired pos to be between 0 to 2pi
        if current_pos > (2*np.pi):
            while current_pos > (2*np.pi):
                current_pos = current_pos - (2*np.pi)
        elif np.sign(current_pos) < 0:
            while np.sign(current_pos) < 0:
                current_pos = current_pos + (2*np.pi)

        if desired_pos > (2*np.pi):
            while desired_pos > (2*np.pi):
                desired_pos = desired_pos - (2*np.pi)
        elif np.sign(desired_pos) < 0:
            while np.sign(desired_pos) < 0:
                desired_pos = desired_pos + (2*np.pi)

        direction_clock = np.abs(0 - current_pos) + (2*np.pi - desired_pos) # Clockwise rotation
        direction_anticlock = (2*np.pi - current_pos) + (0 + desired_pos) # Anticlockwise rotation

        if direction_clock < direction_anticlock:
            return 1
        else:
            return -1

    def repeller_policy(self):
        """
        This uses the repeller policy to move the opponent.
        """
        return self.repeller_stochastic()

    def sample_opponent_policy(self):
        """
        Takes in three probabilities and returns the policies with the given frequency.
        """
        rand_num = self.rng.uniform()
        if rand_num < self.opponent_probabilities[0]:
            self.opponent_policy = 'static_stationary'
        elif rand_num < self.opponent_probabilities[0] + self.opponent_probabilities[1]:
            self.opponent_policy = 'stationary'
        elif rand_num < self.opponent_probabilities[0] + self.opponent_probabilities[1] + self.opponent_probabilities[2]:
            self.opponent_policy = 'random'
        else:
            self.opponent_policy = 'repeller'

    def update_opponent_state(self):
        """
        This function executes an opponent step with
        one of the control policies.
        """
        if self.opponent_policy == 'stationary' or self.opponent_policy == 'static_stationary':
            opponent_vel = np.zeros(2,)

        elif self.opponent_policy == 'random':
            opponent_vel = self.random_movement()

        elif self.opponent_policy == 'repeller':
            opponent_vel = self.repeller_policy()

        elif self.opponent_policy == 'chase_player':
            opponent_vel = self.chase_player()
        else:
            raise NotImplementedError(f"This opponent policy doesn't exist. Chose: static_stationary, stationary or random. Policy was: {self.opponent_policy}")
        self.move_opponent(opponent_vel)


class ChaseTagEnvV0(WalkEnvV0):

    DEFAULT_OBS_KEYS = [
        'internal_qpos',
        'internal_qvel',
        'grf',
        'torso_angle',
        'opponent_pose',
        'opponent_vel',
        'model_root_pos',
        'model_root_vel',
        'muscle_length',
        'muscle_velocity',
        'muscle_force',
    ]

    # You can change reward weights here
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "distance": -0.1,
        "lose": -1000,
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
               win_distance=0.5,
               min_spawn_distance=2,
               task_choice='CHASE',
               terrain='FLAT',
               hills_range=(0,0),
               rough_range=(0,0),
               relief_range=(0,0),
               repeller_opponent=False,
               chase_vel_range=(1.0, 1.0),
               random_vel_range=(1.0, 1.0),
               repeller_vel_range=(1.0, 1.0),
               opponent_probabilities=(0.1, 0.45, 0.45),
               **kwargs,
               ):


        self._setup_convenience_vars()
        # check that this works everywhere and is efficient.
        self.heightfield = ChaseTagField(sim=self.sim,
                                       rng=self.np_random,
                                       rough_range=rough_range,
                                       hills_range=hills_range,
                                       relief_range=relief_range) if terrain != 'FLAT' else None
        self.reset_type = reset_type
        self.task_choice = task_choice
        self.terrain = terrain
        self.maxTime = 20
        if repeller_opponent:
            self.opponent = RepellerChallengeOpponent(sim=self.sim,
                                                      rng=self.np_random,
                                                      probabilities=opponent_probabilities,
                                                      min_spawn_distance=min_spawn_distance,
                                                      chase_vel_range=chase_vel_range,
                                                      random_vel_range=random_vel_range,
                                                      repeller_vel_range=repeller_vel_range)
        else:
            self.opponent = ChallengeOpponent(sim=self.sim,
                                              rng=self.np_random,
                                              probabilities=opponent_probabilities,
                                              min_spawn_distance=min_spawn_distance,
                                              chase_vel_range=chase_vel_range,
                                              random_vel_range=random_vel_range)

        self.win_distance = win_distance
        self.grf_sensor_names = ['r_foot', 'r_toes', 'l_foot', 'l_toes']
        self.success_indicator_sid = self.sim.model.site_name2id("opponent_indicator")
        self.current_task = Task.CHASE
        self.repeller_opponent = repeller_opponent
        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       reset_type=reset_type,
                       **kwargs
                       )
        self.init_qpos[:] = self.sim.model.key_qpos[0]
        self.init_qvel[:] = 0.0
        self.startFlag = True
        self.assert_settings()
        self.opponent.dt = self.sim.model.opt.timestep * self.frame_skip



    def assert_settings(self):
        # chase always positive
        assert self.opponent.chase_vel_range[0] >= 0 and self.opponent.chase_vel_range[1] > 0, f"Chase velocity range should be positive. {self.opponent.chase_vel_range}"
        # others assert that range end is bigger than range start
        assert self.opponent.chase_vel_range[0] <= self.opponent.chase_vel_range[1], f"Chase velocity range is not valid. {self.opponent.chase_vel_range}"
        assert self.opponent.random_vel_range[0] <= self.opponent.random_vel_range[1], f"Random movement velocity range is not valid {self.opponent.random_vel_range}"
        if hasattr(self.opponent, 'repeller_vel_range'):
            assert self.opponent.repeller_vel_range[0] <= self.opponent.repeller_vel_range[1], f"Repeller velocity range is not valid {self.opponent.repeller_vel_range}"
        if self.repeller_opponent == True:
            assert len(self.opponent.opponent_probabilities) == 4, "Repeller opponent requires 4 probabilities"
        else:
            assert len(self.opponent.opponent_probabilities) == 3, "Standard opponent requires 3 probabilities"
        for x in self.opponent.opponent_probabilities:
            assert 0 <= x <= 1, "Probabilities should be between 0 and 1"

    def get_obs_dict(self, sim):
        obs_dict = {}

        # Time
        obs_dict['time'] = np.array([sim.data.time])

        # proprioception
        obs_dict['internal_qpos'] = sim.data.qpos[7:35].copy()
        obs_dict['internal_qvel'] = sim.data.qvel[6:34].copy() * self.dt
        obs_dict['grf'] = self._get_grf().copy()
        obs_dict['torso_angle'] = self.sim.data.body('pelvis').xquat.copy()

        obs_dict['muscle_length'] = self.muscle_lengths()
        obs_dict['muscle_velocity'] = self.muscle_velocities()
        obs_dict['muscle_force'] = self.muscle_forces()

        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()

        # exteroception
        obs_dict['opponent_pose'] = self.opponent.get_opponent_pose()[:].copy()
        obs_dict['opponent_vel'] = self.opponent.opponent_vel[:].copy()
        obs_dict['model_root_pos'] = sim.data.qpos[:2].copy()
        obs_dict['model_root_vel'] = sim.data.qvel[:2].copy()

        # active task
        obs_dict['task'] = np.array(self.current_task.value, ndmin=2, dtype=np.int16)
        # heightfield view of 10x10 grid of points around agent. Reshape to (10, 10) for visual inspection
        if not self.heightfield is None:
            obs_dict['hfield'] = self.heightfield.get_heightmap_obs()

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
        win_cdt = self._win_condition()
        lose_cdt = self._lose_condition()
        if self.current_task.name == 'CHASE':
            score = self._get_score(float(self.obs_dict['time'])) if win_cdt else 0
            self.obs_dict['time'] = self.maxTime if lose_cdt else self.obs_dict['time']
        elif self.current_task.name == 'EVADE':
            score = self._get_score(float(self.obs_dict['time'])) if (win_cdt or lose_cdt) else 0
        # ----------------------

        # Example reward, you should change this!
        distance = np.linalg.norm(obs_dict['model_root_pos'][...,:2] - obs_dict['opponent_pose'][...,:2])

        rwd_dict = collections.OrderedDict((
            # Perform reward tuning here --
            # Update Optional Keys section below
            # Update reward keys (DEFAULT_RWD_KEYS_AND_WEIGHTS) accordingly to update final rewards

            # Example: simple distance function

                # Optional Keys
                ('act_reg', act_mag),
                ('distance', distance),
                ('lose', lose_cdt),
                # Must keys
                ('sparse',  score),
                ('solved',  win_cdt),
                ('done',  self._get_done()),
            ))
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)

        # Success Indicator
        self.sim.model.site_rgba[self.success_indicator_sid, :] = np.array([0, 2, 0, 0.2]) if rwd_dict['solved'] else np.array([2, 0, 0, 0])
        return rwd_dict

    def get_metrics(self, paths):
        """
        Evaluate paths and report metrics
        """
        # average sucess over entire env horizon
        score = np.mean([np.sum(p['env_infos']['rwd_dict']['sparse']) for p in paths])
        points = np.mean([np.sum(p['env_infos']['rwd_dict']['solved']) for p in paths])
        times = np.mean([np.round(p['env_infos']['obs_dict']['time'][-1],2) for p in paths])
        # average activations over entire trajectory (can be shorter than horizon, if done) realized

        metrics = {
            'score': score,
            'points': points,
            'times': times,
            }
        return metrics

    def step(self, *args, **kwargs):
        self.opponent.update_opponent_state()
        results = super().step(*args, **kwargs)
        return results

    def reset(self, **kwargs):
        # randomized terrain types
        self._maybe_sample_terrain()
        # randomized tasks
        self._sample_task()
        # randomized initial state
        qpos, qvel = self._get_reset_state()
        self._maybe_flatten_agent_patch(qpos)
        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = super(WalkEnvV0, self).reset(reset_qpos=qpos, reset_qvel=qvel, **kwargs)
        self.opponent.reset_opponent(player_task=self.current_task.name, rng=self.np_random)
        self.sim.forward()
        return obs

    def _maybe_flatten_agent_patch(self, qpos):
        """
        Ensure that initial state patch is flat.
        """
        if self.heightfield is not None:
            self.heightfield.flatten_agent_patch(qpos)
            if hasattr(self.sim, 'renderer') and not self.sim.renderer._window is None:
                self.sim.renderer._window.update_hfield(0)

    def _sample_task(self):
        if self.task_choice == 'random':
            self.current_task = self.np_random.choice(Task)
        else:
            self.current_task = getattr(Task, self.task_choice)

    def _maybe_sample_terrain(self):
        """
        Sample a new terrain if the terrain type asks for it.
        """
        if not self.heightfield is None:
            self.heightfield.sample(self.np_random)
            self.sim.model.geom_rgba[self.sim.model.geom_name2id('terrain')][-1] = 1.0
            self.sim.model.geom_pos[self.sim.model.geom_name2id('terrain')] = np.array([0, 0, 0])
        else:
            # move heightfield down if not used
            self.sim.model.geom_rgba[self.sim.model.geom_name2id('terrain')][-1] = 0.0
            self.sim.model.geom_pos[self.sim.model.geom_name2id('terrain')] = np.array([0, 0, -10])

    def _randomize_position_orientation(self, qpos, qvel):
        qpos[:2]  = self.np_random.uniform(-5, 5, size=(2,))
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
        if self.heightfield is not None:
                map_i, map_j = self.heightfield.cart2map(qpos[:2])
                hfield_val = self.heightfield.hfield.data[map_i, map_j]
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
        qpos[:] = qpos[:] + self.np_random.normal(0, 0.02, size=qpos.shape)
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
        if self.current_task.name == 'CHASE':
            return self._chase_win_condition()
        elif self.current_task.name == 'EVADE':
            return self._evade_win_condition()
        else:
            raise NotImplementedError

    def _lose_condition(self):
        # falling on knees is always termination
        if self._get_fallen_condition() and self.current_task.name == 'CHASE':
            return 1
        if self.current_task.name == 'CHASE':
            return self._chase_lose_condition()
        elif self.current_task.name == 'EVADE':
            return self._evade_lose_condition()
        else:
            raise NotImplementedError

    def _chase_lose_condition(self):
        root_pos = self.sim.data.body('pelvis').xpos[:2]
        # didnt manage to tag
        if self.obs_dict['time'] >= self.maxTime:
            return 1
        # out-of-bounds
        if np.abs(root_pos[0]) > 6.5 or np.abs(root_pos[1]) > 6.5:
            return 1
        return 0

    def _evade_lose_condition(self):
        root_pos = self.sim.data.body('pelvis').xpos[:2]
        opp_pos = self.obs_dict['opponent_pose'][..., :2]

        # got caught
        if np.linalg.norm(root_pos - opp_pos) <= self.win_distance and self.startFlag:
            return 1
        # out-of-bounds
        if np.abs(root_pos[0]) > 6.5 or np.abs(root_pos[1]) > 6.5:
            return 1
        return 0

    def _chase_win_condition(self):
        root_pos = self.sim.data.body('pelvis').xpos[:2]
        opp_pos = self.obs_dict['opponent_pose'][..., :2]
        if np.linalg.norm(root_pos - opp_pos) <= self.win_distance and self.startFlag:
            return 1
        return 0

    def _evade_win_condition(self):
        # evade long enough
        if self.obs_dict['time'] >= self.maxTime:
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
        time = np.round(time, 2)
        if self.current_task.name == 'CHASE':
            return 1 - (time / self.maxTime)
        elif self.current_task.name == 'EVADE':
            return time / self.maxTime
        else:
            raise NotImplementedError

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
