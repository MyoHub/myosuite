""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/mj_envs
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import gym
import numpy as np
import os
import time as timer

from myosuite.utils.obj_vec_dict import ObsVecDict
from myosuite.utils import tensor_utils
from myosuite.robot.robot import Robot
from os import path
import skvideo.io

# TODO
# remove rwd_mode
# convet obs_keys to obs_keys_wt
# Seed the random number generator in the __init__
# Pass model_path and model_obsd_path to the __init__ so the use has a choice to make partially observed envs

try:
    import mujoco_py
    from mujoco_py import load_model_from_path, MjSim, MjViewer, load_model_from_xml, ignore_mujoco_warnings
except ImportError as e:
    raise gym.error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

def get_sim(model_path:str=None, model_xmlstr=None):
    """
    Get sim using model_path or model_xmlstr.
    """
    if model_path:
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        model = load_model_from_path(fullpath)
    elif model_xmlstr:
        model = load_model_from_xml(model_xmlstr)
    else:
        raise TypeError("Both model_path and model_xmlstr can't be None")

    return MjSim(model)

class MujocoEnv(gym.Env, gym.utils.EzPickle, ObsVecDict):
    """
    Superclass for all MuJoCo environments.
    """

    def __init__(self, model_path):
        # Get a random number generator incase its needed in pre_setup phase
        self.input_seed = None
        self.seed(0)

        # sims
        self.sim = get_sim(model_path)
        self.sim_obsd = get_sim(model_path)
        ObsVecDict.__init__(self)

    def _setup(self,
               obs_keys,
               weighted_reward_keys,
               reward_mode = "dense",
               frame_skip = 1,
               normalize_act = True,
               obs_range = (-10, 10),
               seed = None,
               rwd_viz = False,
               device_id = 0, # device id for rendering
               **kwargs,
        ):

        if self.sim is None or self.sim_obsd is None:
            raise TypeError("sim and sim_obsd must be instantiated for setup to run")

        # seed the random number generator
        self.input_seed = None
        self.seed(seed)
        self.mujoco_render_frames = False
        self.device_id = device_id
        self.rwd_viz = rwd_viz

        # resolve robot config
        self.robot = Robot(mj_sim=self.sim,
                           random_generator=self.np_random,
                           **kwargs)

        #resolve action space
        self.frame_skip = frame_skip
        self.normalize_act = normalize_act
        act_low = -np.ones(self.sim.model.nu) if self.normalize_act else self.sim.model.actuator_ctrlrange[:,0].copy()
        act_high = np.ones(self.sim.model.nu) if self.normalize_act else self.sim.model.actuator_ctrlrange[:,1].copy()
        self.action_space = gym.spaces.Box(act_low, act_high, dtype=np.float32)

        # resolve rewards
        self.rwd_dict = {}
        self.rwd_mode = reward_mode
        self.rwd_keys_wt = weighted_reward_keys

        # resolve obs
        self.obs_dict = {}
        self.obs_keys = obs_keys
        observation, _reward, done, _info = self.step(np.zeros(self.sim.model.nu))
        assert not done, "Check initialization. Simulation starts in a done state."
        self.obs_dim = observation.size
        self.observation_space = gym.spaces.Box(obs_range[0]*np.ones(self.obs_dim), obs_range[1]*np.ones(self.obs_dim), dtype=np.float32)

        # resolve initial state
        self.init_qvel = self.sim.data.qvel.ravel().copy()
        self.init_qpos = self.sim.data.qpos.ravel().copy() # has issues with initial jump during reset
        # self.init_qpos = np.mean(self.sim.model.actuator_ctrlrange, axis=1) if self.normalize_act else self.sim.data.qpos.ravel().copy() # has issues when nq!=nu
        # self.init_qpos[self.sim.model.jnt_dofadr] = np.mean(self.sim.model.jnt_range, axis=1) if self.normalize_act else self.sim.data.qpos.ravel().copy()
        if self.normalize_act:
            linear_jnt_qposids = self.sim.model.jnt_qposadr[self.sim.model.jnt_type>1] #hinge and slides
            linear_jnt_ids = self.sim.model.jnt_type>1
            self.init_qpos[linear_jnt_qposids] = np.mean(self.sim.model.jnt_range[linear_jnt_ids], axis=1)

        return

    def step(self, a):
        """
        Step the simulation forward (t => t+1)
        Uses robot interface to safely step the forward respecting pos/ vel limits
        """
        a = np.clip(a, self.action_space.low, self.action_space.high)
        self.last_ctrl = self.robot.step(ctrl_desired=a,
                                        ctrl_normalized=self.normalize_act,
                                        step_duration=self.dt,
                                        realTimeSim=self.mujoco_render_frames,
                                        render_cbk=self.mj_render if self.mujoco_render_frames else None)

        # observation
        obs = self.get_obs()

        # rewards
        self.expand_dims(self.obs_dict) # required for vectorized rewards calculations
        self.rwd_dict = self.get_reward_dict(self.obs_dict)
        self.squeeze_dims(self.rwd_dict)
        self.squeeze_dims(self.obs_dict)

        # finalize step
        env_info = self.get_env_infos()

        # returns obs(t+1), rew(t), done(t), info(t+1)
        return obs, env_info['rwd_'+self.rwd_mode], bool(env_info['done']), env_info


    def get_obs(self):
        """
        Get observations from the environemnt.
        Uses robot to get sensors, reconstructs the sim and recovers the sensors.
        """
        # get sensor data from robot
        sen = self.robot.get_sensors()

        # reconstruct (partially) observed-sim using (noisy) sensor data
        self.robot.sensor2sim(sen, self.sim_obsd)

        # get obs_dict using the observed information
        self.obs_dict = self.get_obs_dict(self.sim_obsd)

        # recoved observation vector from the obs_dict
        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs


    def get_visual_obs_dict(self, sim, device_id=None):
        """
        Recover visual observation dict corresponding to the 'rgba:cam_name:HxW' keys in obs_keys
        """
        if device_id is None:
            device_id = self.device_id

        visual_obs_dict = {}
        visual_obs_dict['t'] = np.array([self.sim.data.time])
        for key in self.obs_keys:
            if key.startswith('rgb'):
                cam = key.split(':')[1]
                height = int(key.split(':')[2])
                width = int(key.split(':')[3])
                img = self.render_camera_offscreen(
                                    height=height,
                                    width=width,
                                    cameras=[cam],
                                    device_id=device_id,
                                    sim=sim,
                                  )
                img = img.reshape(-1)
                visual_obs_dict.update({key:img})
        return visual_obs_dict


    # VIK??? Its getting called twice. Once in step and sampler calls it as well
    def get_env_infos(self):
        """
        Get information about the environment.
        - Essential keys are added below. Users can add more keys
        - Requires necessary keys (dense, sparse, solved, done) in rwd_dict to be populated
        - Note that entries belongs to different MDP steps
        """
        env_info = {
            'time': self.obs_dict['t'][()],             # MDP(t)
            'rwd_dense': self.rwd_dict['dense'][()],    # MDP(t-1)
            'rwd_sparse': self.rwd_dict['sparse'][()],  # MDP(t-1)
            'solved': self.rwd_dict['solved'][()],      # MDP(t-1)
            'done': self.rwd_dict['done'][()],          # MDP(t-1)
            'obs_dict': self.obs_dict,                  # MDP(t)
            'rwd_dict': self.rwd_dict,                  # MDP(t-1)
        }
        return env_info


    # Methods on paths =======================================================

    def compute_path_rewards(self, paths):
        """
        Compute vectorized rewards for paths and check for done conditions
        path has two keys: observations and actions
        path["observations"] : (num_traj, horizon, obs_dim)
        path["rewards"] should have shape (num_traj, horizon)
        """
        obs_dict = self.obsvec2obsdict(paths["observations"])
        rwd_dict = self.get_reward_dict(obs_dict)

        rewards = rwd_dict[self.rwd_mode]
        done = rwd_dict['done']
        # time align rewards. last step is redundant
        done[...,:-1] = done[...,1:]
        rewards[...,:-1] = rewards[...,1:]
        paths["done"] = done if done.shape[0] > 1 else done.ravel()
        paths["rewards"] = rewards if rewards.shape[0] > 1 else rewards.ravel()
        return paths


    def truncate_paths(self, paths):
        """
        truncate paths as per done condition
        """
        hor = paths[0]['rewards'].shape[0]
        for path in paths:
            if path['done'][-1] == False:
                path['terminated'] = False
                terminated_idx = hor
            elif path['done'][0] == False:
                terminated_idx = sum(~path['done'])+1
                for key in path.keys():
                    path[key] = path[key][:terminated_idx+1, ...]
                path['terminated'] = True
        return paths


    def evaluate_success(self, paths, logger=None, successful_steps=5):
        """
        Evaluate paths and log metrics to logger
        """
        num_success = 0
        num_paths = len(paths)

        # Record success if solved for provided successful_steps
        for path in paths:
            if np.sum(path['env_infos']['solved'] * 1.0) > successful_steps:
                # sum of truth values may not work correctly if dtype=object, need to * 1.0
                num_success += 1
        success_percentage = num_success*100.0/num_paths

        # log stats
        if logger:
            rwd_sparse = np.mean([np.mean(p['env_infos']['rwd_sparse']) for p in paths]) # return rwd/step
            rwd_dense = np.mean([np.sum(p['env_infos']['rwd_dense'])/self.horizon for p in paths]) # return rwd/step
            logger.log_kv('rwd_sparse', rwd_sparse)
            logger.log_kv('rwd_dense', rwd_dense)
            logger.log_kv('success_percentage', success_percentage)

        return success_percentage


    def seed(self, seed=None):
        """
        Set random number seed
        """
        self.input_seed = seed
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]


    def get_input_seed(self):
        return self.input_seed


    def reset(self, reset_qpos=None, reset_qvel=None):
        """
        Reset the environment
        Default implemention provided. Override if env needs custom reset
        """
        qpos = self.init_qpos.copy() if reset_qpos is None else reset_qpos
        qvel = self.init_qvel.copy() if reset_qvel is None else reset_qvel
        self.robot.reset(qpos, qvel)
        return self.get_obs()


    @property
    def _step(self, a):
        return self.step(a)


    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.frame_skip


    @property
    def horizon(self):
        return self.spec.max_episode_steps # paths could have early termination before horizon


    # state utilities ========================================================

    def set_state(self, qpos=None, qvel=None, act=None):
        """
        Set MuJoCo sim state
        """
        assert qpos.shape == (self.sim.model.nq,) and qvel.shape == (self.sim.model.nv,)
        old_state = self.sim.get_state()
        if qpos is None:
            qpos = old_state.qpos
        if qvel is None:
            qvel = old_state.qvel
        if act is None:
            act = old_state.act
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel, act)
        self.sim.set_state(new_state)
        self.sim.forward()


    def get_env_state(self):
        """
        Get full state of the environemnt
        Default implemention provided. Override if env has custom state
        """
        qp = self.sim.data.qpos.ravel().copy()
        qv = self.sim.data.qvel.ravel().copy()
        act = self.sim.data.act.ravel().copy() if self.sim.model.na>0 else None
        mocap_pos = self.sim.data.mocap_pos.copy() if self.sim.model.nmocap>0 else None
        mocap_quat = self.sim.data.mocap_quat.copy() if self.sim.model.nmocap>0 else None
        site_pos = self.sim.model.site_pos[:].copy() if self.sim.model.nsite>0 else None
        site_quat = self.sim.model.site_quat[:].copy() if self.sim.model.nsite>0 else None
        body_pos = self.sim.model.body_pos[:].copy()
        body_quat = self.sim.model.body_quat[:].copy()
        return dict(qpos=qp,
                    qvel=qv,
                    act=act,
                    mocap_pos=mocap_pos,
                    mocap_quat=mocap_quat,
                    site_pos=site_pos,
                    site_quat=site_quat,
                    body_pos=body_pos,
                    body_quat=body_quat)


    def set_env_state(self, state_dict):
        """
        Set full state of the environemnt
        Default implemention provided. Override if env has custom state
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        act = state_dict['act']
        self.set_state(qp, qv, act)
        if self.sim.model.nmocap>0:
            self.sim.model.mocap_pos[:] = state_dict['mocap_pos']
            self.sim.model.mocap_quat[:] = state_dict['mocap_quat']
        if self.sim.model.nsite>0:
            self.sim.model.site_pos[:] = state_dict['site_pos']
            self.sim.model.site_quat[:] = state_dict['site_quat']
        self.sim.model.body_pos[:] = state_dict['body_pos']
        self.sim.model.body_quat[:] = state_dict['body_quat']
        self.sim.forward()

    # def state_vector(self):
    #     state = self.sim.get_state()
    #     return np.concatenate([
    #         state.qpos.flat, state.qvel.flat])


    # Vizualization utilities ================================================

    def mj_render(self):
        try:
            # self.viewer.cam.azimuth+=.1 # trick to rotate camera for 360 videos
            self.viewer.render()
        except:
            self.viewer = MjViewer(self.sim)
            self.viewer._run_speed = 0.5
            self.viewer.cam.elevation = -30
            self.viewer.cam.azimuth = 90
            self.viewer.cam.distance = 2.5
            # self.viewer.lookat = np.array([-0.15602934,  0.32243594,  0.70929817])
            #self.viewer._run_speed /= self.frame_skip
            self.viewer_setup()
            self.viewer.render()


    def update_camera(self, camera=None, distance=None, azimuth=None, elevation=None, lookat=None):
        """
        Updates the given camera to move to the provided settings.
        """
        if not camera:
            if hasattr(self, 'viewer'):
                camera = self.viewer.cam
            else:
                return
        if distance is not None:
            camera.distance = distance
        if azimuth is not None:
            camera.azimuth = azimuth
        if elevation is not None:
            camera.elevation = elevation
        if lookat is not None:
            camera.lookat[:] = lookat

    def render_camera_offscreen(self, cameras:list, width:int=640, height:int=480, device_id:int=0, sim=None):
        """
        Render images(widthxheight) from a list_of_cameras on the specified device_id.
        """
        if sim is None:
            sim = self.sim_obsd
        imgs = np.zeros((len(cameras), height, width, 3), dtype=np.uint8)
        for ind, cam in enumerate(cameras) :
            img = sim.render(width=width, height=height, mode='offscreen', camera_name=cam, device_id=device_id)
            img = img[::-1, :, : ] # Image has to be flipped
            imgs[ind, :, :, :] = img
        return imgs


    def examine_policy(self,
            policy,
            horizon=1000,
            num_episodes=1,
            mode='exploration', # options: exploration/evaluation
            render=None,        # options: onscreen/offscreen/none
            camera_name=None,
            frame_size=(640,480),
            output_dir='/tmp/',
            filename='newvid',
            device_id:int=0
            ):
        """
            Examine a policy for behaviors;
            - either onscreen, or offscreen, or just rollout without rendering.
            - return resulting paths
        """
        exp_t0 = timer.time()

        # configure renderer
        if render == 'onscreen':
            self.mujoco_render_frames = True
        elif render =='offscreen':
            self.mujoco_render_frames = False
            frames = np.zeros((horizon, frame_size[1], frame_size[0], 3), dtype=np.uint8)
        elif render == None:
            self.mujoco_render_frames = False

        # start rollouts
        paths = []
        for ep in range(num_episodes):
            ep_t0 = timer.time()
            observations=[]
            actions=[]
            rewards=[]
            agent_infos = []
            env_infos = []

            print("Episode %d" % ep, end=":> ")
            o = self.reset()
            done = False
            t = 0
            ep_rwd = 0.0
            while t < horizon and done is False:
                a = policy.get_action(o)[0] if mode == 'exploration' else policy.get_action(o)[1]['evaluation']
                next_o, rwd, done, env_info = self.step(a)
                ep_rwd += rwd
                # render offscreen visuals
                if render =='offscreen':
                    curr_frame = self.render_camera_offscreen(
                        sim=self.sim,
                        cameras=[camera_name],
                        width=frame_size[0],
                        height=frame_size[1],
                        device_id=device_id
                    )
                    frames[t,:,:,:] = curr_frame[0]
                    print(t, end=', ', flush=True)
                observations.append(o)
                actions.append(a)
                rewards.append(rwd)
                # agent_infos.append(agent_info)
                env_infos.append(env_info)
                o = next_o
                t = t+1

            print("Total reward = %3.3f, Total time = %2.3f" % (ep_rwd, ep_t0-timer.time()))
            path = dict(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            # agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
            env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
            terminated=done
            )
            paths.append(path)

            # save offscreen buffers as video
            if render =='offscreen':
                file_name = output_dir + filename + str(ep) + ".mp4"
                skvideo.io.vwrite(file_name, np.asarray(frames))
                print("saved", file_name)

        self.mujoco_render_frames = False
        print("Total time taken = %f"% (timer.time()-exp_t0))
        return paths


    # methods to override ====================================================

    def get_obs_dict(self, sim):
        """
        Get observation dictionary
        Implement this in each subclass.
        If visual keys (rgba:cam_name:HxW) are present use get_visual_obs_dict() to get visual inputs, process it (typically passed through an encoder to reduce dims), and then update the obs_dict. For example
            > visual_obs_dict = self.get_visual_obs_dict(sim=sim)
            > obs_dict.update(visual_obs_dict)
        """
        raise NotImplementedError


    def get_reward_dict(self, obs_dict):
        """
        Compute rewards dictionary
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        Due to specifics of new mujoco rendering, the standard viewer cannot be used
        with this set-up. Instead we use this mujoco specific function. Customize your viewer here
        """
        pass