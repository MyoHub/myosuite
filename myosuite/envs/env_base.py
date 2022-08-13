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
import platform
# import torch #removed as not using visual inputs
# import torchvision.transforms as T #removed as not using visual inputs

from myosuite.utils.obj_vec_dict import ObsVecDict
from myosuite.utils import tensor_utils
from myosuite.robot.robot import Robot
from os import path
import skvideo.io

# from r3m import load_r3m #removed as not using visual inputs

# TODO
# remove rwd_mode
# convet obs_keys to obs_keys_wt
# batch images before passing them through the encoder

try:
    import mujoco_py
    from mujoco_py import load_model_from_path, MjSim, MjViewer, load_model_from_xml
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

# Removed as not using visual inputs
# class IdentityEncoder(torch.nn.Module):
#     def __init__(self):
#         super(IdentityEncoder, self).__init__()

#     def forward(self, x):
#         return x

class MujocoEnv(gym.Env, gym.utils.EzPickle, ObsVecDict):
    """
    Superclass for all MuJoCo environments.
    """

    def __init__(self, model_path, obsd_model_path=None, seed=None):
        """
        Create a gym env
        INPUTS:
            model_path: ground truth model
            obsd_model_path : observed model (useful for partially observed envs)
                            : observed model (useful to propagate noisy sensor through env)
                            : use model_path; if None
            seed: Random number generator seed
        """

        # Seed and initialize the random number generator
        self.seed(seed)

        # sims
        self.sim = get_sim(model_path)
        self.sim_obsd = get_sim(obsd_model_path) if obsd_model_path else self.sim
        self.sim.forward()
        self.sim_obsd.forward()
        ObsVecDict.__init__(self)

    def _setup(self,
               obs_keys,
               weighted_reward_keys,
               reward_mode = "dense",
               frame_skip = 1,
               normalize_act = True,
               obs_range = (-10, 10),
               rwd_viz = False,
               device_id = 0, # device id for rendering
               **kwargs,
        ):

        if self.sim is None or self.sim_obsd is None:
            raise TypeError("sim and sim_obsd must be instantiated for setup to run")

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

        # resolve initial state
        self.init_qvel = self.sim.data.qvel.ravel().copy()
        self.init_qpos = self.sim.data.qpos.ravel().copy() # has issues with initial jump during reset
        # self.init_qpos = np.mean(self.sim.model.actuator_ctrlrange, axis=1) if self.normalize_act else self.sim.data.qpos.ravel().copy() # has issues when nq!=nu
        # self.init_qpos[self.sim.model.jnt_dofadr] = np.mean(self.sim.model.jnt_range, axis=1) if self.normalize_act else self.sim.data.qpos.ravel().copy()
        if self.normalize_act:
            # find all linear+actuated joints. Use mean(jnt_range) as init position
            actuated_jnt_ids = self.sim.model.actuator_trnid[self.sim.model.actuator_trntype==mujoco_py.generated.const.TRN_JOINT, 0]
            linear_jnt_ids = np.logical_or(self.sim.model.jnt_type==mujoco_py.generated.const.JNT_SLIDE, self.sim.model.jnt_type==mujoco_py.generated.const.JNT_HINGE)
            linear_jnt_ids = np.where(linear_jnt_ids==True)[0]
            linear_actuated_jnt_ids = np.intersect1d(actuated_jnt_ids, linear_jnt_ids)
            # assert np.any(actuated_jnt_ids==linear_actuated_jnt_ids), "Wooho: Great evidence that it was important to check for actuated_jnt_ids as well as linear_actuated_jnt_ids"
            linear_actuated_jnt_qposids = self.sim.model.jnt_qposadr[linear_actuated_jnt_ids]
            self.init_qpos[linear_actuated_jnt_qposids] = np.mean(self.sim.model.jnt_range[linear_actuated_jnt_ids], axis=1)

        # resolve rewards
        self.rwd_dict = {}
        self.rwd_mode = reward_mode
        self.rwd_keys_wt = weighted_reward_keys

        # resolve obs
        self.obs_dict = {}
        self.obs_keys = obs_keys
        # self._setup_rgb_encoders(obs_keys, device=None) # Removed as not using visual inputs
        self.rgb_encoder = None # To compensate for above
        observation, _reward, done, _info = self.step(np.zeros(self.sim.model.nu))
        assert not done, "Check initialization. Simulation starts in a done state."
        self.obs_dim = observation.size
        self.observation_space = gym.spaces.Box(obs_range[0]*np.ones(self.obs_dim), obs_range[1]*np.ones(self.obs_dim), dtype=np.float32)

        return

    def _setup_rgb_encoders(self, obs_keys, device=None):
        """
        Setup the supported visual encoders: 1d /2d / r3m18/ r3m34/ r3m50
        """
        if device is None:
            self.device_encoder = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device_encoder=device

        # ensure that all keys use the same encoder and image sizes
        id_encoders = []
        for key in obs_keys:
            if key.startswith('rgb'):
                id_encoder = key.split(':')[-2]+":"+key.split(':')[-1] # HxW:encoder
                id_encoders.append(id_encoder)
        if len(id_encoders) > 1 :
            unique_encoder = all(elem == id_encoders[0] for elem in id_encoders)
            assert unique_encoder, "Env only supports single encoder. Multiple in use ({})".format(id_encoders)

        # prepare encoder and transforms
        self.rgb_encoder = None
        self.rgb_transform = None
        if len(id_encoders) > 0:
            wxh, id_encoder = id_encoders[0].split(':')

            # load appropriate encoders
            # if 'r3m' in id_encoder:
            #     print("Loading r3m...")
            #     from r3m import load_r3m

            # Load encoder
            print("Using {} visual inputs with {} encoder".format(wxh, id_encoder))
            if id_encoder == "1d":
                self.rgb_encoder = IdentityEncoder()
            elif id_encoder == "2d":
                self.rgb_encoder = IdentityEncoder()
            elif id_encoder == "r3m18":
                self.rgb_encoder = load_r3m("resnet18")
            elif id_encoder == "r3m34":
                self.rgb_encoder = load_r3m("resnet34")
            elif id_encoder == "r3m50":
                self.rgb_encoder = load_r3m("resnet50")
            else:
                raise ValueError("Unsupported visual encoder: {}".format(id_encoder))
            self.rgb_encoder.eval()
            self.rgb_encoder.to(self.device_encoder)

            # Load tranfsormms
            if id_encoder[:3] == 'r3m':
                if wxh == "224x224":
                    self.rgb_transform = T.Compose([T.ToTensor()]) # ToTensor() divides by 255
                else:
                    print("HxW = 224x224 recommended")
                    self.rgb_transform = T.Compose([T.Resize(256),
                                        T.CenterCrop(224),
                                        T.ToTensor()]) # ToTensor() divides by 255


    def step(self, a):
        """
        Step the simulation forward (t => t+1)
        Uses robot interface to safely step the forward respecting pos/ vel limits
        Accepts a(t) returns obs(t+1), rwd(t+1), done(t+1), info(t+1)
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

        # returns obs(t+1), rwd(t+1), done(t+1), info(t+1)
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

        if self.rgb_encoder:
            visual_obs_dict = self.get_visual_obs_dict(sim=self.sim_obsd)
            self.obs_dict.update(visual_obs_dict)

        # recoved observation vector from the obs_dict
        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs


    def get_visual_obs_dict(self, sim, device_id=None):
        """
        Recover visual observation dict corresponding to the visual keys in obs_keys
        Acceptable visual keys:
            - 'rgb:cam_name:HxW:1d'
            - 'rgb:cam_name:HxW:2d'
            - 'rgb:cam_name:HxW:r3m18'
            - 'rgb:cam_name:HxW:r3m34'
            - 'rgb:cam_name:HxW:r3m50'
        """
        if device_id is None:
            device_id = self.device_id

        visual_obs_dict = {}
        visual_obs_dict['t'] = np.array([self.sim.data.time])
        # find keys with rgb tags
        for key in self.obs_keys:
            if key.startswith('rgb'):
                _, cam, wxh, rgb_encoder_id = key.split(':')
                height = int(wxh.split('x')[0])
                width = int(wxh.split('x')[1])
                # render images ==> returns (ncams, height, width, 3)
                img, dpt = self.robot.get_visual_sensors(
                                    height=height,
                                    width=width,
                                    cameras=[cam],
                                    device_id=device_id,
                                    sim=sim,
                                  )
                # encode images
                if rgb_encoder_id == '1d':
                    rgb_encoded = img.reshape(-1)
                elif rgb_encoder_id == '2d':
                    rgb_encoded = img
                elif rgb_encoder_id[:3] == 'r3m':
                    with torch.no_grad():
                        rgb_encoded = 255.0 * self.rgb_transform(img[0]).reshape(-1, 3, 224, 224)
                        rgb_encoded.to(self.device_encoder)
                        rgb_encoded = self.rgb_encoder(rgb_encoded).cpu().numpy()
                        rgb_encoded = np.squeeze(rgb_encoded)
                else:
                    raise ValueError("Unsupported visual encoder: {}".format(rgb_encoder_id))

                visual_obs_dict.update({key:rgb_encoded})
                # add depth observations if requested in the keys (assumption d will always be accompanied by rgb keys)
                d_key = 'd:'+key[4:]
                if d_key in self.obs_keys:
                    visual_obs_dict.update({d_key:dpt})

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
            'rwd_dense': self.rwd_dict['dense'][()],    # MDP(t)
            'rwd_sparse': self.rwd_dict['sparse'][()],  # MDP(t)
            'solved': self.rwd_dict['solved'][()],      # MDP(t)
            'done': self.rwd_dict['done'][()],          # MDP(t)
            'obs_dict': self.obs_dict,                  # MDP(t)
            'rwd_dict': self.rwd_dict,                  # MDP(t)
            'state': self.get_env_state(),              # MDP(t)
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
        new_state = mujoco_py.MjSimState(old_state.time, qpos=qpos, qvel=qvel, act=act, udd_state={})
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
        for ind, cam in enumerate(cameras):
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

            print("Total reward = %3.3f, Total time = %2.3f" % (ep_rwd, timer.time()-ep_t0))
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
                # check if the platform is OS -- make it compatible with quicktime
                if platform == "darwin":
                    skvideo.io.vwrite(file_name, np.asarray(frames),outputdict={"-pix_fmt": "yuv420p"})
                else:
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
        Note: for visual keys (rgb:cam_name:HxW:encoder) use get_visual_obs_dict()
            to get visual inputs, process it (typically passed through an encoder
            to reduce dims), and then update the obs_dict. For example -
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
