""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/robohive
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

from myosuite.utils import gym
import numpy as np
import os
import time as timer

from myosuite.envs.obs_vec_dict import ObsVecDict
from myosuite.utils import tensor_utils
from myosuite.robot.robot import Robot
from myosuite.utils.implement_for import implement_for
from myosuite.utils.prompt_utils import prompt, Prompt
import skvideo.io
from sys import platform
from myosuite.physics.sim_scene import SimScene
import myosuite.utils.import_utils as import_utils
from myosuite.envs.env_variants import gym_registry_specs
from myosuite.utils import seed_envs

# TODO
# remove rwd_mode
# convert obs_keys to obs_keys_wt
# batch images before passing them through the encoder
# should path methods(compute_path_rewards, truncate_paths, evaluate_success) be moved to paths_utils?

class MujocoEnv(gym.Env, gym.utils.EzPickle, ObsVecDict):
    """
    Superclass for all MuJoCo environments.
    """

    DEFAULT_CREDIT = """\
        MyoSuite: a collection of environments/tasks to be solved by musculoskeletal models | https://sites.google.com/view/myosuite
        Code: https://github.com/MyoHub/myosuite/stargazers (add a star to support the project)
    """

    def __init__(self,  model_path, obsd_model_path=None, seed=None, env_credits=DEFAULT_CREDIT):
        """
        Create a gym env
        INPUTS:
            model_path: ground truth model
            obsd_model_path : observed model (useful for partially observed envs)
                            : observed model (useful to propagate noisy sensor through env)
                            : use model_path; if None
            seed: Random number generator seed

        """

        prompt("MyoSuite:> For environment credits, please cite -")
        prompt(env_credits, color="cyan", type=Prompt.ONCE)

        # Seed and initialize the random number generator
        self.seed(seed)

        # sims
        self.sim = SimScene.get_sim(model_path)
        self.sim_obsd = SimScene.get_sim(obsd_model_path) if obsd_model_path else self.sim
        self.sim.forward()
        self.sim_obsd.forward()
        ObsVecDict.__init__(self)


    def _setup(self,
               obs_keys:dict,               # Keys from obs_dict that forms the obs vector returned by get_obs()
               weighted_reward_keys:dict,   # Keys and weight that sums up to build the reward
               proprio_keys:list = None,    # Keys from obs_dict that forms the proprioception vector returned by get_proprio()
               visual_keys:list = None,     # Keys that specify visual_dict returned by get_visual()
               reward_mode:str = "dense",   # Configure env to return dense/sparse rewards
               frame_skip:int = 1,          # Number of mujoco frames/steps per env step
               normalize_act:bool = True,   # Ask env to normalize the action space
               obs_range:tuple = (-10, 10), # Permissible range of values in obs vector returned by get_obs()
               rwd_viz:bool = False,        # Visualize rewards (WIP, needs vtils)
               device_id:int = 0,           # Device id for rendering
               **kwargs,                    # Additional arguments
        ):

        if self.sim is None or self.sim_obsd is None:
            raise TypeError("sim and sim_obsd must be instantiated for setup to run")

        # Resolve viewer
        self.mujoco_render_frames = False
        self.device_id = device_id
        self.rwd_viz = rwd_viz
        self.viewer_setup()

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
            actuated_jnt_ids = self.sim.model.actuator_trnid[self.sim.model.actuator_trntype==self.sim.lib.mjtTrn.mjTRN_JOINT, 0] # dm
            linear_jnt_ids = np.logical_or(self.sim.model.jnt_type==self.sim.lib.mjtJoint.mjJNT_SLIDE, self.sim.model.jnt_type==self.sim.lib.mjtJoint.mjJNT_HINGE)
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

        # resolve proprio
        self.proprio_dict = {}
        self.proprio_keys = proprio_keys if type(proprio_keys)==list or proprio_keys==None else [proprio_keys]

        # resolve visuals
        self.visual_dict = {}
        self.visual_keys = visual_keys if type(visual_keys)==list or visual_keys==None else [visual_keys]
        self._setup_rgb_encoders(self.visual_keys, device=None)

        # reset to get the env ready
        observation, _reward, done, *_, _info = self.step(np.zeros(self.sim.model.nu))
        # Question: Should we replace above with following? Its specially helpful for hardware as it forces a env reset before continuing, without which the hardware will make a big jump from its position to the position asked by step.
        # observation = self.reset()
        assert not done, "Check initialization. Simulation starts in a done state."
        self.observation_space = gym.spaces.Box(obs_range[0]*np.ones(observation.size), obs_range[1]*np.ones(observation.size), dtype=np.float32)

        return


    def _setup_rgb_encoders(self, visual_keys, device=None):
        """
        Setup the supported visual encoders: 1d /2d / r3m18/ r3m34/ r3m50
        """
        if self.visual_keys == None:
            return
        else:
            # import torch only if environment with visual keys are used.
            import_utils.torch_isavailable()
            global torch; import torch

        if device is None:
            self.device_encoder = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device_encoder=device

        # ensure that all keys use the same encoder and image sizes
        id_encoders = []
        for key in visual_keys:
            if key.startswith('rgb'):
                id_encoder = key.split(':')[-2]+":"+key.split(':')[-1] # HxW:encoder
                id_encoders.append(id_encoder)
        if len(id_encoders) > 1 :
            unique_encoder = all(elem == id_encoders[0] for elem in id_encoders)
            assert unique_encoder, "Env only supports single encoder. Multiple in use ({})".format(id_encoders)

        # prepare encoder and transforms
        class IdentityEncoder(torch.nn.Module):
            def __init__(self):
                super(IdentityEncoder, self).__init__()

            def forward(self, x):
                return x

        self.rgb_encoder = None
        self.rgb_transform = None
        if len(id_encoders) > 0:
            wxh, id_encoder = id_encoders[0].split(':')

            if "rrl" in id_encoder or "resnet" in id_encoder:
                import_utils.torchvision_isavailable()
                import torchvision.transforms as T
                from torchvision.models import resnet50, ResNet50_Weights, resnet34, ResNet34_Weights, resnet18, ResNet18_Weights

            if "r3m" in id_encoder:
                import_utils.torchvision_isavailable()
                import torchvision.transforms as T
                import_utils.r3m_isavailable()
                from r3m import load_r3m

            if "vc1" in id_encoder:
                import_utils.vc_isavailable()
                from vc_models.models.vit import model_utils as vc

            # Load encoder
            prompt("Using {} visual inputs with {} encoder".format(wxh, id_encoder), type=Prompt.INFO)
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
            elif id_encoder == "rrl18":
                model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
                self.rgb_encoder = torch.nn.Sequential(*(list(model.children())[:-1])).float()
            elif id_encoder == "rrl34":
                model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
                self.rgb_encoder = torch.nn.Sequential(*(list(model.children())[:-1])).float()
            elif id_encoder == "rrl50":
                model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
                self.rgb_encoder = torch.nn.Sequential(*(list(model.children())[:-1])).float()
            elif id_encoder == "vc1s" or id_encoder == "vc1l":
                if id_encoder == "vc1s":
                    model,embd_size,model_transforms,model_info = vc.load_model(vc.VC1_BASE_NAME)
                else:
                    model,embd_size,model_transforms,model_info = vc.load_model(vc.VC1_LARGE_NAME)
                self.rgb_encoder = model
                self.rgb_transform = model_transforms
            else:
                raise ValueError("Unsupported visual encoder: {}".format(id_encoder))
            self.rgb_encoder.eval()
            self.rgb_encoder.to(self.device_encoder)

            # Load tranfsormms
            if id_encoder[:3] == 'rrl':
                if wxh == "224x224":
                    self.rgb_transform = T.Compose([T.ToTensor(),  # ToTensor() divides by 255
                                                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                else:
                    prompt("HxW = 224x224 recommended", type=Prompt.WARN)
                    self.rgb_transform = T.Compose([T.Resize(256),
                                                    T.CenterCrop(224),
                                                    T.ToTensor(),  # ToTensor() divides by 255
                                                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            elif id_encoder[:3] == 'r3m':
                if wxh == "224x224":
                    self.rgb_transform = T.Compose([T.ToTensor(),  # ToTensor() divides by 255
                                                   ])
                else:
                    print("HxW = 224x224 recommended")
                    self.rgb_transform = T.Compose([T.Resize(256),
                                                    T.CenterCrop(224),
                                                    T.ToTensor(),  # ToTensor() divides by 255
                                                   ])


    def step(self, a, **kwargs):
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
        return self.forward(**kwargs)

    @implement_for("gym", None, "0.24")
    def forward(self, **kwargs):
        return self._forward(**kwargs)

    @implement_for("gym", "0.24", None)
    def forward(self, **kwargs):
        obs, reward, done, info = self._forward(**kwargs)
        terminal = done
        return obs, reward, terminal, False, info

    @implement_for("gymnasium")
    def forward(self, **kwargs):
        obs, reward, done, info = self._forward(**kwargs)
        terminal = done
        return obs, reward, terminal, False, info

    def _forward(self, **kwargs):
        """
        Forward propagate env to recover env details
        Returns current obs(t), rwd(t), done(t), info(t)
        """

        # render the scene
        if self.mujoco_render_frames:
            self.mj_render()

        # observation
        obs = self.get_obs(**kwargs)

        # rewards
        self.expand_dims(self.obs_dict) # required for vectorized rewards calculations
        self.rwd_dict = self.get_reward_dict(self.obs_dict)
        self.squeeze_dims(self.rwd_dict)
        self.squeeze_dims(self.obs_dict)

        # finalize step
        env_info = self.get_env_infos()

        # returns obs(t+1), rwd(t+1), done(t+1), info(t+1)
        return obs, env_info['rwd_'+self.rwd_mode], bool(env_info['done']), env_info


    def get_obs(self, update_proprioception=True, update_exteroception=False):
        """
        Get state based observations from the environemnt.
        Uses robot to get sensors, reconstructs the sim and recovers the sensors.
        """
        # get sensor data from robot
        sen = self.robot.get_sensors()

        # reconstruct (partially) observed-sim using (noisy) sensor data
        self.robot.sensor2sim(sen, self.sim_obsd)

        # get obs_dict using the observed information
        self.obs_dict = self.get_obs_dict(self.sim_obsd)

        # get proprioception
        if update_proprioception:
            self.proprio_dict = self.get_proprioception(self.obs_dict)[2]

        # Don't update extero keys by default for efficiency
        # User should make an explicit call to get_visual_dict when needed
        if update_exteroception:
            self.visual_dict = self.get_visuals(sim=self.sim_obsd)

        # recoved observation vector from the obs_dict
        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs


    def get_visuals(self, sim=None, visual_keys:list=None, device_id:int=None)->dict:
        """
        Recover visual dict corresponding to the visual keys
        visual_keys
            = self.visual_keys if None
        Acceptable visual keys:
            - 'rgb:cam_name:HxW:1d'
            - 'rgb:cam_name:HxW:2d'
            - 'rgb:cam_name:HxW:r3m18'
            - 'rgb:cam_name:HxW:r3m34'
            - 'rgb:cam_name:HxW:r3m50'
        """
        # return if no visual configured
        if self.visual_keys == None:
            return None

        # default to observed sim
        if sim is None:
            sim = self.sim_obsd

        # default to all visual keys
        if visual_keys == None:
            visual_keys = self.visual_keys

        # default to env device
        if device_id is None:
            device_id = self.device_id

        # collect visuals
        visual_dict = {}
        visual_dict['time'] = np.array([self.sim.data.time])
        for key in visual_keys:
            if key.startswith('rgb'):

                # _, cam, wxh, rgb_encoder_id = key.split(':') # faces issues if camera names have : in them

                # get encoder
                key_payload = key[4:]
                rgb_encoder_id = key_payload.split(':')[-1]

                # get image resolution
                key_payload = key_payload[:-(len(rgb_encoder_id)+1)]
                wxh = key_payload.split(':')[-1]
                height = int(wxh.split('x')[0])
                width = int(wxh.split('x')[1])

                # get camera name
                cam = key_payload[:-(len(wxh)+1)]

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
                    rgb_encoded = img[0].reshape(-1)
                elif rgb_encoder_id == '2d':
                    rgb_encoded = img[0]
                elif rgb_encoder_id[:3] == 'r3m' or rgb_encoder_id[:3] == 'rrl':
                    with torch.no_grad():
                        rgb_encoded = 255.0 * self.rgb_transform(img[0]).reshape(-1, 3, 224, 224)
                        rgb_encoded = rgb_encoded.to(self.device_encoder)
                        rgb_encoded = self.rgb_encoder(rgb_encoded).cpu().numpy()
                        rgb_encoded = np.squeeze(rgb_encoded)
                elif rgb_encoder_id[:3] == 'vc1':
                    with torch.no_grad():
                        rgb_encoded = self.rgb_transform(torch.Tensor(img.transpose(0,3,1,2)))
                        rgb_encoded = rgb_encoded.to(self.device_encoder)
                        rgb_encoded = self.rgb_encoder(rgb_encoded).cpu().numpy()
                        rgb_encoded = np.squeeze(rgb_encoded)
                else:
                    raise ValueError("Unsupported visual encoder: {}".format(rgb_encoder_id))

                visual_dict.update({key:rgb_encoded})
                # add depth observations if requested in the keys (assumption d will always be accompanied by rgb keys)
                d_key = 'd:'+key[4:]
                if d_key in visual_keys:
                    visual_dict.update({d_key:dpt})

        return visual_dict


    def get_proprioception(self, obs_dict=None)->dict:
        """
        Get robot proprioception data. Usually incudes robot's onboard kinesthesia sensors (pos, vel, accn, etc)
        """
        # return if no proprio configured
        if self.proprio_keys == None:
            return None, None, None

        # pull out prioprio from the obs_dict
        if obs_dict==None: obs_dict = self.obs_dict
        proprio_vec = np.zeros(0)
        proprio_dict = {}
        proprio_dict['time'] = obs_dict['time']

        for key in self.proprio_keys:
            proprio_vec = np.concatenate([proprio_vec, obs_dict[key]])
            proprio_dict[key] = obs_dict[key]

        return proprio_dict['time'], proprio_vec, proprio_dict


    def get_exteroception(self, **kwargs)->dict:
        """
        Get robot exteroception data. Usually incudes robot's onboard (visual, tactile, acoustic) sensors
        """
        return self.get_visuals(**kwargs)


    # VIK??? Its getting called twice for mjrl agent. Once in step and sampler calls it as well
    def get_env_infos(self):
        """
        Get information about the environment.
        - NOTE: Returned dict contains pointers that will be updated by the env. Deepcopy returned data if you want it to persist
        - Essential keys are added below. Users can add more keys by overriding this function in their task-env
        - Requires necessary keys (dense, sparse, solved, done) in rwd_dict to be populated
        - Visual_dict can be {} if users hasn't explicitly updated it explicitly for current time
        """

        # resolve if current visuals are available
        if self.visual_dict and "time" in self.visual_dict.keys() and self.visual_dict['time']==self.obs_dict['time']:
            visual_dict = self.visual_dict
        else:
            visual_dict = {}

        env_info = {
            'time': self.obs_dict['time'][()],          # MDP(t)
            'rwd_dense': self.rwd_dict['dense'][()],    # MDP(t)
            'rwd_sparse': self.rwd_dict['sparse'][()],  # MDP(t)
            'solved': self.rwd_dict['solved'][()],      # MDP(t)
            'done': self.rwd_dict['done'][()],          # MDP(t)
            'obs_dict': self.obs_dict,                  # MDP(t)
            'visual_dict': visual_dict,                 # MDP(t), will be {} if user hasn't explicitly updated self.visual_dict at the current time
            'proprio_dict': self.proprio_dict,          # MDP(t)
            'rwd_dict': self.rwd_dict,                  # MDP(t)
            'state': self.get_env_state(),              # MDP(t)
        }
        return env_info


    def seed(self, seed=None):
        """
        Set random number seed
        """
        self.input_seed = seed
        self.np_random, seed = seed_envs(seed)
        return [seed]


    def get_input_seed(self):
        return self.input_seed


    def _reset(self, reset_qpos=None, reset_qvel=None, seed=None, **kwargs):
        """
        Reset the environment (Default implemention provided).
        Override if env needs custom reset. Carefully handle return type for gym/gymnasium compatibility
        """
        qpos = self.init_qpos.copy() if reset_qpos is None else reset_qpos
        qvel = self.init_qvel.copy() if reset_qvel is None else reset_qvel
        self.robot.reset(reset_pos=qpos, reset_vel=qvel, seed=seed, **kwargs)
        return self.get_obs()
    @implement_for("gym", None, "0.26")
    def reset(self, reset_qpos=None, reset_qvel=None, **kwargs):
        return self._reset(reset_qpos=reset_qpos, reset_qvel=reset_qvel, **kwargs)
    @implement_for("gym", "0.26", None)
    def reset(self, reset_qpos=None, reset_qvel=None, **kwargs):
        return self._reset(reset_qpos=reset_qpos, reset_qvel=reset_qvel, **kwargs), {}
    @implement_for("gymnasium")
    def reset(self, reset_qpos=None, reset_qvel=None, seed=None, **kwargs):
        return self._reset(reset_qpos=reset_qpos, reset_qvel=reset_qvel, seed=seed, **kwargs), {}

    # @property
    # def _step(self, a):
    #     return self.step(a)


    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.frame_skip


    @property
    def time(self):
        return self.sim_obsd.data.time


    @property
    def id(self):
        return self.spec.id


    @implement_for("gym")
    def _horizon(self):
        return self.spec.max_episode_steps # paths could have early termination before horizon
    @implement_for("gymnasium")
    def _horizon(self):
        return gym_registry_specs()[self.spec.id].max_episode_steps # gymnasium unwrapper overrides specs (https://github.com/Farama-Foundation/Gymnasium/issues/871)
    @property
    def horizon(self):
        return self._horizon()


    def get_env_state(self):
        """
        Get full state of the environemnt
        Default implemention provided. Override if env has custom state
        """
        time = self.sim.data.time
        qp = self.sim.data.qpos.ravel().copy()
        qv = self.sim.data.qvel.ravel().copy()
        act = self.sim.data.act.ravel().copy() if self.sim.model.na>0 else None
        mocap_pos = self.sim.data.mocap_pos.copy() if self.sim.model.nmocap>0 else None
        mocap_quat = self.sim.data.mocap_quat.copy() if self.sim.model.nmocap>0 else None
        site_pos = self.sim.model.site_pos[:].copy() if self.sim.model.nsite>0 else None
        site_quat = self.sim.model.site_quat[:].copy() if self.sim.model.nsite>0 else None
        body_pos = self.sim.model.body_pos[:].copy()
        body_quat = self.sim.model.body_quat[:].copy()
        return dict(time=time,
                    qpos=qp,
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
        time = state_dict['time']
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        act = state_dict['act'] if 'act' in state_dict.keys() else None
        self.sim.set_state(time=time, qpos=qp, qvel=qv, act=act)
        self.sim_obsd.set_state(time=time, qpos=qp, qvel=qv, act=act)
        if self.sim.model.nmocap>0:
            self.sim.data.mocap_pos[:] = state_dict['mocap_pos']
            self.sim.data.mocap_quat[:] = state_dict['mocap_quat']
            self.sim_obsd.data.mocap_pos[:] = state_dict['mocap_pos']
            self.sim_obsd.data.mocap_quat[:] = state_dict['mocap_quat']
        if self.sim.model.nsite>0:
            self.sim.model.site_pos[:] = state_dict['site_pos']
            self.sim.model.site_quat[:] = state_dict['site_quat']
            self.sim_obsd.model.site_pos[:] = state_dict['site_pos']
            self.sim_obsd.model.site_quat[:] = state_dict['site_quat']
        self.sim.model.body_pos[:] = state_dict['body_pos']
        self.sim.model.body_quat[:] = state_dict['body_quat']
        self.sim.forward()
        self.sim_obsd.model.body_pos[:] = state_dict['body_pos']
        self.sim_obsd.model.body_quat[:] = state_dict['body_quat']
        self.sim_obsd.forward()


    # Methods on paths (should it be a part of path_utils?) =================================

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


    # Vizualization utilities ================================================

    def mj_render(self):
        """
        Render the default camera
        """
        self.sim.renderer.render_to_window()


    def viewer_setup(self, distance=2.5, azimuth=90, elevation=-30, lookat=None, render_actuator=None, render_tendon=None):
        """
        Setup the default camera
        """
        self.sim.renderer.set_free_camera_settings(
                distance=distance,
                azimuth=azimuth,
                elevation=elevation,
                lookat=lookat
        )
        self.sim.renderer.set_viewer_settings(
            render_actuator=render_actuator,
            render_tendon=render_tendon
        )


    # Methods on policy (should it be a part of utils?) =================================
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

        if render == 'onscreen':
            self.mujoco_render_frames = True
        elif render =='offscreen':
            self.mujoco_render_frames = False
            frames = np.zeros((horizon, frame_size[1], frame_size[0], 3), dtype=np.uint8)
        elif render == None or render == 'None' or render == 'none':
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

            prompt("Episode %d" % ep, end=":> ", type=Prompt.INFO)
            o = self.reset()
            done = False
            t = 0
            ep_rwd = 0.0
            while t < horizon and done is False:
                a = policy.get_action(o)[0] if mode == 'exploration' else policy.get_action(o)[1]['evaluation']
                next_o, rwd, done, *_, env_info = self.step(a)
                ep_rwd += rwd
                # render offscreen visuals
                if render =='offscreen':
                    curr_frame = self.sim.renderer.render_offscreen(
                        width=frame_size[0],
                        height=frame_size[1],
                        camera_id=camera_name,
                        device_id=device_id)

                    frames[t,:,:,:] = curr_frame
                    prompt(t, end=', ', flush=True, type=Prompt.INFO)
                observations.append(o)
                actions.append(a)
                rewards.append(rwd)
                # agent_infos.append(agent_info)
                env_infos.append(env_info)
                o = next_o
                t = t+1

            prompt("Total reward = %3.3f, Total time = %2.3f" % (ep_rwd, timer.time()-ep_t0), type=Prompt.INFO)
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
                prompt("saved", file_name, type=Prompt.INFO)

        self.mujoco_render_frames = False
        prompt("Total time taken = %f"% (timer.time()-exp_t0), type=Prompt.INFO)
        return paths


    def examine_policy_new(self,
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

        # from myosuite.logger.roboset_logger import RoboSet_Trace as Trace
        from myosuite.logger.grouped_datasets import Trace

        trace = Trace(self.id+"_rollouts")

        exp_t0 = timer.time()

        if render == 'onscreen':
            self.mujoco_render_frames = True
        elif render =='offscreen':
            self.mujoco_render_frames = False
            frames = np.zeros((horizon, frame_size[1], frame_size[0], 3), dtype=np.uint8)
        elif render == None or render == 'None' or render == 'none':
            self.mujoco_render_frames = False

        # start rollouts
        for ep in range(num_episodes):

            # initialize -----------------------------
            ep_t0 = timer.time()
            group_key='Trial'+str(ep); trace.create_group(group_key)
            prompt(f"Episode {ep}", end=":> ", type=Prompt.INFO)
            obs = self.reset()
            done = False
            t = 0
            ep_rwd = 0.0

            # Rollout --------------------------------
            obs, rwd, done, *_, env_info = self.forward(update_exteroception=True) # t=0
            while t < horizon and done is False:

                # print(t, t*self.dt, self.time, t*self.dt-self.time)
                # Get step's actions ----------------------
                act = policy.get_action(obs)[0] if mode == 'exploration' else policy.get_action(obs)[1]['evaluation']

                # render offscreen visuals ----------------------
                if render =='offscreen':
                    curr_frame = self.sim.renderer.render_offscreen(
                        width=frame_size[0],
                        height=frame_size[1],
                        camera_id=camera_name,
                        device_id=device_id)

                    frames[t,:,:,:] = curr_frame
                    prompt(str(t), end=', ', flush=True, type=Prompt.INFO)

                # log values at time=t ----------------------------------
                datum_dict = dict(
                        time=self.time,
                        observations=obs,
                        actions=act.copy(),
                        rewards=rwd,
                        env_infos=env_info,
                        done=done,
                    )
                trace.append_datums(group_key=group_key, dataset_key_val=datum_dict)


                # step env using actions from t=>t+1 ----------------------
                obs, rwd, done, *_, env_info = self.step(act, update_exteroception=True)
                t = t+1
                ep_rwd += rwd

            # record last step and finalize the rollout --------------------------------
            act = np.nan*np.ones(self.action_space.shape)
            datum_dict = dict(
                        time=self.time,
                        observations=obs,
                        actions=act.copy(),
                        rewards=rwd,
                        env_infos=env_info,
                        done=done,
                    )
            trace.append_datums(group_key=group_key, dataset_key_val=datum_dict)
            prompt(f"Episode {ep}:> Finished in {(timer.time()-ep_t0):0.4} sec. Total rewards {ep_rwd}", type=Prompt.INFO)

            # save offscreen buffers as video --------------------------------
            if render =='offscreen':
                file_name = output_dir + filename + str(ep) + ".mp4"
                # check if the platform is OS -- make it compatible with quicktime
                if platform == "darwin":
                    skvideo.io.vwrite(file_name, np.asarray(frames),outputdict={"-pix_fmt": "yuv420p"})
                else:
                    skvideo.io.vwrite(file_name, np.asarray(frames))
                prompt("saved: "+file_name, type=Prompt.ALWAYS)

        self.mujoco_render_frames = False
        prompt("Total time taken = %f"% (timer.time()-exp_t0), type=Prompt.INFO)
        trace.stack()
        return trace


    # methods to override ====================================================

    def get_obs_dict(self, sim):
        """
        Get observation dictionary
        Implement this in each subclass.
        Note: Visual observations are automatically calculated via call to get_visual_obs_dict() from within get_obs()
            visual obs can be specified via visual keys of the form rgb:cam_name:HxW:encoder where cam_name is the name
            of the camera, HxW is the frame size and encode is the encoding of the image (can be 1d/2d as well as image encoders like rrl/r3m etc)
        """
        raise NotImplementedError


    def get_reward_dict(self, obs_dict):
        """
        Compute rewards dictionary
        Implement this in each subclass.
        """
        raise NotImplementedError
