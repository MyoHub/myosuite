""" =================================================
# Copyright (c) MyoSuite Authors
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com), Balint Hodossy (bkh16@ic.ac.uk), 
            Eric Lyu (shirui.lyu@kcl.ac.uk), Cheryl Wang (cheryl.wang.huiyi@gmail.com), Guillaume Durandau (guillaume.durandau@mcgill.ca)
================================================= """

import collections
import enum
import os, time

from scipy.spatial.transform import Rotation as R
import mujoco
from myosuite.utils import gym
import numpy as np
from myosuite.utils.quat_math import mat2euler, euler2quat
from typing import List

from myosuite.envs.myo.base_v0 import BaseV0

CONTACT_TRAJ_MIN_LENGTH = 100
GOAL_CONTACT = 10
MAX_TIME = 10.0


class BimanualEnvV1(BaseV0):
    DEFAULT_OBS_KEYS = ["time", "myohand_qpos", "myohand_qvel", "pros_hand_qpos", "pros_hand_qvel", "object_qpos",
                        "object_qvel", "touching_body"]

    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "reach_dist": -.1,
        "act": 0,
        "fin_dis": -0.5,
        # "fin_open": -1,
        # "lift_height": 2,
        "pass_err": -1,
        # "lift_bonus": 1,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        # Two step construction (init+setup) is required for pickling to work correctly.
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)
        self._setup(**kwargs)

    """
    TODO update set up function for the new environments
    """

    def _setup(self,
               frame_skip: int = 10,
               start_center=np.array([-0.4, -0.25, 1.05]),  # Start and goal centers, pos = center + shift * [0, 1]
               goal_center=np.array([0.4, -0.25, 1.05]),
               max_force=1500,  # Max force against throwing

               proximity_th=0.17,  # object-target proximity threshold, based on 10cm in each axis in Euclidean Distance

               start_shifts=np.array([0.055, 0.055, 0]),
               # shift factor for start/goal random generation with z-axis fixed
               goal_shifts=np.array([0.098, 0.098, 0]),

               obj_scale_change=None,  # object size change (relative to initial size)
               obj_mass_change=None,  # object size change (relative to initial size)
               obj_friction_change=None,  # object friction change (relative to initial size)
               # {'high': [1.2, 0.006, 0.00012], 'low': [0.8, 0.004, 0.00008]},  # friction change
               task_choice='fixed',  # fixed/ random
               obs_keys: list = DEFAULT_OBS_KEYS,
               weighted_reward_keys: list = DEFAULT_RWD_KEYS_AND_WEIGHTS,
               start_pos=(-0.4, -0.25),
               goal_pos=(0.4, -0.25),
               **kwargs,
               ):

        # user parameters
        self.task_choice = task_choice
        self.proximity_th = proximity_th

        # start position centers (before changes)
        self.start_center = start_center
        self.goal_center = goal_center

        self.start_shifts = start_shifts
        self.goal_shifts = goal_shifts
        self.PILLAR_HEIGHT = 1.09

        self.id_info = IdInfo(self.sim.model)

        self.start_bid = self.id_info.start_id
        self.goal_bid = self.id_info.goal_id

        self.obj_bid = self.id_info.manip_body_id
        self.obj_sid = self.sim.model.site_name2id('touch_site')
        self.obj_gid = self.sim.model.body(self.obj_bid).geomadr + 1
        self.obj_mid = next(i
                            for i in range(self.sim.model.nmesh)
                            if "box" in self.sim.model.mesh(i).name)
        self.init_obj_z = self.sim.data.site_xpos[self.obj_sid][-1]
        self.target_z = 0.2

        # define the palm and tip site id.
        self.palm_sid = self.sim.model.site_name2id('S_grasp')
        self.init_palm_z = self.sim.data.site_xpos[self.palm_sid][-1]
        self.fin0 = self.sim.model.site_name2id("THtip")
        self.fin1 = self.sim.model.site_name2id("IFtip")
        self.fin2 = self.sim.model.site_name2id("MFtip")
        self.fin3 = self.sim.model.site_name2id("RFtip")
        self.fin4 = self.sim.model.site_name2id("LFtip")

        self.Rpalm1_sid = self.sim.model.site_name2id('prosthesis/palm_thumb')
        self.Rpalm2_sid = self.sim.model.site_name2id('prosthesis/palm_pinky')

        self.start_pos = self.start_center
        self.goal_pos = self.goal_center

        self.sim.model.body_pos[self.start_bid] = self.start_pos
        self.sim.model.body_pos[self.goal_bid] = self.goal_pos

        # check whether the object experience force over max force
        self.over_max = False
        self.max_force = 0
        self.goal_touch = 0
        self.TARGET_GOAL_TOUCH = GOAL_CONTACT


        self.touch_history = []

        # setup for task randomization
        self.obj_mass_range = ({'low': self.sim.model.body_mass[self.obj_bid]+obj_mass_change[0],
                                'high': self.sim.model.body_mass[self.obj_bid]+obj_mass_change[1]}
                               if obj_mass_change else None)
        self.obj_scale_range = ({'low': -np.array(obj_scale_change), 'high': obj_scale_change}
                                if obj_scale_change else None)
        self.obj_friction_range = ({'low': self.sim.model.geom_friction[self.obj_gid] - obj_friction_change,
                                    'high': self.sim.model.geom_friction[self.obj_gid] + obj_friction_change}
                                   if obj_friction_change else None)
        # We'll center the mesh on the box to have an easier time scaling it:
        if obj_scale_change:
            self.__center_box_mesh()

        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       frame_skip=frame_skip,
                       **kwargs,
                       )
        self.init_qpos[:] = self.sim.model.key_qpos[2].copy()
        # adding random disturbance to start and goal positions, coefficients might need to be adaptable
        self.initialized_pos = False

    def _obj_label_to_obs(self, touching_body):
        # Function to convert touching body set to a binary observation vector
        # order follows the definition in enum class
        obs_vec = np.array([0, 0, 0, 0, 0])
        for i in touching_body:
            if i == ObjLabels.MYO:
                obs_vec[0] += 1
            elif i == ObjLabels.PROSTH:
                obs_vec[1] += 1
            elif i == ObjLabels.START:
                obs_vec[2] += 1
            elif i == ObjLabels.GOAL:
                obs_vec[3] += 1
            else:
                obs_vec[4] += 1

        return obs_vec

    def __center_box_mesh(self):
        """
        Adjusts the mesh geom's transform and vertices so scaling is straightforward afterwards. Only makes sense
        to call this method within setup after relevant ids have been identified.
        """
        self.obj_size0 = self.sim.model.geom_size[self.obj_gid].copy()
        self.obj_vert_addr = np.arange(self.sim.model.mesh(self.obj_mid).vertadr,
                                       self.sim.model.mesh(self.obj_mid).vertadr + self.sim.model.mesh(0).vertnum)
        q = self.sim.model.geom(self.obj_gid - 1).quat
        r = R.from_quat([q[1], q[2], q[3], q[0]])
        self.sim.model.mesh_vert[self.obj_vert_addr] = r.apply(self.sim.model.mesh_vert[self.obj_vert_addr])
        self.sim.model.mesh_normal[self.obj_vert_addr] = r.apply(self.sim.model.mesh_normal[self.obj_vert_addr])
        self.sim.model.geom(self.obj_gid - 1).quat = [1, 0, 0, 0]
        self.sim.model.mesh_vert[self.obj_vert_addr] += (self.sim.model.geom(self.obj_gid - 1).pos
                                                         - self.sim.model.geom(self.obj_gid).pos)[None, :]

        self.sim.model.geom(self.obj_gid - 1).pos = self.sim.model.geom(self.obj_gid).pos
        self.mesh_vert0 = self.sim.model.mesh_vert[self.obj_vert_addr].copy()
        self.ignore_first_scale = True

    def get_obs_dict(self, sim):
        obs_dict = {}

        obs_dict["time"] = np.array([self.sim.data.time])
        obs_dict["qp"] = sim.data.qpos.copy()
        obs_dict["qv"] = sim.data.qvel.copy()

        # MyoHand data
        obs_dict["myohand_qpos"] = sim.data.qpos[self.id_info.myo_joint_range].copy()
        obs_dict["myohand_qvel"] = sim.data.qvel[self.id_info.myo_dof_range].copy()

        # Prosthetic hand data and velocity
        obs_dict["pros_hand_qpos"] = sim.data.qpos[self.id_info.prosth_joint_range].copy()
        obs_dict["pros_hand_qvel"] = sim.data.qvel[self.id_info.prosth_dof_range].copy()

        # One more joint for qpos due to </freejoint>
        obs_dict["object_qpos"] = sim.data.qpos[self.id_info.manip_joint_range].copy()
        obs_dict["object_qvel"] = sim.data.qvel[self.id_info.manip_dof_range].copy()

        obs_dict["start_pos"] = self.start_pos
        obs_dict["goal_pos"] = self.goal_pos
        obs_dict["elbow_fle"] = self.sim.data.joint('elbow_flexion').qpos.copy()

        this_model = sim.model
        this_data = sim.data

        # Get touching object in terms of binary encoding
        touching_objects = set(get_touching_objects(this_model, this_data, self.id_info))
        self.touch_history.append(touching_objects)

        current_force = sim.data.sensordata[0]
        if current_force > self.max_force:
            self.max_force = current_force
        obs_dict['max_force'] = np.array([self.max_force])

        obs_vec = self._obj_label_to_obs(touching_objects)
        obs_dict["touching_body"] = obs_vec
        obs_dict["palm_pos"] = sim.data.site_xpos[self.palm_sid]
        obs_dict['fin0'] = sim.data.site_xpos[self.fin0]
        obs_dict['fin1'] = sim.data.site_xpos[self.fin1]
        obs_dict['fin2'] = sim.data.site_xpos[self.fin2]
        obs_dict['fin3'] = sim.data.site_xpos[self.fin3]
        obs_dict['fin4'] = sim.data.site_xpos[self.fin4]

        obs_dict["Rpalm_pos"] = (sim.data.site_xpos[self.Rpalm1_sid] + sim.data.site_xpos[self.Rpalm2_sid]) / 2

        obs_dict['MPL_ori'] = mat2euler(np.reshape(self.sim.data.site_xmat[self.Rpalm1_sid], (3, 3)))
        obs_dict['MPL_ori_err'] = obs_dict['MPL_ori'] - np.array([np.pi, 0, np.pi])

        obs_dict["obj_pos"] = sim.data.site_xpos[self.obj_sid]
        obs_dict["reach_err"] = obs_dict["palm_pos"] - obs_dict["obj_pos"]
        obs_dict["pass_err"] = obs_dict["Rpalm_pos"] - obs_dict["obj_pos"]

        if sim.model.na > 0:
            obs_dict["act"] = sim.data.act[:].copy()

        return obs_dict

    def get_reward_dict(self, obs_dict):

        reach_dist = np.abs(np.linalg.norm(obs_dict['reach_err'], axis=-1))
        pass_dist = np.abs(np.linalg.norm(obs_dict['pass_err'], axis=-1))

        obj_pos = obs_dict["obj_pos"][0][0] if obs_dict['obj_pos'].ndim == 3 else obs_dict['obj_pos']
        palm_pos = obs_dict["palm_pos"][0][0] if obs_dict["palm_pos"].ndim == 3 else obs_dict["palm_pos"]
        goal_pos = obs_dict["goal_pos"][0][0] if obs_dict["goal_pos"].ndim == 3 else obs_dict["goal_pos"]
        goal_pos = np.concatenate((goal_pos[:2], np.array([self.PILLAR_HEIGHT])))

        lift_height = np.linalg.norm(np.array([[[obj_pos[-1], palm_pos[-1]]]]) -
                                     np.array([[[self.init_obj_z, self.init_palm_z]]]), axis=-1)
        lift_height = 5 * np.exp(-10 * (lift_height - self.target_z) ** 2) - 5

        act = np.linalg.norm(obs_dict['act'], axis=-1) / self.sim.model.na if self.sim.model.na != 0 else 0
        fin_keys = ['fin0', 'fin1', 'fin2', 'fin3', 'fin4']
        fin_open = sum(np.linalg.norm(obs_dict[fin] - obs_dict['palm_pos'], axis=-1) for fin in fin_keys)
        fin_dis = sum(np.linalg.norm(obs_dict[fin] - obs_dict['obj_pos'], axis=-1) for fin in fin_keys)

        elbow_err = 5 * np.exp(-10 * (obs_dict['elbow_fle'][0] - 1.) ** 2) - 5
        goal_dis = np.array(
            [[np.abs(np.linalg.norm(obj_pos - goal_pos, axis=-1))]])
        
        touching_vec = obs_dict["touching_body"][0][0] if obs_dict['touching_body'].ndim == 3 else obs_dict['touching_body']

        if touching_vec[3] == 1:
            self.goal_touch += 1
        rwd_dict = collections.OrderedDict(
            (
                # Optional Keys
                ("reach_dist", reach_dist + np.log(reach_dist + 1e-6)),
                ("act", act),
                ("fin_open", np.exp(-5 * fin_open)),  # fin_open + np.log(fin_open +1e-8)
                ("fin_dis", fin_dis + np.log(fin_dis + 1e-6)),
                ("lift_bonus", elbow_err),
                ("lift_height", lift_height),
                ("pass_err", pass_dist + np.log(pass_dist + 1e-3)),
                # Must keys
                ("sparse", 0),
                ("goal_dist", goal_dis), 
                ("solved", goal_dis < self.proximity_th and self.goal_touch >= self.TARGET_GOAL_TOUCH),
                ("done", self._get_done(obj_pos[-1])),
            )
        )

        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0
        )

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

    def step(self, a, **kwargs):
        # We unnormalize robotic actuators, muscle ones are handled in the parent implementation
        processed_controls = a.copy()
        if self.normalize_act:
            robotic_act_ind = self.sim.model.actuator_dyntype != mujoco.mjtDyn.mjDYN_MUSCLE
            processed_controls[robotic_act_ind] = (np.mean(self.sim.model.actuator_ctrlrange[robotic_act_ind], axis=-1)
                                                   + processed_controls[robotic_act_ind]
                                                   * (self.sim.model.actuator_ctrlrange[robotic_act_ind, 1]
                                                      - self.sim.model.actuator_ctrlrange[robotic_act_ind, 0]) / 2.0)
        return super().step(processed_controls, **kwargs)


    def get_metrics(self, paths, successful_steps=5):
        """
        Evaluate paths and report metrics
        """
        num_success = 0
        num_paths = len(paths)

        # average sucess over entire env horizon
        for path in paths:
            # record success if solved for provided successful_steps, check how the path is stored
            if np.sum(path['env_infos']['rwd_dict']['solved'] * 1.0) > successful_steps and evaluate_contact_trajectory(
                    path['env_infos']['touch_history']) is None:
                num_success += 1
        score = num_success / num_paths

        times = np.mean([np.round(p['env_infos']['obs_dict']['time'][-1], 5) for p in paths])
        max_force = np.mean([np.round(p['env_infos']['obs_dict']['max_force'][-1], 5) for p in paths])
        goal_dist = np.mean([np.mean(p['env_infos']['rwd_dict']['goal_dist']) for p in paths])

        # average activations over entire trajectory (can be shorter than horizon, if done) realized
        effort = 1.0 * np.mean([np.mean(p['env_infos']['rwd_dict']['act']) for p in paths])

        metrics = {
            'score': score,
            'time': times,
            'effort': effort,
            'peak force': max_force,
            'goal dist': goal_dist, 
        }
        return metrics

    def reset(self, **kwargs):
        self.start_pos = self.start_center + self.start_shifts * (2 * self.np_random.random(3) - 1)
        self.goal_pos = self.goal_center + self.goal_shifts * (2 * self.np_random.random(3) - 1)
        #
        self.sim.model.body_pos[self.start_bid] = self.start_pos
        self.sim.model.body_pos[self.goal_bid] = self.goal_pos
        self.touch_history = []
        self.over_max = False
        self.goal_touch = 0

        # box mass changes
        if self.obj_mass_range:
            self.sim.model.body_mass[self.obj_bid] = self.np_random.uniform(
                **self.obj_mass_range)  # call to mj_setConst(m,d) is being ignored. Derive quantities wont be updated. Die is simple shape. So this is reasonable approximation.

        # box friction changes
        if self.obj_friction_range:
            self.sim.model.geom_friction[self.obj_gid] = self.np_random.uniform(**self.obj_friction_range)

        # box size changes
        if self.obj_scale_range and not self.ignore_first_scale:
            obj_scales = self.np_random.uniform(**self.obj_scale_range) + 1
            self.sim.model.geom(self.obj_gid).size = self.obj_size0 * obj_scales

            if self.sim.renderer._window:
                self.sim.model.mesh_vert[self.obj_vert_addr] = obj_scales[None, :] * self.mesh_vert0
                self.sim.renderer._window.update_mesh(self.obj_mid)
        else:
            self.ignore_first_scale = False
        self.sim.forward()

        self.init_qpos[:] = self.sim.model.key_qpos[2].copy()
        # self.init_qpos[:-14] *= 0 # Use fully open as init pos

        obs = super().reset(
            reset_qpos=self.init_qpos, reset_qvel=self.init_qvel, **kwargs
        )
        object_qpos_adr = self.sim.model.body(self.obj_bid).jntadr[0]
        self.sim.data.qpos[object_qpos_adr:object_qpos_adr + 3] = self.start_pos + np.array([0, 0, 0.1])
        self.init_obj_z = self.sim.data.site_xpos[self.obj_sid][-1]
        self.init_palm_z = self.sim.data.site_xpos[self.palm_sid][-1]
        return obs


class ObjLabels(enum.Enum):
    MYO = 0
    PROSTH = 1
    START = 2
    GOAL = 3
    ENV = 4


class ContactTrajIssue(enum.Enum):
    MYO_SHORT = 0
    PROSTH_SHORT = 1
    NO_GOAL = 2  # Maybe can enforce implicitly, and only declare success is sufficient consecutive frames with only
    # goal contact.
    ENV_CONTACT = 3


class IdInfo:
    def __init__(self, model: mujoco.MjModel):
        self.manip_body_id = model.body("manip_object").id

        myo_bodies = [model.body(i).id for i in range(model.nbody)
                      if not model.body(i).name.startswith("prosthesis")
                      and not model.body(i).name in ["start", "goal", "manip_object"]]
        self.myo_body_range = (min(myo_bodies), max(myo_bodies))

        prosth_bodies = [model.body(i).id for i in range(model.nbody) if model.body(i).name.startswith("prosthesis/")]
        self.prosth_body_range = (min(prosth_bodies), max(prosth_bodies))

        self.myo_joint_range = np.concatenate([model.joint(i).qposadr for i in range(model.njnt)
                                               if not model.joint(i).name.startswith("prosthesis")
                                               and not model.joint(i).name == "manip_object/freejoint"])

        self.myo_dof_range = np.concatenate([model.joint(i).dofadr for i in range(model.njnt)
                                             if not model.joint(i).name.startswith("prosthesis")
                                             and not model.joint(i).name == "manip_object/freejoint"])

        self.prosth_joint_range = np.concatenate([model.joint(i).qposadr for i in range(model.njnt)
                                                  if model.joint(i).name.startswith("prosthesis")])

        self.prosth_dof_range = np.concatenate([model.joint(i).dofadr for i in range(model.njnt)
                                                if model.joint(i).name.startswith("prosthesis")])

        self.manip_joint_range = np.arange(model.joint("manip_object/freejoint").qposadr,
                                           model.joint("manip_object/freejoint").qposadr + 7)

        self.manip_dof_range = np.arange(model.joint("manip_object/freejoint").dofadr,
                                         model.joint("manip_object/freejoint").dofadr + 6)

        self.start_id = model.body("start").id
        self.goal_id = model.body("goal").id


def get_touching_objects(model: mujoco.MjModel, data: mujoco.MjData, id_info: IdInfo):
    for con in data.contact:
        if model.geom(con.geom1).bodyid == id_info.manip_body_id:
            yield body_id_to_label(model.geom(con.geom2).bodyid, id_info)
        elif model.geom(con.geom2).bodyid == id_info.manip_body_id:
            yield body_id_to_label(model.geom(con.geom1).bodyid, id_info)


def body_id_to_label(body_id, id_info: IdInfo):
    if id_info.myo_body_range[0] <= body_id <= id_info.myo_body_range[1]:
        return ObjLabels.MYO
    elif id_info.prosth_body_range[0] <= body_id <= id_info.prosth_body_range[1]:
        return ObjLabels.PROSTH
    elif body_id == id_info.start_id:
        return ObjLabels.START
    elif body_id == id_info.goal_id:
        return ObjLabels.GOAL
    else:
        return ObjLabels.ENV


def evaluate_contact_trajectory(contact_trajectory: List[set]):
    for s in contact_trajectory:
        if ObjLabels.ENV in s:
            return ContactTrajIssue.ENV_CONTACT

    myo_frames = np.nonzero([ObjLabels.MYO in s for s in contact_trajectory])[0]
    prosth_frames = np.nonzero([ObjLabels.PROSTH in s for s in contact_trajectory])[0]

    if len(myo_frames) < CONTACT_TRAJ_MIN_LENGTH:
        return ContactTrajIssue.MYO_SHORT
    elif len(prosth_frames) < CONTACT_TRAJ_MIN_LENGTH:
        return ContactTrajIssue.PROSTH_SHORT

    # Check if only goal was touching object for the last CONTACT_TRAJ_MIN_LENGTH frames
    elif not np.all([{ObjLabels.GOAL} == s for s in contact_trajectory[-GOAL_CONTACT + 2:]]): # Subtract 2 from the calculation to maintain a buffer zone around trajectory boundaries for safety/accuracy.
        return ContactTrajIssue.NO_GOAL
