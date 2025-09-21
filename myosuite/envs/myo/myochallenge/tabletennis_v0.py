""" =================================================
# Copyright (c) MyoSuite Authors
Authors  :: Cheryl Wang (cheryl.wang.huiyi@gmail.com), Balint Hodossy (bkh16@ic.ac.uk), 
            Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com), Guillaume Durandau (guillaume.durandau@mcgill.ca)
================================================= """

import collections
from typing import List
import enum

from dm_control.mujoco.wrapper import MjModel as dm_MjModel

import mujoco
import numpy as np
from myosuite.utils import gym
import h5py
import os
from scipy.spatial.transform import Rotation as R

from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.utils.spec_processing import recursive_immobilize, recursive_remove_contacts, recursive_mirror


MAX_TIME = 3.0


class TableTennisEnvV0(BaseV0):

    DEFAULT_OBS_KEYS = ['pelvis_pos', 'body_qpos', 'body_qvel', 'ball_pos', 'ball_vel', 'paddle_pos', "paddle_vel", 'paddle_ori', 'reach_err' , "touching_info"]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "reach_dist": 1,
        "palm_dist": 1,
        "paddle_quat": 2,
        "act_reg": .5,
        'torso_up': 2,
        #"ref_qpos_err": 1,
        #"ref_qvel_err": .1,
        "sparse": 100,
        "solved": 1000,
        'done': -10
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        # Two step construction (init+setup) is required for pickling to work correctly.
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        preproc_kwargs = {"remove_body_collisions": kwargs.pop("remove_body_collisions", True),
                          "add_left_arm": kwargs.pop("add_left_arm", True)}
        spec: mujoco.MjSpec = mujoco.MjSpec.from_file(model_path)
        spec = self._preprocess_spec(spec, **preproc_kwargs)  # TODO: confirm this doesn't break pickling
        model_handle = dm_MjModel(spec.compile())
        super().__init__(model_path=model_handle, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)
        self._setup(**kwargs)


    def _setup(self,
            frame_skip: int = 10,
            qpos_noise_range = None, # Noise in joint space for initialization
            obs_keys:list = DEFAULT_OBS_KEYS,
            ball_xyz_range = None,
            ball_qvel = None,
            ball_friction_range = None,
            paddle_mass_range = None,
            rally_count = 1,
            weighted_reward_keys:list = DEFAULT_RWD_KEYS_AND_WEIGHTS,
            **kwargs,
        ):
        self.ball_xyz_range = ball_xyz_range
        self.ball_qvel = ball_qvel
        self.qpos_noise_range = qpos_noise_range
        self.paddle_mass_range = paddle_mass_range
        self.ball_friction_range = ball_friction_range
        self.init_paddle_quat = R.from_euler('xyz', np.array([-0.3, 1.57, 0]), degrees = False).as_quat()[[3, 0, 1, 2]]
        self.contact_trajectory = []

        self.id_info = IdInfo(self.sim.model)
        self.ball_dofadr = self.sim.model.body_dofadr[self.id_info.ball_bid]
        self.ball_posadr = self.sim.model.joint("pingpong_freejoint").qposadr[0]

        super()._setup(obs_keys=obs_keys,
                    weighted_reward_keys=weighted_reward_keys,
                    frame_skip=frame_skip,
                    **kwargs,
        )
        keyFrame_id = 0
        self.init_qpos[:] = self.sim.model.key_qpos[keyFrame_id].copy()
        self.start_vel = np.array([[5.6, 1.6, 0.1] ]) #np.array([[5.5, 1, -2.8] ])
        self.init_qvel[self.ball_dofadr : self.ball_dofadr + 3] = self.start_vel
        self.rally_count = rally_count
        self.cur_rally = 0

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict['time'] = np.array([sim.data.time])

        obs_dict['pelvis_pos'] = sim.data.site_xpos[self.sim.model.site_name2id("pelvis")]

        obs_dict['body_qpos'] = sim.data.qpos[self.id_info.myo_joint_range].copy()
        obs_dict['body_qvel'] = sim.data.qvel[self.id_info.myo_dof_range].copy()

        obs_dict["ball_pos"] = sim.data.site_xpos[self.id_info.ball_sid]
        obs_dict["ball_vel"] = self.get_sensor_by_name(sim.model, sim.data, "pingpong_vel_sensor")

        obs_dict["paddle_pos"] = sim.data.site_xpos[self.id_info.paddle_sid]
        obs_dict["paddle_vel"] = self.get_sensor_by_name(sim.model, sim.data, "paddle_vel_sensor")
        obs_dict["paddle_ori"] = sim.data.body_xquat[self.id_info.paddle_bid]
        obs_dict['padde_ori_err'] = obs_dict["paddle_ori"] - self.init_paddle_quat

        obs_dict['reach_err'] = obs_dict['paddle_pos'] - obs_dict['ball_pos']

        obs_dict['palm_pos'] = self.sim.data.site_xpos[self.sim.model.site_name2id('S_grasp')]
        obs_dict['palm_err'] = obs_dict['palm_pos'] - obs_dict['paddle_pos']

        this_model = sim.model
        this_data = sim.data

        touching_objects = set(get_ball_contact_labels(this_model, this_data, self.id_info))
        self.contact_trajectory.append(touching_objects)

        obs_vec = self._ball_label_to_obs(touching_objects)
        obs_dict["touching_info"] = obs_vec

        if sim.model.na>0:
            obs_dict['act'] = sim.data.act[:].copy()
        return obs_dict


    def get_reward_dict(self, obs_dict):
        reach_dist = np.abs(np.linalg.norm(self.obs_dict['reach_err'], axis=-1))
        palm_dist = np.abs(np.linalg.norm(self.obs_dict['palm_err'], axis=-1))
        act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1)/self.sim.model.na if self.sim.model.na !=0 else 0
        ball_pos = obs_dict["ball_pos"][0][0] if obs_dict['ball_pos'].ndim == 3 else obs_dict['ball_pos']
        solved = evaluate_pingpong_trajectory(self.contact_trajectory) == None
        paddle_quat_err = np.linalg.norm(obs_dict['padde_ori_err'], axis=-1)
        torso_err = abs(self.sim.data.qpos[self.sim.model.jnt_qposadr[self.sim.model.joint_name2id('flex_extension')]])
        paddle_touch = obs_dict['touching_info'][0][0] if obs_dict['touching_info'].ndim == 3 else obs_dict['touching_info']
        #=========== for the baseline, we provide an h5 file in which you could perform simple imitation learning ===========
            #======== uncomment to load the files and rewards =======================
        #qpos_ref, qvel_ref, qpos_err, qvel_err = self.ref_traj()()
        #ref_qpos_err = np.linalg.norm(qpos_err)
        #ref_qvel_err = np.linalg.norm(qvel_err)

        rwd_dict = collections.OrderedDict((
            # Perform reward tuning here --
            # Update Optional Keys section below
            # Update reward keys (DEFAULT_RWD_KEYS_AND_WEIGHTS) accordingly to update final rewards
            # Examples: Env comes pre-packaged with two keys pos_dist and rot_dist
            # Optional Keys
            ('reach_dist', np.exp(-1. * reach_dist)),
            ('palm_dist', np.exp(-5. * palm_dist)),
            ('paddle_quat', np.exp(- 5 * paddle_quat_err)),
            ('torso_up', np.exp(-5 * torso_err)),
            #('ref_qpos_err', -1 * ref_qpos_err), use these for your imitation learning script
            #('ref_qvel_err', -1 * ref_qvel_err),
            # Must keys
            ('act_reg', -1.*act_mag),
            ('sparse', paddle_touch[0] == 1), #paddle_touching
            ('solved', np.array([[solved]])),
            ('done', np.array([[self._get_done(ball_pos[-1], solved)]])),
        ))

        rwd_dict['dense'] = sum(float(wt) * float(np.array(rwd_dict[key]).squeeze())
                            for key, wt in self.rwd_keys_wt.items()
                                )

        if rwd_dict['solved']:
            self.cur_rally += 1
        if rwd_dict['solved'] and self.cur_rally < self.rally_count:
            rwd_dict['done'] = False
            rwd_dict['solved'] = False
            self.obs_dict['time'] = 0
            self.sim.data.time = 0
            self.contact_trajectory = []
            self.relaunch_ball()
        return rwd_dict
    
    def ref_traj(self, traj_path= r"your_h5.h5"):
        """
        Returns a function that provides reference (qpos, qvel) and their errors.
        After the end of the reference trajectory, errors are zero and the agent is unconstrained.
        """
        if not hasattr(self, "_ref_traj_cache"):
            if not os.path.isfile(traj_path):
                raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

            with h5py.File(traj_path, 'r') as f:
                qpos_ref = np.array(f['qpos'])  # shape (T, nq)
                qvel_ref = np.array(f['qvel'])  # shape (T, nv)

            self._ref_dt = self.sim.model.opt.timestep
            self._ref_traj_cache = {
                "qpos": qpos_ref,
                "qvel": qvel_ref,
                "T": len(qpos_ref),
                "nq": qpos_ref.shape[1],
                "nv": qvel_ref.shape[1],
            }

        def _get_ref():
            t = self.sim.data.time
            idx = int(t // self._ref_dt)

            if idx < self._ref_traj_cache["T"]:
                qpos_ref = self._ref_traj_cache["qpos"][idx]
                qvel_ref = self._ref_traj_cache["qvel"][idx]
                qpos_err = qpos_ref - self.sim.data.qpos
                qvel_err = qvel_ref - self.sim.data.qvel
            else:
                nq = self._ref_traj_cache["nq"]
                nv = self._ref_traj_cache["nv"]
                qpos_ref = np.zeros(nq)
                qvel_ref = np.zeros(nv)
                qpos_err = np.zeros(nq)
                qvel_err = np.zeros(nv)

            return qpos_ref, qvel_ref, qpos_err, qvel_err

        return _get_ref

    def _get_done(self, z, solved):
        if self.obs_dict['time'] > MAX_TIME:
            return 1
        elif z < 0.3:
            self.obs_dict['time'] = MAX_TIME
            return 1
        elif solved:
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


    def get_metrics(self, paths, successful_steps=1):
        """
        Evaluate paths and report metrics
        """
        num_success = 0
        num_paths = len(paths)

        # average sucess over entire env horizon
        for path in paths:
            # record success if solved for provided successful_steps
            if np.sum(path['env_infos']['rwd_dict']['solved'] * 1.0) >= successful_steps:
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
        self.contact_trajectory = []
        self.init_qpos[:] = self.sim.model.key_qpos[0].copy()

        # the mass of the paddle slightly changes 
        if self.paddle_mass_range:
            self.sim.model.body_mass[self.id_info.paddle_bid] = self.np_random.uniform(
                *self.paddle_mass_range) 

        # friction of the ball changes 
        if self.ball_friction_range:
            self.sim.model.geom_friction[self.id_info.ball_gid] = self.np_random.uniform(**self.ball_friction_range)

        if self.ball_xyz_range is not None:
            ball_pos = self.np_random.uniform(**self.ball_xyz_range)
            self.sim.model.body_pos[self.id_info.ball_bid] = ball_pos
            self.init_qpos[self.ball_posadr : self.ball_posadr + 3] = ball_pos
        
        if self.qpos_noise_range is not None:
            joint_ranges = self.sim.model.jnt_range[:, 1] - self.sim.model.jnt_range[:, 0]
            noise_fraction = self.np_random.uniform(**self.qpos_noise_range, size=joint_ranges.shape)

            reset_qpos_local = self.init_qpos.copy()

            # apply noise to all but the last two joints for paddle and pingpong
            for j, adr in enumerate(self.sim.model.jnt_qposadr[:-2]):
                reset_qpos_local[adr] += noise_fraction[j] * joint_ranges[j]

                reset_qpos_local[adr] = np.clip(
                    reset_qpos_local[adr],
                    self.sim.model.jnt_range[j, 0],
                    self.sim.model.jnt_range[j, 1],
                )
        else:
            reset_qpos_local = reset_qpos if reset_qpos is not None else self.init_qpos

        if self.ball_qvel:            
            v_bounds = self.cal_ball_qvel(ball_pos)
            v_low, v_high = v_bounds[1], v_bounds[0]
            ball_vel = self.np_random.uniform(low=v_low, high=v_high)
            self.init_qvel[self.ball_dofadr : self.ball_dofadr + 3] = ball_vel
        obs = super().reset(reset_qpos=reset_qpos_local, reset_qvel=self.init_qvel,**kwargs)

        self.cur_rally = 0

        return obs

    def cal_ball_qvel(self, ball_qpos):
        """
        Returns a range of velocity for the given ball_qpos
        The calculated qvel will make sure the table tennis lands on the model's side of the table
        """
        table_upper = [1.35, 0.70, 0.785] #set the position's range on the model's side of the table
        table_lower = [0.5, -0.60, 0.785]
        gravity = 9.81
        v_z = self.np_random.uniform(*(-0.1, 0.1))

        a = -0.5 * gravity
        b = v_z
        c = ball_qpos[2] - table_upper[2]

        discriminant = b**2 - 4 * a * c
        t = (-b - discriminant**0.5) / (2 * a)

        if discriminant < 0:
            raise ValueError(f"No real t: z0={ball_qpos[2]}, z_target={table_upper[2]}, v_z_init={v_z}")

        v_upper = [(table_upper[i] - ball_qpos[i]) / t for i in range(2)]
        v_lower = [(table_lower[i] - ball_qpos[i]) / t for i in range(2)]

        return [
            [v_upper[0], v_upper[1], v_z],
            [v_lower[0], v_lower[1], v_z]
        ]

    def relaunch_ball(self):

        ball_pos = self.init_qpos[self.ball_posadr: self.ball_dofadr + 3]
        ball_vel = self.init_qvel[self.ball_dofadr: self.ball_dofadr + 6]  # 6 dof to reset spin
        if self.ball_xyz_range is not None:
            ball_pos = self.np_random.uniform(**self.ball_xyz_range)
            self.sim.model.body_pos[self.id_info.ball_bid] = ball_pos
            self.init_qpos[self.ball_posadr: self.ball_posadr + 3] = ball_pos

        if self.ball_qvel:
            v_bounds = self.cal_ball_qvel(ball_pos)
            v_low, v_high = v_bounds[1], v_bounds[0]
            ball_vel[:3] = self.np_random.uniform(low=v_low, high=v_high)
            self.init_qvel[self.ball_dofadr: self.ball_dofadr + 3] = ball_vel[:3]
        self.sim.data.qpos[self.ball_posadr: self.ball_posadr + 3] = ball_pos
        self.sim.data.qvel[self.ball_dofadr: self.ball_dofadr + 6] = ball_vel

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

    def _preprocess_spec(self,
                         spec,
                         remove_body_collisions=True,
                         add_left_arm=True):
        # We'll process the string path to:
        # - add contralateral limb
        # - immobilize leg
        # - optionally alter physics
        # - we could attach the paddle at this point too
        # and compile it to a (wrapped) model - the SimScene can now take that as an input

        for paddle_b in spec.bodies:
            if "paddle" in paddle_b.name and paddle_b.parent != spec.worldbody:
                import warnings
                warnings.warn("A paddle was found that was not a free body. Confirm this is intended.")
        for s in spec.sensors:
            if "pingpong" not in s.name and "paddle" not in s.name:
                s.delete()
        temp_model = spec.compile()

        removed_ids = recursive_immobilize(spec, temp_model, spec.body("femur_l"), remove_eqs=True)
        removed_ids.extend(recursive_immobilize(spec, temp_model, spec.body("femur_r"), remove_eqs=True))

        for key in spec.keys:
            key.qpos = [j for i, j in enumerate(key.qpos) if i not in removed_ids]

        if remove_body_collisions:
            recursive_remove_contacts(spec.body("full_body"), return_condition=lambda b: "radius" in b.name)

        if add_left_arm:
            torso = spec.body("torso")

            spec_copy: mujoco.MjSpec = spec.copy()
            attachment_frame = torso.add_frame(quat=[0.5, 0.5, -0.5, 0.5],
                                               pos=[0.05, 0.373, -0.04])
            [k.delete() for k in spec_copy.keys]
            [t.delete() for t in spec_copy.textures]
            [m.delete() for m in spec_copy.materials]
            [t.delete() for t in spec_copy.tendons]
            [a.delete() for a in spec_copy.actuators]
            [e.delete() for e in spec_copy.equalities]
            [s.delete() for s in spec_copy.sensors]
            [c.delete() for c in spec_copy.cameras]
            recursive_immobilize(spec, temp_model, spec_copy.worldbody)
            recursive_remove_contacts(spec_copy.worldbody, return_condition=None)

            meshes_to_mirror = set()

            recursive_mirror(meshes_to_mirror, spec_copy, spec_copy.body("clavicle"))
            for mesh in spec_copy.meshes:
                if mesh.name in meshes_to_mirror:
                    mesh.name += "_mirrored"
                    mesh.scale[1] *= -1
                else:
                    mesh.delete()

            attachment_frame.attach_body(spec_copy.body("clavicle_mirrored"))
            spec.body("ulna_mirrored").quat =[0.546, 0, 0, -0.838]
            spec.body("humerus_mirrored").quat = [ 0.924, 0.383, 0, 0]
            pass
        return spec



class IdInfo:
    def __init__(self, model: mujoco.MjModel):
        self.paddle_sid = model.site("paddle").id
        self.paddle_bid = model.body("paddle").id
        self.ball_sid = model.site("pingpong").id
        self.ball_bid = model.body("pingpong").id

        self.ball_bid = model.body("pingpong").id
        self.ball_gid = model.geom("pingpong").id
        self.own_half_gid = model.geom("coll_own_half").id
        self.paddle_gid = model.geom("pad").id
        self.opponent_half_gid = model.geom("coll_opponent_half").id
        self.ground_gid = model.geom("ground").id
        self.net_gid = model.geom("coll_net").id

        myo_bodies = [model.body(i).id for i in range(model.nbody)
                      if not model.body(i).name.startswith("ping")
                      and "paddle" not in model.body(i).name
                      and not model.body(i).name in ["pingpong"]]
        self.myo_body_range = (min(myo_bodies), max(myo_bodies))

        # TODO add locomotion joint ids

        self.myo_joint_range = np.concatenate([model.joint(i).qposadr for i in range(model.njnt)
                                            if not model.joint(i).name.startswith("ping")
                                            and not model.joint(i).name == "pingpong_freejoint"
                                            and not model.joint(i).name == "paddle_freejoint"])

        self.myo_dof_range = np.concatenate([model.joint(i).dofadr for i in range(model.njnt)
                                            if not model.joint(i).name.startswith("ping")
                                            and not model.joint(i).name == "paddle_freejoint"])


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


def get_ball_contact_labels(model: mujoco.MjModel, data: mujoco.MjData, id_info: IdInfo):
    for con in data.contact:
        if model.geom(con.geom1).bodyid == id_info.ball_bid:
            yield geom_id_to_label(con.geom2, id_info)
        elif model.geom(con.geom2).bodyid == id_info.ball_bid:
            yield geom_id_to_label(con.geom1, id_info)


def geom_id_to_label(body_id, id_info: IdInfo):
    if body_id == id_info.paddle_gid:
        return PingpongContactLabels.PADDLE
    elif body_id == id_info.own_half_gid:
        return PingpongContactLabels.OWN
    elif body_id == id_info.opponent_half_gid:
        return PingpongContactLabels.OPPONENT
    elif body_id == id_info.net_gid:
        return PingpongContactLabels.NET
    elif body_id == id_info.ground_gid:
        return PingpongContactLabels.GROUND
    else:
        return PingpongContactLabels.ENV


def evaluate_pingpong_trajectory(contact_trajectory: List[set]):
    has_hit_paddle = False
    has_bounced_from_paddle = False
    has_bounced_from_table = False
    own_contact_count = 0
    own_contact_phase_done = False

    for s in contact_trajectory:
        if PingpongContactLabels.PADDLE not in s and has_hit_paddle:
            has_bounced_from_paddle = True
        if PingpongContactLabels.PADDLE in s and has_bounced_from_paddle:
            return ContactTrajIssue.DOUBLE_TOUCH
        if PingpongContactLabels.PADDLE in s:
            has_hit_paddle = True
        if PingpongContactLabels.OWN in s:
            if not has_bounced_from_table:
                # Start of initial bounce from serving
                has_bounced_from_table = True
                own_contact_count = 1
            elif not own_contact_phase_done:
                own_contact_count += 1
                if own_contact_count > 2: #initial serving bounce has contact for 2 timesteps
                    own_contact_phase_done = True
                    return ContactTrajIssue.OWN_HALF
            else:
                return ContactTrajIssue.OWN_HALF
        elif has_bounced_from_table:
            # Exit the initial own bounce phase
            own_contact_phase_done = True

        if PingpongContactLabels.OPPONENT in s:
            if has_hit_paddle:
                return None
            else:
                return ContactTrajIssue.NO_PADDLE

    return ContactTrajIssue.MISS
