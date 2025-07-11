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
import mujoco
from scipy.spatial.transform import Rotation as R

from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.utils.spec_processing import recursive_immobilize, recursive_remove_contacts, recursive_mirror


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
        preproc_kwargs = {"remove_body_collisions": kwargs.pop("remove_body_collisions", True),
                          "add_left_arm": kwargs.pop("add_left_arm", True)}
        model_handle = self._preprocess_spec(model_path, **preproc_kwargs)  # TODO: confirm this doesn't break pickling
        super().__init__(model_path=model_handle, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)
        self._setup(**kwargs)


    def _setup(self,
            frame_skip: int = 10,
            qpos_noise_range = None, # Noise in joint space for initialization
            obs_keys:list = DEFAULT_OBS_KEYS,
            ball_xyz_range = None,
            weighted_reward_keys:list = DEFAULT_RWD_KEYS_AND_WEIGHTS,
            **kwargs,
        ):
        self.ball_xyz_range = ball_xyz_range
        self.qpos_noise_range = qpos_noise_range
        self.contact_trajectory = []

        self.id_info = IdInfo(self.sim.model)
        self.ball_dofadr = self.sim.model.body_dofadr[self.id_info.ball_bid]

        super()._setup(obs_keys=obs_keys,
                    weighted_reward_keys=weighted_reward_keys,
                    frame_skip=frame_skip,
                    **kwargs,
        )
        keyFrame_id = 0
        self.init_qpos[:] = self.sim.model.key_qpos[keyFrame_id].copy()
        self.start_vel = np.array([[5.6, 1.6, 0.1] ]) #np.array([[5.5, 1, -2.8] ])
        self.init_qvel[self.ball_dofadr : self.ball_dofadr + 3] = self.start_vel

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

        obs_dict['reach_err'] = obs_dict['paddle_pos'] - obs_dict['ball_pos']

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
        rwd_dict['dense'] = sum(float(wt) * float(np.array(rwd_dict[key]).squeeze())
                            for key, wt in self.rwd_keys_wt.items()
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
            self.sim.model.body_pos[self.id_info.ball_bid] = self.np_random.uniform(**self.ball_xyz_range)

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

    def _preprocess_spec(self,
                         model_path,
                         remove_body_collisions=True,
                         add_left_arm=True):
        # We'll process the string path to:
        # - add contralateral limb
        # - immobilize leg
        # - optionally alter physics
        # - we could attach the paddle at this point too
        # and compile it to a (wrapped) model - the SimScene can now take that as an input
        spec: mujoco.MjSpec = mujoco.MjSpec.from_file(model_path)
        for paddle_b in spec.bodies:
            if "paddle" in paddle_b.name and paddle_b.parent != spec.worldbody:
                spec.detach_body(paddle_b)
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
        return dm_MjModel(spec.compile())



class IdInfo:
    def __init__(self, model: mujoco.MjModel):
        self.paddle_sid = model.site("paddle").id
        self.ball_sid = model.site("pingpong").id
        self.ball_bid = model.body("pingpong").id

        self.ball_bid = model.body("pingpong").id
        self.own_half_gid = model.geom("coll_own_half").id
        self.paddle_gid = model.geom("ping_pong_paddle").id
        self.opponent_half_gid = model.geom("coll_opponent_half").id
        self.ground_gid = model.geom("ground").id
        self.net_gid = model.geom("coll_net").id

        myo_bodies = [model.body(i).id for i in range(model.nbody)
                    if not model.body(i).name.startswith("ping")
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


if __name__ == '__main__':
    pingpong_env = PingPongEnvV0(r"../assets/arm/myoarm_tabletennis.xml")
    from mujoco import viewer
    pingpong_env.reset()
    viewer.launch(pingpong_env.sim.model._model, pingpong_env.sim.data._data)
    pass