"""=================================================
# Copyright (c) MyoSuite Authors
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================="""

import collections

import numpy as np

from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.utils import gym
from myosuite.utils.quat_math import euler2quat
from myosuite.utils.vector_math import calculate_cosine


class PenTwirlFixedEnvV0(BaseV0):

    DEFAULT_OBS_KEYS = [
        "hand_jnt",
        "obj_pos",
        "obj_vel",
        "obj_rot",
        "obj_des_rot",
        "obj_err_pos",
        "obj_err_rot",
    ]
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pos_align": 1.0,
        "rot_align": 1.0,
        "act_reg": 5.0,
        "drop": 5.0,
        "bonus": 10.0,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):

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
        super().__init__(
            model_path=model_path,
            obsd_model_path=obsd_model_path,
            seed=seed,
            env_credits=self.MYO_CREDIT,
        )

        self._setup(**kwargs)

    def _setup(
        self,
        obs_keys: list = DEFAULT_OBS_KEYS,
        weighted_reward_keys: list = DEFAULT_RWD_KEYS_AND_WEIGHTS,
        **kwargs,
    ):
        self.target_obj_bid = self.mj_model.body("target").id
        self.S_grasp_sid = self.mj_model.site("S_grasp").id
        self.obj_bid = self.mj_model.body("Object").id
        self.eps_ball_sid = self.mj_model.site("eps_ball").id
        self.obj_t_sid = self.mj_model.site("object_top").id
        self.obj_b_sid = self.mj_model.site("object_bottom").id
        self.tar_t_sid = self.mj_model.site("target_top").id
        self.tar_b_sid = self.mj_model.site("target_bottom").id
        self.pen_length = np.linalg.norm(
            self.mj_model.site_pos[self.obj_t_sid]
            - self.mj_model.site_pos[self.obj_b_sid]
        )
        self.tar_length = np.linalg.norm(
            self.mj_model.site_pos[self.tar_t_sid]
            - self.mj_model.site_pos[self.tar_b_sid]
        )

        super()._setup(
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            **kwargs,
        )
        self.init_qpos[:-6] *= 0  # Use fully open as init pos
        self.init_qpos[0] = -1.5  # place palm up

    def get_obs_vec(self):
        # qpos for hand, xpos for obj, xpos for target
        self.obs_dict["time"] = np.array([self.mj_data.time])
        self.obs_dict["hand_jnt"] = self.mj_data.qpos[:-6].copy()
        self.obs_dict["obj_pos"] = self.mj_data.xpos[self.obj_bid].copy()
        self.obs_dict["obj_des_pos"] = self.mj_data.site_xpos[self.eps_ball_sid].ravel()
        self.obs_dict["obj_vel"] = self.mj_data.qvel[-6:].copy() * self.dt
        self.obs_dict["obj_rot"] = (
            self.mj_data.site_xpos[self.obj_t_sid]
            - self.mj_data.site_xpos[self.obj_b_sid]
        ) / self.pen_length
        self.obs_dict["obj_des_rot"] = (
            self.mj_data.site_xpos[self.tar_t_sid]
            - self.mj_data.site_xpos[self.tar_b_sid]
        ) / self.tar_length
        self.obs_dict["obj_err_pos"] = (
            self.obs_dict["obj_pos"] - self.obs_dict["obj_des_pos"]
        )
        self.obs_dict["obj_err_rot"] = (
            self.obs_dict["obj_rot"] - self.obs_dict["obj_des_rot"]
        )
        if self.mj_model.na > 0:
            self.obs_dict["act"] = self.mj_data.act[:].copy()

        t, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs

    def get_obs_dict(self, mj_model, mj_data):
        obs_dict = {}
        # qpos for hand, xpos for obj, xpos for target
        obs_dict["time"] = np.array([mj_data.time])
        obs_dict["hand_jnt"] = mj_data.qpos[:-6].copy()
        obs_dict["obj_pos"] = mj_data.xpos[self.obj_bid].copy()
        obs_dict["obj_des_pos"] = mj_data.site_xpos[self.eps_ball_sid].ravel()
        obs_dict["obj_vel"] = mj_data.qvel[-6:].copy() * self.dt
        obs_dict["obj_rot"] = (
            mj_data.site_xpos[self.obj_t_sid] - mj_data.site_xpos[self.obj_b_sid]
        ) / self.pen_length
        obs_dict["obj_des_rot"] = (
            mj_data.site_xpos[self.tar_t_sid] - mj_data.site_xpos[self.tar_b_sid]
        ) / self.tar_length
        obs_dict["obj_err_pos"] = obs_dict["obj_pos"] - obs_dict["obj_des_pos"]
        obs_dict["obj_err_rot"] = obs_dict["obj_rot"] - obs_dict["obj_des_rot"]
        if mj_model.na > 0:
            obs_dict["act"] = mj_data.act[:].copy()

        return obs_dict

    def get_reward_dict(self, obs_dict):
        pos_err = obs_dict["obj_err_pos"]
        pos_align = np.linalg.norm(pos_err, axis=-1)
        rot_align = calculate_cosine(obs_dict["obj_rot"], obs_dict["obj_des_rot"])
        # dropped = obs_dict['obj_pos'][:,:,2] < 0.075 if obs_dict['obj_pos'].ndim==3 else obs_dict['obj_pos'][2] < 0.075
        dropped = pos_align > 0.075
        act_mag = (
            np.linalg.norm(self.obs_dict["act"], axis=-1) / self.mj_model.na
            if self.mj_model.na != 0
            else 0
        )
        rwd_dict = collections.OrderedDict(
            (
                # Optional Keys
                ("pos_align", -1.0 * pos_align),
                ("rot_align", rot_align),
                ("act_reg", -1.0 * act_mag),
                ("drop", -1.0 * dropped),
                (
                    "bonus",
                    1.0 * (rot_align > 0.9) * (pos_align < 0.075)
                    + 5.0 * (rot_align > 0.95) * (pos_align < 0.075),
                ),
                # Must keys
                ("sparse", -1.0 * pos_align + rot_align),
                ("solved", (rot_align > 0.95) * (~dropped)),
                ("done", dropped),
            )
        )
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0
        )
        return rwd_dict


class PenTwirlRandomEnvV0(PenTwirlFixedEnvV0):

    def reset(self, **kwargs):
        # randomize target
        desired_orien = np.zeros(3)
        desired_orien[0] = self.np_random.uniform(low=-1, high=1)
        desired_orien[1] = self.np_random.uniform(low=-1, high=1)
        self.mj_model.body_quat[self.target_obj_bid] = euler2quat(desired_orien)
        self.robot.sync_sims(
            self.mj_model, self.mj_data, self.obsd_mj_model, self.obsd_mj_data
        )
        obs = super().reset(**kwargs)
        return obs
