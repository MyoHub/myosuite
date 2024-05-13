""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

import logging

import mujoco
import numpy as np

from myosuite.envs import env_base
from myosuite.envs.myo.fatigue import CumulativeFatigue

from myosuite.envs import env_base
from myosuite.envs.myo.fatigue import CumulativeFatigue


class BaseV0(env_base.MujocoEnv):

    MYO_CREDIT = """\
    MyoSuite: A contact-rich simulation suite for musculoskeletal motor control
        Vittorio Caggiano, Huawei Wang, Guillaume Durandau, Massimo Sartori, Vikash Kumar
        L4DC-2019 | https://sites.google.com/view/myosuite
    """

    def _setup(
        self,
        obs_keys: list,
        weighted_reward_keys: dict,
        sites: list = None,
        frame_skip=10,
        muscle_condition="",
        fatigue_reset_vec=None,
        fatigue_reset_random=False,
        **kwargs,
    ):
        if self.sim.model.na > 0 and "act" not in obs_keys:
            obs_keys = (
                obs_keys.copy()
            )  # copy before editing incase other envs are using the defaults
            obs_keys.append("act")

        # ids
        self.tip_sids = []
        self.target_sids = []
        if sites:
            for site in sites:
                self.tip_sids.append(self.sim.model.site_name2id(site))
                self.target_sids.append(self.sim.model.site_name2id(site + "_target"))

        self.muscle_condition = muscle_condition
        self.fatigue_reset_vec = fatigue_reset_vec
        self.fatigue_reset_random = fatigue_reset_random
        self.frame_skip = frame_skip
        self.initializeConditions()
        super()._setup(
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            frame_skip=frame_skip,
            **kwargs,
        )
        self.viewer_setup(azimuth=90, distance=1.5, render_actuator=True)

    def initializeConditions(self):
        # for muscle weakness we assume that a weaker muscle has a
        # reduced maximum force
        if self.muscle_condition == "sarcopenia":
            for mus_idx in range(self.sim.model.actuator_gainprm.shape[0]):
                self.sim.model.actuator_gainprm[mus_idx, 2] = (
                    0.5 * self.sim.model.actuator_gainprm[mus_idx, 2].copy()
                )

        # for muscle fatigue we used the 3CC-r model
        elif self.muscle_condition == "fatigue":
            self.muscle_fatigue = CumulativeFatigue(
                self.sim.model, self.frame_skip, seed=self.get_input_seed()
            )

        # Tendon transfer to redirect EIP --> EPL
        # https://www.assh.org/handcare/condition/tendon-transfer-surgery
        elif self.muscle_condition == "reafferentation":
            self.EPLpos = self.sim.model.actuator_name2id("EPL")
            self.EIPpos = self.sim.model.actuator_name2id("EIP")

    # step the simulation forward
    def step(self, a, **kwargs):
        muscle_a = a.copy()
        muscle_act_ind = self.sim.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
        # Explicitely project normalized space (-1,1) to actuator space (0,1) if muscles
        if self.sim.model.na and self.normalize_act:
            # find muscle actuators
            muscle_a[muscle_act_ind] = 1.0 / (
                1.0 + np.exp(-5.0 * (muscle_a[muscle_act_ind] - 0.5))
            )
            # TODO: actuator space may not always be (0,1) for muscle or (-1, 1) for others
            isNormalized = (
                False  # refuse internal reprojection as we explicitly did it here
            )
        else:
            isNormalized = self.normalize_act  # accept requested reprojection

        # implement abnormalities
        if self.muscle_condition == "fatigue":
            # import ipdb; ipdb.set_trace()
            muscle_a[muscle_act_ind], _, _ = self.muscle_fatigue.compute_act(
                muscle_a[muscle_act_ind]
            )
        elif self.muscle_condition == "reafferentation":
            # redirect EIP --> EPL
            muscle_a[self.EPLpos] = muscle_a[self.EIPpos].copy()
            # Set EIP to 0
            muscle_a[self.EIPpos] = 0
        # step forward
        self.last_ctrl = self.robot.step(
            ctrl_desired=muscle_a,
            ctrl_normalized=isNormalized,
            step_duration=self.dt,
            realTimeSim=self.mujoco_render_frames,
            render_cbk=self.mj_render if self.mujoco_render_frames else None,
        )

        return self.forward(**kwargs)

    def reset(self, fatigue_reset=True, *args, **kwargs):
        if fatigue_reset:
            if self.muscle_condition == "fatigue":
                self.muscle_fatigue.reset(
                    fatigue_reset_vec=self.fatigue_reset_vec,
                    fatigue_reset_random=self.fatigue_reset_random,
                )
            else:
                pass

        return super().reset(*args, **kwargs)

    def set_fatigue_reset_random(self, fatigue_reset_random):  #
        if self.muscle_condition != "fatigue":
            logging.warning("This has no effect, as no fatigue model is provided.")
        self.fatigue_reset_random = fatigue_reset_random
