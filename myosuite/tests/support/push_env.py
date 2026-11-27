# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Native dynamic-body fixture shared by push protocol and replay tests."""

from dataclasses import replace

import gymnasium as gym
import mujoco
import numpy as np

from myosuite.utils.push_protocol import PushProtocol


class PushEnv(gym.Env):
    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_string("""<mujoco><option timestep=".01"/>
          <worldbody><geom type="plane" size="2 2 .1"/><body name="pelvis" pos="0 0 1">
          <freejoint/><geom type="sphere" size=".1" mass="2"/></body></worldbody></mujoco>""")
        self.data = mujoco.MjData(self.model)
        self._ctrl_dt = 0.01
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (13,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return np.r_[self.data.qpos, self.data.qvel].astype(np.float32), {}

    def step(self, action):
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return (
            np.r_[self.data.qpos, self.data.qvel].astype(np.float32),
            float(action[0]),
            False,
            False,
            {},
        )


def make_push_protocol(**kwargs) -> PushProtocol:
    return replace(
        PushProtocol("pelvis", 1.0, steps=35, start_s=0.1, duration_s=0.1, force_n=4.0),
        **kwargs,
    )
