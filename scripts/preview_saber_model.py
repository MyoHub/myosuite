# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
import mujoco
import mujoco.viewer

from myosuite.envs.myo.tasks.challenge.saber_vs.saber_vs_config import SaberVsConfig
from myosuite.envs.myo.tasks.challenge.saber_vs.saber_vs_model import (
    build_saber_vs_model,
)

model, data, _ = build_saber_vs_model(SaberVsConfig())
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as v:
    v.cam.azimuth = 160
    v.cam.elevation = -15
    v.cam.distance = 2.5
    v.cam.lookat[:] = [0.0, 0.0, 1.2]
    while v.is_running():
        v.sync()
