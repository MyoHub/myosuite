import mujoco
import mujoco.viewer

from myosuite.envs.myo.tasks.challenge.boxing_vs.boxing_vs_config import BoxingVsConfig
from myosuite.envs.myo.tasks.challenge.boxing_vs.boxing_vs_model import (
    build_boxing_vs_model,
)

model, data, _ = build_boxing_vs_model(BoxingVsConfig())
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as v:
    v.cam.azimuth = 160
    v.cam.elevation = -15
    v.cam.distance = 2.5
    v.cam.lookat[:] = [0.0, 0.0, 1.0]
    while v.is_running():
        v.sync()
