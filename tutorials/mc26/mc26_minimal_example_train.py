from dataclasses import replace

import myosuite

myosuite.register_all_envs()
from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
    bootstrap_myosuite_mjlab_registry,
)

bootstrap_myosuite_mjlab_registry()
from mjlab.scripts.train import TrainConfig, launch_training

cfg = TrainConfig.from_task("myoChallengeSaberP0-v0")
cfg.env.scene.num_envs = 8
cfg.agent.max_iterations = 1
cfg.agent.num_steps_per_env = 8
cfg.agent.save_interval = 1
cfg = replace(cfg, gpu_ids=None, video=False)
launch_training("myoChallengeSaberP0-v0", cfg)
