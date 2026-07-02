from mjlab.tasks.registry import register_mjlab_task
from mjlab.rl.runner import MjlabOnPolicyRunner

from .env_cfg import stand_env_cfg
from .rl_cfg import torque_body_ppo_runner_cfg

register_mjlab_task(
    task_id="Myo-Stand",
    env_cfg=stand_env_cfg(),
    play_env_cfg=stand_env_cfg(play=True),
    rl_cfg=torque_body_ppo_runner_cfg(),
    runner_cls=MjlabOnPolicyRunner,
)
