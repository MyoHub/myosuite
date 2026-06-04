from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
import gymnasium as gym
import myosuite

myosuite.register_all_envs()
from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
    bootstrap_myosuite_mjlab_registry,
)

bootstrap_myosuite_mjlab_registry()
from myosuite.core.registry import make_env
from mjlab.rl import RslRlVecEnvWrapper, MjlabOnPolicyRunner
from mjlab.tasks.registry import load_rl_cfg

ckpt = Path("logs/rsl_rl/exp1/2026-05-12_20-53-29/model_0.pt")
rl_cfg = load_rl_cfg("myoChallengeSaberP0-v0")
mj_env = make_env("myoChallengeSaberP0-v0", backend="mjlab", device="cpu")
wrapped = RslRlVecEnvWrapper(mj_env)
runner = MjlabOnPolicyRunner(wrapped, asdict(rl_cfg), None, "cpu")
runner.load(str(ckpt))
policy = runner.get_inference_policy(device="cpu")

cpu_env = gym.make("myoChallengeSaberP0-v0")
obs_cpu, _ = cpu_env.reset(seed=0)
obs_mj, _ = mj_env.reset(seed=0)
for step in range(5):
    act_cpu = (
        policy(torch.as_tensor(obs_cpu, dtype=torch.float32).unsqueeze(0))
        .detach()
        .cpu()
        .numpy()[0]
    )
    act_mj = policy(obs_mj["policy"]).detach().cpu().numpy()[0]
    print(
        "step",
        step,
        "raw max diff",
        float(np.max(np.abs(act_cpu - act_mj))),
        "clipped max diff",
        float(np.max(np.abs(np.clip(act_cpu, 0, 1) - np.clip(act_mj, 0, 1)))),
    )
    obs_cpu, *_ = cpu_env.step(np.clip(act_cpu, 0.0, 1.0))
    obs_mj, *_ = mj_env.step(
        torch.clamp(torch.as_tensor(act_mj).unsqueeze(0), 0.0, 1.0)
    )
