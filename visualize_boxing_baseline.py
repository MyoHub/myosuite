
import time
import numpy as np
import mujoco.viewer

from myosuite import make_env

env = make_env("myoChallengeBoxingVs-v0")
obs, info = env.reset(seed=0)

print("Boxing environment loaded!")
print("Agents:", list(obs.keys()))
print("Starting baseline simulation...")

# Constant muscle activation, not a trained policy
ACTIVATION = 0.0

try:
    with mujoco.viewer.launch_passive(
        env.unwrapped.model,
        env.unwrapped.data,
    ) as viewer:

        while viewer.is_running():
            actions = {
                agent: np.full(
                    env.action_space[agent].shape,
                    ACTIVATION,
                    dtype=np.float32,
                )
                for agent in obs
            }

            obs, rewards, terminated, truncated, info = env.step(actions)

            viewer.sync()
            time.sleep(env.unwrapped.dt)

            if any(terminated.values()) or any(truncated.values()):
                print("Episode finished:", info)
                time.sleep(1)
                obs, info = env.reset()
                viewer.sync()

finally:
    env.close()
