import gymnasium as gym
import myosuite
from mujoco import viewer
import time


def run_env():
    env = gym.make("myoChallengeRunTrack-v0")
    # window = viewer.launch_passive(env.sim.mj_model, env.sim.mj_data)
    env.reset()
    env.unwrapped.mj_render()
    env.reset()
    for _ in range(10):
        env.reset()
        for i in range(100):
            action = env.action_space.sample()
            env.step(action)
            env.unwrapped.mj_render()
            # window.sync()
            time.sleep(0.01)
            
    


if __name__ == "__main__":
    run_env()

    
