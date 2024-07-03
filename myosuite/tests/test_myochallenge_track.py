import gymnasium 
import myosuite
from mujoco import viewer
import time


def run_env():
    env = gymnasium.make("myoChallengeRunTrack-v0")
    env.reset()
    env.unwrapped.mj_render()
    env.reset()
    for _ in range(100):
        env.reset()
        for i in range(100):
            action = env.action_space.sample()
            state, reward, done, *_ = env.step(action)
            env.unwrapped.mj_render()
            time.sleep(0.01)
            if done:
                break
            
    


if __name__ == "__main__":
    run_env()

    
