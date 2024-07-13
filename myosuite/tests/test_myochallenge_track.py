from myosuite.utils import gym
import myosuite
from mujoco import viewer
import time


def run_env():
    env = gym.make("myoChallengeRunTrackP1-v0")
    # env = gym.make("myoChallengeChaseTagP2-v0")
    env.reset()
    env.unwrapped.mj_render()
    env.reset()
    for _ in range(5):
        env.reset()
        for i in range(1000):
            action = env.action_space.sample()
            state, reward, done, *_ = env.step(action)
            env.unwrapped.mj_render()
            time.sleep(0.01)
            print(f"{reward=}")
            print(f"{env.unwrapped.rwd_dict['solved']=}")
            if done:
                print(f"{done=}")
                break

            
    


if __name__ == "__main__":
    run_env()

    
