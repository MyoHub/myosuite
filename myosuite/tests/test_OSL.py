from myosuite.utils import gym
import myosuite
from mujoco import viewer
import time


def run_env():
    env = gym.make("myoChallengeRunTrackP1-v0", reset_type='osl_init')
    # window = viewer.launch_passive(env.sim.mj_model, env.sim.mj_data)
    env.reset()
    env.unwrapped.mj_render()
    env.reset()

    print(env.unwrapped.OSL_FSM)
    print(env.unwrapped.OSL_PARAM_LIST)

    for _ in range(10):
        env.reset()
        for i in range(100000):
            action = env.action_space.sample()
            state, reward, done, *_ = env.step(action)
            env.unwrapped.mj_render()
            # window.sync()
            time.sleep(0.01)
            if done:
                break
            
    


if __name__ == "__main__":
    run_env()

    
