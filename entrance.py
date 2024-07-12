import gym
import myosuite

from myosuite.utils import gym

env = gym.make('myoChallengeBimanual-v0')
env.reset()

for i in range(1000):
    
    env.mj_render()
    env.step(env.action_space.sample()) # take a random action
    if i % 50 == 0:
        env.reset()


env.close()