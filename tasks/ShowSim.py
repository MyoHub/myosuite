from myosuite.utils import gym
env = gym.make('myoHandObjHoldFixed-v0')
env.reset()
for _ in range(5000):
  env.mj_render()
  env.step(env.action_space.sample()) # take a random action
env.close()