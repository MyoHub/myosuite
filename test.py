import gym
import myosuite
from tqdm import tqdm
import skvideo
import skvideo.io
import os
import random
from stable_baselines3 import PPO
import numpy as np
from stable_baselines3.common.env_util import make_vec_env

from myosuite.utils import gym
import myosuite.envs.myo.myochallenge

nb_seed = 1

movie = True
path = '.'#/MPL_myo_bimanual'
env = make_vec_env('myoChallengeBimanual-v0', 4)
#env = gym.make('myoChallengeBimanual-v0')
#model = myosuite.load_baseline(env)
#print(model)
env.reset()

random.seed() 

frames = []
view = 'front'
for _ in tqdm(range(2)):
    done = False
    obs = env.reset()
    step = 0
    for _ in tqdm(range(250)):
          obs = env.obsdict2obsvec(env.obs_dict, env.obs_keys)[1]
          action, _ = model.predict(obs, deterministic=True)
          obs, reward, done, info, _ = env.step(action)
          if movie:
                  geom_1_indices = np.where(env.sim.model.geom_group == 1)
                  env.sim.model.geom_rgba[geom_1_indices, 3] = 0
                  frame = env.sim.renderer.render_offscreen(width= 1080, height=880,camera_id=f'{view}_view')
                  frame = (frame).astype(np.uint8)
                  frame = np.flipud(frame)
                  frames.append(frame[::-1,:,:])
          step += 1

# evaluate policy
all_rewards = []
ep = 5
for _ in tqdm(range(ep)): 
  ep_rewards = []
  done = False
  obs = env.reset()
  step = 0
  while (not done) and (step < 250):
      # get the next action from the policy
      obs = env.obsdict2obsvec(env.obs_dict, env.obs_keys)[1]
      action, _ = model.predict(obs, deterministic= True)
      obs, reward, done, info, _ = env.step(action)
      ep_rewards.append(reward)
      step += 1
  all_rewards.append(np.sum(ep_rewards))
env.close()
print(f"Average reward: {np.mean(all_rewards)} over {ep} episodes")


if movie:
    skvideo.io.vwrite('trained_baseline.mp4', np.asarray(frames), inputdict = {'-r':'100'} , outputdict={"-pix_fmt": "yuv420p"})
	
