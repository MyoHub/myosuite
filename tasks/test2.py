
# import sys

# print("sys.modules 里有没有 gym:", "gym" in sys.modules)

# try:
#     import gym
#     print("✅ 真正的 gym 包存在", gym.__file__)
# except Exception as e:
#     print("❌ 真正的 gym 包不存在:", e)


from myosuite.utils import gym
import skvideo.io
import numpy as np
import os

from IPython.display import HTML
from base64 import b64encode

def show_video(video_path, video_width = 400):

  video_file = open(video_path, "r+b").read()

  video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
  return HTML(f"""<video autoplay width={video_width} controls><source src="{video_url}"></video>""")

pth = 'myosuite/agents/baslines_NPG/'

policy = pth+"myoElbowPose1D6MExoRandom-v0/2022-02-26_21-16-27/36_env=myoElbowPose1D6MExoRandom-v0,seed=1/iterations/best_policy.pickle"

import pickle
pi = pickle.load(open(policy, 'rb'))

env = gym.make('myoElbowPose1D6MExoRandom-v0')

env.reset()

# define a discrete sequence of positions to test
AngleSequence = [60, 30, 30, 60, 80, 80, 60, 30, 80, 30, 80, 60]
env.reset()
frames = []
for ep in range(len(AngleSequence)):
    print("Ep {} of {} testing angle {}".format(ep, len(AngleSequence), AngleSequence[ep]))
    env.unwrapped.target_jnt_value = [np.deg2rad(AngleSequence[int(ep)])]
    env.unwrapped.target_type = 'fixed'
    env.unwrapped.weight_range=(0,0)
    env.unwrapped.update_target()
    for _ in range(40):
        frame = env.unwrapped.sim.renderer.render_offscreen(
                        width=400,
                        height=400,
                        camera_id=0)
        frames.append(frame)
        o = env.unwrapped.get_obs()
        a = pi.get_action(o)[0]
        next_o, r, done, *_, ifo = env.step(a) # take an action based on the current observation
env.close()

os.makedirs('videos', exist_ok=True)
# make a local copy
skvideo.io.vwrite('videos/exo_arm.mp4', np.asarray(frames),outputdict={"-pix_fmt": "yuv420p"})

show_video('videos/exo_arm.mp4')