import myosuite
import gym

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import MinMaxScaler
from collections import deque as dq

import os
import skvideo.io
from IPython.display import HTML
from base64 import b64encode

import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib

from typing import Callable
import numpy as np
import os
from base64 import b64encode
from IPython.display import HTML
import joblib
import matplotlib.pyplot as plt
import warnings

os.environ['MUJOCO_GL'] = 'egl'
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
plt.rcParams["font.family"] = "Latin Modern Roman"

def show_video(video_path, video_width=600):
    """
    Displays any mp4 video within the notebook.
    
    video_path: str; path to mp4
    video_width: str; optional; size to render video
    """
    video_file = open(video_path, "r+b").read()
    video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
    return HTML(f"""<video autoplay width={video_width} controls><source src="{video_url}"></video>""")

class SaveSuccesses(BaseCallback):
    """
    sb3 callback used to calculate and monitor success statistics. Used in training functions.
    """
    def __init__(self, check_freq: int, log_dir: str, env_name: str, verbose: int = 1):
        super(SaveSuccesses, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'ignore')
        self.check_for_success = []
        self.success_buffer = dq(maxlen=100)
        self.success_results = []
        self.env_name = env_name
        
    def _on_rollout_start(self) -> None:
        self.check_for_success = []
        
    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            
    def _on_rollout_end(self) -> None:
        if sum(self.check_for_success) > 0:
            self.success_buffer.append(1)
        else:
            self.success_buffer.append(0)
        
        if len(self.success_buffer) > 0:
            self.success_results.append(sum(self.success_buffer)/len(self.success_buffer))

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self.check_for_success.append(self.locals['infos'][0]['solved'])
        return True
    
    def _on_training_end(self) -> None:
        np.save(os.path.join(self.log_dir, f'success_{self.env_name}'), np.array(self.success_results))
#         plt.plot(range(len(self.success_results)), self.success_results)
#         plt.show()
        pass    

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plot_zeroshot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    n_bars = len(data)
    bar_width = total_width / n_bars
    bars = []
    for i, (name, values) in enumerate(data.items()):
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, 
                         color=colors[i % len(colors)],capsize=2)
        bars.append(bar[0])
    if legend:
        ax.legend(bars, data.keys(), fontsize=10)
    plt.xticks(range(2), ["Reorient-ID", "Reorient-OOD"])
    plt.title("Zero-shot generalization", size=18)
    plt.xlabel('generalization sets', fontsize=15)
    plt.ylabel('mean success', fontsize=15)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
def plot_results(smoothing=1000, experiment='locomotion', terrain=None):
    if not isinstance(smoothing, int) or smoothing < 1:
        raise ValueError("The smoothing value must be an integer greater than or equal to 1")

    smth = smoothing

    if experiment == 'locomotion':
        print(terrain)
        sar_rl_file = f'SAR-RL_results_myoLeg{terrain}TerrainWalk-v0_0/progress.csv'
        rl_e2e_file = f'RL-E2E_results_myoLeg{terrain}TerrainWalk-v0_0/progress.csv'

        if os.path.isfile(sar_rl_file):
            a_df = pd.read_csv(sar_rl_file)
            a_timesteps = a_df['time/total_timesteps'][:-smth]
            a_reward_mean = smooth(a_df['rollout/ep_rew_mean'], smth)[:-smth]
            plt.plot(a_timesteps, a_reward_mean, linewidth=3, label='SAR-RL')

        if os.path.isfile(rl_e2e_file):
            b_df = pd.read_csv(rl_e2e_file)
            b_timesteps = b_df['time/total_timesteps'][:-smth]
            b_reward_mean = smooth(b_df['rollout/ep_rew_mean'], smth)[:-smth]
            plt.plot(b_timesteps, b_reward_mean, linewidth=3, label='RL-E2E')

        plt.title(f'MyoLeg {terrain} locomotion task success comparison', size=14)

    elif experiment == 'manipulation':
        sar_rl_file = './SAR-RL_successes_myoHandReorient100-v0_0/success_myoHandReorient100-v0_0.npy'
        rl_e2e_file = './RL-E2E_successes_myoHandReorient100-v0_0/success_myoHandReorient100-v0_0.npy'

        if os.path.isfile(sar_rl_file):
            suc = np.load(sar_rl_file)
            suc = smooth(suc, smth)[:-smth]
            plt.plot(range(len(suc)), suc, linewidth=2.5, label='SAR-RL')

        if os.path.isfile(rl_e2e_file):
            suc = np.load(rl_e2e_file)
            suc = smooth(suc, smth)[:-smth]
            plt.plot(range(len(suc)), suc, linewidth=2.5, label='RL-E2E')

        plt.title(f'Success comparison on Reorient100', size=17)

    else:
        raise ValueError("experiment must be either 'locomotion' or 'manipulation'")

    plt.grid()

    plt.xlabel('environment iterations', fontsize=14)
    plt.ylabel('success/reward metric', fontsize=14)

    plt.legend(fontsize=11, loc='upper left')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()


def load_manipulation_SAR():    
    ica = joblib.load('./SAR_pretrained/manipulation/ica.pkl')
    pca = joblib.load('./SAR_pretrained/manipulation/pca.pkl')
    normalizer = joblib.load('./SAR_pretrained/manipulation/normalizer.pkl')
    
    return ica, pca, normalizer

def load_locomotion_SAR():    
    ica = joblib.load('./SAR_pretrained/locomotion/ica.pkl')
    pca = joblib.load('./SAR_pretrained/locomotion/pca.pkl')
    normalizer = joblib.load('./SAR_pretrained/locomotion/normalizer.pkl')
    
    return ica, pca, normalizer

class SynNoSynWrapper(gym.ActionWrapper):
    """
    gym.ActionWrapper that reformulates the action space as the combination of a task-general synergy space and a
    task-specific orginal space, and uses this mix to step the environment in the original action space.
    """
    def __init__(self, env, ica, pca, scaler, phi):
        super().__init__(env)
        self.ica = ica
        self.pca = pca
        self.scaler = scaler
        self.weight = phi
        
        self.syn_act_space = self.pca.components_.shape[0]
        self.no_syn_act_space = env.action_space.shape[0]
        self.full_act_space = self.syn_act_space + self.no_syn_act_space
        
        self.action_space = gym.spaces.Box(low=-1., high=1., shape=(self.full_act_space,),dtype=np.float32)
    def action(self, act):
        syn_action = act[:self.syn_act_space]
        no_syn_action = act[self.syn_act_space:]
        
        syn_action = self.pca.inverse_transform(self.ica.inverse_transform(self.scaler.inverse_transform([syn_action])))[0]
        final_action = self.weight * syn_action + (1 - self.weight) * no_syn_action
        
        return final_action

class SynergyWrapper(gym.ActionWrapper):
    """
    gym.ActionWrapper that reformulates the action space as the synergy space and inverse transforms
    synergy-exploiting actions back into the original muscle activation space.
    """
    def __init__(self, env, ica, pca, w_scaler):
        super().__init__(env)
        self.ica = ica
        self.pca = pca
        self.scaler = w_scaler
        
        self.action_space = gym.spaces.Box(low=-1., high=1., shape=(self.pca.components_.shape[0],),dtype=np.float32)
    
    def action(self, act):
        action = self.pca.inverse_transform(self.ica.inverse_transform(self.scaler.inverse_transform([act])))
        return action[0]

def get_vid(name, env_name, seed, episodes, video_name, determ=False, 
            pca=None, ica=None, scaler=None, phi=None, is_sar=False, syn_nosyn=False):
    frames = []
    if is_sar:
        if syn_nosyn:
            env = SynNoSynWrapper(gym.make(env_name), ica, pca, scaler, phi)
        else:
            env = SynergyWrapper(gym.make(env_name), ica, pca, scaler, phi)
    else:
        env = gym.make(env_name)

    if 'Leg' in env_name:
        camera = 'side_view'
    else:
        camera = 'front'
        
    for i,__ in tqdm(enumerate(range(episodes))):              
        env.reset()

        model = SAC.load(f'{name}_model_{env_name}_{seed}.zip')
        vec = VecNormalize.load(f'{name}_env_{env_name}_{seed}', DummyVecEnv([lambda: env]))
        
        rs = 0
        is_solved = []
        done = False
        while not done:
            o = vec.normalize_obs(env.get_obs())
            a, __ = model.predict(o, deterministic=determ)
            
            frame = env.sim.renderer.render_offscreen(width=640, height=480,camera_id=camera)
            frames.append(frame)

            next_o, r, done, info = env.step(a)
            is_solved.append(info['solved'])
            
            rs+=r
    env.close()
    skvideo.io.vwrite(f'{video_name}.mp4', np.asarray(frames),outputdict={"-pix_fmt": "yuv420p"})