""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Cameron Berg (cameronberg@fb.com), Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

"""
This is a job script for running SAC on myosuite tasks.
"""

import os
import json
import time as timer
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from collections import deque as dq
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import numpy as np
import myosuite
import gym
import numpy as np
import pandas as pd

class SaveSuccesses(BaseCallback):
    """
    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, timesteps: int, check_freq: int, log_dir: str, env_name: str, verbose: int = 1):
        super(SaveSuccesses, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = log_dir
        self.success_buffer = dq(maxlen=200)
        self.success_results = np.zeros(timesteps)
        self.env_name = env_name
        
    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self.success_buffer.append(self.locals['infos'][0]['solved'])
        if len(self.success_buffer) > 0:
            self.success_results[self.n_calls-1] = np.mean(self.success_buffer)
            success = f'./successes_{self.env_name}'
            if os.path.isfile(success):
                os.remove(success)
            np.save(success, self.success_results)
        return True

def train_loop(job_data) -> None:
    
    if job_data.algorithm == 'SAC':
        print("========================================")
        print("Starting policy learning")
        print("========================================")

        ts = timer.time()
        
        # specify shape of actor and critic networks
        policy_kwargs = dict(net_arch=dict(pi=[400, 300], qf=[400, 300]))

        log = configure(f'results_{job_data.env}')
        # make the env
        env = gym.make(job_data.env)

        # wrap env in monitor object to see reward data
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])

        # Automatically normalize the input features
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
        env.reset()
        
        model = SAC(job_data.policy, env, learning_rate=job_data.learning_rate, buffer_size=job_data.buffer_size, learning_starts=job_data.learning_starts, batch_size=job_data.batch_size, tau=job_data.tau, gamma=job_data.gamma, **job_data.alg_hyper_params)
        
        model.set_logger(log)
    
        callback = SaveSuccesses(timesteps=job_data.total_timesteps, check_freq=50, env_name=job_data.env, log_dir='./')
        
        model.learn(total_timesteps=job_data.total_timesteps, callback=callback, log_interval=job_data.log_interval)
        model.save(f"{job_data.env}_model")
        env.save(f'{job_data.env}')
        
        print("========================================")
        print("Job Finished. Time taken = %f" % (timer.time()-ts))
        print("========================================")

    else:
        NotImplementedError("This is for SAC only, make sure your algorithm is specified as 'SAC'.")