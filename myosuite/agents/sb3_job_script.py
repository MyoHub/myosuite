""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Cameron Berg (cameronberg@fb.com), Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

"""
This is a job script for running SB3 on myosuite tasks.
"""

import os
import json
import time as timer
import myosuite
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize
import torch
from omegaconf import OmegaConf

import functools
from in_callbacks import InfoCallback, FallbackCheckpoint, SaveSuccesses, EvalCallback

IS_WnB_enabled = False
try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    IS_WnB_enabled = True
except ImportError as e:
    pass 

def train_loop(job_data) -> None:
    
    config = {
            "policy_type": job_data.policy,
            "total_timesteps": job_data.total_timesteps,
            "env_name": job_data.env,
    }
    if IS_WnB_enabled:
        run = wandb.init(
            project="sb3_hand",
            config=config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )

    log = configure(f'results_{job_data.env}')
    # Create the vectorized environment and normalize ob
    env = make_vec_env(job_data.env, n_envs=job_data.n_env)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    eval_env = make_vec_env(job_data.env, n_envs=job_data.n_eval_env)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.)
    
    algo = job_data.algorithm
    if algo == 'PPO':
        # Load activation function from config
        policy_kwargs = OmegaConf.to_container(job_data.policy_kwargs, resolve=True)

        model = PPO(job_data.policy, env,  verbose=1,
                    learning_rate=job_data.learning_rate, 
                    batch_size=job_data.batch_size, 
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=f"wandb/{run.id}",
                    gamma=job_data.gamma, **job_data.alg_hyper_params)
    elif algo == 'SAC':
        model = SAC(job_data.policy, env, 
                    learning_rate=job_data.learning_rate, 
                    buffer_size=job_data.buffer_size, 
                    learning_starts=job_data.learning_starts, 
                    batch_size=job_data.batch_size, 
                    tau=job_data.tau, 
                    tensorboard_log=f"wandb/{run.id}",
                    gamma=job_data.gamma, **job_data.alg_hyper_params)
    
    

    if job_data.job_name =="checkpoint.pt":
        foldername = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"baseline_SB3/myoChal24/{job_data.env}")
        file_path = os.path.join(foldername, job_data.job_name)
        if os.path.isfile(file_path):
            print("Loading weights from checkpoint")
            model.policy.load_state_dict(torch.load(file_path))
        else:
            raise FileNotFoundError(f"No file found at the specified path: {file_path}. See https://github.com/MyoHub/myosuite/blob/dev/myosuite/agents/README.md to download one.")
    else:
        print("No checkpoint loaded, training starts.")

    if IS_WnB_enabled:
        callback = [WandbCallback(
                model_save_path=f"models/{run.id}",
                verbose=2,
            )]
    else:
        callback = []
    
    callback += [EvalCallback(job_data.eval_freq, eval_env)]
    callback += [InfoCallback()]
    callback += [FallbackCheckpoint(job_data.restore_checkpoint_freq)]
    callback += [CheckpointCallback(save_freq=job_data.save_freq, save_path=f'logs/',
                                            name_prefix='rl_models')]

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callback,
    )
    
    model.set_logger(log)

    model.save(f"{job_data.env}_"+algo+"_model")
    env.save(f'{job_data.env}_'+algo+'_env')

    if IS_WnB_enabled:
        run.finish()
