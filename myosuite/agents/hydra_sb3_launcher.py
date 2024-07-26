""" =================================================
# Copyright (c) Facebook, Inc. and its affiliates
Authors  :: Cameron Berg (cameronberg@fb.com), Vikash Kumar (vikashplus@gmail.com), Vittorio Caggiano (caggiano@gmail.com)
================================================= """

"""
This is a launcher script for launching SAC training using hydra.
"""

import os
import time as timer
import hydra
from omegaconf import DictConfig, OmegaConf
from sb3_job_script import train_loop
from myosuite.utils import gym

# ===============================================================================
# Process Inputs and configure job
# ===============================================================================
@hydra.main(config_name="hydra_sb3_config", config_path="config")
def configure_jobs(job_data):
    print("========================================")
    print("Job Configuration")
    print("========================================")

    assert 'algorithm' in job_data.keys()
    
    assert any([job_data.algorithm == a for a in ['SAC', 'PPO']])
    
    job_data.alg_hyper_params = dict() if 'alg_hyper_params' not in job_data.keys() else job_data.alg_hyper_params

    with open('job_config.json', 'w') as fp:
        OmegaConf.save(config=job_data, f=fp.name)

    print(OmegaConf.to_yaml(job_data, resolve=True))
    train_loop(job_data)

if __name__ == "__main__":
    configure_jobs()