import os

import gdown
import torch
import yaml
from stable_baselines3 import PPO
import os



def load_baseline(environment):
    identifier = (
        environment.env_name
        if hasattr(environment, "env_name")
        else str(environment)
    )
    if "myoChallengeBimanual" in identifier:
        print("Load Bimanual Baseline")
        return load_baseline_bimanual(environment)


def load_baseline_bimanual(environment):
    modelurl = (
        "https://drive.google.com/uc?id=1e7bj2Lrk50FU7nSRnIDg2JmYjGLBNChs"
    )
    configurl = (
        "https://drive.google.com/uc?id=1vJ8WxzU49SxqYpB6JocixFJCZ4-EfrYj"
    )
    cwd = os.path.dirname(os.path.realpath(__file__))
    foldername = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../agents/baselines_SB3/myoChal24/bimanual")
    
    if not os.path.exists(foldername):
        os.makedirs(foldername)
        os.makedirs(os.path.join(foldername, "checkpoints"))
    modelpath = os.path.join(foldername, "checkpoints/step_10000000.pt")
    configpath = os.path.join(foldername, "model_config.yaml")
    if not os.path.exists(modelpath):
        gdown.download(modelurl, modelpath, quiet=False)
        gdown.download(configurl, configpath, quiet=False)
    return load(foldername, environment)


# Function to dynamically load activation function from a string
def load_activation_fn(name):
    components = name.split('.')
    mod = __import__('.'.join(components[:-1]), fromlist=[components[-1]])
    return getattr(mod, components[-1])

# Load configurations from YAML file
def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

# Initialize and return the PPO model based on configuration and environment
def load(foldername, env):
    config_path = os.path.join(foldername, "model_config.yaml")
    weights_path = os.path.join(foldername, "checkpoints/step_10000000.pt")
    
    # Load config
    config = load_config(config_path)

    # Load activation function from config
    policy_kwargs_config = config['ppo_args']['policy_kwargs']
    activation_fn = load_activation_fn(policy_kwargs_config['activation_fn'])

    # Policy kwargs setup
    policy_kwargs = {
        'activation_fn': activation_fn,
        'net_arch': {
            'pi': policy_kwargs_config['net_arch']['pi'],
            'vf': policy_kwargs_config['net_arch']['vf']
        }
    }

    # Create PPO model with specified configurations
    model = PPO("MlpPolicy", env, verbose=1,
                policy_kwargs=policy_kwargs,
                learning_rate=config['ppo_args']['learning_rate'],
                gamma=config['ppo_args']['gamma'],
                n_steps=config['ppo_args']['n_steps'],
                batch_size=config['ppo_args']['batch_size'],
                n_epochs=config['ppo_args']['n_epochs'],
                clip_range=config['ppo_args']['clip_range'],
                ent_coef=config['ppo_args']['ent_coeff'])

    # Load the model weights
    model.policy.load_state_dict(torch.load(weights_path))

    return model