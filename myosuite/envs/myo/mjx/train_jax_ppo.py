"""Train a PPO agent using brax."""

import functools
import time
import jax

print(f"Current backend: {jax.default_backend()}")
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from myosuite.envs.myo.mjx import ppo_config

from myosuite.envs.myo.mjx import make, get_default_config
from mujoco_playground import wrapper

import wandb
import argparse


def main(env_name, impl, log_to_wandb):
    """Run training and evaluation for the specified environment."""

    env, ppo_params, network_factory = load_env_and_network_factory(env_name, impl)

    if log_to_wandb:
        wandb_run = wandb.init(project=env_name, config=ppo_params)

    # Train the model
    make_inference_fn, params, _ = ppo.train(
        environment=env,
        num_envs=env._config.num_envs,
        episode_length=env._config.max_episode_steps,
        progress_fn=functools.partial(progress, log_to_wandb=log_to_wandb),
        network_factory=network_factory,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        num_eval_envs=ppo_params.pop("num_eval_envs"),
        **ppo_params,
    )

    print(f"Time to JIT compile: {times[1] - times[0]}")
    print(f"Time to train: {times[-1] - times[1]}")

def load_env_and_network_factory(env_name, impl):
    env = make(env_name, config_overrides={"impl": impl})
    config = get_default_config(env_name)
    ppo_params = dict(ppo_config)

    print(f"Training on environment:\n{env_name}")
    print(f"Using backend:\n{impl}")
    print(f"Environment Config:\n{config}")
    print(f"PPO Training Parameters:\n{ppo_config}")

    if "network_factory" in ppo_params:
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks, **ppo_params.pop("network_factory")
        )
    else:
        network_factory = ppo_networks.make_ppo_networks

    return env, ppo_params, network_factory


times = [time.monotonic()]
total_steps = [0]


# Progress function for logging
def progress(num_steps, metrics, log_to_wandb):
    times.append(time.monotonic())
    total_steps.append(num_steps)
    print(
        f"Step {num_steps} at {times[-1]}: reward={metrics['eval/episode_reward']:.3f}"
    )
    if log_to_wandb:
        wandb.log(metrics, step=num_steps)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train PPO agent with Brax")
    parser.add_argument(
        "--env_name",
        type=str,
        default="MjxFingerPoseRandom-v0",
    )
    parser.add_argument(
        "--impl",
        type=str,
        default="jax",
        help='Implementation to use: "jax" (MJX) or "warp" (MJWarp)',
    )
    parser.add_argument(
        "--log_to_wandb",
        action="store_true",
    )

    args = parser.parse_args()

    main(args.env_name, args.impl, args.log_to_wandb)
