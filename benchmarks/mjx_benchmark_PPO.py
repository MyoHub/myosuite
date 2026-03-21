import os
import timeit
import functools

from ml_collections.config_dict import ConfigDict
import numpy as np

## for standard myosuite-MJX envs
from mujoco_playground import registry as pg_registry
from myosuite.envs.myo.mjx import get_default_config, make as mjx_make
from myosuite.envs.myo.mjx.train_jax_ppo import progress
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from myosuite.envs.myo.mjx import ppo_config
from mujoco_playground import wrapper
import argparse

def measure_num_env_training_steps(meta_seed=0, env_name="MjxElbowPoseRandom-v0", impl="jax", num_timesteps=5_000_000, num_envs=8192):
  """Measure how number of envs influences execution time (total number of steps)"""
  os.makedirs("mjx_benchmark_PPO_checkpoints", exist_ok=True)
  np_default_rng = np.random.default_rng(seed=meta_seed)

  env = mjx_make(env_name, config_overrides={"impl": impl})  # Overwrite with your env's name
  config = get_default_config(env_name)
  ppo_params = dict(ppo_config)

  ## override rl params
  ppo_params["num_timesteps"] = num_timesteps
  ppo_params["num_evals"] = 2
  if "num_envs" in ppo_params:
    del ppo_params["num_envs"]
  if "seed" in ppo_params:
    del ppo_params["seed"]

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

  print(f"Testing env num: {num_envs}")

  def run_benchmark():
    # Train the model
    make_inference_fn, params, _ = ppo.train(
        environment=env,
        num_envs=num_envs,
        episode_length=env._config.max_episode_steps,
        progress_fn=functools.partial(progress, log_to_wandb=False),
        network_factory=network_factory,
        save_checkpoint_path=os.path.join(os.path.abspath(f"mjx_benchmark_PPO_checkpoints/mjx_benchmark_PPO_{env_name}_{impl}_{num_envs}")),
        seed=np_default_rng.integers(0, 10000),  # Use a random seed for each run
        wrap_env_fn=wrapper.wrap_for_brax_training,
        **ppo_params,
    )

  results = timeit.repeat(run_benchmark, number=1, repeat=3)
  print(f"Results for {num_envs} envs: PPO training for {num_timesteps} total steps take {results} seconds.")
  res = np.mean(results)
  print(res)

  return results

def main():
  parser = argparse.ArgumentParser(description="Benchmark PPO training for MyoSuite MJX environments.")
  parser.add_argument("--env_name", type=str, default="MjxElbowPoseRandom-v0", help="Environment name")
  parser.add_argument("--impl", type=str, default="warp", help="Backend implementation")
  parser.add_argument("--num_envs", type=int, default=8192, help="Number of environments")
  args = parser.parse_args()

  results = {}
  print(f"MyoSuite env {args.env_name} -- Testing implementation: {args.impl} ({args.num_envs} envs)")
  results[f"{args.env_name}_{args.impl}_{args.num_envs}"] = measure_num_env_training_steps(
    meta_seed=0,
    env_name=args.env_name,
    impl=args.impl,
    num_envs=args.num_envs
  )
  print(results)

  np.save(f"mjx_benchmark_PPO_results_{args.env_name}_{args.impl}_{args.num_envs}.npy", results, allow_pickle=True)

if __name__ == "__main__":
  main()
