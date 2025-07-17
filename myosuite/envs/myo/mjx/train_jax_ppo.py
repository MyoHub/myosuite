"""Train a PPO agent using brax."""

import functools
import time
import pickle
import jax
print(f"Current backend: {jax.default_backend()}")
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from myosuite.envs.myo.mjx import ppo_config

from myosuite.envs.myo.mjx import make, get_default_config
from mujoco_playground import wrapper

def main(env_name):
  """Run training and evaluation for the specified environment."""

  env, ppo_params, network_factory = load_env_and_network_factory(env_name)

  # Train the model
  make_inference_fn, params, _ = ppo.train(
      environment=env,
      progress_fn=progress,
      network_factory=network_factory,
      wrap_env_fn=wrapper.wrap_for_brax_training,
      num_eval_envs=ppo_params.pop("num_eval_envs"),
      **ppo_params,
  )

  print(f"Time to JIT compile: {times[1] - times[0]}")
  print(f"Time to train: {times[-1] - times[1]}")

  with open('playground_params.pickle', 'wb') as handle:
      pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_env_and_network_factory(env_name):
  env = make(env_name)
  ppo_params = dict(ppo_config)

  print(f"Training on environment:\n{env_name}")
  print(f"Environment Config:\n{get_default_config(env_name)}")
  print(f"PPO Training Parameters:\n{ppo_config}")

  if "network_factory" in ppo_params:
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **ppo_params.pop("network_factory")
    )
  else:
    network_factory = ppo_networks.make_ppo_networks

  return env, ppo_params, network_factory


times = [time.monotonic()]
# Progress function for logging
def progress(num_steps, metrics):
  times.append(time.monotonic())
  print(f"Step {num_steps} at {times[-1]}: reward={metrics['eval/episode_reward']:.3f}")


if __name__ == "__main__":
  main("MjxElbowPoseFixed-v0")
