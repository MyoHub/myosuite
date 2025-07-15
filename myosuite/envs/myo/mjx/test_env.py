# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train a PPO agent using JAX on the specified environment."""

# pip install 'nvidia-cublas-cu12==12.9.0.13' # https://github.com/jax-ml/jax/issues/29042#issuecomment-2916978884

from datetime import datetime
import functools
import json
import os
import time
import warnings
import pickle
import h5py

import jax

from absl import app
from absl import flags
from absl import logging
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo
from etils import epath
from flax.training import orbax_utils


import jax.numpy as jp
from ml_collections import config_dict
from orbax import checkpoint as ocp
from tensorboardX import SummaryWriter
import wandb

import mujoco_playground
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import dm_control_suite_params
from mujoco_playground.config import locomotion_params
from mujoco_playground.config import manipulation_params

# from playground_elbow_v0 import default_config
# from playground_elbow_v0 import MjxElbow as PlaygroundElbow

# from playground_finger_v0 import default_config
# # from playground_finger_v0 import MjxFinger as PlaygroundElbow

from playground_handreach_v0 import MjxHand as PlaygroundElbow
from playground_handreach_v0 import default_config

# from playground_hand_v0 import MjxHand as PlaygroundElbow
# from playground_hand_v0 import default_config

def main(argv):
  """Run training and evaluation for the specified environment."""
  
  del argv
  print(f"Current backend: {jax.default_backend()}")
  registry.locomotion.register_environment("MyoElbow", PlaygroundElbow, default_config)
  env_cfg = default_config()
  env = registry.load("MyoElbow", config=env_cfg)
  env.reset(jax.random.PRNGKey(0))

  breakpoint()

if __name__ == "__main__":
  app.run(main)


# env_cfg = default_config()
# env = registry.load("MyoElbow", config_overrides=env_cfg)
# env.reset(jax.random.PRNGKey(0))