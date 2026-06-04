Machine Learning Quick Start
=============================

This guide gets ML and RL researchers running experiments with MyoSuite in minutes.

.. contents:: Contents
   :local:
   :depth: 2

Installation
------------

Install the core package plus the RL extra (Stable-Baselines3):

.. code-block:: bash

   pip install -e ".[rl]"
   # or with uv:
   uv sync -p 3.10 --extra rl

Verify:

.. code-block:: bash

   python -m myosuite.tests.test_myo   # prints all registered env IDs


Environment API
---------------

MyoSuite environments follow the standard `Gymnasium <https://gymnasium.farama.org/>`_ API.
All environments return 5-tuple ``step()`` results (Gymnasium ≥ 0.26 style):

.. code-block:: python

   import gymnasium as gym
   import myosuite  # registers all environments

   env = gym.make('myoElbowPose1D6MRandom-v0')
   obs, info = env.reset(seed=42)

   for _ in range(1000):
       action = env.action_space.sample()  # random policy
       obs, reward, terminated, truncated, info = env.step(action)
       if terminated or truncated:
           obs, info = env.reset()

   env.close()

**Action space**: all environments expose a continuous ``Box`` action space.
For musculoskeletal envs the actions are normalised muscle excitations in ``[-1, 1]``;
the environment maps these to ``[0, 1]`` via a sigmoid internally.

**Observation space**: a flat ``Box`` vector. The exact contents depend on the
environment's ``obs_keys`` (joint positions, velocities, muscle activations, task
error, etc.).  Access the full dictionary per-step via ``info['obs_dict']``.

**Reward**: ``info['rwd_dict']`` exposes individual reward components
(``'pose'``, ``'bonus'``, ``'act_reg'``, ``'solved'``, etc.) for detailed analysis.


Training with Stable-Baselines3
--------------------------------

.. code-block:: python

   from stable_baselines3 import SAC
   from stable_baselines3.common.env_util import make_vec_env
   import myosuite

   # Single environment
   env = make_vec_env('myoElbowPose1D6MRandom-v0', n_envs=4)

   model = SAC(
       'MlpPolicy',
       env,
       verbose=1,
       learning_rate=3e-4,
       buffer_size=1_000_000,
       batch_size=256,
   )
   model.learn(total_timesteps=1_000_000)
   model.save("sac_elbow_pose")

See ``tutorials/4c_Train_SB_policy.ipynb`` for a complete walkthrough including
evaluation and rendering.


Reproducing Pre-trained Baselines
----------------------------------

Pre-trained policies are available under ``myosuite/agents/``.
Load and roll out a baseline:

.. code-block:: python

   from myosuite.utils.evaluate import load_policy_and_rollout
   # see tutorials/2_Load_policy.ipynb for the full workflow


Available Environments
-----------------------

MyoSuite registers **125+ environment IDs** across six musculoskeletal models
(myoFinger, myoElbow, myoHand, myoArm, myoLeg, myoTorso), each with fixed and
random-target variants, plus sarcopenia, fatigue, and reafferentation
pathological variants auto-generated for every base environment.

See :doc:`environments` for the complete annotated listing, organised by model
and task type, including challenge and GPU-accelerated (MJX / mjlab) IDs.

A few representative starting points:

.. code-block:: python

   import myosuite, gymnasium as gym

   gym.make("myoElbowPose1D6MRandom-v0")    # Elbow — 2 DoF, 6 muscles
   gym.make("myoHandPoseRandom-v0")          # Hand  — 23 DoF, 39 muscles
   gym.make("myoLegWalk-v0")                 # Leg   — 20 DoF, 80 muscles
   gym.make("myoChallengeBaodingP2-v1")      # Baoding balls challenge
   gym.make("myoSarcElbowPose1D6MRandom-v0") # Sarcopenia variant

Run ``python -m myosuite.tests.test_myo`` to print all IDs registered on your system.


GPU-Accelerated Training with MJX
----------------------------------

For large-scale experiments, use the MJX backend to run thousands of
environments in parallel on GPU via JAX/Brax:

.. code-block:: bash

   # CPU-only MJX stack
   pip install -e ".[mjx]"

   # Recommended for NVIDIA GPUs (CUDA 12 wheels):
   pip install -e ".[mjx-cuda]"
   # or with uv:
   uv sync -p 3.10 --extra mjx-cuda

.. code-block:: python

   from myosuite.envs.myo.backends.mjx import make
   from mujoco_playground import wrapper as pg_wrapper

   env = make("MjxElbowPoseRandom-v0")
   brax_env = pg_wrapper.wrap_for_brax_training(env)  # Brax-compatible

For a full benchmark comparing CPU, MJX (JAX), and MuJoCo Warp throughput:

.. code-block:: bash

   python benchmarks/sar_backends/run_benchmark.py --steps 50000

See ``tutorials/Walk_Backends_Demo.ipynb`` for an end-to-end walkthrough.

Training MJX envs with Brax PPO
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MyoSuite ships a lightweight helper around the upstream
`mujoco_playground train_jax_ppo.py <https://github.com/google-deepmind/mujoco_playground/blob/main/learning/train_jax_ppo.py>`_
script. It uses the same Brax PPO implementation but is pre-wired to
MyoSuite's MJX envs.

1. **Install the MJX + GPU stack** (see above):

   .. code-block:: bash

      # In a cloned myosuite4 repo
      uv sync -p 3.10 --extra mjx-cuda

   Verify that JAX sees your GPU:

   .. code-block:: bash

      uv run python -c "import jax; print(jax.devices())"

   On GPU machines you can optionally force CUDA:

   .. code-block:: bash

      export JAX_PLATFORMS=cuda
      export MUJOCO_GL=egl

2. **Run PPO training on an MJX env** (example: finger pose):

   .. code-block:: bash

      # Train Brax PPO on MjxFingerPoseRandom-v0
      uv run python -m myosuite.envs.myo.backends.mjx.train_jax_ppo

   By default this trains on ``MjxFingerPoseRandom-v0`` and writes
   a ``playground_params.pickle`` checkpoint plus Weights & Biases logs
   (W&B is installed via the ``[mjx]`` extra).

3. **Change the environment or training horizon** from Python:

   .. code-block:: python

      from myosuite.envs.myo import mjx as mjx_module
      from myosuite.envs.myo.backends.mjx import train_jax_ppo

      # Shorter smoke run (e.g. for debugging)
      mjx_module.ppo_config.num_timesteps = 200_000
      mjx_module.ppo_config.num_evals = 4

      # Train on an elbow MJX task instead of finger
      train_jax_ppo.main("MjxElbowPoseRandom-v0", render_evaluations=False)

   The :mod:`myosuite.envs.myo.backends.mjx` module exposes the shared
   ``ppo_config`` object used by the helper; adjusting its fields before
   calling :func:`train_jax_ppo.main` lets you control the number of
   timesteps, evaluation frequency, network sizes, and other PPO
   hyperparameters while keeping the Brax training loop unchanged.

For advanced configuration, refer to the original
``train_jax_ppo.py`` in `mujoco_playground` (linked above), which
demonstrates additional options such as domain randomisation, vision
observations, and TensorBoard / W&B logging flags.


Pathological Condition Variants
---------------------------------

Every base task has variants modelling physiological conditions.
These are useful for studying robustness and curriculum learning:

.. code-block:: python

   import myosuite, gymnasium as gym

   # Sarcopenia: muscles generate only 50 % of their peak force
   env = gym.make('myoSarcElbowPose1D6MFixed-v0')

   # Cumulative fatigue: sustained activation degrades muscle output
   env = gym.make('myoElbowPoseFatigue1D6MFixed-v0')

   # Tendon transfer (reafferentation) — altered muscle routing
   env = gym.make('myoElbowPoseReaff1D6MFixed-v0')


Next Steps
----------

* :doc:`quickstart_biomechanics` — extract raw simulation data for analysis
* :doc:`quickstart_neuroscience` — add custom controllers or sensory models
* :doc:`quickstart_rehabilitation` — clinical condition modelling
* ``tutorials/4_Train_policy.ipynb`` — DEPRL / reflex baseline training
* ``myosuite/agents/`` — pre-trained policy weights and training configs
