Machine Learning Quick Start
=============================

CPU training uses Gymnasium + Stable-Baselines3. GPU training uses the same
``env_id`` on mjlab (MuJoCo Warp + RSL-RL).

.. contents:: Contents
   :local:
   :depth: 2


Installation
------------

.. code-block:: bash

   pip install -e ".[rl]"
   # GPU (Linux + CUDA): pip install -e ".[mjlab]"

See :doc:`install` for matching the torch build to your driver's CUDA version.


Environment API
---------------

.. code-block:: python

   import gymnasium as gym
   import myosuite  # registers environments

   env = gym.make("myoElbowPose1D6MRandom-v0")
   obs, info = env.reset(seed=42)

   for _ in range(1000):
       action = env.action_space.sample()
       obs, reward, terminated, truncated, info = env.step(action)
       if terminated or truncated:
           obs, info = env.reset()

   env.close()

Muscle actions are a continuous ``Box``, typically ``[-1, 1]``, mapped internally
to ``[0, 1]`` excitation. ``info["obs_dict"]`` and ``info["rwd_dict"]`` hold the
named breakdowns.


Training (CPU)
--------------

.. code-block:: python

   from stable_baselines3 import PPO
   import gymnasium as gym
   import myosuite

   env = gym.make("myoElbowPose1D6MRandom-v0")
   model = PPO("MlpPolicy", env, verbose=1)
   model.learn(total_timesteps=100_000)
   model.save("ppo_elbow_pose")

Walkthrough: ``tutorials/4c_Train_SB_policy.ipynb``.


Training (GPU)
--------------

Same ``env_id`` as CPU::

   python scripts/train_mjlab.py myoElbowPose1D6MRandom-v0 --render onscreen \
       --agent.max-iterations 1000 --env.scene.num-envs 2048

Replace "onscreen" with "offscreen" when running on a remote, headless machine.
The flag ``--agent.max-iterations`` sets the number of PPO update iterations (default is
task-specific); see ``--help`` for the full flag list, including
``--agent.num-steps-per-env`` and ``--env.scene.num-envs``.

.. important::

   ``--env.scene.num-envs`` defaults to **1** if you don't pass it. PPO's batch size
   per update is ``num_envs * num_steps_per_env``, so training with the default
   collects only ``num_steps_per_env`` (24) transitions per iteration — the reward
   curve is dominated by single-trajectory noise, the KL-adaptive learning-rate
   schedule sees noisy KL estimates and collapses toward its floor, and with little
   policy-gradient signal left to oppose it, the entropy bonus keeps inflating the
   action std over time instead of the policy converging. Always set
   ``--env.scene.num-envs`` explicitly — 1024–4096, depending on GPU memory and
   model size (larger musculoskeletal models need more memory per env).

Walk-through: ``tutorials/directional_leg_gpu_training.py``.

Resuming a run
^^^^^^^^^^^^^^^

.. code-block:: bash

   python scripts/train_mjlab.py myoElbowPose1D6MRandom-v0 --agent.resume True \
       --agent.max-iterations 5000 --env.scene.num-envs 2048

``--agent.resume True`` continues training from a checkpoint of the same
``--agent.experiment-name`` (default: task-specific, e.g. ``myo_elbow_pose``),
by default the most recent run's latest ``model_*.pt`` under
``logs/rsl_rl/<experiment_name>/``. Use ``--agent.load-run <regex>`` and
``--agent.load-checkpoint <regex>`` to pick a specific run or checkpoint instead.

**Only the network weights and optimizer state are resumed.** The environment
config (``--env.*``, including ``--env.scene.num-envs``) is rebuilt fresh from
this invocation's CLI flags before the checkpoint loads — it is **not** read
back from the original run. Repeat every ``--env.*`` flag you used originally
(especially ``--env.scene.num-envs``) on every resumed run, or the run silently
falls back to the defaults above.

An MJX (JAX) backend also exists (``pip install -e ".[mjx]"``). It is
experimental — prefer mjlab for new GPU work.


Environments
------------

List registered CPU IDs::

   python -c "import myosuite; print('\n'.join(myosuite.myosuite_env_suite))"

Starting points:

.. code-block:: python

   import gymnasium as gym
   import myosuite

   gym.make("myoElbowPose1D6MRandom-v0")
   gym.make("myoHandPoseRandom-v0")
   gym.make("myoLegWalk-v0")
   gym.make("myoChallengeBaodingP2-v1")
   gym.make("myoSarcElbowPose1D6MRandom-v0")
   gym.make("myoFatiElbowPose1D6MFixed-v0")

Pathological prefixes: ``myoSarc…`` (sarcopenia), ``myoFati…`` (fatigue),
``myoReaf…`` (tendon transfer, hands). See :doc:`environments`.

Pretrained NPG trees under ``myosuite/agents/`` are not shipped. Train your own
policy, or see ``tutorials/2_Load_policy.ipynb`` (falls back to a random policy
when weights are missing).


Next steps
----------

* :doc:`tutorials` — notebook index
* :doc:`architecture` — CPU vs mjlab
* :doc:`quickstart_biomechanics` — kinematics and forces
