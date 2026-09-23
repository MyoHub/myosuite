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

   python scripts/train_mjlab.py myoElbowPose1D6MRandom-v0 --render onscreen

Replace "onscreen" with "offscreen" when running on a remote, headless machine.
Walk-through: ``tutorials/directional_leg_gpu_training.py``.

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
