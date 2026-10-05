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

Walkthrough: ``tutorials/2.1_Train_SB3_Policy.ipynb``.

From the command line (the CPU counterpart of ``scripts/train_mjlab.py``)::

   python scripts/train_sb3.py myoElbowPose1D6MRandom-v0 --timesteps 500000

It supports ``--algo ppo|sac|td3``, ``--n-envs``, ``--normalize`` and ``--tensorboard``, and prints the
deterministic success rate after training.


Training (GPU)
--------------

Same ``env_id`` as CPU::

   python scripts/train_mjlab.py myoElbowPose1D6MRandom-v0 \
       --agent.max-iterations 1000 --env.scene.num-envs 1024

Add ``--video True`` to record videos during training (it renders offscreen, so it also works on a remote, headless machine).
The flag ``--agent.max-iterations`` sets the number of PPO update iterations (default is
task-specific); see ``--help`` for the full flag list, including
``--agent.num-steps-per-env`` and ``--env.scene.num-envs``.

.. important::

   ``--env.scene.num-envs`` defaults to **1** if you don't pass it. PPO's batch size
   per update is ``num_envs * num_steps_per_env``, so training with the default
   collects only ``num_steps_per_env`` (24 by default) transitions per iteration — the reward
   curve is dominated by single-trajectory noise, the KL-adaptive learning-rate
   schedule sees noisy KL estimates and collapses toward its floor, and with little
   policy-gradient signal left to oppose it, the entropy bonus keeps inflating the
   action std over time instead of the policy converging. Always set
   ``--env.scene.num-envs`` explicitly — 1024–4096, depending on GPU memory and
   model size (larger musculoskeletal models need more memory per env).

By default a run stops early once the ``Episode_Metrics/success`` rate of an iteration
exceeds 95% (``--success-threshold``); a checkpoint of that iteration is stored first.
Pass ``--stop-on-success False`` to always train for ``--agent.max-iterations``. The
early stop only applies to tasks that log a ``success`` metric and to single-GPU runs.

Walk-through: ``tutorials/2.2_Train_MjLab_Policy.ipynb``.

Resuming a run
^^^^^^^^^^^^^^^

.. code-block:: bash

   python scripts/train_mjlab.py myoElbowPose1D6MRandom-v0 --agent.resume True \
       --agent.max-iterations 5000 --env.scene.num-envs 4096

``--agent.resume True`` continues training from a checkpoint of the same
``--agent.experiment-name`` (default: task-specific, e.g. ``myo_elbow_pose``),
by default the latest ``model_*.pt`` of the most recent run *of this env id* under
``logs/rsl_rl/<experiment_name>/``. Several env ids share an experiment name (e.g.
the 12 ``myo_elbow_pose`` envs, whose Exo members have 7 actuators instead of 6), so
runs whose ``params/env.yaml`` records another env id are skipped. Use
``--agent.load-run <regex>`` and ``--agent.load-checkpoint <regex>`` to pick a
specific run or checkpoint instead; a run named exactly is used even when it was
trained on another env id (e.g. to warm-start ``...Fixed`` from ``...Random``).

**Only the network weights and optimizer state are resumed.** The environment
config (``--env.*``, including ``--env.scene.num-envs``) is rebuilt fresh from
this invocation's CLI flags before the checkpoint loads — it is **not** read
back from the original run. Repeat every ``--env.*`` flag you used originally
(especially ``--env.scene.num-envs``) on every resumed run, or the run silently
falls back to the defaults above.

Success metric
^^^^^^^^^^^^^^^

Every mjlab env logs ``Episode_Metrics/success``: the env's standard "solved" flag
(0/1) on the **last step** of an episode, averaged over the finished episodes. Solved
means

* **pose** tasks: the joint errors (target minus current angle) have a norm below the
  task's ``pose_thd`` (rad; e.g. 0.7 for the hand numeral poses);
* **reach** tasks: every tip site is within the task's reach threshold of its target;
* **locomotion** (``myoLegWalk``, ``myoLegDirectional*``, terrain walks): upright and
  the planar COM velocity within 0.5 m/s of the commanded velocity (speed times heading);
* ``myoLegStandRandom``: the pelvis has reached its (relative) target position.

The training log uses the *sampled* policy (with exploration noise). For muscle tasks
the *deterministic* policy (mean action) can behave very differently, and it is what
you deploy, so ``--stop-on-success`` also requires the deterministic success rate to
exceed the threshold (measured on the first two episodes of each of 256 separate envs,
so envs that fail early do not count more often than the others).


Evaluating a policy
^^^^^^^^^^^^^^^^^^^^

``scripts/eval_mjlab_policy.py`` rolls a trained checkpoint out on either backend and
prints the number of episodes, the mean return and length, and the success rate:

.. code-block:: bash

   python scripts/eval_mjlab_policy.py myoElbowPose1D6MRandom-v0 \
       --checkpoint logs/rsl_rl/myo_elbow_pose/<run> --backend cpu     # or --backend mjlab
   # video of a grid of parallel envs (5 x 3 envs, 2 episodes each), offscreen (EGL):
   python scripts/eval_mjlab_policy.py myoElbowPose1D6MRandom-v0 \
       --checkpoint logs/rsl_rl/myo_elbow_pose/<run> --video out.mp4 --backend mjlab \
       --num-cols 5 --num-rows 3 --episodes-per-env 2

``--checkpoint`` is a ``model_<iter>.pt`` file or a run directory (its newest
checkpoint). By default the deterministic (mean-action) policy is used; add
``--stochastic`` to sample actions like the training log does. The video marks the
targets of reach tasks, colours the joint errors of pose tasks and draws the target
velocity of walking tasks as an arrow. ``--backend cpu`` requires the mjlab twin to have
the same observation as the CPU env (the torso exosuit twins do so through a
conversion that is still experimental, see :doc:`environments`); a checkpoint that does
not fit stops the script with a clear "expects N-d observations" error.


Default policies
^^^^^^^^^^^^^^^^^

A ready-made mjlab policy for many envs is hosted on
`myohub/myosuite-3-baselines <https://huggingface.co/myohub/myosuite-3-baselines>`_ on Hugging
Face (see `docs/baseline_checkpoints.md
<https://github.com/MyoHub/myosuite/blob/ms3/docs/baseline_checkpoints.md>`_ for their
deterministic success rates and caveats); a preview video of each is on the same page. Evaluate
one directly with ``scripts/eval_mjlab_policy.py <env_id> --backend cpu`` (no ``--checkpoint``
needed); ``tutorials/1.2_Load_Policy.ipynb`` finds and downloads them automatically. Some are
unconverged snapshots, and some envs have no default policy yet; that page lists which.

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
policy, or see ``tutorials/1.2_Load_Policy.ipynb`` (falls back to a random policy
when weights are missing).


Next steps
----------

* :doc:`tutorials` — notebook index
* :doc:`architecture` — CPU vs mjlab
* :doc:`quickstart_biomechanics` — kinematics and forces
