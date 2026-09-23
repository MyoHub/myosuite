Tutorials
=========

Notebooks live in the repository ``tutorials/`` directory, not under
``docs/source/``. Index and install notes: ``tutorials/ReadMe.md``.

Start with:

1. ``tutorials/1_Get_Started.ipynb`` — create an env and step it
2. ``tutorials/4c_Train_SB_policy.ipynb`` — train PPO with Stable-Baselines3
3. ``tutorials/3_Analyse_movements.ipynb`` — kinematics and synergies

Muscle-condition IDs:

* Fatigue: ``myoFatiElbowPose1D6MRandom-v0`` (not ``…Fatigue…``)
* Sarcopenia: ``myoSarcElbowPose1D6MRandom-v0``
* Tendon transfer (hands): ``myoReafHandKeyTurnFixed-v0``

GPU training (Linux + CUDA)::

   python scripts/train_mjlab.py myoElbowPose1D6MRandom-v0 --render onscreen

Replace "onscreen" with "offscreen" when running on a remote, headless machine.
Walk-through: ``tutorials/directional_leg_gpu_training.py``.

New CPU tasks subclass :class:`~myosuite.envs.gymnasium_env.MyoGymnasiumEnv`
and register with ``registry.register_env`` — see
``docs/wiki/adding-a-new-task.md`` and :doc:`architecture`.
