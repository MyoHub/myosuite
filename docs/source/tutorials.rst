Tutorials
=========

Notebooks live in the repository ``tutorials/`` directory, not under
``docs/source/``. Index and install notes: ``tutorials/README.md``.

Notebooks are numbered ``<track>.<number>``; extra files of ``X.Y`` live in ``tutorials/files/X.Y/``.

1. **Basics** — ``1.1_Get_Started`` (create an env and step it), ``1.2_Load_Policy`` (run an mjlab or SB3 checkpoint)
2. **Training** — ``2.1_Train_SB3_Policy`` (PPO, CPU), ``2.2_Train_MjLab_Policy`` (GPU), ``2.3_SAR``, ``2.4_DEP_RL``, ``2.5_MyoReflex_Walk``
3. **Analysis** — ``3.1_Analyse_Movements``, ``3.2_Inverse_Kinematics``, ``3.3_Inverse_Dynamics``, ``3.4_Computed_Muscle_Control``, ``3.5_Playback_Mot_File``
4. **Modelling and conditions** — ``4.1_Move_Hand_Fingers``, ``4.2_Fatigue_Modeling``, ``4.3_Modular_Task_Config``
5. **MuscleMimic** — ``5.1_Fullbody_Load_Policy``, ``5.2_Fullbody_Train_Policy``, ``5.3_Fullbody_Train_MjLab_Policy``, ``5.4_MuscleMimic_Directional_Locomotion``, ``5.5_MuscleMimic_SAR``

Muscle-condition IDs:

* Fatigue: ``myoFatiElbowPose1D6MRandom-v0`` (not ``…Fatigue…``)
* Sarcopenia: ``myoSarcElbowPose1D6MRandom-v0``
* Tendon transfer (hands): ``myoReafHandKeyTurnFixed-v0``

GPU training (Linux + CUDA)::

   python scripts/train_mjlab.py myoElbowPose1D6MRandom-v0 --render onscreen --env.scene.num-envs 1024

Replace "onscreen" with "offscreen" when running on a remote, headless machine.
Walk-through: ``tutorials/2.2_Train_MjLab_Policy.ipynb``.

New CPU tasks subclass :class:`~myosuite.envs.gymnasium_env.MyoGymnasiumEnv`
and register with ``registry.register_env`` — see
``docs/wiki/adding-a-new-task.md`` and :doc:`architecture`.
