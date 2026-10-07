References
==========

Papers, datasets and software that MyoSuite builds on, grouped by topic. The page
:doc:`publications` lists the MyoSuite papers; this page also covers the sources behind the
models, the muscle conditions, the controllers and the motion data. If you use one of these
parts of MyoSuite, please cite the corresponding work.

.. contents::
   :local:
   :depth: 1

MyoSuite and the MuscleMimic integration
----------------------------------------

* Caggiano, V., Wang, H., Durandau, G., Sartori, M., Kumar, V. (2022). MyoSuite -- A contact-rich
  simulation suite for musculoskeletal motor control. *Learning for Dynamics and Control (L4DC)*.
  https://arxiv.org/abs/2205.13600
* Wang, H., Caggiano, V., Durandau, G., Sartori, M., Kumar, V. (2022). MyoSim: Fast and
  physiologically realistic MuJoCo models for musculoskeletal and exoskeletal studies. *IEEE
  International Conference on Robotics and Automation (ICRA)*.
  https://ieeexplore.ieee.org/abstract/document/9811684
* Caggiano, V., Dasari, S., Kumar, V. (2023). MyoDex: A generalizable prior for dexterous
  manipulation. *International Conference on Machine Learning (ICML)*.
  https://arxiv.org/abs/2309.03130
* Berg, C., Caggiano, V., Kumar, V. (2023). SAR: Generalization of physiological dexterity via
  synergistic action representation. *Robotics: Science and Systems (RSS)*.
  https://arxiv.org/abs/2307.03716
* Caggiano, V. et al. (2023). MyoChallenge 2022: Learning contact-rich manipulation using a
  musculoskeletal hand. *Proceedings of the NeurIPS 2022 Competitions Track*, PMLR 220, 233-250.
  https://proceedings.mlr.press/v220/caggiano23a.html
* Li, C., Wang, C., Ziliotto, B., Simos, M., Kovecses, J., Durandau, G., Mathis, A. (2026). Towards
  Embodied AI with MuscleMimic: Unlocking full-body musculoskeletal motor learning at scale.
  arXiv:2603.25544. https://arxiv.org/abs/2603.25544

Work built on MyoSuite
----------------------

* Schumacher, P., Haeufle, D.F.B., Büchler, D., Schmitt, S., Martius, G. (2023). DEP-RL: Embodied
  exploration for reinforcement learning in overactuated and musculoskeletal systems.
  *International Conference on Learning Representations (ICLR)*.
  https://openreview.net/forum?id=C-xa_D3oTj6
* Chiappa, A.S., Marin Vargas, A., Huang, A.Z., Mathis, A. (2023). Latent exploration for
  reinforcement learning. *Advances in Neural Information Processing Systems (NeurIPS)*.
  https://arxiv.org/abs/2305.20065
* Hodossy, B.K., Crotti, M., Pace, A., Catalano, M.G., Aszmann, O.C., Bicchi, A., Farina, D. (2026).
  Towards a Virtual Gait Lab: Testing Prosthetics with Dynamically Simulated Users. *IEEE Transactions
  on Medical Robotics and Bionics*. https://ieeexplore.ieee.org/abstract/document/11606461
* Bhattarai, A., Selder, H., Fischer, F., Fleig, A., Kristensson, P.O. (2026). MyoInteract: A
  framework for fast prototyping of biomechanical HCI tasks using reinforcement learning. *ACM
  Designing Interactive Systems Conference (DIS)*. https://arxiv.org/abs/2602.15245,
  https://doi.org/10.1145/3800645.3812899 (builds on MyoSuite; trains and evaluates muscle-actuated
  simulated users from a GUI)

Motion data and retargeting (MuscleMimic, tutorials 5.1-5.5)
------------------------------------------------------------

The MuscleMimic motions are derived from the AMASS archive, whose KIT subset provides the
locomotion clips (for example ``KIT/314/walking_medium09_poses``), and retargeted to the MyoFullBody
and MyoBimanualArm models with GMR. The retargeted datasets are non-commercial research data
under the AMASS license; the dataset cards ask you to cite AMASS and the MuscleMimic paper.

* Mahmood, N., Ghorbani, N., Troje, N.F., Pons-Moll, G., Black, M.J. (2019). AMASS: Archive of
  motion capture as surface shapes. *International Conference on Computer Vision (ICCV)*, 5442-5451.
  https://arxiv.org/abs/1904.03278 (license and data: https://amass.is.tue.mpg.de)
* Mandery, C., Terlemez, Ö., Do, M., Vahrenkamp, N., Asfour, T. (2015). The KIT whole-body human
  motion database. *International Conference on Advanced Robotics (ICAR)*, 329-336.
* Mandery, C., Terlemez, Ö., Do, M., Vahrenkamp, N., Asfour, T. (2016). Unifying representations
  and large-scale whole-body motion databases for studying human motion. *IEEE Transactions on
  Robotics* 32(4), 796-809. https://doi.org/10.1109/TRO.2016.2572685
  (AMASS asks to cite one of the two KIT papers when KIT motions are used; both are listed here.)
* Araújo, J.P., Ze, Y., Xu, P., Wu, J., Liu, C.K. (2025). Retargeting matters: General motion
  retargeting for humanoid motion tracking (GMR). arXiv:2510.02252.
  https://arxiv.org/abs/2510.02252 (MuscleMimic uses the fork https://github.com/amathislab/gmr_plus)
* Simos, M., Chiappa, A.S., Mathis, A. (2025). KINESIS: Motion imitation for human musculoskeletal
  locomotion. arXiv:2503.14637. https://arxiv.org/abs/2503.14637 (source of the KIT training and
  testing splits ``KIT_KINESIS_TRAINING_MOTIONS`` / ``KIT_KINESIS_TESTING_MOTIONS``)
* ULTRA-MoCap v1 upper-limb dataset (Fritsche, O. et al.), figshare,
  https://doi.org/10.6084/m9.figshare.28751156.v1, CC BY 4.0. Source of the 208 ULTRA-MoCap
  trajectories of the bimanual retargeted dataset.

Models and biomechanics
-----------------------

* Delp, S.L. et al. (2007). OpenSim: Open-source software to create and analyze dynamic simulations
  of movement. *IEEE Transactions on Biomedical Engineering* 54(11), 1940-1950.
  https://doi.org/10.1109/TBME.2007.901024
* Seth, A. et al. (2018). OpenSim: Simulating musculoskeletal dynamics and neuromuscular control to
  study human and animal movement. *PLoS Computational Biology* 14(7), e1006223.
  https://doi.org/10.1371/journal.pcbi.1006223
* Rajagopal, A. et al. (2016). Full-body musculoskeletal model for muscle-driven simulation of human
  gait. *IEEE Transactions on Biomedical Engineering* 63(10), 2068-2079.
  https://ieeexplore.ieee.org/document/7505900
* Xu, Z., Kumar, V., Matsuoka, Y., Todorov, E. (2012). Design of an anthropomorphic robotic finger
  system with biomimetic artificial joints. *IEEE RAS & EMBS International Conference on Biomedical
  Robotics and Biomechatronics (BioRob)*. https://doi.org/10.1109/BioRob.2012.6290710
* Thelen, D.G., Anderson, F.C., Delp, S.L. (2003). Generating dynamic simulations of movement using
  computed muscle control. *Journal of Biomechanics* 36(3), 321-328.
  https://doi.org/10.1016/S0021-9290(02)00432-3 (tutorial 3.4)
* Todorov, E., Erez, T., Tassa, Y. (2012). MuJoCo: A physics engine for model-based control. *IEEE/RSJ
  International Conference on Intelligent Robots and Systems (IROS)*, 5026-5033.
  https://doi.org/10.1109/IROS.2012.6386109

Muscle fatigue (``myoFati*``, tutorial 4.2)
-------------------------------------------

The model and its parameters are validated on the page :doc:`fatigue_validation`, which also lists
the endurance-time sources.

* Xia, T., Frey-Law, L.A. (2008). A theoretical approach for modeling peripheral muscle fatigue and
  recovery. *Journal of Biomechanics* 41(14), 3046-3052. https://doi.org/10.1016/j.jbiomech.2008.07.013
* Frey-Law, L.A., Avin, K.G. (2010). Endurance time is joint-specific: A modelling and meta-analysis
  investigation. *Ergonomics* 53(1), 109-129. https://doi.org/10.1080/00140130903389068
* Frey-Law, L.A., Looft, J.M., Heitsman, J. (2012). A three-compartment muscle fatigue model
  accurately predicts joint-specific maximum endurance times for sustained isometric tasks.
  *Journal of Biomechanics* 45(10), 1803-1808. https://doi.org/10.1016/j.jbiomech.2012.04.018
* Looft, J.M., Herkert, N., Frey-Law, L. (2018). Modification of a three-compartment muscle fatigue
  model to predict peak torque decline during intermittent tasks. *Journal of Biomechanics* 77,
  16-25. https://doi.org/10.1016/j.jbiomech.2018.06.005
* Looft, J.M., Frey-Law, L.A. (2020). Adapting a fatigue model for shoulder flexion fatigue:
  Enhancing recovery rate during intermittent rest intervals. *Journal of Biomechanics* 106, 109762.
  https://doi.org/10.1016/j.jbiomech.2020.109762
* Rakshit, R., Xiang, Y., Yang, J. (2021). Functional muscle group- and sex-specific parameters for a
  three-compartment controller muscle fatigue model applied to isometric contractions. *Journal of
  Biomechanics* 127, 110695. https://doi.org/10.1016/j.jbiomech.2021.110695
* Cheema, N., Frey-Law, L.A., Naderi, K., Lehtinen, J., Slusallek, P., Hämäläinen, P. (2020).
  Predicting mid-air interaction movements and fatigue using deep reinforcement learning. *ACM CHI
  Conference on Human Factors in Computing Systems*. https://doi.org/10.1145/3313831.3376701
  (implementation of the fatigue model in the User-in-the-Box framework, which the CPU model follows)
* Daly, M., Vidt, M.E., Eggebeen, J.D., Simpson, W.G., Miller, M.E., Marsh, A.P., Saul, K.R. (2013).
  Upper extremity muscle volumes and functional strength after resistance training in older adults.
  *Journal of Aging and Physical Activity* 21(2), 186-207. https://doi.org/10.1123/japa.21.2.186
  (functional muscle groups of the shoulder, wrist and elbow)

Motor control, locomotion and synergies
---------------------------------------

* Harris, C.M., Wolpert, D.M. (1998). Signal-dependent noise determines motor planning. *Nature*
  394, 780-784. https://doi.org/10.1038/29528 (signal-dependent motor noise)
* van Beers, R.J., Haggard, P., Wolpert, D.M. (2004). The role of execution noise in movement
  variability. *Journal of Neurophysiology* 91(2), 1050-1063. https://doi.org/10.1152/jn.00652.2003
  (default motor-noise levels, ``MotorNoiseCfg.van_beers_2004()``)
* Song, S., Geyer, H. (2015). A neural circuitry that emphasizes spinal feedback generates diverse
  behaviours of human locomotion. *The Journal of Physiology* 593(16), 3493-3511.
  https://doi.org/10.1113/JP270228 (reflex walking controller of tutorial 2.4)
* Tresch, M.C., Cheung, V.C.K., d'Avella, A. (2006). Matrix factorization algorithms for the
  identification of muscle synergies: Evaluation on simulated and experimental data sets. *Journal of
  Neurophysiology* 95(4), 2199-2212. https://doi.org/10.1152/jn.00222.2005 (tutorial 2.3)

Reinforcement learning and numerics
-----------------------------------

* Peng, X.B., Abbeel, P., Levine, S., van de Panne, M. (2018). DeepMimic: Example-guided deep
  reinforcement learning of physics-based character skills. *ACM Transactions on Graphics* 37(4),
  143. https://doi.org/10.1145/3197517.3201311 (multi-term tracking reward and reference state
  initialization of the Mimic envs)
* Pardo, F., Tavakoli, A., Levdik, V., Kormushev, P. (2018). Time limits in reinforcement learning.
  *International Conference on Machine Learning (ICML)*. https://arxiv.org/abs/1712.00378
  (truncation handling in ``tutorials/files/5.2/train_mimic.py``)
* Fischer, F., Bachinski, M., Klar, M., Fleig, A., Müller, J. (2021). Reinforcement learning control
  of a biomechanical model of the upper extremity. *Scientific Reports* 11, 14445.
  https://doi.org/10.1038/s41598-021-93760-1 (motor-noise levels 0.103 and 0.185)
* Ikkala, A., Fischer, F., Klar, M., Bachinski, M., Fleig, A., Howes, A., Hämäläinen, P., Müller, J.,
  Murray-Smith, R., Oulasvirta, A. (2022). Breathing life into biomechanical user models. *ACM
  Symposium on User Interface Software and Technology (UIST)*. https://doi.org/10.1145/3526113.3545689
  (User-in-the-Box; motor noise on the controls)
* Timmer, J., Koenig, M. (1995). On generating power law noise. *Astronomy & Astrophysics* 300,
  707-710. (colored exploration noise)
* Buss, S.R. (2004). Introduction to inverse kinematics with Jacobian transpose, pseudoinverse and
  damped least squares methods. Technical note (``myosuite.physics.inverse_kinematics``).
