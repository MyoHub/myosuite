Welcome to MyoSuite's documentation!
=====================================

`MyoSuite <https://sites.google.com/view/myosuite>`_ is a collection of musculoskeletal
environments and tasks simulated with the `MuJoCo <https://mujoco.org/>`_ physics engine.
It serves researchers and practitioners across biomechanics, neuroscience, machine learning,
sports medicine, and physical rehabilitation.

`GitHub <https://github.com/MyoHub/myosuite>`__ |
`Paper (arXiv) <https://arxiv.org/abs/2205.13600>`__ |
`Slack <https://join.slack.com/t/myosuite/shared_invite/zt-1zkpw2zzk-NhVhVlSDxhoMHbzROD8gMA>`__

.. note::

   This project is under active development.

What's new in MyoSuite 3
-------------------------

MyoSuite 3 brings the whole suite to fast, scalable training while keeping the simple
interface you know:

* **One task, several backends.** The same ``env_id`` runs on your **CPU** through the
  standard `Gymnasium <https://gymnasium.farama.org/>`_ interface (to explore, debug and
  replay policies, or to train with libraries such as Stable-Baselines3), on **mjlab** for
  massively parallel GPU training, and on an **experimental MJX** (JAX) path. See :doc:`quickstart_ml`.
* **MuscleMimic support.** Run, evaluate and train full-body and bimanual **MuscleMimic**
  policies, with ready-to-use checkpoints and motion datasets.
* **The complete MyoChallenge suite** as Gymnasium environments: Baoding, Bimanual, Chase
  Tag, Die Reorient, OSL Run, Relocate, Soccer and Table Tennis.
* **Much faster learning.** Train with thousands of environments in parallel on a single
  GPU, then replay the policy on the CPU.
* **New tasks in a few lines.** Describe a task with a compact spec and reuse the shared
  observation, reward and model-building blocks instead of writing an environment class.
* **Some existing environments changed.** Observations are no longer clipped and are read
  after a fresh forward step (55 env ids), the ``motorFinger*`` envs have 4x stronger motors,
  the Random finger-reach tasks now sample only targets the fingertip can reach, and several
  reset and seed behaviours were corrected. Policies trained with MyoSuite 2.x or earlier
  snapshots may need retraining; the repository's ``CHANGELOG.md`` lists the changes.
* **Ready to use.** Default trained policies with evaluation videos (see
  :doc:`baselines`), plus updated tutorials from the first rollout to GPU training and
  MuscleMimic (:doc:`tutorials`). The repository's ``CHANGELOG.md`` lists everything that
  changed since v2.12.

Choose your path
-----------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - I am a…
     - Start here
   * - **Biomechanist**
     - :doc:`quickstart_biomechanics` — kinematics, muscle forces, inverse dynamics, OpenSim
   * - **Neuroscientist**
     - :doc:`quickstart_neuroscience` — proprioception, reflex controllers, fatigue
   * - **ML / RL Researcher**
     - :doc:`quickstart_ml` — Gymnasium API, SB3, mjlab GPU training
   * - **Sports / Rehab Clinician**
     - :doc:`quickstart_rehabilitation` — pathological conditions, clinical metrics


.. toctree::
   :maxdepth: 1
   :caption: Quick Start by Audience

   quickstart_biomechanics
   quickstart_neuroscience
   quickstart_ml
   quickstart_rehabilitation

.. toctree::
   :maxdepth: 1
   :caption: Installation & Tutorials

   install
   tutorials

.. toctree::
   :maxdepth: 1
   :caption: Reference

   architecture
   environments
   model_builder
   backend_parity
   fatigue_validation

.. toctree::
   :maxdepth: 1
   :caption: Advanced Features

   suite

.. toctree::
   :maxdepth: 1
   :caption: Projects with MyoSuite

   projects
   baselines
   challenge-doc
   challenge-doc2025

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: References

   publications
   references


How to cite
-----------

If you use MyoSuite, please cite the MyoSuite paper:

.. code-block:: bibtex

   @inproceedings{Caggiano2022MyoSuite,
      title     = {{MyoSuite} -- A contact-rich simulation suite for musculoskeletal motor control},
      author    = {Caggiano, Vittorio and Wang, Huawei and Durandau, Guillaume and Sartori, Massimo and Kumar, Vikash},
      booktitle = {Learning for Dynamics and Control (L4DC)},
      year      = {2022},
      doi       = {10.48550/ARXIV.2205.13600},
      url       = {https://arxiv.org/abs/2205.13600},
   }

For MyoSuite 3, cite it as software:

.. code-block:: bibtex

   @misc{MyoSuite2026,
      title        = {{MyoSuite} 3.0 -- A multimodal platform for efficient and scalable musculoskeletal motor control},
      author       = {Caggiano, Vittorio and Hodossy, Balint and Fischer, Florian and Wang, Cheryl and {MyoSuite Team}},
      year         = {2026},
      howpublished = {\url{https://github.com/myohub/myosuite}},
   }

For the full-body MuscleMimic models, checkpoints, retargeted motions and tutorials 5.1-5.5, also cite:

.. code-block:: bibtex

   @article{Li2026MuscleMimic,
      title   = {Towards Embodied AI with {MuscleMimic}: Unlocking full-body musculoskeletal motor learning at scale},
      author  = {Li, Chengkun and Wang, Cheryl and Ziliotto, Bianca and Simos, Merkourios and Kovecses, Jozsef and Durandau, Guillaume and Mathis, Alexander},
      journal = {arXiv preprint arXiv:2603.25544},
      year    = {2026},
   }

The motion data come from AMASS and its KIT subset, retargeted with GMR; the datasets are for
non-commercial research and ask you to cite AMASS and MuscleMimic. See :doc:`references` for these
and all other sources (models, muscle fatigue, controllers, motion data).
