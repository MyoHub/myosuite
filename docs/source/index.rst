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
     - :doc:`quickstart_ml` — gym API, SB3 training, GPU backends, benchmarks
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


How to cite
-----------

.. code-block:: bibtex

   @article{MyoSuite2022,
      author =       {Vittorio, Caggiano AND Huawei, Wang AND Guillaume, Durandau AND Massimo, Sartori AND Vikash, Kumar},
      title =        {MyoSuite -- A contact-rich simulation suite for musculoskeletal motor control},
      publisher = {arXiv},
      year = {2022},
      howpublished = {\url{https://github.com/facebookresearch/myosuite}},
      doi = {10.48550/ARXIV.2205.13600},
      url = {https://arxiv.org/abs/2205.13600},
   }
