Welcome to MyoSuite's documentation!
=====================================

`MyoSuite <https://sites.google.com/view/myosuite>`_  is a collection of musculoskeletal environments and tasks simulated with the `MuJoCo <http://www.mujoco.org/>`_ physics engine and wrapped in the OpenAI ``gym`` API to enable the application of Machine Learning to bio-mechanic control problems.

Check our `github repository <https://github.com/MyoHub/myosuite>`__ for more technical details.

Our paper can be found at: `https://arxiv.org/abs/2205.13600 <https://arxiv.org/abs/2205.13600>`__

Users are expected to have a foundational understanding of Reinforcement Learning and to review the `OpenAI Gym API <https://gymnasium.farama.org/>`__ to have the best experience with Myosuite.

.. note::

   This project is under active development.


Main features
============================================
   * Bio-mechanic based simulation of musculoskeletal
   * Task-driven controller for motion synthesis



.. toctree::
   :maxdepth: 1
   :caption: Get started

   install
   tutorials

.. toctree::
   :maxdepth: 1
   :caption: Advanced Features

   suite
   

.. toctree::
   :maxdepth: 1
   :caption: Documentations

   api

.. toctree::
   :maxdepth: 1
   :caption: Projects with Myosuite

   projects
   baselines 


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
      year =         {2022}
      doi = {10.48550/ARXIV.2205.13600},
      url = {https://arxiv.org/abs/2205.13600},
   }
