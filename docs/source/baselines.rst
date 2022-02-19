RL Baselines
============

.. _baselines:


Installation
~~~~~~~~~~~~

1. We use `mjrl <https://github.com/aravindr93/mjrl>`_ for our baselines. Please install it with ``python -m pip install git+https://github.com/aravindr93/mjrl.git``
2. install hydra ``pip install hydra-core --upgrade``
3. install `submitit <https://github.com/facebookincubator/submitit>`_ launcher hydra plugin to launch jobs on cluster/ local ``pip install hydra-submitit-launcher --upgrade``

.. code-block:: bash

    pip install git+https://github.com/aravindr93/mjrl.git
    pip install hydra-core --upgrade
    pip install hydra-submitit-launcher --upgrade
    pip install submitit

Launch training
~~~~~~~~~~~~~~~~

1. Get commands to lunch training by running `train_myosuite.sh` located in the `myosuite/agents` folder:

.. code-block:: bash

    sh train_myosuite.sh myo         # runs natively
    sh train_myosuite.sh myo local   # use local launcher
    sh train_myosuite.sh myo slurm   # use slurm launcher

2. Further customize the prompts from the previous step and execute.
