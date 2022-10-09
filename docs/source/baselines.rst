RL Baselines
============

.. _baselines:


For ease of getting started, MyoSuite comes prepackaged with a set of pre-trained baselines.
See `here <https://github.com/facebookresearch/myosuite/tree/main/myosuite/agents>`_ for our complete set of baselines.


Try baselines
~~~~~~~~~~~~~~~~

We use model-free on-policy algorithm - Natural Policy Gradient (from `mjrl <https://github.com/aravindr93/mjrl>`_) as our baselines.

1. install `mjrl` with ``python -m pip install git+https://github.com/aravindr93/mjrl.git``
2. you can try our baselines using ``python myosuite/utils/examine_env.py --env_name <env_name> --policy_path <policy_path>```
3. alternatively, follow our `load policy tutorial <https://github.com/facebookresearch/myosuite/blob/main/docs/source/tutorials/2_Load_policy.ipynb>`_ or load the `colab <https://colab.research.google.com/drive/1U6vo6Q_rPhDaq6oUMV7EAZRm6s0fD1wn?usp=sharing>`_ to try the pretrained behaviors.


Reproduce Baselines
~~~~~~~~~~~~~~~~~~~~

Installation Steps -

1. install mjrl ``python -m pip install git+https://github.com/aravindr93/mjrl.git``
2. install hydra ``pip install hydra-core --upgrade``
3. install `submitit <https://github.com/facebookincubator/submitit>`_ launcher hydra plugin to launch jobs on cluster/ local ``pip install hydra-submitit-launcher --upgrade``

.. code-block:: bash

    pip install tabulate matplotlib torch git+https://github.com/aravindr93/mjrl.git
    pip install hydra-core --upgrade
    pip install hydra-submitit-launcher --upgrade
    pip install submitit

Trainign steps -

1. Get commands to lunch training by running `train_myosuite.sh` located in the `myosuite/agents` folder:

.. code-block:: bash

    sh train_myosuite.sh myo         # runs natively
    sh train_myosuite.sh myo local   # use local launcher
    sh train_myosuite.sh myo slurm   # use slurm launcher

2. Further customize the prompts from the previous step and execute.

Installation
~~~~~~~~~~~~

1. We use `mjrl <https://github.com/aravindr93/mjrl>`_ and PyTorch for our baselines. Please install it with ``python -m pip install git+https://github.com/aravindr93/mjrl.git``
2. install hydra ``pip install hydra-core --upgrade``
3. install `submitit <https://github.com/facebookincubator/submitit>`_ launcher hydra plugin to launch jobs on cluster/ local ``pip install hydra-submitit-launcher --upgrade``

.. code-block:: bash

    pip install tabulate matplotlib torch git+https://github.com/aravindr93/mjrl.git
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
