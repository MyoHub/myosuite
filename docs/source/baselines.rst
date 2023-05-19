RL Baselines
============

.. _baselines:


For ease of getting started, MyoSuite comes prepackaged with a set of pre-trained baselines.
See `here <https://github.com/facebookresearch/myosuite/tree/main/myosuite/agents>`_ for our complete set of baselines.

MJRL baselines
```````````````

We use a model-free on-policy algorithm - Natural Policy Gradient (from `mjrl <https://github.com/aravindr93/mjrl>`_) as our baselines.

Try baselines
~~~~~~~~~~~~~~~~

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


DEPRL baseline
`````````````````
We provide `deprl <https://github.com/martius-lab/depRL>`_ as an additional baseline for locomotion policies.
See `here <https://github.com/facebookresearch/myosuite/blob/main/docs/source/tutorials/4a_deprl.ipynb>`_ for more detailed tutorials.

Installation
~~~~~~~~~~~
Simply run

.. code-block:: bash

    python -m pip install deprl

after installing the myosuite.

Train new policy
~~~~~~~~~~~~~~~~~

For training from scratch, navigate to the `agents folder <https://github.com/facebookresearch/myosuite/tree/main/myosuite/agents/>`_ folder and run the following code

.. code-block:: bash

    python -m deprl.main baselines_DEPRL/myoLegWalk.json

and training should start. Inside the json-file, you can set a custom output folder with ``working_dir=myfolder``. Be sure to adapt the ``sequential`` and ``parallel`` settings. During a training run, `sequential x parallel` environments are spawned, which consumes over 30 GB of RAM with the default settings for the myoLeg. Reduce this number if your workstation has less memory.

Visualize, log and plot
~~~~~~~~~~~~~~~~~~~~
We provide several utilities in the ``deprl`` package.
To visualize a trained policy, run

.. code-block:: bash

    python -m deprl.play --path baselines_DEPRL/myoLegWalk_20230514/myoLeg/


where the last folder contains the ``checkpoints`` and ``config.yaml`` files. This runs the policy for several episodes and returns scores.
You can also log some settings to `wandb <https://wandb.ai/)>`_ . Set it up and afterwards run

.. code-block:: bash

    python -m deprl.log --path baselines_DEPRL/myoLegWalk_20230514/myoLeg/log.csv --project myoleg_deprl_baseline

which will log all the training metrics to your ``wandb`` project.

If you want to plot your training run, use

.. code-block:: bash

    python -m deprl.plot --path baselines_DEPRL/myoLegWalk_20230514/


For more instructions on how to use the plot feature, checkout `TonicRL <https://github.com/fabiopardo/tonic>`_, which is the general-purpose RL library deprl was built on.

Credit
~~~~~~~~~~~~~~~~~~~~
The DEPRL baseline for the myoLeg was developed by Pierre Schumacher, Daniel Häufle and Georg Martius as members of the Max Planck Institute for Intelligent Systems and the Hertie Institute for Clinical Brain Research.

Please cite `this paper <https://openreview.net/forum?id=C-xa_D3oTj6>`_ if you are using our work.


