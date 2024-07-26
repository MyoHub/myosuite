.. _baselines:

RL Baselines
=====================================



For ease of getting started, MyoSuite comes prepackaged with a set of pre-trained baselines. Tutorials in this sections aims to show how different 
RL baselines can be integrated in the myosuite environment.
See `here <https://github.com/facebookresearch/myosuite/tree/main/myosuite/agents>`_ for our complete set of baselines.

In total, three different types of baselines are provided:

* :ref:`mjrl_baseline`
* :ref:`deprl_baseline`
* :ref:`reflex_controller`


.. _mjrl_baseline:

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

Training steps -

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


.. _deprl_baseline:

DEP-RL baseline
```````````````
We provide `deprl <https://github.com/martius-lab/depRL>`_ as an additional baseline for locomotion policies. You can find more detailed explanations and documentation on how to use it `here <https://deprl.readthedocs.io/en/latest/index.html>`__. The controller was adapted from the original paper and produces robust locomotion policies with the MyoLeg through the use of a self-organizing exploration method.
While DEP-RL can be used for any kind of RL task, we provide a pre-trained controller and training settings for the `myoLegWalk-v0` task.
See `this tutorial <https://github.com/facebookresearch/myosuite/blob/main/docs/source/tutorials/4a_deprl.ipynb>`_ for more detailed tutorials.

Installation
~~~~~~~~~~~~
Simply run

.. code-block:: bash

    python -m pip install deprl

after installing the myosuite.

Train new policy
~~~~~~~~~~~~~~~~

For training from scratch, navigate to the `agents folder <https://github.com/facebookresearch/myosuite/tree/main/myosuite/agents/>`_ folder and run the following code

.. code-block:: bash

    python -m deprl.main baselines_DEPRL/myoLegWalk.json

and training should start. Inside the json-file, you can set a custom output folder with ``working_dir=myfolder``. Be sure to adapt the ``sequential`` and ``parallel`` settings. During a training run, `sequential x parallel` environments are spawned, which consumes over 30 GB of RAM with the default settings for the myoLeg. Reduce this number if your workstation has less memory.

Visualize, log and plot
~~~~~~~~~~~~~~~~~~~~~~~
We provide several utilities in the ``deprl`` package.
To visualize a trained policy, run

.. code-block:: bash

    python -m deprl.play --path baselines_DEPRL/myoLegWalk_20230514/myoLeg/


where the last folder contains the ``checkpoints`` and ``config.yaml`` files. This runs the policy for several episodes and returns scores.
You can also log some settings to `wandb <https://wandb.ai/>`_ . Set it up and afterwards run

.. code-block:: bash

    python -m deprl.log --path baselines_DEPRL/myoLegWalk_20230514/myoLeg/log.csv --project myoleg_deprl_baseline

which will log all the training metrics to your ``wandb`` project.

If you want to plot your training run, use

.. code-block:: bash

    python -m deprl.plot --path baselines_DEPRL/myoLegWalk_20230514/


For more instructions on how to use the plot feature, checkout `TonicRL <https://github.com/fabiopardo/tonic>`_, which is the general-purpose RL library deprl was built on.



.. _reflex_controller:

MyoLegReflex baseline
`````````````````````

MyoLegReflex is a reflex-based walking controller for MyoLeg. With the provided set of 46 control parameters, MyoLeg generates steady walking patterns. Users have the freedom to discover alternative parameter sets for generating diverse walking behaviors or design a higher-level controller that modulates these parameters dynamically, thereby enabling navigation within dynamic environments.

Examples
~~~~~~~~~~~~~~

MyoLegReflex is bundled as a wrapper around MyoLeg. To run MyoLegReflex with default parameters, you can either utilize the Jupyter notebook found in ``myosuite/docs/source/tutorials/4b_reflex`` or execute the following code snippet:

.. code-block:: python

    import ReflexCtrInterface
    import numpy as np

    sim_time = 5  # in seconds
    dt = 0.01
    steps = int(sim_time/dt)
    frames = []

    params = np.loadtxt('baseline_params.txt')

    Myo_env = ReflexCtrInterface.MyoLegReflex()
    Myo_env.reset()

    Myo_env.set_control_params(params)

    for timstep in range(steps):
        frame = Myo_env.env.mj_render()
        Myo_env.run_reflex_step()
    Myo_env.env.close()

Note: This code snippet only works in the folder ``myosuite/docs/source/tutorials/4b_reflex``, where the MyoLegReflex wrapper resides.

Reflex-based Controller
-----------------------

MyoLegReflex is adapted from the neural circuitry model proposed by Song and Geyer in "A neural circuitry that emphasizes spinal feedback generates diverse behaviours of human locomotion" (The Journal of Physiology, 2015). The original model is capable of producing a variety of human-like locomotion behaviors, utilizing a musculoskeletal model with 22 leg muscles (11 per leg).

To make the controller more straightforward, we first modified the circuits that operate based on muscle lengths and velocities to work with joint angles and angular velocities instead.

Subsequently, we adapted this controller to be compatible with MyoLeg, which features 80 leg muscles. We achieved this by merging sensory data from each functional muscle group into one, processing the combined sensory data through the adapted reflex circuits to generate muscle stimulation signals, and then distributing these signals to the individual muscles within each group. The grouping of muscles is defined in `ReflexCtrInterface.py <https://github.com/facebookresearch/myosuite/blob/main/docs/source/tutorials/4b_reflex/ReflexCtrInterface.py#L212-L345>`_.



Credit
``````

DEP-RL
~~~~~~

The DEP-RL baseline for the myoLeg was developed by

* Pierre Schumacher <schumacherpier@gmail.com>
* Daniel Häufle <daniel.haeufle@uni-tuebingen.de>
* Georg Martius <georg.martius@tuebingen.mpg.de>

as members of the Max Planck Institute for Intelligent Systems and the Hertie Institute for Clinical Brain Research.

Please cite `this paper <https://openreview.net/forum?id=C-xa_D3oTj6>`__ if you are using our work.

.. code-block:: bibtex

    @inproceedings{
    schumacher2023deprl,
    title={{DEP}-{RL}: Embodied Exploration for Reinforcement Learning in Overactuated and Musculoskeletal Systems},
    author={Pierre Schumacher and Daniel Haeufle and Dieter B{\"u}chler and Syn Schmitt and Georg Martius},
    booktitle={The Eleventh International Conference on Learning Representations },
    year={2023},
    url={https://openreview.net/forum?id=C-xa_D3oTj6}
    }

MyoLegReflex
~~~~~~~~~~~~

The MyoLegReflex controller was developed by

* Seungmoon Song <ssm0445@gmail.com>
* Chun Kwang Tan <cktan.neumove@gmail.com>

as members of the Northeastern University.

Please cite `this paper <https://physoc.onlinelibrary.wiley.com/doi/full/10.1113/JP270228>`__ if you are using our work.

.. code-block:: bibtex

    @article{https://doi.org/10.1113/JP270228,
    author = {Song, Seungmoon and Geyer, Hartmut},
    title = {A neural circuitry that emphasizes spinal feedback generates diverse behaviours of human locomotion},
    journal = {The Journal of Physiology},
    volume = {593},
    number = {16},
    pages = {3493-3511},
    doi = {https://doi.org/10.1113/JP270228},
    url = {https://physoc.onlinelibrary.wiley.com/doi/abs/10.1113/JP270228},
    eprint = {https://physoc.onlinelibrary.wiley.com/doi/pdf/10.1113/JP270228},
    year = {2015}
    }
