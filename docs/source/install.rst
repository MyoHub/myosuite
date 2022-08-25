Installation
============

.. _installation:

MyoSuite uses git submodules to resolve dependencies.
Please follow steps exactly as below to install correctly.

Requirements
~~~~~~~~~~~~
* python >= 3.7.1 (if needed follow instructions `here <https://docs.conda.io/en/latest/miniconda.html>`_ for installing python and conda)
* free-mujoco-py >= 2.1.6

At this moment, the library works only on MacOs and Linux


Installing the pip package
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   conda create --name MyoSuite python=3.7.1
   conda activate MyoSuite
   pip install -U myosuite


(alternative) Installing from source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
❗IMPORTANT❗ Install MuJoCo 2.1 before installing MyoSuite

To get started with MyoSuite, clone this repo with pre-populated submodule dependencies

.. code-block:: bash

   git clone --recursive https://github.com/facebookresearch/myosuite.git
   cd myosuite
   pip install -e -r requirements

OR Add repo to pythonpath by updating `~/.bashrc` or `~/.bash_profile`

.. code-block:: bash

   export PYTHONPATH="<path/to/myosuite>:$PYTHONPATH"

Testing the installation
~~~~~~~~~~~~~~~~~~~~~~~~

You can test the installation using

.. code-block:: bash

   python myosuite/tests/test_myo.py

You can visualize the environments with random controls using the below command

.. code-block:: bash

   python myosuite/utils/examine_env.py --env_name myoElbowPose1D6MRandom-v0

.. note::
   If the visualization results in a GLFW error, this is because ``mujoco-py`` does not see some graphics drivers correctly.
   This can usually be fixed by explicitly loading the correct drivers before running the python script.
   See `this page <https://github.com/aravindr93/mjrl/tree/master/setup#known-issues>`_ for details.

Examples
~~~~~~~~~

It is possible to create and interface with MyoSuite environments like any other OpenAI gym environments.
For example, to use the ``myoElbowPose1D6MRandom-v0`` environment it is possible simply to run:

.. code-block:: python

   import myosuite
   import gym
   env = gym.make('myoElbowPose1D6MRandom-v0')
   env.reset()
   for _ in range(1000):
      env.sim.render(mode='window')
      env.step(env.action_space.sample()) # take a random action
   env.close()
