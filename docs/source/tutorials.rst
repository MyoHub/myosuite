Tutorials
###########

.. _tutorials:


Here a set of examples on how to use different MyoSuite models and non-stationarities.
Jupyter-Notebooks can be found `here <https://github.com/facebookresearch/myosuite/tree/main/docs/source/tutorials>`__

* :ref:`run_myosuite`
* :ref:`run_visualize_index_movements`
* :ref:`run_trained_policy`
* :ref:`test_muscle_fatigue`
* :ref:`test_sarcopenia`
* :ref:`test_tendon_transfer`
* :ref:`resume_training`
* :ref:`load_deprl_baseline`
* :ref:`load_MyoReflex_baseline`

.. _run_myosuite:

Test Environment
======================
Example on how to use an environment e.g. send random movements

.. code-block:: python

    from myosuite.utils import gym
    env = gym.make('myoElbowPose1D6MRandom-v0')
    env.reset()
    for _ in range(1000):
        env.mj_render()
        env.step(env.action_space.sample()) # take a random action
    env.close()


.. _run_visualize_index_movements:

Activate and visualize finger movements
============================================
Example on how to generate and visualize a movement e.g. index flexion, and visualize the results

.. code-block:: python

    from myosuite.utils import gym
    env = gym.make('myoHandPoseRandom-v0')
    env.reset()
    for _ in range(1000):
        env.mj_render()
        env.step(env.action_space.sample()) # take a random action
    env.close()

.. _run_trained_policy:

Test trained policy
======================
Example on using a policy e.g. elbow flexion, and change non-stationaries

.. code-block:: python

    from myosuite.utils import gym
    policy = "iterations/best_policy.pickle"

    import pickle
    pi = pickle.load(open(policy, 'rb'))

    env = gym.make('myoElbowPose1D6MRandom-v0')
    env.reset()
    for _ in range(1000):
        env.mj_render()
        env.step(env.action_space.sample()) # take a random action



.. _test_muscle_fatigue:

Test Muscle Fatigue
======================
This example shows how to add fatigue to a model. It tests random actions on a model without and then with muscle fatigue.

.. code-block:: python

    from myosuite.utils import gym
    env = gym.make('myoElbowPose1D6MRandom-v0')
    env.reset()
    for _ in range(1000):
        env.mj_render()
        env.step(env.action_space.sample()) # take a random action

    # Add muscle fatigue
    env = gym.make('myoFatiElbowPose1D6MRandom-v0')
    env.reset()
    for _ in range(1000):
        env.mj_render()
        env.step(env.action_space.sample()) # take a random action
    env.close()


.. _test_sarcopenia:

Test Sarcopenia
======================
This example shows how to add sarcopenia or muscle weakness to a model. It tests random actions on a model without and then with muscle weakness.

.. code-block:: python

    from myosuite.utils import gym
    env = gym.make('myoElbowPose1D6MRandom-v0')
    env.reset()
    for _ in range(1000):
        env.mj_render()
        env.step(env.action_space.sample()) # take a random action

    # Add muscle weakness
    env = gym.make('myoSarcElbowPose1D6MRandom-v0')
    env.reset()
    for _ in range(1000):
        env.mj_render()
        env.step(env.action_space.sample()) # take a random action
    env.close()


.. _test_tendon_transfer:

Test Physical tendon transfer
==============================

This example shows how load a model with physical tendon transfer.

.. code-block:: python

    from myosuite.utils import gym
    env = gym.make('myoHandKeyTurnFixed-v0')
    env.reset()
    for _ in range(1000):
        env.mj_render()
        env.step(env.action_space.sample()) # take a random action

    # Add tendon transfer
    env = gym.make('myoTTHandKeyTurnFixed-v0')
    env.reset()
    for _ in range(1000):
        env.mj_render()
        env.step(env.action_space.sample()) # take a random action
    env.close()


.. _resume_training:

Resume Learning of policies
==============================
When using ``mjrl`` it might be needed to resume training of a policy locally. It is possible to use the following instruction

.. code-block:: bash

    python3 hydra_mjrl_launcher.py --config-path config --config-name hydra_biomechanics_config.yaml hydra/output=local hydra/launcher=local env=myoHandPoseRandom-v0 job_name=[Absolute Path of the policy] rl_num_iter=[New Total number of iterations]

.. _load_deprl_baseline:

Load DEP-RL Baseline
====================
See `here <https://deprl.readthedocs.io/en/latest/index.html>`__ for more detailed documentation of ``deprl``.

If you want to load and execute the pre-trained DEP-RL baseline. Make sure that the ``deprl`` package is installed.

.. code-block:: python

    from myosuite.utils import gym
    import deprl

    # we can pass arguments to the environments here
    env = gym.make('myoLegWalk-v0', reset_type='random')
    policy = deprl.load_baseline(env)
    obs = env.reset()
    for i in range(1000):
        env.mj_render()
        action = policy(obs)
        obs, *_ = env.step(action)
    env.close()

.. _load_MyoReflex_baseline:

Load MyoReflex Baseline
=======================

To load and execute the MyoReflex controller with baseline parameters.
Run the MyoReflex tutorial `here <https://github.com/facebookresearch/myosuite/tree/main/docs/source/tutorials/4b_reflex>`__



Customizing Tasks
=================

In order to create a new customized task, there are two places where you need to act:

1. Set up a new environment class for the new task

2. Register the new task

Set up a new environment
+++++++++++++++++++++++++

Environment classes are developed according to the OpenAI Gym definition
and contain all the information specific for a task,
to interact with the environment, to observe it and to
act on it. In addition, each environment class contains
a reward function which converts the observation into a
number that establishes how good the observation is with
respect to the task objectives. In order to create a new
task, a new environment class needs to be generated eg.
reach2_v0.py (see for example how `reach_v0.py <https://github.com/MyoHub/myosuite/blob/main/myosuite/envs/myo/myobase/reach_v0.py>`__ is structured).
In this file, it is possible to specify the type of observation (eg. joint angles, velocities, forces), actions (e.g. muscle, motors), goal, and reward.


.. code-block:: python

    from myosuite.envs.myo.base_v0 import BaseV0
    import deprl

    # Class extends Basev0
    class NewReachEnvV0(BaseV0):
        ....

    # defines the observation
    def get_obs_dict(self, sim):
        ....

    # defines the rewards
    def get_reward_dict(self, obs_dict):
        ...

    #reset condition that
    def reset(self):
        ...

    # step the simulation forward by acting on the environment
    def step(self, a, **kwargs):
       ...
       return observation, reward, task_terminated, environment_information

.. _setup_base_class:


Register the new environment
++++++++++++++++++++++++++++++

Once defined the task `reach2_v0.py`, the new environment needs to be registered to be
visible when importing `myosuite`. This is achieved by introducing the new environment in
the `__init__.py` (called when the library is imported) where the registration routine happens.
The registration of the new enviornment is obtained adding:

.. code-block:: python

   register_env_with_variants(id='newReachTask-v0',
        entry_point='myosuite.envs.myo.myobase.reach_v0:NewReachEnvV0', # where to find the new Environment Class
        max_episode_steps=200, # duration of the episode
        kwargs={
            'model_path': curr_dir+'/../assets/hand/myohand_pose.xml', # where the xml file of the environment is located
            'target_reach_range': {'IFtip': ((0.1, 0.05, 0.20), (0.2, 0.05, 0.20)),}, # this is used in the setup to define the goal e.g. rando position of the team between 0.1 and 0.2 in the x coordinates
            'normalize_act': True, # if to use normalized actions using a sigmoid function.
            'frame_skip': 5, # collect a sample every 5 iteration step
        }
    )


.. _register_new_environment:

