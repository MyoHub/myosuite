Tutorials
###########

.. _tutorials:


Here a set of examples on how to use different MyoSuite models and non-stationarities.
Jupyter-Notebooks can be found `here <https://github.com/facebookresearch/myosuite/tree/main/docs/source/tutorials>`_

* :ref:`run_myosuite`
* :ref:`run_visualize_index_movements`
* :ref:`run_trained_policy`
* :ref:`test_muscle_fatigue`
* :ref:`test_sarcopenia`
* :ref:`test_tendon_transfer`
* :ref:`resume_training`

.. _run_myosuite:

Test Environment
======================
Example on how to use an environment e.g. send random movements

.. code-block:: python

    import myosuite
    import gym
    env = gym.make('myoElbowPose1D6MRandom-v0')
    env.reset()
    for _ in range(1000):
        env.sim.render(mode='window')
        env.step(env.action_space.sample()) # take a random action
    env.close()


.. _run_visualize_index_movements:

Activate and visualize finger movements
============================================
Example on how to generate and visualize a movement e.g. index flexion, and visualize the results

.. code-block:: python

    import myosuite
    import gym
    env = gym.make('myoHandPoseRandom-v0')
    env.reset()
    for _ in range(1000):
        env.sim.render(mode='window')
        env.step(env.action_space.sample()) # take a random action
    env.close()

.. _run_trained_policy:

Test trained policy
======================
Example on using a policy e.g. elbow flexion, and change non-stationaries

.. code-block:: python

    import myosuite
    import gym
    policy = "iterations/best_policy.pickle"

    import pickle
    pi = pickle.load(open(policy, 'rb'))

    env = gym.make('myoElbowPose1D6MRandom-v0')
    env.reset()
    for _ in range(1000):
        env.sim.render(mode='window')
        env.step(env.action_space.sample()) # take a random action



.. _test_muscle_fatigue:

Test Muscle Fatigue
======================
This example shows how to add fatigue to a model. It tests random actions on a model without and then with muscle fatigue.

.. code-block:: python

    import myosuite
    import gym
    env = gym.make('myoElbowPose1D6MRandom-v0')
    env.reset()
    for _ in range(1000):
        env.sim.render(mode='window')
        env.step(env.action_space.sample()) # take a random action

    # Add muscle fatigue
    env = gym.make('myoFatiElbowPose1D6MRandom-v0')
    env.reset()
    for _ in range(1000):
        env.sim.render(mode='window')
        env.step(env.action_space.sample()) # take a random action
    env.close()


.. _test_sarcopenia:

Test Sarcopenia
======================
This example shows how to add sarcopenia or muscle weakness to a model. It tests random actions on a model without and then with muscle weakness.

.. code-block:: python

    import myosuite
    import gym
    env = gym.make('myoElbowPose1D6MRandom-v0')
    env.reset()
    for _ in range(1000):
        env.sim.render(mode='window')
        env.step(env.action_space.sample()) # take a random action

    # Add muscle weakness
    env = gym.make('myoSarcElbowPose1D6MRandom-v0')
    env.reset()
    for _ in range(1000):
        env.sim.render(mode='window')
        env.step(env.action_space.sample()) # take a random action
    env.close()


.. _test_tendon_transfer:

Test Physical tendon transfer
==============================

This example shows how load a model with physical tendon transfer.

.. code-block:: python

    import myosuite
    import gym
    env = gym.make('myoHandKeyTurnFixed-v0')
    env.reset()
    for _ in range(1000):
        env.sim.render(mode='window')
        env.step(env.action_space.sample()) # take a random action

    # Add tendon transfer
    env = gym.make('myoTTHandKeyTurnFixed-v0')
    env.reset()
    for _ in range(1000):
        env.sim.render(mode='window')
        env.step(env.action_space.sample()) # take a random action
    env.close()


.. _resume_training:

Resume Learning of policies
==============================
When using ``mjrl`` it might be needed to resume training of a policy locally. It is possible to use the following instruction

.. code-block:: bash

    python3 hydra_mjrl_launcher.py --config-path config --config-name hydra_biomechanics_config.yaml hydra/output=local hydra/launcher=local env=myoHandPoseRandom-v0 job_name=[Absolute Path of the policy] rl_num_iter=[New Total number of iterations]
