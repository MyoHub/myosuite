Tutorials
###########

.. _tutorials:


Here a set of examples on how to use different MyoSuite models and non-stationarities.

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

    import myoSuite
    import gym
    env = gym.make('XXX')
    env.reset()


.. _run_visualize_index_movements:

Activate and visualize finger movements
============================================
Example on how to generate and visualize a movement e.g. index flexion, and visualize the results

.. code-block:: python

    import myoSuite
    import gym
    env = gym.make('XXX')
    env.reset()

.. _run_trained_policy:

Test trained policy
======================
Example on using a policy e.g. elbow flexion, and change non-stationaries

.. code-block:: python

    import myoSuite
    import gym
    env = gym.make('XXX')
    env.reset()


.. _test_muscle_fatigue:

Test Muscle Fatigue
======================
This example shows how to add fatigue to a model. It tests random actions on a model without and then with muscle fatigue.

.. code-block:: python

    import mj_envs
    import gym
    env = gym.make('ElbowPose1D6MRandom-v0')
    env.reset()
    env.sim.render(mode='window')
    for _ in range(1000):
        env.step(env.action_space.sample()) # take a random action
    # Add muscle fatigue
    env.env.muscle_condition = 'fatigue'
    for _ in range(1000):
        env.step(env.action_space.sample()) # take a random action
    env.close()


.. _test_sarcopenia:

Test Sarcopenia
======================
This example shows how to add sarcopenia or muscle weakness to a model. It tests random actions on a model without and then with muscle weakness.

.. code-block:: python

    import mj_envs
    import gym
    env = gym.make('ElbowPose1D6MRandom-v0')
    env.reset()
    env.sim.render(mode='window')
    for _ in range(1000):
    env.step(env.action_space.sample()) # take a random action
    # Add muscle weakness
    env.env.muscle_condition = 'weakness'
    for _ in range(1000):
        env.step(env.action_space.sample()) # take a random action
    env.close()


.. _test_tendon_transfer:

Test Physical tendon transfer
==============================

This example shows how load a model with physical tendon transfer.

.. code-block:: python

    import myoSuite
    import gym
    env = gym.make('XXX')
    env.reset()
    env.sim.render(mode='window')
    for _ in range(1000):
        env.step(env.action_space.sample()) # take a random action
    env.close()


.. _resume_training:

Resume Learning of policies
==============================
When using ``mjrl`` it might be needed to resume training of a policy locally. It is possible to use the following instruction

.. code-block:: bash

    python3 hydra_mjrl_launcher.py --config-path config --config-name hydra_biomechanics_config.yaml hydra/output=local hydra/launcher=local env=HandPoseMuscleRandom-v0 job_name=[Absolute Path of the policy] rl_num_iter=[New Total number of iterations]
