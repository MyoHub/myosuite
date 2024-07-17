MyoChallenge-2024
#############################################


* :ref:`challenge24_manipulation`
* :ref:`challenge24_locomotion`
* :ref:`challenge24_tutorial`




.. _challenge24_manipulation:

Prosthesis Co-Manipulation
--------------------------------------------------------------

Task Overview
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A myoHand :ref:`myoHand` model and a Modular Prosthetic Limb (`MPL <https://www.jhuapl.edu/work/projects-and-missions/revolutionizing-prosthetics/research>`__)
involved in moving an object between two tables with a handover. This task requires delicate coordination of the 
object without dropping it or destroying it (maximum force on the object for success) and a mandatory handover between 
the MyoArm and the MPL to move the objects between two locations.


.. image:: images/MyoChallenge-manip.png
    :width: 450
    :align: center



Objective
^^^^^^^^^^^^^^^^^^^^^^^^^^^


Move the object between two locations with a handover between a hand and a prosthesis. The object parameter is
randomized with object starting location, object weight, and friction if necessary. 


**Success conditions**:
  *Moving the object to the end location without dropping it, destroying it, and having a handover.*


Maximum object resistance.
The object first has to touch the MyoArm (x ms), then the MPL (x ms), and then the end location (x ms).



Action Space
^^^^^^^^^^^^^^^^^^^^^^^^

The whole set of muscles and actuated joints [0, 1]. Prosthesis action value range 


Observation Space
^^^^^^^^^^^^^^^^^^^^^^^^^

All joints angles [-,-]
Myohand Data for hand joint positions and velocities, the MPL position and velocity. The object’s position and velocity. The starting and goal position. Contact information 
of object with myohand/MPL/start/goal/env. 





Disclaimer on challenge realism
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



A single policy is used to map observations about the entire bimanual task to the control of both arms. This differs from real prosthetic control tasks, where the two separate 
agents (human and device) communicate through a narrow bandwidth human-machine interface (HMI). Usually only 1-4 degrees of freedom are communicated between the two systems, often 
thresholded into a binary channels.
Since bimanual manipulation is already a complex and challenging task, we did not place such a restriction into the environment or submission rules. Single agent solutions should 
indicate an upper limit to performance with a perfect HMI. We encourage you to investigate implementing more realistic multi-agent scenarios after the challenge!



.. _challenge24_locomotion:


Prosthesis Locomotion
---------------------------------




A trans-femoral myoLeg model and a Open Source Leg (`OSL <https://neurobionics.robotics.umich.edu/research/wearable-robotics/open-source-leg/>`__)  involved 
in walking over different terrain types. The task requires learning the dynamics and control of a powered prosthetic leg that has its own controller. 
This is similar to how people with limb loss learn to adapt to a prosthetic leg over time. This task also requires navigation over different terrain 
with increasing difficulty.


.. image:: images/Myotrack_promo_1.png
  :width: 350
  :align: center
  :alt: Text




Objective
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Traverse over different terrain types with a prosthetic leg. Randomization will be done with:

    - Terrain Types:
        - Flat Ground
        - Rough Ground
        - Slopes
        - Stairs
    - Difficulty of Terrain
        - Rough: Increasing roughness
        - Slopes: Increasing steepness of Slopes
        - Stairs: Increasing height of stairs

.. figure:: images/Myotrack_promo_2.png
    :width: 600
    :align: center

    Example of increasing difficulty of obstacles


Only 1 terrain type will be present in each episode. Mixed terrains in a single episode may be implemented to increase the 
difficulty of the challenge for the purposes of tie-breaking.




**Learning interactions with prosthetic leg**


The primary way to interact with the prosthetic leg is via socket interaction forces on the residual limb (which are provided 
in the observations). A state-based impedance controller would provide the commands to move the prosthetic limb and participants 
are provided with the corresponding APIs to update the impedance controller.




For task evaluation, there are no direct observations and control over the prosthetic leg. Angles, angular velocities and torque 
of the prosthetic leg will not be available in the observations. Similarly, there is no commanded position, velocity or torques 
for the prosthetic leg.



Evaluation Criteria
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Submission are evaluated on the distance traveled over a fixed time horizon on the pre-defined track. The submission must stay on
the track to receive full credits.



Action Space
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Muscles control values are given as continuous values between  :math:`[-1, 1]`. Normalization to a range of :math:`[0, 1]` is done in the environment 
according to the equation

.. math::

    1 / ( 1 + exp(-5 * (muscleCtrl - 0.5) ) )


For participants that do not wish to use this normalization feature, it can be done during environment initialization with:

:code:`env = gym.make(“myoChallengeRunTrackP1-v0”, normalize_act=False)`


where in this case, the control range of the muscles are set between :math:`[0, 1]` without any normalization performed.
Commanded torque values are generated by an embedded State Machine :ref:`challenge24_state_machine`. Refer to the section below for more information.






Observation Space
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. TODO : Is it better to make it a table?

+-----------------------------------------+-----------------------------+-----------------+
| **Description**                         |        **Access**           |   **Dimension** |
+-----------------------------------------+-----------------------------+-----------------+
|Time                                     |      obs_dict['time']       |        (1x1)    |
+-----------------------------------------+-----------------------------+-----------------+
| Terrain type (see below)                |   obs_dict['terrain']       | (1x1)           |
+-----------------------------------------+-----------------------------+-----------------+
| Torso angle                             |                             |                 |
| (quaternion in world frame)             |   obs_dict['torso_angle']   |  (4x1)          |
+-----------------------------------------+-----------------------------+-----------------+
| Joint positions                         |                             |                 |
| (except those from the prosthetic leg)  | obs_dict['internal_qpos']   |  (21x1)         | 
+-----------------------------------------+-----------------------------+-----------------+
| Joint velocities                        |                             |                 | 
| (except those from the prosthetic leg)  | obs_dict['internal_qvel']   | (21x1)          | 
+-----------------------------------------+-----------------------------+-----------------+
| Ground reaction forces                  | obs_dict['grf']             |  (2x1)          |
| (only for biological leg)               |                             |                 |
+-----------------------------------------+-----------------------------+-----------------+
| Socket forces (see below)               | obs_dict['socket_force']    | (3x1)           |
+-----------------------------------------+-----------------------------+-----------------+
| Muscle activations                      | obs_dict['act']             | (54x1)          |
+-----------------------------------------+-----------------------------+-----------------+
| Muscle length                           | obs_dict['muscle_length']   |  (54x1)         |
+-----------------------------------------+-----------------------------+-----------------+
| Muscle velocities                       | obs_dict['muscle_velocity'] | (54x1)          |
+-----------------------------------------+-----------------------------+-----------------+
| Muscle forces                           | obs_dict['muscle_force']    | (54x1)          |
+-----------------------------------------+-----------------------------+-----------------+
| Model center of mass position           |                             |  (3x1)          |
| (in world frame)                        |  obs_dict['model_root_pos'] |                 |
+-----------------------------------------+-----------------------------+-----------------+
| Model center of mass velocity           |  obs_dict['model_root_vel'] |   (3x1)         |
| (in world frame)                        |                             |                 |
+-----------------------------------------+-----------------------------+-----------------+
| Height map                              |  obs_dict['hfield']         | (100x1)         |
+-----------------------------------------+-----------------------------+-----------------+


.. TODO: decide which observation table is better


Observations from the environment are
    1. Time, obs_dict['time'] (1x1)
    2. Terrain type (see below) obs_dict['terrain'] (1x1)
    3. Torso angle (quaternion in world frame) obs_dict['torso_angle'] (4x1)
    4. Joint positions (except those from the prosthetic leg) obs_dict['internal_qpos'] (21x1)
    5. Joint velocities (except those from the prosthetic leg) obs_dict['internal_qvel'] (21x1)
    6. Ground reaction forces (only for biological leg) obs_dict['grf'] (2x1)
    7. Socket forces (see below) obs_dict['socket_force'](3x1)
    8. Muscle properties
        a. Muscle activations obs_dict['act'] (54x1)
        b. Muscle length obs_dict['muscle_length'] (54x1)
        c. Muscle velocities obs_dict['muscle_velocity'] (54x1)
        d. Muscle forces obs_dict['muscle_force'] (54x1)
    9. Model center of mass position (in world frame) obs_dict['model_root_pos'] (3x1)
    10. Model center of mass velocity (in world frame) obs_dict['model_root_vel'] (3x1)
    11. Height map obs_dict['hfield'] (100x1)


**Description of observations**

    - Terrain type codes are given as:

        - FLAT = 0
        - HILLY = 1
        - ROUGH = 2
        - STAIRS = 3

    - Socket forces

        - Represented as a 3-DOF force vector. Note that the direction of the force sensor is from the bottom of the socket projecting to the residual limb (i.e. the vertical axis force into the residual limb is negative). Processing of the observations is left to the participant’s discretion.
    
    Height Map

        - The height map is a 10x10 grid (flattened to a 100x1), centered around the center of the MyoOSL model. This is a simple representation of a visual input of the terrain around the model.


.. _challenge24_state_machine:

State Machine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A simple 4-state state machine is created to track the gait phase of the prosthetic leg. Each state contains the gain parameters 
for an impedance controller, which in turn, provides the required torques to the prosthetic actuators. The code for the state machine 
is released together with MyoChallenge. Interested participants are invited to examine the code at 
`myoosl_control <https://github.com/MyoHub/myosuite/blob/dev/myosuite/envs/myo/assets/leg/myoosl_control.py>`__


Parameters of the impedance controller are taken from `finite_state_machine <https://opensourceleg.readthedocs.io/en/latest/examples/finite_state_machine.html>`__



Gait phases in the state machine are divide into:

    1. Early Stance (e_stance)
    2. Late Stance (l_stance)
    3. Early Swing (e_swing)
    4. Late Swing (l_swing)


List of states variables:

    - States

        - ["e_stance", "l_stance", "e_swing", "l_swing"]

    - Impedance controller parameters (for both knee and ankle actuators)

        - Stiffness
        - Damping
        - Target angle

    - State transition thresholds

        - Load
        - Knee angle
        - Knee velocity
        - Ankle angle
        - Ankle velocity



.. _challenge24_tutorial:


Challenge Tutorial
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This page aims to provide an basic knowledge of the challenge.

For a step-by-step tutorial, please check our :ref:`tutorials` page for more advanced info.


.. code-block:: python

    from myosuite.utils import gym
    # Include the locomotion track environment, uncomment to select the manipulation challenge
    env = gym.make('myoChallengeRunTrackP1-v0')
    #env = gym.make('myoChallengeBimanual-v0')
    

    env.reset()

    # Repeat 1000 time steps
    for _ in range(1000):

        # Activate mujoco rendering window
        env.mj_render()


        # Get observation from the envrionment, details are described in the above docs
        obs = env.get_obs()
        current_time = obs['time']
        #print(current_time)


        # Take random actions
        action = env.action_space.sample()


        # Environment provides feedback on action
        next_obs, reward, terminated, truncated, info = env.step(action)


        # Reset training if env is terminated
        if terminated:
            next_obs, info = env.reset()


To obtain a more in-depth understanding of the challenge, we have prepared baselines for both of the challenges.
Links are available for `manipulation <https://colab.research.google.com/drive/1AqC1Y7NkRnb2R1MgjT3n4u02EmSPem88#scrollTo=-mAnRvYjIS4d>`__, 
locomotion.

.. TODO: locomotion colab page is missing