Projects with Myosuite
#########################################

.. _projects:

* :ref:`myochallenge`
    * :ref:`myo_challenge_22`
    * :ref:`myo_challenge_23`
    * :ref:`myo_challenge_24`
* :ref:`pub_with_myosuite`
    * :ref:`ref_myodex`
    * :ref:`ref_deprl`
    * :ref:`ref_lattic`
    * :ref:`ref_sar`


.. _myochallenge:

MyoChallenge
========================================
The Myosuite Team organised MyoChallenge to tackle difficult problems in top-level machine learning conference.
Our latest challenge is accepted to NeuIPs 2024.

.. _myo_challenge_22:

MyoChallenge-2022: Learning Physiological Dexterity
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Introducing `MyoChallenge - a NeurIPS 2022 <https://sites.google.com/view/myochallenge>`__ competition track on learning contact-rich manipulation skills for a physiologically 
realistic musculo-skeletal hand. The goal of MyoChallenge is to push our understanding of physiological motor-control responsible
for nimble and agile movements of the human body. In the current edition of MyoChallenge, 
we are focusing on developing controllers for contact rich dexterous manipulation behaviors. 
This challenge builds upon the MyoSuite ecosystem -- a fast (>4000x faster) and contact-rich framework 
for musculoskeletal motor control. 




Competition Tracks
The MyoChallenge consists of two tracks:

.. _myo_challenge_23:

MyoChallenge-2023: Towards Human-Level Dexterity and Agility
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Introducing `MyoChallenge 2023 <https://sites.google.com/view/myosuite/myochallenge/myochallenge-2023>`__: Towards Human-Level Dexterity and Agility

Humans effortlessly grasp objects of diverse shapes and properties and execute 
agile locomotion without overwhelming their cognitive capacities. This ability was acquired 
through millions of years of evolution, which honed the symbiotic relationship between the central and 
peripheral nervous systems and the musculoskeletal structure. Consequently, it is not surprising that 
uncovering the intricacies of these complex, evolved systems underlying human movement remains a formidable 
challenge. Advancements in neuromechanical simulations and data driven methods offer promising avenues to 
overcome these obstacles. This year’s competition will feature two tracks: the manipulation track and the locomotion track. 

.. _myo_challenge_24:

MyoChallenge 2024
+++++++++++++++++++++++++++++++++++++
About to be released



Prosthesis co-manipulation
--------------------------------------------------------------

Task Overview:
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A myoHand :ref:`myoHand` model and a Modular Prosthetic Limb (`MPL <https://www.jhuapl.edu/work/projects-and-missions/revolutionizing-prosthetics/research>`__)
involved in moving an object between two tables with a handover. This task requires delicate coordination of the 
object without dropping it or destroying it (maximum force on the object for success) and a mandatory handover between 
the MyoArm and the MPL to move the objects between two locations.


Objective:
^^^^^^^^^^^^^^^^^^^^^^^^^^^


Move the object between two locations with a handover between a hand and a prosthesis. The object parameter is
randomized with object starting location, object weight, and friction if necessary. 


**Success conditions**:
  *Moving the object to the end location without dropping it, destroying it, and having a handover.*


Maximum object resistance.
The object first has to touch the MyoArm (x ms), then the MPL (x ms), and then the end location (x ms).



Action Space:
^^^^^^^^^^^^^^^^^^^^^^^^

The whole set of muscles and actuated joints [0, 1]. Prosthesis action value range 


Observation Space:
^^^^^^^^^^^^^^^^^^^^^^^^^

All joints angles [-,-]
Myohand Data for hand joint positions and velocities, the MPL position and velocity. The object’s position and velocity. The starting and goal position. Contact information of object with myohand/MPL/start/goal/env. 

.. image:: images/manipulation_myo24.png
  :width: 300

Tutorials:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from myosuite.utils import gym
    #env = gym.make('myoElbowPose1D6MRandom-v0')
    env.reset()
    for _ in range(1000):
        env.mj_render()
        env.step(env.action_space.sample()) # take a random actio
    
    # Add code on how to run the baselines



Prosthesis Locomotion
---------------------------------


Task Overview:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A trans-femoral myoLeg model and a Open Source Leg (`OSL <https://neurobionics.robotics.umich.edu/research/wearable-robotics/open-source-leg/>`__)  involved 
in walking over different terrain types. The task requires learning the dynamics and control of a powered 
prosthetic leg that has its own controller. This is similar to how people with limb loss learn to adapt to 
a prosthetic leg over time. This task also requires navigation over different terrain with increasing difficulty. 


Objective:
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


Only 1 terrain type will be present in each episode. Mixed terrains in a single episode may be implemented to increase the 
difficulty of the challenge for the purposes of tie-breaking.



Learning interactions with prosthetic leg




The primary way to interact with the prosthetic leg is via socket interaction forces on the residual limb (which is 
provided in the observations). A state-based impedance controller would provide the commands to move the prosthetic 
limb. Participants are also provided APIs to update the parameters of the impedance controller. The State Machine description 
is provided `here <https://opensourceleg.readthedocs.io/en/latest/examples/finite_state_machine.html>`__, and the code for the 
State Machine is given in the environment.



For task evaluation, there are no direct observations and control over the prosthetic leg. This means angles, 
angular velocities and torque of the prosthetic leg will not be available in the observations. Similarly, there is no 
commanded position, velocity or torques for the prosthetic leg.


Task Evaluations:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Submission are evaluated on the distance traveled over a fixed time horizon.

Action Space:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Observation Space:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^











.. _pub_with_myosuite:

Publications with Myosuite
========================================


Please feel free to create a PR for your own project with Myosuite

.. _ref_myodex:

MyoDex: A Generalizable Prior for Dexterous Manipulation
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Link avaiable at `here <https://sites.google.com/view/myodex>`__



.. _ref_deprl:

DEP-RL: Embodied Exploration for Reinforcement Learning in Overactuated and Musculoskeletal Systems
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Link avaiable at `here <https://github.com/martius-lab/depRL>`__



.. _ref_lattic:

Lattice: Latent Exploration for Reinforcement Learning
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Link avaiable at `here <https://github.com/amathislab/lattice>`__



.. _ref_sar:

SAR: Generalization of Physiological Agility and Dexterity via Synergistic Action Representation
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


Link avaiable at `here <https://sites.google.com/view/sar-rl>`__