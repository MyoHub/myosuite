Sports Medicine & Rehabilitation Quick Start
=============================================

This guide is for sports medicine practitioners and physical rehabilitation
researchers who want to use MyoSuite to model pathological conditions,
assistive devices, and clinical rehabilitation scenarios.

No machine learning background is required.

.. contents:: Contents
   :local:
   :depth: 2

Key Concepts
------------

MyoSuite provides physiologically validated musculoskeletal models that can
simulate conditions commonly encountered in clinical practice:

* **Sarcopenia** — age-related loss of muscle mass and strength (50 % force reduction).
* **Cumulative fatigue** — neuromuscular fatigue accumulating during sustained effort.
* **Tendon transfer (reafferentation)** — surgical re-routing of tendons altering the
  muscle-to-joint mapping.

These conditions are built into MyoSuite as environment variants, accessible
with a single argument — no programming of the physics required.


Installation
------------

.. code-block:: bash

   pip install -U myosuite
   python -m myosuite.tests.test_myo   # verify and list all environments


Simulating Sarcopenia
----------------------

Sarcopenia reduces peak muscle force by 50 %, modelling the strength loss
typical of older adults:

.. code-block:: python

   import gymnasium as gym
   import myosuite
   import numpy as np

   env_normal = gym.make('myoElbowPose1D6MRandom-v0')
   env_sarco  = gym.make('myoSarcElbowPose1D6MRandom-v0')

   results = {}
   for label, env in [("Normal", env_normal), ("Sarcopenia", env_sarco)]:
       obs, info = env.reset(seed=0)
       peak_forces = []
       for _ in range(300):
           obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
           peak_forces.append(np.abs(env.unwrapped.data.actuator_force).max())
           if terminated or truncated:
               obs, info = env.reset()
       results[label] = np.max(peak_forces)
       env.close()

   print(f"Peak force — Normal: {results['Normal']:.1f} N")
   print(f"Peak force — Sarcopenia: {results['Sarcopenia']:.1f} N  "
         f"({100*results['Sarcopenia']/results['Normal']:.0f} % of normal)")


Simulating Neuromuscular Fatigue
----------------------------------

The cumulative fatigue model tracks motor unit pool depletion during
sustained activation — useful for studying exercise tolerance and
work capacity:

.. code-block:: python

   import gymnasium as gym
   import myosuite
   import numpy as np

   env = gym.make('myoFatiElbowPose1D6MFixed-v0')
   obs, info = env.reset(seed=0)

   force_over_time = []
   # Apply constant 60 % excitation — observe force decline
   ctrl_level = np.full(env.action_space.shape, 0.2)  # maps to ~60 % excitation

   for step in range(3000):
       obs, reward, terminated, truncated, info = env.step(ctrl_level)
       force_over_time.append(np.abs(env.unwrapped.data.actuator_force).mean())
       if terminated or truncated:
           obs, info = env.reset()

   force = np.array(force_over_time)
   print(f"Force at start: {force[:50].mean():.2f} N")
   print(f"Force at end:   {force[-50:].mean():.2f} N  "
         f"({100*force[-50:].mean()/force[:50].mean():.0f} % of initial)")

   env.close()

See ``tutorials/7_Fatigue_Modeling.ipynb`` for plots and recovery dynamics.


Clinical Metrics
-----------------

Useful metrics you can extract from any simulation:

.. code-block:: python

   import gymnasium as gym
   import myosuite
   import numpy as np

   env = gym.make('myoLegWalk-v0')
   obs, info = env.reset(seed=0)

   joint_angles, forces, coms = [], [], []

   for _ in range(1000):
       obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
       d = env.unwrapped.data
       joint_angles.append(d.qpos.copy())
       forces.append(d.actuator_force.copy())
       coms.append(d.subtree_com[0].copy())  # whole-body centre of mass
       if terminated or truncated:
           obs, info = env.reset()

   joint_angles = np.array(joint_angles)
   forces       = np.array(forces)
   coms         = np.array(coms)

   model = env.unwrapped.model
   print("Joint range of motion (deg):")
   for i in range(model.njnt):
       rom = np.degrees(np.ptp(joint_angles[:, i]))
       if rom > 0.1:
           print(f"  {model.joint(i).name:30s}  {rom:.1f}")

   # Metabolic cost proxy: integrated squared muscle activation
   metabolic_proxy = np.mean(forces**2)
   print(f"\nMetabolic cost proxy (N²): {metabolic_proxy:.2f}")

   # Walking symmetry: CoM lateral deviation
   if coms.shape[1] >= 2:
       lateral_std = np.std(coms[:, 1])
       print(f"Lateral CoM std (symmetry): {lateral_std*100:.2f} cm")

   env.close()


Simulating Tendon Transfer Surgery
------------------------------------

The reafferentation variant models a tendon transfer that redirects the
EIP (extensor indicis proprius) to the EPL (extensor pollicis longus),
as performed in radial nerve palsy rehabilitation:

.. code-block:: python

   import gymnasium as gym
   import myosuite

   # Normal finger extension
   env_normal = gym.make('myoFingerPoseRandom-v0')

   # Post-surgical — altered muscle routing
   env_reaff  = gym.make('myoReafHandPoseFixed-v0')

   # An RL or reflex controller trained on normal anatomy must re-adapt —
   # useful for studying motor re-learning after surgery.


Rehabilitation Progression / Curriculum
-----------------------------------------

You can simulate progressive rehabilitation by changing the task difficulty
between episodes.  A simple example for elbow flexion ROM progression:

.. code-block:: python

   import gymnasium as gym
   import myosuite
   import numpy as np

   # Start with a restricted target range, progressively widen it
   rom_stages = [
       {"r_elbow_flex": (1.0, 1.5)},   # Stage 1: limited ROM
       {"r_elbow_flex": (0.5, 2.0)},   # Stage 2: moderate ROM
       {"r_elbow_flex": (0.0, 2.5)},   # Stage 3: full ROM
   ]

   for stage, target_range in enumerate(rom_stages, 1):
       env = gym.make(
           'myoElbowPose1D6MRandom-v0',
           target_jnt_range=target_range,
       )
       obs, info = env.reset(seed=0)
       successes = 0
       for ep in range(20):
           obs, info = env.reset()
           for _ in range(200):
               obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
               if info.get("rwd_dict", {}).get("solved", False):
                   successes += 1
                   break
               if terminated or truncated:
                   break
       print(f"Stage {stage}: {successes}/20 episodes solved")
       env.close()


Assessing Walking and Gait
----------------------------

The ``myoLegWalk-v0`` environment models a full lower limb including hip,
knee, and ankle with 80 muscles.  Useful for:

* Gait analysis under normal and pathological conditions
* Exoskeleton assistance simulation
* Fall risk assessment

.. code-block:: python

   import gymnasium as gym
   import myosuite

   env = gym.make('myoLegWalk-v0')
   obs, info = env.reset(seed=0)

   # The reward dictionary includes gait-relevant components
   for _ in range(100):
       obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
       rwd = info.get("rwd_dict", {})
       # Components: 'vel_reward', 'cyclic_hip', 'ref_rot', 'joint_angle_rew', etc.

   env.close()

For challenge-level locomotion tasks including obstacle courses, see
``myoChallengeOslRunRandom-v0`` and ``myoChallengeSoccerP2-v0``.


Sports Tasks
-------------

MyoSuite includes several sports-inspired environments:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Environment
     - Description
   * - ``myoChallengeOslRunRandom-v0``
     - Running on random terrain (prosthetic limb)
   * - ``myoChallengeChaseTagP1-v0``
     - Whole-body chase task (two agents)
   * - ``myoChallengeBaodingP2-v1``
     - Dexterous manipulation — Baoding ball rotation
   * - ``myoChallengeTableTennisP2-v0``
     - Arm swing for table-tennis striking

These environments stress specific physical capacities (power, dexterity,
agility, endurance) and can be used to study sports performance limits.


Environment Index for Clinical Research
-----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Clinical question
     - Condition
     - Environment to use
   * - Upper-limb strength loss with age
     - Sarcopenia
     - ``myoSarcElbowPose1D6MRandom-v0``
   * - Work capacity under fatigue (elbow)
     - Cumulative fatigue
     - ``myoFatiElbowPose1D6MFixed-v0``
   * - Work capacity under fatigue (hand)
     - Cumulative fatigue
     - ``myoFatiHandPoseRandom-v0``
   * - Post-surgical motor re-learning (hand)
     - Reafferentation
     - ``myoReafHandPoseFixed-v0``
   * - Walking endurance / fall risk
     - Normal
     - ``myoLegWalk-v0``
   * - Walking on uneven ground
     - Normal
     - ``myoLegRoughTerrainWalk-v0``
   * - Stair climbing
     - Normal
     - ``myoLegStairTerrainWalk-v0``
   * - Hand function / dexterity
     - Normal
     - ``myoHandPoseRandom-v0``
   * - Key turning (pinch grip)
     - Normal
     - ``myoHandKeyTurnRandom-v0``
   * - Object grasping & placement
     - Normal
     - ``myoChallengeRelocateP1-v0``
   * - Lumbar spine / back posture
     - Normal
     - ``myoTorsoPoseFixed-v0``
   * - Elbow exoskeleton assistance
     - Normal + exo
     - ``myoElbowPose1D6MExoFixed-v0``
   * - Sarcopenic leg walking
     - Sarcopenia
     - ``myoSarcLegWalk-v0``

See :doc:`environments` for the complete listing of all 125+ registered IDs,
including challenge environments and GPU-accelerated variants.


Next Steps
----------

* ``tutorials/7_Fatigue_Modeling.ipynb`` — detailed fatigue dynamics
* ``tutorials/3_Analyse_movements.ipynb`` — kinematic analysis
* ``tutorials/6_Inverse_Dynamics.ipynb`` — joint torque estimation
* :doc:`quickstart_biomechanics` — extract raw simulation data for analysis
* :doc:`quickstart_neuroscience` — add reflex controllers and sensory models
