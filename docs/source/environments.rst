Environment Reference
======================

MyoSuite registers environments across three execution paths:

* **CPU** — standard Gymnasium environments, accessible via ``gym.make()``.
* **MJX** — JAX/GPU environments, via ``myosuite.envs.myo.backends.mjx.make()``.
* **mjlab** — MuJoCo Warp / Isaac Lab environments (subset).

The tables below cover all registered environments.
Run ``python -m myosuite.tests.test_myo`` to verify registration on your system.

.. contents:: Contents
   :local:
   :depth: 2

Naming Conventions
-------------------

Base environment IDs follow the pattern::

    myo<Model><Task>[Difficulty]-v<N>

Pathological variants are auto-registered for every base ``myo*`` CPU environment:

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Prefix pattern
     - Condition
     - Example
   * - ``myoSarc<…>``
     - Sarcopenia (50 % peak force)
     - ``myoSarcElbowPose1D6MRandom-v0``
   * - ``myoFati<…>``
     - Cumulative neuromuscular fatigue
     - ``myoFatiHandPoseRandom-v0``
   * - ``myoReaf<…>``
     - Tendon transfer / reafferentation *(hand envs only)*
     - ``myoReafHandPoseFixed-v0``

The suffix ``Fixed`` indicates a fixed (non-random) target;
``Random`` indicates a randomly sampled target each episode.


myoFinger  (4 DoF, 5–6 muscles)
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 45 15 15 15 15

   * - Environment ID
     - Task
     - Diff.
     - Sarc
     - Fati
   * - ``myoFingerReachFixed-v0``
     - Fingertip reach
     - Easy
     - ✓
     - ✓
   * - ``myoFingerReachRandom-v0``
     - Fingertip reach
     - Hard
     - ✓
     - ✓
   * - ``myoFingerPoseFixed-v0``
     - Joint pose
     - Easy
     - ✓
     - ✓
   * - ``myoFingerPoseRandom-v0``
     - Joint pose
     - Hard
     - ✓
     - ✓

Each base environment also exposes ``myoSarc…`` and ``myoFati…`` variants
(8 total IDs for this model).


myoElbow  (2 DoF, 6 muscles)
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 45 20 10 10 10

   * - Environment ID
     - Task
     - Diff.
     - Sarc
     - Fati
   * - ``myoElbowPose1D6MFixed-v0``
     - Joint pose
     - Easy
     - ✓
     - ✓
   * - ``myoElbowPose1D6MRandom-v0``
     - Joint pose
     - Hard
     - ✓
     - ✓
   * - ``myoElbowPose1D6MExoFixed-v0``
     - Pose + elbow exoskeleton
     - Easy
     - ✓
     - ✓
   * - ``myoElbowPose1D6MExoRandom-v0``
     - Pose + elbow exoskeleton
     - Hard
     - ✓
     - ✓

Each row additionally has ``myoSarc…`` and ``myoFati…`` variants (12 total IDs).


myoHand  (23 DoF, 39 muscles)
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 45 20 10 10 10 10

   * - Environment ID
     - Task
     - Diff.
     - Sarc
     - Fati
     - Reaf
   * - ``myoHandPoseFixed-v0``
     - 23-DoF joint pose
     - Easy
     - ✓
     - ✓
     - ✓
   * - ``myoHandPoseRandom-v0``
     - 23-DoF joint pose
     - Hard
     - ✓
     - ✓
     - ✓
   * - ``myoHandReachFixed-v0``
     - Fingertip spatial reach
     - Easy
     - ✓
     - ✓
     - ✓
   * - ``myoHandReachRandom-v0``
     - Fingertip spatial reach
     - Hard
     - ✓
     - ✓
     - ✓
   * - ``myoHandKeyTurnFixed-v0``
     - Key rotation (thumb + index)
     - Easy
     - ✓
     - ✓
     - ✓
   * - ``myoHandKeyTurnRandom-v0``
     - Key rotation, random init
     - Hard
     - ✓
     - ✓
     - ✓
   * - ``myoHandObjHoldFixed-v0``
     - Object repositioning (no drop)
     - Easy
     - ✓
     - ✓
     - ✓
   * - ``myoHandObjHoldRandom-v0``
     - Random object, random target
     - Hard
     - ✓
     - ✓
     - ✓
   * - ``myoHandPenTwirlFixed-v0``
     - Pen twirl to fixed orientation
     - Easy
     - ✓
     - ✓
     - ✓
   * - ``myoHandPenTwirlRandom-v0``
     - Pen twirl to random orientation
     - Hard
     - ✓
     - ✓
     - ✓
   * - ``myoHandReorient8-v0``
     - Reorient 1 of 8 objects
     - Easy
     - ✓
     - ✓
     - ✓
   * - ``myoHandReorient100-v0``
     - Reorient 1 of 100 objects
     - Medium
     - ✓
     - ✓
     - ✓
   * - ``myoHandReorientID-v0``
     - Reorient 1 of 1000 (in-domain)
     - Hard
     - ✓
     - ✓
     - ✓
   * - ``myoHandReorientOOD-v0``
     - Reorient 1 of 1000 (out-of-domain)
     - Hardest
     - ✓
     - ✓
     - ✓

With all three variants each row generates ``myoSarc…``, ``myoFati…``, and
``myoReaf…`` IDs.  The full hand environment count is **14 × 4 = 56 IDs**.

**MyoChallenge hand tasks** (no automatic variant registration):

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Environment ID
     - Task
   * - ``myoChallengeDieReorientDemo-v0``
     - Die reorientation demo
   * - ``myoChallengeDieReorientP1-v0``
     - Die reorientation (limited goal range)
   * - ``myoChallengeDieReorientP2-v0``
     - Die reorientation (full range + friction/size variation)
   * - ``myoChallengeBaodingP1-v1``
     - Baoding balls — swap positions
   * - ``myoChallengeBaodingP2-v1``
     - Baoding balls — full rotation + size/friction variation
   * - ``myoChallengeRelocateP1-v0``
     - Grasp & place object (phase 1)
   * - ``myoChallengeRelocateP2-v0``
     - Grasp & place (phase 2, harder)
   * - ``myoChallengeRelocateP2eval-v0``
     - Relocate phase 2 — evaluation split
   * - ``myoChallengeTableTennisP0-v0``
     - Table tennis swing (warm-up)
   * - ``myoChallengeTableTennisP1-v0``
     - Table tennis (phase 1)
   * - ``myoChallengeTableTennisP2-v0``
     - Table tennis (phase 2, full task)
   * - ``myoChallengeBimanual-v0``
     - Bimanual object manipulation


myoArm  (27 DoF, 63 muscles — hand-free variant: 20 DoF, 32 muscles)
-----------------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 45 35 10 10 10

   * - Environment ID
     - Task
     - Diff.
     - Sarc
     - Fati
   * - ``myoArmReachFixed-v0``
     - Index fingertip reach (fixed)
     - Easy
     - ✓
     - ✓
   * - ``myoArmReachRandom-v0``
     - Index fingertip reach (random)
     - Hard
     - ✓
     - ✓

The arm model used here is the hand-free variant (extrinsic + intrinsic hand
muscles removed) to isolate reaching without manipulation.


myoLeg  (10 joints, 20 DoF, 80 muscles)
------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 45 35 10 10 10

   * - Environment ID
     - Task
     - Diff.
     - Sarc
     - Fati
   * - ``myoLegStandRandom-v0``
     - Static balance — random init
     - Easy
     - ✓
     - ✓
   * - ``myoLegWalk-v0``
     - Flat-ground forward walking
     - Medium
     - ✓
     - ✓
   * - ``myoLegRoughTerrainWalk-v0``
     - Walking on rough terrain
     - Hard
     - ✓
     - ✓
   * - ``myoLegHillyTerrainWalk-v0``
     - Walking on hilly terrain
     - Hard
     - ✓
     - ✓
   * - ``myoLegStairTerrainWalk-v0``
     - Stair climbing
     - Hardest
     - ✓
     - ✓

**MyoChallenge leg / whole-body tasks:**

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Environment ID
     - Task
   * - ``myoChallengeChaseTagP1-v0``
     - Chase-tag locomotion (phase 1)
   * - ``myoChallengeChaseTagP2-v0``
     - Chase-tag (phase 2, two-agent)
   * - ``myoChallengeChaseTagP2eval-v0``
     - Chase-tag phase 2 — evaluation split
   * - ``myoChallengeOslRunFixed-v0``
     - OSL prosthetic running (fixed terrain)
   * - ``myoChallengeOslRunRandom-v0``
     - OSL prosthetic running (random terrain)
   * - ``myoChallengeSoccerP1-v0``
     - Soccer ball kicking (phase 1)
   * - ``myoChallengeSoccerP2-v0``
     - Soccer ball kicking (phase 2)


myoTorso  (18 joints, 210 muscles)
-------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 45 35 10 10 10

   * - Environment ID
     - Task
     - Diff.
     - Sarc
     - Fati
   * - ``myoTorsoPoseFixed-v0``
     - Lumbar spine pose
     - Fixed
     - ✓
     - ✓
   * - ``myoTorsoExoPoseFixed-v0``
     - Lumbar spine pose + exoskeleton
     - Fixed
     - ✓
     - ✓


MJX Environments (JAX / GPU)
------------------------------

These environments run on GPU via MuJoCo MJX and require the ``[mjx]`` extra.
Access via ``myosuite.envs.myo.backends.mjx.make()``.

.. list-table::
   :header-rows: 1
   :widths: 45 30 25

   * - Environment ID
     - Model
     - CPU analogue
   * - ``MjxElbowPoseFixed-v0``
     - myoElbow
     - ``myoElbowPose1D6MFixed-v0``
   * - ``MjxElbowPoseRandom-v0``
     - myoElbow
     - ``myoElbowPose1D6MRandom-v0``
   * - ``MjxFingerPoseFixed-v0``
     - myoFinger
     - ``myoFingerPoseFixed-v0``
   * - ``MjxFingerPoseRandom-v0``
     - myoFinger
     - ``myoFingerPoseRandom-v0``
   * - ``MjxHandReachFixed-v0``
     - myoHand
     - ``myoHandReachFixed-v0``
   * - ``MjxHandReachRandom-v0``
     - myoHand
     - ``myoHandReachRandom-v0``
   * - ``MjxLegWalk-v0``
     - myoLeg
     - ``myoLegWalk-v0``

.. code-block:: python

   from myosuite.envs.myo.backends.mjx import make
   env = make("MjxElbowPoseRandom-v0")


mjlab Environments (MuJoCo Warp / Isaac Lab)
----------------------------------------------

These environments use the Isaac Lab manager API with MuJoCo Warp as the
physics backend.  They require the ``[mjlab]`` extra and are accessed via
``mjlab.envs.make()``.

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Environment ID (same as CPU)
     - Config class
   * - ``myoElbowPose1D6MFixed-v0``
     - ``ElbowPoseCfg``
   * - ``myoElbowPose1D6MRandom-v0``
     - ``ElbowPoseCfg``
   * - ``myoHandPoseRandom-v0``
     - ``HandReachCfg``
   * - ``myoLegWalk-v0``
     - ``WalkCfg``
   * - ``myoChallengeBaodingP2-v1``
     - ``BaodingCfg``


Summary Counts
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Category
     - Base IDs
     - Notes
   * - myoFinger
     - 4
     - ×3 (base + Sarc + Fati) = 12
   * - myoElbow
     - 4
     - ×3 = 12; includes Exo variants
   * - myoHand (base)
     - 14
     - ×4 (base + Sarc + Fati + Reaf) = 56
   * - myoHand (challenge)
     - 12
     - no auto-variants
   * - myoArm
     - 2
     - ×3 = 6
   * - myoLeg (base)
     - 5
     - ×3 = 15
   * - myoLeg (challenge)
     - 7
     - no auto-variants
   * - myoTorso
     - 2
     - ×3 = 6
   * - MJX
     - 7
     - JAX/GPU path
   * - mjlab
     - 5
     - MuJoCo Warp / Isaac Lab path
   * - **Total (CPU)**
     - **≥ 125**
     - counting all variant IDs
