Environment Reference
======================

* **CPU** — ``gym.make(env_id)`` after ``import myosuite``. Playback and SB3.
* **mjlab** — the same ``env_id`` for GPU training (``scripts/train_mjlab.py``).

List every CPU ID on your install::

   python -c "import myosuite; print('\n'.join(myosuite.myosuite_env_suite))"

Tables below are the common CPU IDs. Pathological prefixes (``myoSarc…``,
``myoFati…``, hand ``myoReaf…``) are auto-registered for ``myo*`` tasks.

.. contents:: Contents
   :local:
   :depth: 2

GPU (mjlab) coverage
---------------------

The mjlab backend registers a twin under the same ``env_id`` for these CPU families
(``myoSarc…``/``myoFati…``/``myoReaf…`` variants exist where the CPU env has them):

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Family
     - mjlab twins
   * - Pose
     - ``myoElbowPose1D6M{Fixed,Random}`` (+ ``Exo``), ``myoFingerPose{Fixed,Random}``,
       ``myoHandPose{0-9}Fixed``, ``myoHandPose{Fixed,Random}``,
       ``myoTorsoPoseFixed``, ``myoTorsoExoPoseFixed``, ``motorFingerPose{Fixed,Random}``
   * - Reach
     - ``myoArmReach``, ``myoFingerReach``, ``myoHandReach``, ``motorFingerReach``
       (each ``{Fixed,Random}``)
   * - Leg
     - ``myoLegWalk``, ``myoLegDirectional{Forward,Backward,Random}``,
       ``myoLegStandRandom``, ``myoLeg{Rough,Hilly,Stair}TerrainWalk``
   * - Challenge
     - ``myoChallengeChaseTagFBP2``, ``myoChallengeTableTennisP{0,1,2}``

Variants: ``myoSarc…`` exist for all twins above except ``myoLegDirectional*`` (no CPU
variant) and the Challenge twins; ``myoFati…`` likewise, and ``myoReaf…`` for the hand
pose and reach twins. **Not on mjlab yet:** the other Challenge tasks, ``myoElbowPoseTask*``,
``myoFullBodyDirectional`` and the other MyoMimic/MuscleMimic envs, and the hand manipulation
families (``myoHandKeyTurn``, ``ObjHold``, ``PenTwirl``, ``Reorient*``) including their
variants.

Differences from the CPU env that matter when moving policies between backends:

* **Torso exosuit** (``myoTorsoExoPoseFixed`` and variants) — *experimental*: the two
  free exosuit bodies are 6-DoF joint chains on mjlab (mjlab allows one freejoint per
  entity); their ``qpos``/``qvel`` are converted back to the CPU layout, so the
  observation is the same 296-d vector (matching CPU to ~1e-3 over short rollouts, see
  ``test_torso_exo_observation_matches_cpu``). Transfer of *trained* policies between
  the backends has not been tested yet.
* **Terrain walks:** the terrain is baked into the model with a fixed seed instead of
  being resampled at each reset (``myoLegRoughTerrainWalk`` uses one fixed sample).
* Leg locomotion twins are checked only for bounded divergence from the CPU env, not
  step-by-step parity (see :doc:`backend_parity`).

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


mjlab (GPU)
-----------

Install ``pip install -e ".[mjlab]"`` (see :doc:`install` for matching the torch
build to your driver's CUDA version) and train with the CPU ``env_id``::

   python scripts/train_mjlab.py myoElbowPose1D6MRandom-v0 --render onscreen --env.scene.num-envs 1024

Registered mjlab IDs live in
``myosuite.envs.myo.backends.mjlab.REGISTERED_TASKS``.


MJX (experimental)
------------------

A JAX path exists (``pip install -e ".[mjx]"``,
``from myosuite.envs.myo.backends.mjx import make``). Do not start new work on
it; use mjlab for GPU training.
