RL baselines & Training Modules
===============================

MyoSuite ships default CPU and GPU training scripts alongside isolated method packages in
`training/ <https://github.com/MyoHub/myosuite/blob/dev/training/README.md>`_. See
`training/README.md <https://github.com/MyoHub/myosuite/blob/dev/training/README.md>`_ for the
side-by-side method comparison table and matched PPO benchmarks across locomotion, motion tracking,
dexterous pose control, and multi-agent tasks.

CPU baselines
-------------

* **Stable-Baselines3 (PPO, SAC, TD3)** — ``scripts/train_sb3.py`` and
  ``tutorials/2.1_Train_SB3_Policy.ipynb`` (``pip install -e ".[rl]"``). Best suited for debugging,
  custom CPU environments, and low-dimensional models (``myoElbow*``, ``myoFinger*``).
* **SAR (Synergistic Action Representation)** — ``tutorials/2.3_SAR.ipynb`` extracts PCA + FastICA
  muscle synergies from short rollouts to accelerate dexterous manipulation on CPU.
* **MyoReflex** — ``tutorials/2.4_MyoReflex_Walk.ipynb`` tunes a Song & Geyer spinal reflex
  controller for bipedal walking without neural-network training.

GPU baselines (``mjlab``)
-------------------------

RSL-RL PPO on MuJoCo Warp (see :doc:`install` for matching the PyTorch build to your CUDA driver)::

   pip install -e ".[mjlab]"
   python scripts/train_mjlab.py myoLegWalk-v0 --env.scene.num-envs 1024

Always set ``--env.scene.num-envs`` explicitly (default is 1 — see :doc:`quickstart_ml` for why
that stalls training) and re-pass it whenever you ``--agent.resume True`` a run.

Specialized training packages (``training/<method>/``)
------------------------------------------------------

Method-specific algorithms keep their dependencies isolated from the core ``myosuite`` wheel:

* **DynSyn muscle synergies** (``training/dynsyn``) — groups muscles whose lengths co-vary on the
  compiled ``MjModel`` (``dynsyn-groups``) and trains RSL-RL PPO through a ``TrainedActionMap``
  (``dynsyn-train``). Reduces iteration count and worst-seed variance on ``myoLegWalk-v0``,
  ``myoLegRoughTerrainWalk-v0``, and ``myoLegHillyTerrainWalk-v0``; checkpoints replay directly via
  ``scripts/eval_mjlab_policy.py`` without ``myosuite-dynsyn`` installed.
* **myoSuperMimic** (``training/supermimic``) — pairs a learned body world model with analytic
  differentiable kinematics, muscle geometry, and pure task reward heads, tracking all 972 AMASS/KIT
  training clips on 354-muscle ``myoMimic-v0`` in ~30 minutes on a single GPU (``0/108`` held-out
  clip failures and ``18.6 mm`` MPJPE with 5-step test-time refinement).
* **ChaseTag teacher & yaw-canonicalised DAgger** (``training/chasetag_teacher``) — steers a
  ``myoSuperMimic`` tracker via motion matching on ``myoChallengeChaseTagFBVs-v0`` and directional
  walking, then distils standalone ``.onnx`` students with SE(2) root-yaw canonicalisation baked
  into the graph (``tutorials/5.4``, ``5.6``, ``5.7``).

Default policies
^^^^^^^^^^^^^^^^

Default mjlab (RSL-RL), MuscleMimic, and ChaseTag checkpoints are hosted on
`myohub/myosuite-3-baselines <https://huggingface.co/myohub/myosuite-3-baselines>`_
on Hugging Face, and are downloaded automatically (cached by ``huggingface_hub``) the first time
they're needed — by the tutorials, and by ``myosuite.utils.checkpoint_utils.find_checkpoint``.
They run on both the CPU and the mjlab backend::

   python scripts/eval_mjlab_policy.py myoFingerPoseRandom-v0 --backend cpu

`docs/baseline_checkpoints.md <https://github.com/MyoHub/myosuite/blob/main/docs/baseline_checkpoints.md>`_
lists every default-run env id's deterministic success rate; some published policies are
unconverged snapshots. See :doc:`quickstart_ml` for how success is defined and evaluated.

Pretrained NPG / DEP-RL weight trees under ``myosuite/agents/`` are **not**
shipped in the pip package and are gitignored. Train your own policy, or use
MuscleMimic and baseline checkpoints from Hugging Face (see
``myosuite/integrations/musclemimic/README.md``).
