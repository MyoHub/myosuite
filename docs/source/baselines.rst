RL baselines
============

CPU
---

* **Stable-Baselines3** — ``tutorials/2.1_Train_SB3_Policy.ipynb`` (``pip install -e ".[rl]"``)
* **DEP-RL** (walk) — ``tutorials/2.4_DEP_RL.ipynb`` (``pip install deprl``, Python ≤3.11.5)
* **MyoReflex** — ``tutorials/2.5_MyoReflex_Walk.ipynb``

GPU
---

RSL-RL PPO on mjlab (see :doc:`install` for matching the torch build to your
driver's CUDA version)::

   pip install -e ".[mjlab]"
   python scripts/train_mjlab.py myoLegWalk-v0 --env.scene.num-envs 1024

Always set ``--env.scene.num-envs`` explicitly (default is 1 — see
:doc:`quickstart_ml` for why that stalls training) and re-pass it whenever you
``--agent.resume True`` a long walk run.

Default policies
^^^^^^^^^^^^^^^^

The repository ships mjlab (RSL-RL) checkpoints of default training runs for many envs
in ``baselines/checkpoints/<env_id>/`` (with their evaluation videos in
``baselines/evals/``). They run on the CPU and the mjlab backend::

   python scripts/eval_mjlab_policy.py myoFingerPoseRandom-v0 \
       --checkpoint baselines/checkpoints/myoFingerPoseRandom-v0 --backend cpu

``baselines/checkpoints/README.md`` lists each policy's deterministic success rate;
some are unconverged snapshots, and the ``myoLeg*`` locomotion policies are
provisional. See :doc:`quickstart_ml` for how success is defined and evaluated.

Pretrained NPG / DEP-RL weight trees under ``myosuite/agents/`` are **not**
shipped in the pip package and are gitignored. Train your own policy, or use
MuscleMimic checkpoints from Hugging Face (see
``myosuite/integrations/musclemimic/README.md``).
