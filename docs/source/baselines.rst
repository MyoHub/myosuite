RL baselines
============

CPU
---

* **Stable-Baselines3** — ``tutorials/2.1_Train_SB3_Policy.ipynb`` (``pip install -e ".[rl]"``)
* **MyoReflex** — ``tutorials/2.4_MyoReflex_Walk.ipynb``

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

Default mjlab (RSL-RL) checkpoints are hosted on `myohub/myosuite-3-baselines <https://huggingface.co/myohub/myosuite-3-baselines>`_
on Hugging Face, and are downloaded automatically (cached by ``huggingface_hub``) the first time
they're needed — by the tutorials, and by ``myosuite.utils.checkpoint_utils.find_checkpoint``.
They run on both the CPU and the mjlab backend::

   python scripts/eval_mjlab_policy.py myoFingerPoseRandom-v0 --backend cpu

`docs/baseline_checkpoints.md <https://github.com/MyoHub/myosuite/blob/dev/docs/baseline_checkpoints.md>`_
lists every default-run env id's deterministic success rate; some published policies are
unconverged snapshots. See :doc:`quickstart_ml` for how success is defined and evaluated.

Pretrained NPG / DEP-RL weight trees under ``myosuite/agents/`` are **not**
shipped in the pip package and are gitignored. Train your own policy, or use
MuscleMimic checkpoints from Hugging Face (see
``myosuite/integrations/musclemimic/README.md``).
