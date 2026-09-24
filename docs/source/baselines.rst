RL baselines
============

CPU
---

* **Stable-Baselines3** — ``tutorials/4c_Train_SB_policy.ipynb`` (``pip install -e ".[rl]"``)
* **DEP-RL** (walk) — ``tutorials/4a_deprl.ipynb`` (``pip install deprl``, Python ≤3.11.5)
* **MyoReflex** — ``tutorials/4b_reflex/MyoSuite_MyoReflex_Walk.ipynb``

GPU
---

RSL-RL PPO on mjlab (see :doc:`install` for matching the torch build to your
driver's CUDA version)::

   pip install -e ".[mjlab]"
   python scripts/train_mjlab.py myoLegWalk-v0 --env.scene.num-envs 2048

Always set ``--env.scene.num-envs`` explicitly (default is 1 — see
:doc:`quickstart_ml` for why that stalls training) and re-pass it whenever you
``--agent.resume True`` a long walk run.

Pretrained NPG / DEP-RL weight trees under ``myosuite/agents/`` are **not**
shipped in the pip package and are gitignored. Train your own policy, or use
MuscleMimic checkpoints from Hugging Face (see
``myosuite/integrations/musclemimic/README.md``).
