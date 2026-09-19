RL baselines
============

CPU
---

* **Stable-Baselines3** — ``tutorials/4c_Train_SB_policy.ipynb`` (``pip install -e ".[rl]"``)
* **DEP-RL** (walk) — ``tutorials/4a_deprl.ipynb`` (``pip install deprl``)
* **MyoReflex** — ``tutorials/4b_reflex/MyoSuite_MyoReflex_Walk.ipynb``

GPU
---

RSL-RL PPO on mjlab::

   pip install -e ".[mjlab]"
   python scripts/train_mjlab.py myoLegWalk-v0

Pretrained NPG / DEP-RL weight trees under ``myosuite/agents/`` are **not**
shipped in the pip package and are gitignored. Train your own policy, or use
MuscleMimic checkpoints from Hugging Face (see
``myosuite/integrations/musclemimic/README.md``).
