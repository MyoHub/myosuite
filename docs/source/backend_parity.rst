Backend parity
==============

CPU (Gymnasium) and mjlab (Warp) implementations of the same ``env_id`` must
agree on observation layout, action mapping, and control timing so a policy
trained on GPU can be played back on CPU. Details:
``docs/wiki/cross-backend-contract.md``.

Check CPU physics against frozen rollouts::

   pytest myosuite/tests/test_parity.py -v

Default absolute tolerance is ``1e-6`` (some contact-rich envs are relaxed).
Regenerate a baseline after an *intentional* env change::

   python scripts/generate_parity_baselines.py --env-id myoElbowPose1D6MRandom-v0

MJX is experimental; do not treat MJX numerical match as a merge gate for new
tasks.
