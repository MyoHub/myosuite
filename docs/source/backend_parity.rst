Backend parity
==============

CPU (Gymnasium) and mjlab (Warp) implementations of the same ``env_id`` must
agree on observation layout, action mapping, and control timing so a policy
trained on GPU can be played back on CPU (verified per family below; treat families
marked *experimental* as untested). Details:
``docs/wiki/cross-backend-contract.md``.

Check CPU physics against frozen rollouts::

   pytest myosuite/tests/test_parity.py -v

Default absolute tolerance is ``1e-6`` (some contact-rich envs are relaxed).
Regenerate a baseline after an *intentional* env change::

   python scripts/generate_parity_baselines.py --env-id myoElbowPose1D6MRandom-v0

What is verified per family (``myosuite/tests/test_mjlab_cpu_twins.py``):

* **Pose, reach, ``myoLegStandRandom``**: one-step parity of observation, reward and
  solved flag against the CPU env (a random reset state written into both).
* **``myoLegWalk``, ``myoLegDirectional*`` and the terrain walks**: only that the
  divergence stays bounded (``test_myo_leg_walk_state_parity.py``, GPU run); trained
  policies can still behave differently on the two backends.
* **Torso exosuit twins** (experimental): the observation is compared with the CPU env
  over short free-running rollouts (``test_torso_exo_observation_matches_cpu``). Their
  two free bodies are 6-DoF chains on mjlab, converted back to the CPU layout. Policy
  transfer between the backends has not been tested yet.

Every twin must log the ``Episode_Metrics/success`` metric (checked for all envs).

MJX is experimental; do not treat MJX numerical match as a merge gate for new
tasks.
