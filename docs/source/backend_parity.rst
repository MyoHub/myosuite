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

* **Pose, reach, ``myoLegStandRandom``, ``myoLegWalk`` (+ Sarc/Fati), ``myoLegDirectional*``
  and the terrain walks**: 25-step parity of observation, reward, termination and solved
  flag against the CPU env (the CPU state is written into both at every step). Tolerances
  are looser where contacts dominate: foot-contact events (walking), joint velocities of
  5-10 rad/s in the first steps (directional) and height-field terrain, where MuJoCo Warp
  creates at most one capsule-hfield contact (see the constants in
  ``myosuite/tests/test_mjlab_cpu_twins.py``). Trajectories of trained locomotion policies
  still drift apart over hundreds of steps, so re-check them on the CPU env.
* **Torso exosuit twins** (experimental): the observation is compared with the CPU env
  over short free-running rollouts (``test_torso_exo_observation_matches_cpu``). Their
  two free bodies are 6-DoF chains on mjlab, converted back to the CPU layout. Policy
  transfer between the backends has not been tested yet.

Every twin must log the ``Episode_Metrics/success`` metric (checked for all envs).

MJX is experimental; do not treat MJX numerical match as a merge gate for new
tasks.
