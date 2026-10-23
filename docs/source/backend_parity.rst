Backend parity
==============

CPU (Gymnasium) and mjlab (Warp) implementations of the same ``env_id`` must
agree on observation layout, action mapping, and control timing so a policy
trained on GPU can be played back on CPU (verified per family below; treat families
marked *experimental* as untested). Details:
``docs/wiki/cross-backend-contract.md``.

Which muscle-command features (noise, fatigue, sarcopenia, ...) each env and backend accepts is
tabulated in the same page ("Which features run where").

Check CPU physics against frozen rollouts::

   pytest myosuite/tests/test_parity.py -v

Default absolute tolerance is ``1e-6`` (some contact-rich envs are relaxed).
Regenerate a baseline after an *intentional* env change::

   python scripts/generate_parity_baselines.py --env-id myoElbowPose1D6MRandom-v0

What is verified per family (``myosuite/tests/test_mjlab_cpu_twins.py``):

* **Pose, reach, ``myoLegStandRandom``, ``myoLegWalk`` (+ Sarc/Fati), ``myoLegDirectional*``
  and the terrain walks**: 25-step parity of observation, reward, termination and solved
  flag against the CPU env (the CPU state is written into both at every step). Random
  actions rarely reach a pose or reach target, so for those the solved flag and bonus are
  also checked with the target moved near the reached state
  (``test_success_parity_near_target``). Tolerances
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

Newton compatibility probe
--------------------------

`NVIDIA Newton <https://github.com/newton-physics/newton>`_ is not a MyoSuite backend.
``scripts/newton_compat_probe.py`` measures how much of a MyoSuite model survives
Newton's MJCF import, so the gap can be followed across Newton releases.

For the elbow, hand, arm and leg env models and the MuscleMimic full-body and
bimanual models, the probe compiles the model with MuJoCo, imports the same file
into Newton, converts it back to MuJoCo with Newton's ``SolverMuJoCo`` and compares
the two. It reports the actuators and muscles that were lost, the muscle length
ranges that changed, how far the muscle forces moved (at the default pose, with all
activations and controls at 0.5), and the bodies, joints, tendons, equality
constraints and sensors on both sides. Each model is tried twice: as MyoSuite writes
it, and rewritten by MuJoCo into one plain file with ``-`` in names replaced by ``_``
(Newton 1.6.1 and 1.7.0 look up a spatial tendon's sites by their sanitized names, so they
drop hyphenated sites; `newton#4566 <https://github.com/newton-physics/newton/issues/4566>`_).
The rewrite also removes geoms using MuJoCo's binary ``.msh`` meshes, which
Newton 1.7.0 cannot read (upstream PR 4903 adds support). The normalized comparison uses that modified model as its
reference; it does not establish that normalization preserved every property of
the authored model. Remaining differences can originate in either Newton's
importer or its conversion.

Run it on the CPU, in an environment with Newton and without mjlab (the two need
different ``mujoco-warp`` versions)::

   uv venv --python 3.12 .venv-newton
   uv pip install --python .venv-newton "newton[sim]==1.7.0" "musclemimic_models==1.0.6" -e .
   .venv-newton/bin/python scripts/newton_compat_probe.py --out newton_probe

It prints a table and writes ``newton_probe/newton_probe.md`` and
``newton_probe.json``. ``--strict`` makes it exit with an error when a model fails
to load or convert, loses named elements or muscles, or changes the measured
structure, options, muscle length ranges or forces. Non-finite values fail the
comparison. Tolerances must be finite and non-negative. A converted model with
differences is labelled ``differences`` rather than ``ok``.
The *Newton probe* GitHub workflow runs the same command on the release pull request
(`dev` into `main`, not as a draft) with the latest Newton, and when started by hand
for a chosen version (Actions > Newton probe > Run workflow). It keeps the report as a
download and in the run summary. Installation and execution failures also produce
a summary and logs; the job is allowed to fail and does not gate the release.

The probe measures model conversion and forces at one pose on CPU MuJoCo. It
does not run Newton's GPU solver, trajectories, policy playback or training.
Newton remains outside MyoSuite's supported CPU and mjlab backends.
Remaining conversion blockers and the acceptance criteria for further validation
are tracked in `#563 <https://github.com/MyoHub/myosuite/issues/563>`_.

The separate manual trajectory diagnostic, matched CPU/CUDA measurements, source
identity maps and upstream fixes are recorded in :doc:`newton_validation`.

.. toctree::
   :hidden:

   newton_validation
