Newton validation
=================

Newton remains an external compatibility experiment. MyoSuite supports CPU
MuJoCo and mjlab; these measurements do not register another backend or establish
policy playback or training compatibility.

Released Newton 1.8.0 (2026-11-15)
----------------------------------

Released ``newton==1.8.0`` was validated on Windows 11, Python 3.12.16,
``warp-lang==1.19.0``, ``mujoco==3.17.0``, ``mujoco-warp==3.17.0``,
``musclemimic_models==1.0.6``, and an NVIDIA GeForce RTX 4070 Ti SUPER (16 GB,
CUDA 13.4).

Newton 1.8.0 incorporates the upstream MJCF fixes for joint equalities
(`Newton PR 5029 <https://github.com/newton-physics/newton/pull/5029>`__), partial
vector defaults and mesh inertia
(`Newton PR 5030 <https://github.com/newton-physics/newton/pull/5030>`__,
`Newton PR 5123 <https://github.com/newton-physics/newton/pull/5123>`__), compiler
mass and inertia bounds
(`Newton PR 3742 <https://github.com/newton-physics/newton/pull/3742>`__), binary
``.msh`` meshes
(`Newton PR 4461 <https://github.com/newton-physics/newton/pull/4461>`__),
hyphenated spatial tendon sites
(`Newton PR 4595 <https://github.com/newton-physics/newton/pull/4595>`__), the
``<muscle>`` actuator shortcut
(`Newton PR 3752 <https://github.com/newton-physics/newton/pull/3752>`__), and
``actuator_lengthrange`` preservation
(`Newton PR 4415 <https://github.com/newton-physics/newton/pull/4415>`__).

**Conversion.** With ``.github/newton_probe/newton_compat_probe.py --strict``, all
twelve authored and normalized cases import with meshes and convert. Body, joint,
DoF, tendon, wrap and authored equality counts, total body mass (18.6, 4.63,
41.8, 75.2, 84.3, and 16.5 kg), every muscle actuator, and every muscle
``lengthrange`` (largest abs diff 2.87e-8 m) match. Static muscle force max
absolute diffs remain 0.000249 N (elbow), 0.00529 N (hand), 0.00935 N (arm),
0.00257 N (leg), 0.00464 N (full body), and 0.00212 N (bimanual). Strict mode
still exits with status 1 due to:

* Hierarchical body/joint/site export naming (`Newton issue 4904 <https://github.com/newton-physics/newton/issues/4904>`__).
* Unexported MJCF ``<sensor>`` definitions (4 on Leg, 110 on Full body, 106 on Bimanual; `Newton issue 5147 <https://github.com/newton-physics/newton/issues/5147>`__, `Newton PR 5157 <https://github.com/newton-physics/newton/pull/5157>`__).
* Default ``SolverMuJoCo`` solver options when not overridden by the caller (``integrator: euler -> implicitfast``, ``timestep: 0.001 -> 0.002``, and a synthetic root ``weld`` row when ``kinematic_switchable_roots=True``).

**Matched 128-step trajectories.** With ``mujoco-warp==3.17.0``
(`MuJoCo-Warp PR 1717 <https://github.com/google-deepmind/mujoco_warp/pull/1717>`__),
the wrist wrap trajectory drift previously observed on ``Hand``, ``Arm``, and
``Bimanual`` (`MuJoCo-Warp issue 1835 <https://github.com/google-deepmind/mujoco_warp/issues/1835>`__)
is resolved: all four upper-limb models pass the fixed bounds (``qpos``
``rtol=1e-4, atol=1e-4``, muscle force ``rtol=1e-3, atol=0.05`` N) across 2 poses
and 128 steps on both CPU MuJoCo and CUDA MuJoCo-Warp.

.. list-table:: Released Newton 1.8.0 (128 steps, 2 poses)
   :header-rows: 1
   :widths: 20 12 16 18 12 16 18

   * - Model
     - CPU
     - CPU max qpos
     - CPU max force (N)
     - CUDA
     - CUDA max qpos
     - CUDA max force (N)
   * - Elbow
     - Pass
     - 9.26e-7
     - 0.000355
     - Pass
     - 3.60e-6
     - 0.000590
   * - Hand
     - Pass
     - 2.58e-5
     - 0.003128
     - Pass
     - 2.57e-5
     - 0.003206
   * - Arm
     - Pass
     - 3.24e-6
     - 0.013120
     - Pass
     - 2.74e-5
     - 0.025074
   * - Leg
     - Fail
     - 0.013759
     - 113.213013
     - Fail
     - 0.014338
     - 113.212899
   * - Full body
     - Fail
     - 0.013615
     - 111.197558
     - Fail
     - 0.013615
     - 111.197333
   * - Bimanual
     - Pass
     - 1.02e-5
     - 0.002992
     - Pass
     - 3.61e-5
     - 0.046053

``Leg`` and ``Full body`` fail at steps 34--36 in released ``newton==1.8.0`` on
both CPU and CUDA because ``SolverMuJoCo`` in 1.8.0 still zeroes authored
``geom_margin=0.001`` m model-wide whenever a box geom is present. Post-1.8
`Newton PR 5125 <https://github.com/newton-physics/newton/pull/5125>`__ preserves
margins on ``use_mujoco_cpu=True``, and
`Newton PR 5378 <https://github.com/newton-physics/newton/pull/5378>`__
(`Newton issue 5377 <https://github.com/newton-physics/newton/issues/5377>`__)
preserves them on the MuJoCo Warp backend (reducing ``Full body`` CUDA max
``qpos`` drift to ``3.54e-4`` rad / ``3.07`` N at step 40, matching CPU
``3.42e-4`` rad / ``3.07`` N). On ``Leg``, CUDA additionally diverges after foot
strike due to single-contact capsule--hfield collision on MuJoCo Warp
(`MuJoCo-Warp issue 1835 <https://github.com/google-deepmind/mujoco_warp/issues/1835>`__).

**Differentiability and environment co-installation.** Recording ``SolverMuJoCo.step``
under ``wp.Tape()`` yields zero ``ctrl.grad`` in Newton 1.8.0 because ``SolverMuJoCo``
runs forward-only graph capture without backward adjoints
(`Newton PR 5271 <https://github.com/newton-physics/newton/pull/5271>`__), and
MuJoCo Warp's analytic adjoint pass (`MuJoCo-Warp PR 1535 <https://github.com/google-deepmind/mujoco_warp/pull/1535>`__)
does not yet support tendon transmissions, spatial geom wrapping, or Hill muscle
gain/bias VJPs (`MuJoCo-Warp issue 1970 <https://github.com/google-deepmind/mujoco_warp/issues/1970>`__).
Installing ``myosuite``, ``mjlab``, and upcoming Newton 1.9 (`Newton PR 5337 <https://github.com/newton-physics/newton/pull/5337>`__)
in one environment is tracked in `MyoSuite issue 818 <https://github.com/MyoHub/myosuite/issues/818>`__.

**Batched rollout probe (experimental).** For external compatibility probes requiring
multi-world rollouts on ``newton>=1.8.0``, :mod:`myosuite.integrations.newton` provides
the experimental ``NewtonBatchSim`` and ``NewtonAccessor`` (an
:class:`~myosuite.core.protocols.EnvAccessor` adapter). It replicates a MyoSuite MJCF
across ``N`` worlds with ``SolverMuJoCo``, remaps merged-joint coordinates, controls,
activations, and sites back into authored MuJoCo order, runs MyoSuite's sigmoid
normalization and portable ``ExcitationStage`` pipeline, and evaluates pure
:mod:`myosuite.terms` observation and reward functions without registering a third
``make_env`` backend.

Historical baseline: 2026-10-23 (PR 4903)
-----------------------------------------

The pre-1.8 baseline below tested Newton revision
``4cd047d3d4095c0b19786469f1a5041d049d703e``
(`upstream PR 4903 <https://github.com/newton-physics/newton/pull/4903>`__), based on
``12d03b030aeaaf38956e62a76a01760ec36f3747``, on Windows, Python 3.12.10, Warp
1.18.0, MuJoCo and MuJoCo-Warp 3.15.0, ``musclemimic_models`` 1.0.6, and an
NVIDIA RTX 4050 Laptop GPU (6 GB, driver 610.60). PR 4903 bundled import fixes
together with local default changes for ``default_shape_cfg.gap``, ``integrator``,
``enable_multiccd``, and CPU margin preservation before PR 4903 was split into
upstream PRs (`Newton PR 5029 <https://github.com/newton-physics/newton/pull/5029>`__,
`Newton PR 5030 <https://github.com/newton-physics/newton/pull/5030>`__).

Conversion and names (PR 4903)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All twelve authored/normalized cases import with their meshes and convert.
Body, joint, DoF, tendon, wrap, equality and sensor counts, mass and the measured
solver options agree. Every muscle is retained and all length ranges agree at
the configured tolerance (the largest absolute range difference is 1.49e-8 m).
Authored inputs preserve the scene meshes; the normalized diagnostic still
removes binary mesh geoms, as described in :doc:`backend_parity`.

The strict conversion command still exits with status 1. Raw body, joint and site
names change, and static muscle forces exceed the narrow default bounds
(``rtol=1e-6``, ``atol=1e-8``). Maximum authored static force errors are
0.000249 N (elbow), 0.005292 N (hand), 0.009347 N (arm), 0.002569 N (leg),
0.004643 N (full body), and 0.002121 N (bimanual). These differences are reported
separately from structural preservation.

The numerical harness records complete one-to-one source-to-converted identity
maps for bodies, joints and sites in all six models. Hierarchical namespace
prefixes and generated joint/shape suffixes explain the changed names; the
scalar joint identities are recovered through per-DoF labels or, where Newton
merges a body's joints without them, through the merged joint's name and DoF
order. A stable public mapping API and exact-name round trips are tracked in
`Newton issue 4904 <https://github.com/newton-physics/newton/issues/4904>`__.
Raw names are not counted as preserved by the strict probe.

Matched trajectories (PR 4903, MuJoCo-Warp 3.15.0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``.github/newton_probe/newton_dynamics_probe.py`` compares the original authored native CPU
model with Newton's converted CPU MuJoCo or CUDA MuJoCo-Warp solver. It uses
the default pose and a deterministic small scalar-joint perturbation, identical
sinusoidal muscle controls, and 128 steps at the authored timestep and integrator.
The bounds were fixed before these runs: qpos ``rtol=1e-4, atol=1e-4``
(radians/metres), and muscle force ``rtol=1e-3, atol=0.05`` N. The force bounds
reject a 10% muscle force/scale mutation in every model and pose. The qpos bounds
reject the earlier incorrect leg contact-gap import by more than three orders of
magnitude.

Each row gives the largest absolute errors over both poses and all steps.
Pass/fail uses the component-wise relative and absolute bounds, rather than
only the maxima shown here. Contact capacity is 2048 and constraint capacity
8192, with Newton's strict capacity checking enabled.

.. list-table:: Native CPU reference versus Newton CPU MuJoCo (PR 4903)
   :header-rows: 1
   :widths: 25 15 20 25

   * - Model
     - Result
     - Max qpos error
     - Max force error (N)
   * - Elbow
     - Pass
     - 9.26e-7
     - 0.000355
   * - Hand
     - Pass
     - 2.58e-5
     - 0.003127
   * - Arm
     - Pass
     - 3.23e-6
     - 0.013120
   * - Leg
     - Pass
     - 2.04e-6
     - 0.013303
   * - Full body
     - Fail
     - 3.42e-4
     - 3.073153
   * - Bimanual
     - Pass
     - 1.02e-5
     - 0.002993

The full-body CPU comparison first exceeds qpos bounds at step 40 in both poses.
Its force peaks occur in the soleus muscles. The precise remaining cause has
not been established; Newton's float32 model storage is a plausible source of
round-trip perturbations, rather than evidence of exact CPU equivalence.

.. list-table:: Native CPU reference versus Newton CUDA MuJoCo-Warp (PR 4903, MuJoCo-Warp 3.15.0)
   :header-rows: 1
   :widths: 25 15 20 25

   * - Model
     - Result
     - Max qpos error
     - Max force error (N)
   * - Elbow
     - Pass
     - 3.60e-6
     - 0.000590
   * - Hand
     - Fail
     - 0.001325
     - 0.359166
   * - Arm
     - Fail
     - 0.004090
     - 4.898774
   * - Leg
     - Fail
     - 0.018801
     - 588.226064
   * - Full body
     - Fail
     - 0.013615
     - 111.197886
   * - Bimanual
     - Fail
     - 0.002998
     - 69.780513

Independent contact and solver checks
-------------------------------------

An omitted MJCF geom gap inherits Newton's speculative ``default_shape_cfg.gap``
of 0.1 m unless the caller sets ``builder.default_shape_cfg.gap = 0.0`` before
``add_mjcf`` (`MyoSuite PR 819 <https://github.com/MyoHub/myosuite/pull/819>`__).
With ``gap = 0.0`` and preserved margins, an independent lowered-root audit on
MyoLeg finds identical contact-pair multisets at 16, 53 and 72 contacts, distance
errors below 9.49e-8 m, and matching collision properties for all 56 named geoms.

Directly transferring the authored upper-limb models to MuJoCo-Warp 3.17.0
without Newton passes 128-step rollouts on both CPU and CUDA for ``Elbow``,
``Hand``, ``Arm``, and ``Bimanual``. The direct ``Leg`` comparison passes with
contacts disabled and diverges after landing with contacts enabled, isolating
capsule--hfield contact differences in
`MuJoCo-Warp issue 1835 <https://github.com/google-deepmind/mujoco_warp/issues/1835>`__.

Reproduce and download
----------------------

Use a separate environment for Newton and install the released package::

   uv venv --python 3.12 .venv-newton
   uv pip install --python .venv-newton "newton[sim]==1.8.0" "musclemimic_models==1.0.6" -e .
   .venv-newton/bin/python .github/newton_probe/newton_compat_probe.py --strict --out newton_conversion
   .venv-newton/bin/python .github/newton_probe/newton_dynamics_probe.py --engine cpu --device cpu --poses 2 --steps 128 --out newton_cpu
   .venv-newton/bin/python .github/newton_probe/newton_dynamics_probe.py --engine warp --device cuda:0 --poses 2 --steps 128 --out newton_cuda

On Windows use ``.venv-newton/Scripts/python.exe``. The numerical harness writes
its report incrementally and exits with status 1 when any case fails the bounds.
It is a manual diagnostic; the release/manual conversion workflow remains
informational and does not run these GPU trajectories.

The reports themselves are not stored in the repository. The commands above
regenerate the ``newton==1.8.0`` conversion and trajectory reports, with
dependency metadata in each JSON report, and the release workflow
(``.github/workflows/newton-probe.yml``) uploads its conversion report as the
``newton-probe-<version>`` run artifact. Broader GPU equivalence, differentiability,
sensor/name export, and real policy playback remain tracked in
`MyoSuite issue 563 <https://github.com/MyoHub/myosuite/issues/563>`__.
