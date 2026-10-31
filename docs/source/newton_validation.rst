Newton validation: 2026-10-23
=============================

Newton remains an external compatibility experiment. MyoSuite supports CPU
MuJoCo and mjlab; these measurements do not register another backend or establish
policy playback or training compatibility.

The tested Newton revision is ``4cd047d3d4095c0b19786469f1a5041d049d703e``
(`upstream PR 4903 <https://github.com/newton-physics/newton/pull/4903>`_), based on
``12d03b030aeaaf38956e62a76a01760ec36f3747``. The environment was Windows,
Python 3.12.10, Warp 1.18.0, MuJoCo and MuJoCo-Warp 3.15.0,
musclemimic_models 1.0.6, and an NVIDIA RTX 4050 Laptop GPU with 6 GB and
driver 610.60. The upstream branch is unreleased; reproducing these results with
the next released Newton remains pending in
`MyoSuite issue 563 <https://github.com/MyoHub/myosuite/issues/563>`_.

Upstream status, 2026-10-31
---------------------------

PR 4903 has been split. `Newton PR 5029 <https://github.com/newton-physics/newton/pull/5029>`_
(joint equality endpoints and partial ``polycoef``) and
`Newton PR 5030 <https://github.com/newton-physics/newton/pull/5030>`_
(partial vector attributes) carry the import fixes onto Newton main
``2e20d82d``. With both, the conversion probe again converts all twelve cases
with every muscle and length range and the static force errors listed below.
Body masses of the elbow and bimanual models, the integrator and the sensors
still differ: the mesh inertia, compiler bound, option, geom gap, multi-CCD and
CPU margin changes of PR 4903 alter Newton defaults and are not part of those
two PRs. The trajectory tables below were measured with all of PR 4903 applied.

Conversion and names
--------------------

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
scalar joint identities can be recovered through per-DoF metadata. A stable
public mapping API and exact-name round trips are tracked in
`Newton issue 4904 <https://github.com/newton-physics/newton/issues/4904>`_.
Raw names are not counted as preserved by the strict probe.

Matched trajectories
--------------------

``.github/newton_probe/newton_dynamics_probe.py`` compares the original authored native CPU
model with Newton's converted CPU MuJoCo or CUDA MuJoCo-Warp solver. It uses
the default pose and a deterministic small scalar-joint perturbation, identical
sinusoidal muscle controls, and 128 steps at the authored timestep. The bounds
were fixed before these runs: qpos ``rtol=1e-4, atol=1e-4`` (radians/metres), and
muscle force ``rtol=1e-3, atol=0.05`` N. The force bounds reject a 10% muscle
force/scale mutation in every model and pose. The qpos bounds reject the earlier
incorrect leg contact-gap import by more than three orders of magnitude.

Each row gives the largest absolute errors over both poses and all steps.
Pass/fail uses the component-wise relative and absolute bounds, rather than
only the maxima shown here. Contact capacity is 2048 and constraint capacity
8192, with Newton's strict capacity checking enabled.

.. list-table:: Native CPU reference versus Newton CPU MuJoCo
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

.. list-table:: Native CPU reference versus Newton CUDA MuJoCo-Warp
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

The CUDA solver also warns that it zeros some authored geom margins for its
native-CCD compatibility path. This changes contact semantics and is retained
in the reports. No bounds were relaxed to make these failures pass.

Independent contact and solver checks
-------------------------------------

An omitted MJCF geom gap previously inherited Newton's speculative gap of 0.1 m.
For MyoLeg this created 115 initial contacts where native MuJoCo had zero.
PR 4903 preserves the source gap of zero. An independent lowered-root audit now
finds identical contact-pair multisets at 16, 53 and 72 contacts, distance errors
below 9.49e-8 m, and matching collision properties for all 56 named geoms.
The leg CPU rollout above reaches 16-18 contacts and passes the fixed bounds.

Directly transferring the same authored model to MuJoCo-Warp, without importing
Newton, reproduces wrist trajectory drift for hand, arm and bimanual models.
The matched control with contacts disabled still fails; GPU buffer overflow
counters remain zero. The direct leg comparison passes with contacts disabled
and diverges after landing with contacts enabled. These observations separate
solver/contact behavior from Newton import errors without assigning an
unproven cause. Wrap sensitivity and run-to-run force-peak variation remain
under investigation in
`MuJoCo-Warp issue 1835 <https://github.com/google-deepmind/mujoco_warp/issues/1835>`_.

Reproduce and download
----------------------

Use a separate environment for Newton and install the exact upstream revision::

   uv venv --python 3.12 .venv-newton
   uv pip install --python .venv-newton "newton[sim] @ git+https://github.com/M-Colley/newton.git@4cd047d3d4095c0b19786469f1a5041d049d703e" "musclemimic_models==1.0.6" -e .
   .venv-newton/bin/python .github/newton_probe/newton_compat_probe.py --strict --out newton_conversion
   .venv-newton/bin/python .github/newton_probe/newton_dynamics_probe.py --engine cpu --device cpu --poses 2 --steps 128 --newton-revision 4cd047d3d4095c0b19786469f1a5041d049d703e --out newton_cpu
   .venv-newton/bin/python .github/newton_probe/newton_dynamics_probe.py --engine warp --device cuda:0 --poses 2 --steps 128 --newton-revision 4cd047d3d4095c0b19786469f1a5041d049d703e --out newton_cuda

On Windows use ``.venv-newton/Scripts/python.exe``. The numerical harness writes
its report incrementally and exits with status 1 when any case fails the bounds.
It is a manual diagnostic; the release/manual conversion workflow remains
informational and does not run these GPU trajectories.

The reports themselves are not stored in the repository. The commands above
regenerate every table on this page, with dependency metadata and the source
revision in each JSON report, and the release workflow
(``.github/workflows/newton-probe.yml``) uploads its conversion report as the
``newton-probe-<version>`` run artifact.

The focused upstream suite ran 118 tests with 6 optional-USD skips; all executed
tests and pre-commit hooks passed. New regressions fail on the unmodified
baseline. Upstream hosted checks require maintainer approval for this fork.
Released-version validation, broader GPU equivalence, timing and real policy
playback remain open acceptance criteria in issue 563.
