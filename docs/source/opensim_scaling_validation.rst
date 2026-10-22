OpenSim scaling validation
==========================

The reproducible comparison in ``scripts/validate_opensim_scaling.py`` measures
what subject scaling adds to the existing MyoLeg/OpenSim conversion difference.
It does not establish equivalence of the two muscle models or accuracy for a
measured person. The committed report contains every sampled value, including
the substantial nominal differences, rather than only a pass/fail result.

Fixture and provenance
----------------------

The reference uses the authoritative
`Rajagopal2016 model <https://github.com/opensim-org/opensim-models/blob/d9b05d470b1a481c222372c85b75772faf8f7792/Models/Rajagopal/Rajagopal2016.osim>`__,
downloaded at commit ``d9b05d470b1a481c222372c85b75772faf8f7792``. Its SHA-256 is
``3f5c5f23e486073f2ad2aa4a4967ffe2fcdd582b1e355512bc54f70c36376bf4``.
Downloads and cached files must match this hash. The model is fetched on demand;
visual meshes are unnecessary for these kinematic and muscle measurements.

The original authors' `SimTK Full Body Model project
<https://simtk.org/projects/full_body>`__ identifies package 1738 as the MIT Use
Agreement, copyright 2015 Stanford University. The complete notice is retained
in ``myosuite/tests/data/opensim_scaling/LICENSE-Rajagopal.txt``. The upstream
`README <https://github.com/opensim-org/opensim-models/blob/d9b05d470b1a481c222372c85b75772faf8f7792/Models/Rajagopal/README.txt>`__
documents the OpenSim format update and muscle damping change. No restricted
body-shape assets or measured subject data are used. The synthetic factors,
static poses and sole landmarks are in ``fixture.json`` alongside the report.

The checked-in ``reference.json`` was generated using OpenSim
``4.6-2026-03-31-d6972f9e6``, MuJoCo ``3.11.0``, MyoSuite ``3.0.0`` and
``myo-sim==0.2.3`` on Windows. It records both Python versions, source-code hashes,
compiled MyoLeg model hashes, the model download hash, the complete fixture and
the complete raw results. MuJoCo model hashes are version-specific. Reproducing
the measurements is preferable to requiring the same binary hash across systems.

Protocol
--------

OpenSim's actual ``ModelScaler`` (the Scale Tool's model-scaling component) runs
in an isolated interpreter. MyoSuite uses the registered ``myoLegWalk-v0`` model
recipe and ``build_subject_model`` with ``RAJAGOPAL_MYOLEG_SEGMENTS``. Both receive
identical local-axis factors: one global 1.1 factor, and one nonuniform set with
separate pelvis, femur, tibia and foot factors. Each set runs with volume-scaled
mass and with preserved mass distribution. Neither pipeline applies a target
total mass; absolute nominal body masses differ and mass scaling ratios are
compared separately.

Twelve static poses include neutral, individual hip/knee/ankle sweeps, and
combined flexion poses. Rotational coordinate names and signs agree. MuJoCo's
fitted polynomial knee equalities are applied before forward kinematics and
before each central finite difference. Its moment arms are ``-dL/dq`` along
those constrained coordinates, using a ``1e-5`` radian perturbation; selecting
only the hinge column of ``actuator_moment`` would omit coupled knee motion.
OpenSim uses ``computeMomentArm`` after coordinate assembly.

Positions are expressed in the pelvis anatomical frame (x forward, y up, z
right), removing different free-root placements and world conventions. Named
joint centres here mean the femur, tibia and talus body origins. The OpenSim
model has no collision sole surfaces. Three explicitly specified common
calcaneal sole landmarks therefore define the cross-model floor-offset
comparison. They scale in the calcaneus frame in both pipelines. MyoSuite's
actual collision-sole root offset is additionally recorded separately; these
landmarks do not establish contact equivalence or dynamic balance.

The six muscles are iliacus, rectus femoris, vastus medialis, soleus, medial
gastrocnemius and tibialis anterior on the right. The report contains their
moment arms, normalized fibre lengths and operating ranges, normalized tendon
forces, and their summed joint torques across the angle sweeps. Activation is
one and coordinate velocities are zero. OpenSim equilibrates compliant Millard
muscles with pennation; MuJoCo uses its inextensible tendon muscle model. Its
fibre coordinate uses the optimal length and slack length derived from
``lengthrange`` and the actuator's normalized range. Torque includes the six
listed muscles only, not all leg muscles or passive joint forces.

Measured differences
--------------------

For each sampled quantity the report stores absolute maximum/RMS disagreement
and the maximum scaling increment discrepancy:

.. code-block:: text

    (MyoSuite_scaled - MyoSuite_nominal) - (OpenSim_scaled - OpenSim_nominal)

The following are maxima over the fixed grid. The mass mode does not change
these kinematic/isometric muscle measurements.

.. list-table:: Cross-model differences on the committed fixture
   :header-rows: 1
   :widths: 34 22 22 22

   * - Quantity
     - Nominal absolute error
     - Uniform increment
     - Nonuniform increment
   * - Joint centres
     - 11.80 mm
     - 1.180 mm
     - 0.357 mm
   * - Common sole landmarks / floor offset
     - 11.80 mm
     - 1.180 mm
     - 0.357 mm
   * - Muscle moment arms
     - 8.747 mm
     - 0.875 mm
     - 0.740 mm
   * - Normalized fibre length
     - 0.502
     - 0.0000052
     - 0.0775
   * - Normalized tendon force
     - 0.731
     - 0.0000092
     - 0.1050
   * - Six-muscle joint torque
     - 92.50 N m
     - 9.253 N m
     - 13.94 N m

The neutral joint centres agree within a micrometre, while the converted knee
motion differs at flexion. Uniform length/torque increments include the
corresponding magnification of the nominal conversion error. Uniform normalized
muscle increments remain very small. Nonuniform scaling materially changes
normalized operating lengths and forces relative to the OpenSim reference.
Those differences remain visible; the tests do not turn them into an assertion
of biomechanical equivalence.

Body mass scaling ratios agree within ``2.3e-16`` for both mass modes. Uniform
inertia scaling ratios also agree to floating-point precision. Nonuniform
diagonal inertia scaling ratios differ by up to 0.184 (volume) and 0.175
(preserved mass) across the sampled bodies. Nominal inertias differ as well.
To isolate the scaling rule, both runtimes also scale the same synthetic body:
mass 10 kg, diagonal inertia ``(0.1, 0.2, 0.25) kg m^2``, factors
``(1.2, 0.8, 1.1)``. In volume mode both produce mass 10.56 kg. MuJoCo's affine
distribution transform produces diagonal inertia
``(0.082632, 0.298056, 0.316800)``; OpenSim's body scaling produces
``(0.092928, 0.274560, 0.274560)``. The report retains both mass modes.

The rotated gluteal pelvis cylinder dimensions agree within 4 nanometres on this
factor set. The converted psoas wrap is already a sphere where OpenSim has a
cylinder: its missing 0.1 m cylinder length is a nominal representation change.
Nonuniform scaling adds a 3.60 mm dimension discrepancy. Matching a single wrap
does not establish arbitrary anisotropic primitive equivalence: shear and
elliptical cross-sections remain outside MuJoCo's primitive representation.

Reproduction and regression checks
----------------------------------

OpenSim is optional and is not added to MyoSuite's runtime dependencies. The
`official installation instructions
<https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53085346/Scripting+in+Python>`__
support an isolated conda environment. For example:

.. code-block:: text

    conda create -n myosuite-opensim --override-channels -c opensim-org -c conda-forge opensim=4.6 python=3.11
    python scripts/validate_opensim_scaling.py --opensim-python <opensim-environment-python>
    python -m pytest myosuite/tests/test_opensim_scaling_validation.py -q

The script writes ``tasks/opensim/report.json`` and exits unsuccessfully if a
documented increment bound is exceeded. It does not overwrite the committed
reference by default. To regenerate that reference deliberately, pass
``--output myosuite/tests/data/opensim_scaling/reference.json`` and review all
provenance, measurement and tolerance changes. Do not relax bounds to hide a
new discrepancy.

Fast tests rerun MyoSuite against the stored independent OpenSim measurements.
``tolerances.json`` records their meaning and measurement-based limits. Every
one of the seven limits in both scaling regimes and both mass modes is shown
to reject a deliberate 5% tibia-factor error, swapped nonuniform femur axes, or
3% muscle ``lengthrange`` error. The nonuniform inertia disagreement and the
nominal wrap substitution are also checked explicitly.

For a fresh independent OpenSim run, set ``OPENSIM_PYTHON`` to that environment's
Python and run:

.. code-block:: text

    python -m pytest myosuite/tests/test_opensim_scaling_validation.py -m slow -q

The live oracle test downloads the pinned model, executes OpenSim, and compares
all raw sampled quantities with the recorded external reference. The fixed
fixture is a reproducible characterization, not a substitute for validating a
new subject's scale estimate, motion, contacts or policy performance.
