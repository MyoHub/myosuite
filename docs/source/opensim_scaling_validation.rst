OpenSim scaling validation
==========================

``scripts/validate_opensim_scaling.py`` compares the OpenSim-file route of
:func:`~myosuite.core.body_scaling.scale_model` with OpenSim's own Scale Tool on
the same synthetic subject. It measures what subject scaling adds to the
existing MyoLeg/OpenSim conversion difference. It does not establish
equivalence of the two muscle models or accuracy for a measured person. The
committed reference keeps every sampled value, including the substantial
nominal differences, rather than only a pass/fail result.

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
in ``myosuite/tests/data/opensim_scaling/LICENSE-Rajagopal.txt``. The
opensim-models repository carries no licence file of its own; its
`README <https://github.com/opensim-org/opensim-models/blob/d9b05d470b1a481c222372c85b75772faf8f7792/Models/Rajagopal/README.txt>`__
documents the OpenSim format update and muscle damping change. No restricted
body-shape assets or measured subject data are used. The synthetic factors,
static poses and calcaneus landmarks are in ``fixture.json`` alongside the report.

``reference.json`` stores both halves with one line per array. The OpenSim half
was generated with OpenSim ``4.6-2026-03-31-d6972f9e6`` (Python 3.11) on
Windows; its source hashes are of the working-tree files of commit ``ca4ad086``.
The MyoSuite half was regenerated for ``scale_model`` with MuJoCo ``3.11.0``,
MyoSuite ``3.0.0`` and ``myo-sim==0.2.3``. It records the SHA-256 of the script
and of ``body_scaling.py`` with LF line endings, which is what git stores, so the
hashes verify against a checkout on any platform.

Protocol
--------

OpenSim's actual ``ModelScaler`` (the Scale Tool's model-scaling component) runs
in an isolated interpreter. MyoSuite builds the registered ``myoLegWalk-v0``
model as its env does and calls ``scale_model(spec, path)`` with an OpenSim
ScaleSet file written from ``fixture.json``: the route a Scale Tool or
AddBiomechanics result takes. ``scale_model`` maps the Rajagopal segments onto
MyoLeg's bodies and permutes each segment's factors onto the body's own axes,
since MyoLeg's root and torso are z-up where OpenSim's frames are y-up. Both
receive one global 1.1 factor and one nonuniform set with separate pelvis,
femur, tibia and foot factors.

OpenSim runs each set with volume-scaled mass and with preserved mass
distribution, both without a target total mass. ``scale_model`` scales mass with
volume, and ``total_mass`` rescales the whole subject. Rescaling to the nominal
mass reproduces the preserved-mass case for the uniform factor, where it keeps
every body's mass. There is no counterpart for the nonuniform preserved-mass
case, so it is not compared; its OpenSim results remain in the reference.

Twelve static poses include neutral, individual hip/knee/ankle sweeps, and
combined flexion poses. Rotational coordinate names and signs agree. MuJoCo's
fitted polynomial knee equalities are applied before forward kinematics and
before each central finite difference. Its moment arms are ``-dL/dq`` along
those constrained coordinates, using a ``1e-5`` radian perturbation; selecting
only the hinge column of ``actuator_moment`` would omit coupled knee motion.
OpenSim uses ``computeMomentArm`` after coordinate assembly. ``scale_model``
measures the muscle-tendon length ratios that set optimal fibre and tendon slack
lengths in the default pose with MyoLeg's knee and patella couplings satisfied,
as OpenSim assembles it.

Positions are expressed in the pelvis anatomical frame (x forward, y up, z
right), removing different free-root placements and world conventions. Named
joint centres here mean the femur, tibia and talus body origins. The OpenSim
model has no collision sole surfaces. Three common calcaneal landmarks, given
in the calcaneus frame and multiplied by the fixture's calcaneus factors by the
script on both sides, add the calcaneus pose and its depth below the pelvis.
They do not measure how either pipeline scales the foot itself. MyoSuite's
collision soles set the root height of each pose, which the MyoSuite half
records and the regression check below covers.

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

With ``m``/``o`` the MyoSuite/OpenSim values and ``s``/``n`` the scaled and
nominal cases, the scaling discrepancy removes the nominal conversion error:

.. code-block:: text

    one factor s:      (m_s - o_s) - s**k (m_n - o_n)
    several factors:   (m_s - m_n) - (o_s - o_n)

``k`` is 1 for positions, moment arms and torque and 0 for normalized fibre
length and force. The uniform form is zero when both pipelines are exact
similarities, whatever the nominal error. The nonuniform form has no such
reference and still carries some nominal error at flexed poses. The following
are maxima over the fixed grid.

.. list-table:: Cross-model differences on the committed fixture
   :header-rows: 1
   :widths: 34 22 22 22

   * - Quantity
     - Nominal absolute error
     - Uniform discrepancy
     - Nonuniform discrepancy
   * - Joint centres
     - 11.80 mm
     - 2.9e-16 m
     - 0.357 mm
   * - Calcaneus landmarks
     - 11.80 mm
     - 4.1e-16 m
     - 0.357 mm
   * - Landmark depth below the pelvis
     - 11.80 mm
     - 2.9e-16 m
     - 0.357 mm
   * - Muscle moment arms
     - 8.747 mm
     - 2.4 µm
     - 0.740 mm
   * - Normalized fibre length
     - 0.502
     - 0.0000052
     - 0.0193
   * - Normalized tendon force
     - 0.731
     - 0.0000092
     - 0.0164
   * - Six-muscle joint torque
     - 92.50 N m
     - 0.0074 N m
     - 3.19 N m

The neutral joint centres agree within a micrometre, while the converted knee
motion differs at flexion. Under the uniform factor the positions agree to
rounding. MyoSuite's own uniform scaling is exact, so the moment-arm and muscle
residuals are the numerical resolution of OpenSim's moment-arm and
muscle-equilibrium solvers.

The nonuniform position discrepancy follows one documented convention. OpenSim
scales a joint's offset in the child body with the child's factors, while
``scale_model`` scales a child body's position, which includes that offset, with
the parent's. At the walker knee the tibia offset is
``(-8.090, -3.535, -1.485)`` mm, and the femur and tibia factors differ by
``(0.04, 0.02, -0.07)``. So in the neutral pose MyoSuite's tibia and talus
origins lie ``(0.324, 0.071, -0.104)`` mm from OpenSim's, which is most of the
0.357 mm.

The muscles' optimal-length scale ratios agree with OpenSim's within 0.12%;
soleus and tibialis anterior agree exactly. The remaining nonuniform muscle
discrepancies peak for rectus femoris at deep knee flexion (0.0193 normalized
fibre length at 90 degrees), where the converted knee and patella kinematics
already differ, and for ankle torque at 20 degrees of dorsiflexion. They also
include the different response of OpenSim's compliant, pennated muscles to
length changes. The tests do not turn them into an assertion of biomechanical
equivalence.

Body mass scaling ratios agree within ``4e-16`` in the compared cases, including
the uniform preserved-mass case. Uniform inertia scaling ratios agree within
``7e-16``. Nonuniform diagonal inertia scaling ratios differ by up to 0.184
across the sampled bodies, because the two programs scale inertia differently.
To isolate the scaling rule, both runtimes also scale the same synthetic body:
mass 10 kg, diagonal inertia ``(0.1, 0.2, 0.25) kg m^2``, factors
``(1.2, 0.8, 1.1)``. In volume mode both produce mass 10.56 kg.
``scale_model`` stretches the mass distribution by the factors, which gives
diagonal inertia ``(0.082632, 0.298056, 0.316800)``. OpenSim models the body as
a cylinder along its smallest principal moment, which gives
``(0.092928, 0.274560, 0.274560)``, with equal transverse moments. The tests
check that MyoSuite follows its rule exactly; OpenSim's values are recorded.

The rotated gluteal pelvis cylinder dimensions agree exactly under the uniform
factor and within 4 nanometres under the nonuniform factors. The converted psoas
wrap is a sphere where OpenSim has a cylinder, so OpenSim's 0.1 m cylinder length
has no counterpart. Scaling multiplies that missing length by the factor along
the cylinder axis: 0.110 m for the uniform factor and 0.1036 m for the
nonuniform factors. The radius follows different rules: the sphere takes the
mean of the three factors (0.05083 m), and OpenSim's rotated cylinder gives
0.05036 m, a 0.48 mm difference. Matching a single wrap does not establish
arbitrary anisotropic primitive equivalence: shear and elliptical
cross-sections remain outside MuJoCo's primitive representation.

Bounds and regression checks
----------------------------

``tolerances.json`` records the bounds, their meaning and the mutations each
must reject. Each mutation multiplies one segment's factor along one OpenSim
axis by 1.05, all three tibia factors by 1.05, swaps the femur's x and y
factors, or multiplies the six muscles' ``lengthrange`` by 1.03.

.. list-table:: Bounds on the scaling discrepancy
   :header-rows: 1
   :widths: 40 30 30

   * - Quantity
     - Uniform bound
     - Nonuniform bound
   * - Joint centres, landmarks, landmark depth
     - 1e-9 m
     - 0.4 mm
   * - Muscle moment arms
     - 4e-6 m
     - 0.8 mm
   * - Normalized fibre length
     - 8e-6
     - 0.022
   * - Normalized tendon force
     - 1.4e-5
     - 0.019
   * - Six-muscle joint torque
     - 0.012 N m
     - 3.6 N m

The uniform bounds reject every 5% axis error of the pelvis, femur, tibia,
patella, talus and calcaneus, the tibia error and the ``lengthrange`` error.
The nonuniform bounds reject the forward and vertical axis errors of every leg
segment, all pelvis axis errors, the femur axis swap, the tibia error and the
``lengthrange`` error. A 1e-8 m bound on the gluteal cylinder rejects every
pelvis axis error. No bound sees the mediolateral factor of the femur, tibia,
patella, talus or calcaneus under nonuniform scaling, or any torso factor,
because the poses flex in the sagittal plane and the OpenSim reference has no
torso.

A tight self-regression covers these. The fast tests compare every regenerated
MyoSuite value with the committed MyoSuite half, within ``1e-7`` relative,
``1e-9`` absolute in SI units, ``1e-6`` N m for torque and ``1e-8`` kg m^2 for
inertia. MuJoCo 3.16 reproduces the committed half within ``2e-8`` N m and
``1e-9`` kg m^2. The half includes the torso mass, inertia and geom positions,
the root's head site, and the root height of each pose, which follows the
scaled collision soles. The tests show that this check rejects every mutation
above, and that it rejects feet whose collision geoms were left unscaled, which
the landmarks cannot see. A separate test scales MyoLeg's z-up root and torso
with distinct factors per OpenSim axis. It checks that each factor lands on the
body axis that points the same way, and that factors applied to the bodies' own
axes fail.

Reproduction
------------

OpenSim is optional and is not added to MyoSuite's runtime dependencies. The
`official installation instructions
<https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53085346/Scripting+in+Python>`__
support an isolated conda environment. For example:

.. code-block:: text

    conda create -n myosuite-opensim --override-channels -c opensim-org -c conda-forge opensim=4.6 python=3.11
    python scripts/validate_opensim_scaling.py --opensim-python <opensim-environment-python>
    python -m pytest myosuite/tests/test_opensim_scaling_validation.py -q

The script writes ``tasks/opensim/reference.json`` and
``tasks/opensim/comparison.json``, and exits unsuccessfully if a bound is
exceeded. ``--refresh-myosuite`` keeps the committed OpenSim half and
regenerates only the MyoSuite half, without OpenSim. Neither overwrites the
committed reference by default. To regenerate it deliberately, pass
``--output myosuite/tests/data/opensim_scaling/reference.json`` and review all
provenance, measurement and tolerance changes. Do not relax bounds to hide a
new discrepancy.

For a fresh independent OpenSim run, set ``OPENSIM_PYTHON`` to that environment's
Python and run:

.. code-block:: text

    python -m pytest myosuite/tests/test_opensim_scaling_validation.py -m slow -q

The live oracle test downloads the pinned model, executes OpenSim, and compares
every stored OpenSim value with the new run. The fixed fixture is a reproducible
characterization, not a substitute for validating a new subject's scale
estimate, motion, contacts or policy performance.
