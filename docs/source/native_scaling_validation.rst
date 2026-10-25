Native source scaling validation
================================

The optional adapters in ``myosuite.integrations.capture.native_sources`` use
upstream model evaluators to turn locally held data into the existing
``MarkerTrial`` interface. They introduce no native-reader dependencies into the
scaling core. Source originals remain local. The validation command publishes
only aggregate errors and hashes: no meshes, joint trajectories, qpos, shape
coefficients, individual scale factors or demographic records are included.

Native source inputs
--------------------

``skel_joint_trial`` evaluates a user-loaded upstream SKEL model in bounded Torch
batches. Supply the native pose, shape and translation arrays from the licensed
workflow and an explicit calibrated rotation, rate, model identity and licence.
The adapter labels the 24 BSM-order output rows using ``model.bone_names``.
The tested model's separate ``joints_name`` field contains 22 SMPL-order names
and is deliberately not used. No third-party vertex indices are bundled.

.. code-block:: python

    import numpy as np
    from myosuite.integrations.capture import skel_joint_trial

    trial = skel_joint_trial(
        model, poses, betas, translations,  # Loaded by your upstream workflow.
        units="m", rotation_to_model=calibrated_rotation,
        frequency_hz=120, source="your parameter sequence and SHA-256",
        model_source="your SKEL model version and SHA-256",
        definition_license="terms applying to your licensed model outputs",
    )

``read_bioamass`` uses native OpenSim forward kinematics for a local fitted
``.osim`` model and ``.mot`` motion. Select joint child-frame origins with
``joint_names`` or body-frame origins with ``body_names``; those point definitions
are different. Coordinate values are applied inside their source model, with
angular-unit conversion and native assembly of coupled coordinates. Omitted
coordinates and their model-default/assembly policy are recorded. Source qpos
columns are never renamed into target coordinates.

.. code-block:: python

    from myosuite.integrations.capture import read_bioamass

    bodies = ("pelvis",) + tuple(
        f"{name}_{side}" for side in "rl"
        for name in ("femur", "tibia", "talus", "calcn", "toes")
    )
    trial = read_bioamass(
        "optimized_scale_and_markers.osim", "01_ik.mot",
        body_names=bodies, rotation_to_model=calibrated_rotation,
        definition_source="your fitted BSM model/version; body-frame origins",
        definition_license="terms applying to your licensed BioAmass data",
    )

``read_addbiomechanics`` uses the optional official Nimble ``SubjectOnDisk``
reader on a supported platform. It returns raw observed markers in SI units,
preserving missing observations. Its label set is the union observed in the
requested window; markers absent throughout that window are not invented.
Source anthropometry, frame indices, file hash and missing-GRF reasons remain
local provenance. Nonfinite/nonpositive height or mass is unknown. A missing
GRF reason is unknown, not proof that forces or inverse dynamics are valid.

.. code-block:: python

    from myosuite.integrations.capture import read_addbiomechanics

    trial = read_addbiomechanics(
        "subject.b3d", 0, rotation_to_model=calibrated_rotation,
        definition_source="source capture marker protocol/version",
        definition_license="release and marker-definition terms",
    )

The adapters do not automatically use a source mass as the target mass, infer
unmeasured segments from height, or turn inferred skeleton origins into measured
anatomical centres. Preserve that distinction when combining these observations
with :doc:`capture_scaling` and :doc:`body_scaling`.

Reproduce the model-origin comparison
-------------------------------------

After local native evaluation, export a pickle-free local NPZ of calibrated
model-world metre positions. Keep this coordinate file with the licensed input;
it is not a redistributable fixture merely because it was exported as NumPy.

.. code-block:: python

    np.savez_compressed(
        "local_native_origins.npz", positions=trial.in_model_frame(),
        labels=np.array(trial.labels), frequency_hz=trial.frequency_hz,
    )

For the evaluated CMU01 sequence, use calibration frames 0--2 and the fixed
held-out indices 50, 150, ..., 2650. These span most of the 2,751-frame sequence
and were selected before examining the larger-window fit:

.. code-block:: bash

    python -m myosuite.integrations.capture.native_validation --trial local_native_origins.npz --source-label "your source/version" --definition-license "your local output terms" --output aggregate_metrics.json --calibration-frames 0 1 2 --heldout-start 50 --heldout-stride 100 --heldout-count 27 --ground-projection

The explicit MyoLeg protocol uses native skeletal/body origins corresponding to
``pelvis`` and bilateral ``femur``, ``tibia``, ``talus``, ``calcn`` and ``toes``.
It adds target sites at those body-frame origins. It does not equate them to
skin markers or use MyoLeg's ``KJC`` site as the knee hinge: that site is about
9 mm from the target knee anchor in the neutral model. Source-origin definitions
and the calibrated coordinate convention must be verified for another model.

Seven local axes are estimated: femur/tibia length along local y, calcn-to-toes
length along local x, and pelvis hip separation along local z. Other axes stay
one. The target is constructed and aligned before the held-out motion is fitted.
A nominal-target fit replayed on the scaled target supplies the negative control.
The recipe remains the shared subject-model construction used by CPU and mjlab.

The command retains failed fits and reports convergence counts, marker RMS/max,
root error, equality residual and sole penetration. ``geometric_contact_geom_frames``
counts near-ground/slow **geometry-frame pairs**, not contact events or measured
forces; the number of selected sole geoms is recorded. Twenty millimetres is the
explicit marker convergence threshold in this experiment, not anatomical ground
truth or a clinical acceptance bound.

``--ground-projection`` separately evaluates per-frame root-height alignment
using the existing geometry helper and quaternion-aware velocity differencing.
It reports both forced-ground alignment and a raise-only variant which never
lowers airborne frames. Neither refits the pose or solves contacts/dynamics.
Raw results remain visible. The ground plane at z = 0 is an explicit diagnostic
assumption: the source floor was not measured or independently registered.

The fixed 27-frame results, in millimetres, are:

.. list-table:: Raw kinematic fits; all held-out frames retained
   :header-rows: 1
   :widths: 21 15 15 16 16 17

   * - Source representation
     - Nominal RMS
     - Scaled RMS
     - Scaled max
     - Converged
     - Sole penetration max
   * - SKEL 1.1.1
     - 28.44
     - 6.13
     - 40.46
     - 25 / 27
     - 75.12
   * - BioAmass / OpenSim
     - 22.79
     - 5.07
     - 10.13
     - 27 / 27
     - 75.44

Replaying the nominal fit on the scaled target gives RMS 44.43 mm for SKEL and
28.80 mm for BioAmass, respectively. Scaling before fitting therefore improves
source-origin tracking in this experiment. These residuals do not establish
correct anatomy or ground interaction. In particular, the raw fits penetrate
the assumed plane by about 75 mm. The raise-only geometric postprocess reduces
penetration to numerical precision but increases RMS/max marker error to
30.37/90.42 mm for SKEL and 19.83/79.85 mm for BioAmass. Both fail the explicit
20 mm maximum-error criterion. Forced-ground alignment performs worse still
(RMS 79.19 and 72.82 mm) because it also lowers airborne frames. No projection
is applied to the primary result or offered as a validated contact solution.

The aggregate :download:`report <_static/scaling/native_validation_metrics.json>`
retains source hashes, solver settings, protocol indices, root/equality errors,
all controls and both projection variants. Retargeting used MuJoCo 3.17.0 and
NumPy 2.5.4. Native evaluation used upstream SKEL revision
``c32cf16581295bff19399379efe5b776d707cd95`` with model version 1.1.1 and Torch
2.11.0 on CPU, and OpenSim 4.6 for the BioAmass model/motion pair.

Measured inputs and limits
--------------------------

Native SKEL 1.1.1 and BioAmass v1.0 CMU01 were evaluated locally with their
upstream readers. They represent the **same subject and source motion**, not
two independent population samples. The larger 27-frame checks retain all
convergence failures; they test representation transfer and temporal held-out
frames rather than subject-disjoint anatomical or dynamic accuracy.

The AddBiomechanics core Lencioni/Tiziana subject used a separate measured
29-marker, 47-frame trial. Three calibration frames estimated bilateral lateral
condyle-to-malleolus shank scales; 37 later frames evaluated the same span
definition. Nominal span RMS error was 105.76 mm and scaled RMS was 4.64 mm,
with maximum scaled span error 11.64 mm. This is measured surface-marker span
consistency, not a claim of true joint-centre length or a full-body reconstruction.
Unmeasured femur dimensions were left nominal. All marker samples were finite;
only 18 of 47 frames had source GRF-valid status and the other 29 remain flagged.
The source header's sampling rate is 61.3043 Hz; it is preserved rather than
rounded to a conventional rate. All 74 held-out bilateral span observations
were finite. The source GRF flags do not validate this marker-only experiment's
forces or dynamics.

Export the measured marker trial locally as above, optionally retaining its
integer source ``missing_grf`` status array (zero means source GRF-valid), then
reproduce that fixed span check with:

.. code-block:: bash

    python -m myosuite.integrations.capture.native_validation --protocol lateral-shank --trial local_measured_markers.npz --source-label "your source/version" --definition-license "your local output terms" --output measured_span_metrics.json

The raw AddBiomechanics sources were acquired selectively from the official core
archive with ZIP CRC/SHA checks. The Windows analysis used the pinned upstream
protobuf schema to inspect those files; this research extractor is not a second
production B3D implementation. Production ``read_addbiomechanics`` delegates to
Nimble, whose native wheel is unavailable in this Windows setup. Independent
contract tests exercise that optional wrapper; the measured-marker check records
the extraction method explicitly.

The source roles remain explicit: SKEL and BioAmass each estimate seven axes
from inferred skeletal/model origins and validate retargeting on their shared
CMU01 motion. The measured Lencioni/Tiziana markers estimate two shank length
axes and validate held-out surface-marker spans. A separately acquired Hamner
running subject was inspected for format, marker availability and GRF status;
it did not contribute factors or the reported scaling benchmarks. These small
checks do not constitute population validation, and missing cross-section
dimensions are not filled from these sources.

Original source archives, native models, marker trials and motion exports are
excluded from version control. Scientific aggregate errors, convergence rates,
source/version hashes and protocol code are the publication artifacts. Existing
licences apply to any further derived models or coordinate exports; recording a
licence string does not certify redistribution rights. See
:doc:`smpl_geometry` for primary source links and access conditions.

Publication scope and citations
--------------------------------

The tables and JSON report present scalar scientific findings, source hashes
and experimental protocols. SKEL/AMASS publication provisions support reporting
research outcomes with appropriate citations. This does not grant permission
to redistribute their models, fitted coordinates, marker definitions, motions
or subject-model derivatives. Native SKEL code and model weights share the
`SKEL noncommercial, no-distribution terms
<https://skel.is.tue.mpg.de/license.html>`_; a public GitHub repository is not a
permissive software licence. BioAmass contains BSM fits to AMASS sequences, so
the `AMASS terms <https://amass.is.tue.mpg.de/license.html>`_ and applicable
underlying-source terms also remain relevant. Neither supplied data archive
contained a separate licence/readme file. Commercial rights or further asset
distribution require the appropriate rightsholders' permission.

SMPL distinguishes its restricted `model/software and shape tools
<https://smpl.is.tue.mpg.de/modellicense.html>`_ from the separately licensed
`SMPL-Body <https://smpl.is.tue.mpg.de/bodylicense.html>`_ subset. The latter
permits CC BY 4.0 sharing of a mesh/rig/pose and dynamic blendshapes without
shape blendshapes or shape-generation tools. That limited permission does not
clear AMASS motion, SKEL/BioAmass or other input rights. No model, regressor,
third-party vertex-index set or upstream implementation is included here.

The measured AddBiomechanics core data is released under
`CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/>`_. The original
Lencioni data descriptor independently specifies the same licence for its
de-identified data. Our changes compute calibration/held-out error summaries
from the source markers; the original trials remain local. Credit the following
work when reusing or describing this experiment:

* Keller, M., Werling, K., Shin, S., Delp, S., Pujades, S., Liu, C. K., and
  Black, M. J. (2023). `From Skin to Skeleton: Towards Biomechanically Accurate
  3D Digital Humans <https://doi.org/10.1145/3618381>`_. ACM Transactions on
  Graphics 42(6), 253:1--253:15, SIGGRAPH Asia.
* Loper, M., Mahmood, N., Romero, J., Pons-Moll, G., and Black, M. J. (2015).
  `SMPL: A Skinned Multi-Person Linear Model <https://smpl.is.tue.mpg.de/>`_.
  ACM Transactions on Graphics 34(6), 248:1--248:16, SIGGRAPH Asia.
* Mahmood, N., Ghorbani, N., Troje, N. F., Pons-Moll, G., and Black, M. J.
  (2019). `AMASS: Archive of Motion Capture as Surface Shapes
  <https://amass.is.tue.mpg.de/>`_. ICCV, pp. 5442--5451.
* Werling, K., et al. `AddBiomechanics Dataset: Capturing the Physics of Human
  Motion at Scale <https://doi.org/10.1007/978-3-031-73223-2_27>`_. ECCV 2024,
  proceedings published 2025, pp. 490--508.
* Werling, K., Bianco, N. A., Raitor, M., Stingel, J., Hicks, J. L., Collins,
  S. H., Delp, S. L., and Liu, C. K. (2023). `AddBiomechanics: Automating model
  scaling, inverse kinematics, and inverse dynamics from human motion data
  through sequential optimization <https://doi.org/10.1371/journal.pone.0295152>`_.
  PLOS ONE 18(11), e0295152.
* Lencioni, T., Carpinella, I., Rabuffetti, M., Marzegan, A., and Ferrarin, M.
  (2019). `Human kinematic, kinetic and EMG data during different walking and
  stair ascending and descending tasks
  <https://doi.org/10.1038/s41597-019-0323-z>`_. Scientific Data 6, 309.

The underlying CMU motion data was obtained from
`mocap.cs.cmu.edu <https://mocap.cs.cmu.edu/>`_; the database was created with
funding from NSF EIA-0196217. CMU's original-motion terms allow broad reuse but
prohibit direct resale, including converted data. They do not replace the
separate terms of AMASS/SKEL fits derived from those motions.
