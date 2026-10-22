Scale estimation and target-specific retargeting
================================================

The optional ``myosuite.integrations.capture`` integration converts labelled
static-marker measurements to :class:`~myosuite.core.subject_model.ScaleEstimate`.
It does not import capture or body-shape packages in the core. Arrays and the
bundled synthetic example need only MyoSuite's existing dependencies; reading a
real C3D file additionally needs ``pip install "myosuite[capture]"``.

Labelled static trials
----------------------

.. code-block:: python

    import numpy as np
    from myosuite.integrations.capture import (
        MarkerTrial, SegmentMeasurement, estimate_marker_scales, read_c3d,
    )

    # Lab coordinates are explicit. Calibrate this proper rotation for your lab.
    trial = read_c3d("subject_static.c3d", rotation_to_model=np.eye(3),
                     confidence=0.95, uncertainty_m=0.003)
    # Alternatively supply positions[frames, markers, 3] and labels directly.
    mapping = [SegmentMeasurement(
        segment="femur_r", marker_pair=("RHIP", "RKNE"),
        reference_length_m=0.42, axis="y",
        expected_direction_model=(0, 0, -1),
    )]
    result = estimate_marker_scales(trial, mapping, mass_kg=72.0)
    estimate, evidence = result.estimate, result.evidence

Reference lengths, body/segment names and **local** scale axes must come from the
unscaled target. The example's ``y`` local axis and world ``-z`` direction are
illustrative; neither is a universal femur convention. The directed marker check
uses the calibrated **static-pose world** direction. Without such checks,
distances alone cannot identify an incorrect proper rotation. Reflections and
nonorthonormal transforms are always refused.

Units are explicitly ``m``, ``cm`` or ``mm``. C3D units, labels and sampling rate
come from POINT metadata; the lab-to-model rotation and optional translation in
metres come from the caller. Negative C3D residuals and NaN array samples are
missing observations. No gap filling or inferred marker relabelling occurs.
Unknown labels or insufficient valid frames omit the affected measurement and
add a warning; if no measurement remains, estimation fails. A median length
uses valid frames, with median absolute deviation and valid fraction recorded.
Default scale bounds [0.5, 2] reject likely units/reference mistakes; they are
configurable and do not establish anatomical validity.

One distance constrains one axis. Unmeasured axes stay one. ``axis=None``
explicitly assumes isotropic scaling and warns that cross sections are unknown.
Several independent local axes can be supplied for a segment. Overlapping scale
definitions are rejected. Height and mass remain separate evidence; height is
not converted into invented per-segment factors. Confidence and uncertainty are
source-reported, retained separately from across-frame dispersion, and are not a
calibrated confidence interval for the reconstructed anatomy. Soft-tissue motion,
marker offsets and pose errors can bias length measurements.

Retarget after construction
---------------------------

Build and align the target first, then solve its marker positions. Retargeting
changes joint and root coordinates by fitting source Cartesian observations to
that target's kinematics. Named-column expansion or replay alone does not do so.

.. code-block:: python

    from myosuite.core.subject_model import (
        GroundAlignment, SubjectModelConfig, build_subject_model,
    )
    from myosuite.integrations.capture.retargeting import (
        retarget_markers, load_retargeted_clip,
    )

    subject = build_subject_model(spec, SubjectModelConfig(
        estimate, ground=GroundAlignment("root", ("right_sole", "left_sole")),
    ))
    fitted = retarget_markers(
        subject, motion_trial, {"RHIP": "hip_marker", "RKNE": "knee_marker"},
        free_joint="root", sole_geoms=("right_sole", "left_sole"),
        scale_evidence=evidence,
    )
    path = fitted.save("outputs/subject_motion.npz")
    clip = load_retargeted_clip(path, subject)

The multi-marker solver uses SciPy's bounded nonlinear least squares and
MuJoCo's quaternion-aware ``mj_integratePos``. Each frame warm-starts from the
previous fit. Scalar joint limits and active equality residuals are checked.
Unlike the legacy single-site physics-wrapper helper, it fits all observed
markers together and exposes failure evidence. Finite differences are taken
through the actual quaternion integration, avoiding an incorrect assumption that
an angular velocity Jacobian is the derivative of a finite rotation vector.
Limited ball joints need a cone-constrained solver and are explicitly refused.
Optional joint selection fixes unselected coordinates at their seed. Ambiguous
or sparse markers can yield a small residual with an incorrect pose; this
baseline is not a validated anatomical retargeter or dynamics optimizer.

``strict=True`` requires each frame's marker and equality errors to converge.
``strict=False`` is for inspecting failures and negative controls. Output contains
named qpos/qvel columns, fitted sites, frequency, source coordinate/provenance
metadata, the full subject identity and scale evidence. NPZ files contain no
pickled Python objects. A JSON sidecar records readable provenance and quality.
``load_retargeted_clip`` rejects a different target model. Velocity is obtained
with ``mj_differentiatePos``; no quaternion component subtraction is used.

Diagnostics measure marker RMS/max error, geometric sole penetration, root
trajectory and, when a marker represents the free-root origin, root tracking
error. Contact labels mean the selected sole geometry is within 1 cm of the
horizontal plane and its centre speed is below 0.2 m/s. These are explicit
geometric heuristics rather than measured contacts or force estimates. Contact
feasibility, muscle forces, joint moments and dynamic balance need additional
validation. Missing observations are excluded, not scored as zero error.

Licensed runnable example
-------------------------

.. code-block:: bash

    python -m myosuite.integrations.capture.example --output outputs/marker_retargeting
    pytest myosuite/tests/test_capture_scaling.py myosuite/tests/test_capture_retargeting.py

The redistributable fixtures in ``myosuite/integrations/capture/fixtures`` were
independently authored under Apache-2.0. Their analytic source has a 0.60 m thigh
and 0.36 m shank; the nominal target has 0.50 m and 0.40 m links. The static-trial
adapter must recover factors 1.20 and 0.90. The motion is an independently
evaluated seven-frame planar stance sequence, with no source qpos supplied to
IK. The command estimates scales, builds/aligns the target, retargets, exports
named motion with subject provenance and replays it through ``mj_forward``.

Tests require marker/root errors below 0.01 mm, negligible sole penetration and
seven contact labels. Fitting the nominal target first and replaying its fitted
qpos on the scaled target must instead produce errors exceeding 3 cm in marker
RMS, root RMS and penetration. This negative control detects the exact workflow
error that a topology/replay check misses. Synthetic wrong-unit/frame fixtures
must be rejected, and a missing-marker fixture must omit only affected scales.
The generated C3D round-trip test runs when the optional capture extra is present.

User-provided virtual markers and licences
------------------------------------------

``virtual_marker_trial`` accepts labelled coordinates and requires
``definition_source`` and ``definition_license`` strings. The same static adapter
then uses your explicit marker-to-segment mapping. This records provenance and
does not certify permission to redistribute those definitions. It loads no
SMPL/SMPL-X package, model, marker-index list or AMASS motion. No restricted
indices, derived meshes or derived fixtures are included here.

The primary licences have distinct scopes (reviewed 2026-10-23):

* `SMPL-Model <https://smpl.is.tue.mpg.de/modellicense.html>`_ grants
  noncommercial research/education/artistic use with restrictions on distribution.
* The separate `SMPL-Body licence <https://smpl.is.tue.mpg.de/license.html>`_
  covers a subset excluding shape blendshapes and shape-generation tools under
  CC BY 4.0. That permission does not establish permission for arbitrary external
  marker definitions, the full model or its software.
* The `smplx software licence <https://github.com/vchoutas/smplx/blob/main/LICENSE>`_
  is noncommercial and prohibits redistribution absent permission.
* `AMASS <https://amass.is.tue.mpg.de/license.html>`_ similarly restricts use and
  redistribution. Capture/model assets and any required permissions stay with
  the user. Only independently authored marker coordinates are bundled here.

No redistribution grant was established for a third-party SMPL virtual-marker
definition, so the integration requires user-supplied definitions instead.
