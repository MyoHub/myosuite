Reusable capture workflows
==========================

Use one explicit target protocol with labelled trials from different readers.
The common workflow estimates scales, constructs the target, fits motion against
that target and evaluates replay. It uses the existing subject recipe shared by
CPU and mjlab; it introduces no environment registration or backend-specific model.

Portable local trials
---------------------

``MarkerTrial`` includes optional ``subject_id`` and ``point_semantics``. Use
``surface_marker``, ``model_origin``, ``anatomical_landmark`` or another reviewed
definition. Equal labels do not make skin markers, body origins and anatomical
joint centres interchangeable. A protocol that requires known semantics rejects
unknown or mismatching declarations.

.. code-block:: python

    from myosuite.integrations.capture import save_marker_trial, load_marker_trial

    save_marker_trial(trial, "outputs/subject/trial.local.npz")
    restored = load_marker_trial("outputs/subject/trial.local.npz")

The versioned NPZ preserves original coordinates, missing samples, units,
rotation/translation, frequency, uncertainty, subject identity, definition terms
and reader-specific metadata such as native source hashes and quality flags.
It uses no pickle and rejects unsupported schemas. Evaluating a native reader
can therefore happen in its compatible environment; fitting can happen in a
separate environment with the ordinary MyoSuite dependencies.

These files contain source observations and may be restricted. Conversion to
NPZ grants no redistribution permission. Keep local trials, build recipes,
scale evidence and fitted motions with their licensed inputs. Only the aggregate
report described below is intended for a separate publication-rights review.

The generic ``read_named_landmarks`` reader is available through ``capture`` for
headerless ``label,x,y,z`` CSVs with explicit units and calibrated transforms.
``LandmarkCorrespondences`` and ``fit_landmark_scales`` are exported there too.

Target protocols
----------------

``TargetProtocol`` declares marker-to-site correspondences, per-body length
measurements and target-local scale axes. Reference lengths come from the
unscaled target and matching point definitions. Profiles can be saved as JSON,
loaded with ``TargetProtocol.load`` and reused for another trial with the same
definitions. Missing target names, overlapping measurements and incompatible
point semantics fail explicitly.

For a fixed-base arm, omit ``free_joint``, ``root_marker`` and ``sole_geoms``.
Root trajectory and penetration are unavailable, rather than fabricated zero
errors. A floating target can select its actual free joint, root marker, sole
geometry and ground height. Ground alignment and geometric near-ground labels
do not establish measured contact forces or dynamic feasibility.

.. code-block:: python

    from myosuite.integrations.capture import TargetProtocol, SegmentMeasurement

    protocol = TargetProtocol(
        name="reviewed-arm",
        marker_sites={"ELBOW": "elbow_point", "WRIST": "wrist_point"},
        measurements=(SegmentMeasurement(
            "forearm", ("ELBOW", "WRIST"), 0.25, axis="x",
        ),),
        point_semantics="anatomical_landmark",
    )
    protocol.save("outputs/arm/profile.json")

The names and 0.25 m length above are illustrative. Supply reviewed names and
measurements for the actual target. An explicit axis constrains only that
dimension; other dimensions remain nominal.

Command-line stages
-------------------

Use ``myosuite-capture`` after installation, or the equivalent
``python -m myosuite.integrations.capture.workflow``. ``doctor`` discovers
optional dependencies without loading licensed native assets. Discovery does
not certify native binary compatibility; execution reports import failures.

.. code-block:: bash

    myosuite-capture doctor --format trial
    myosuite-capture inspect --trial outputs/subject/trial.local.npz
    myosuite-capture estimate --trial outputs/subject/trial.local.npz --profile profile.json --calibration-frames 0 1 2 --output outputs/subject/estimate.local.json
    myosuite-capture build --model local_target.xml --profile profile.json --estimate outputs/subject/estimate.local.json --output outputs/subject/build.local.json
    myosuite-capture retarget --trial outputs/subject/motion.local.npz --build outputs/subject/build.local.json --output outputs/subject/motion_fit.local.npz
    myosuite-capture validate --trial outputs/subject/motion.local.npz --build outputs/subject/build.local.json --clip outputs/subject/motion_fit.local.npz --marker-max-m 0.005 --output outputs/subject/aggregate.json

``build`` can use ``--env`` instead of ``--model`` for an existing registered
environment. It records a rebuild recipe and model identity, without copying
source model assets. ``retarget`` rebuilds and checks that identity before
fitting. Calibration and motion with different declared subject IDs are refused.
``validate`` checks trial/protocol identity, coordinates, sampling rate and the
saved fitted poses before reusing historical solver-convergence counts. Equality
errors are recomputed during replay.

``export`` converts local CSV/C3D and available TRC/native readers to the common
trial format. Raw exports require explicit calibration, definition provenance,
terms and point semantics. CSV also requires units. ``doctor --format b3d`` or
``--format bioamass`` checks the optional reader environment. For SKEL, evaluate
the user-loaded model with the native Python adapter, then call
``save_marker_trial``; the CLI does not load pickled models or download weights.

Independent fixed-base example
------------------------------

Generate the fictional arm inputs without downloads or native model packages:

.. code-block:: python

    from myosuite.integrations.capture.arm_example import write_example_inputs

    inputs = write_example_inputs("outputs/fixed_arm/inputs")

Run the complete workflow with explicit quality limits:

.. code-block:: bash

    myosuite-capture run --trial outputs/fixed_arm/inputs/trial.npz --profile outputs/fixed_arm/inputs/protocol.json --model outputs/fixed_arm/inputs/arm_model.xml --calibration-frames 0 1 2 --marker-max-m 0.00001 --min-converged-fraction 1 --output outputs/fixed_arm/result

``--motion-trial`` selects a separate motion trial, allowing disjoint calibration
and evaluation data. The standalone ``python -m
myosuite.integrations.capture.arm_example --output outputs/fixed_arm/control``
also evaluates seven held-out motion frames, a missing wrist observation,
named-motion export/reload and a nominal-fit replay negative control. Analytic
link equations independently generate the observations. The known upper/lower
length factors are 1.2 and 0.8. All example inputs are independently authored
synthetic Apache-2.0 assets, not captured people or restricted body-model exports.

Quality and aggregate exports
-----------------------------

There are no universal clinical or contact acceptance limits. Callers supply
``QualityThresholds`` for the intended use. Each requested check is ``pass``,
``fail`` or ``unknown``; a missing root/contact metric or absent threshold does
not become a passing result. CLI validation returns 0 for all requested checks
passing, 1 for failure and 2 for unknown results or invalid inputs. A strict fit
fails immediately on nonconvergence; ``--keep-failed`` retains those poses for
inspection and reports their counts.

``aggregate_validation_report`` and ``write_aggregate_report`` admit only named
scalar errors, counts, conditions, explicit thresholds and approved
``SourceAttribution`` records. They exclude source coordinates, rigid alignment,
residual vectors, individual scale factors, anthropometry, trajectories and local
paths. The writer revalidates the report instead of dumping arbitrary metadata.
Source identifiers, licence identifiers, public citation URLs/DOIs and SHA-256
hashes are explicitly supplied; they do not certify sharing permission.

Provide ``--attributions`` with a JSON list of approved source records when
publishing a report. For example, an independently authored synthetic input can
use ``[{"source_id": "synthetic-arm", "license": "Apache-2.0"}]``. Real datasets
require their own attribution, terms and publication-rights review. Full local
fitting evidence remains separate from that report.
