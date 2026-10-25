SMPL geometry and marker imports
================================

User-owned body geometry and marker trials feed :doc:`capture_scaling` and
:doc:`body_scaling` before target-specific retargeting.

For native SKEL/BioAmass/AddBiomechanics inputs and measured scaling checks,
see :doc:`native_scaling_validation`.

See the :download:`source-specific licence and attribution notices <../../THIRD_PARTY_DATA.md>`
for external inputs and published validation results.

Continue a user-owned SMPL fit
------------------------------

``vertex_marker_trial`` extracts user-defined virtual markers directly from
static ``(vertices, 3)`` or moving ``(frames, vertices, 3)`` arrays. The arrays can
be exported by a separately licensed SMPL, SMPL-X or SKEL workflow. MyoSuite
loads no model package and bundles no model or third-party vertex-index list.

.. code-block:: python

    import json
    from pathlib import Path
    import numpy as np
    from myosuite.integrations.capture import (
        SegmentMeasurement, VertexMarkerSet, estimate_marker_scales,
        vertex_marker_trial,
    )

    # Your JSON specifies topology, vertex_count, indices,
    # definition_source and definition_license for your actual vertex order.
    definitions = VertexMarkerSet(**json.loads(
        Path("my_marker_definitions.json").read_text(encoding="utf-8")
    ))
    static_trial = vertex_marker_trial(
        np.load("my_static_vertices.npy", allow_pickle=False), definitions,
        topology="my independently verified model/version/vertex-order ID",
        units="m", rotation_to_model=np.eye(3), source="my static body fit",
    )
    # Illustrative names/length/axis: derive from your unscaled target sites.
    measurements = [SegmentMeasurement(
        "femur_r", ("THIGH_PROXIMAL", "THIGH_DISTAL"), 0.42, axis="y",
    )]
    result = estimate_marker_scales(static_trial, measurements, mass_kg=72.0)
    motion_trial = vertex_marker_trial(
        np.load("my_motion_vertices.npy", allow_pickle=False), definitions,
        topology="my independently verified model/version/vertex-order ID",
        units="m", rotation_to_model=np.eye(3), source="my motion fit",
        frequency_hz=60.0,
    )

The declared topology must exactly match the definitions and vertex count.
SMPL and SMPL-X mappings are not interchangeable. A caller who mislabels a
reordered array with the same identity/count defeats these checks: vertex
coordinates alone cannot identify the ordering. Topology and definition
provenance survive in ``trial.metadata()["vertex_markers"]``. This records
source-reported permissions; it does not certify redistribution rights.
NaN vertices propagate as missing markers. Surface markers require matching
target surface sites or a calibrated anatomical offset; they do not directly
measure hip, knee or other internal joint centres.

Use ``result.estimate`` with ``SubjectModelConfig`` and
``build_subject_model``. Then fit ``motion_trial`` to sites on the **constructed
target** with ``retarget_markers``, passing ``scale_evidence=result.evidence``.
The retargeted export retains the input definitions, scale evidence and target
identity. See :doc:`capture_scaling` for the solver and :doc:`subject_leg_workflow`
for the existing registered ``myoLegWalk-v0`` CPU/mjlab subject recipe, training,
resume and policy replay. The same opt-in recipe serves both backends.

Import external OpenSim markers
--------------------------------

The `SMPL2AddBiomechanics workflow
<https://github.com/MarilynKeller/SMPL2AddBiomechanics#quick-start>`_ exports
virtual marker trajectories as TRC files. The same interchange format is used
by experimental capture and OpenCap. ``read_trc`` brings those coordinates into
the existing estimator and retargeter without loading a body-model package:

.. code-block:: python

    from myosuite.integrations.capture import read_trc

    trial = read_trc(
        "subject_static.trc", rotation_to_model=np.eye(3),
        definition_source="my exported marker protocol/version",
        definition_license="terms applying to those marker definitions",
    )
    result = estimate_marker_scales(trial, measurements)
    print(trial.metadata()["trc"])

Calibrate the proper source-to-target world rotation; identity above is an
example. The reader follows the `OpenSim TRC format
<https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53089972/Marker+.trc+Files>`_.
It preserves blank coordinate cells and NaN as missing data, checks declared
frame/marker counts and units, and refuses nonconsecutive frame numbers or
nonuniform times. The timing check allows the six-significant-digit timestamp
and rate rounding used by Nimble's TRC exporter, with a one-microsecond baseline
allowance and an overall cap of a quarter sample interval. More precise tokens
reduce the allowance. It retains the
declared rate and does not resample or infer labels.
The trial retains the source rate, original frame/time range, header and exact
file SHA-256. Definition terms are source-reported and do not certify permission
to redistribute a dataset.

Data that strengthens validation
--------------------------------

Shape and pose diversity alone cannot validate internal anatomy, muscle forces
or dynamic balance. Use the sources below for distinct acceptance checks,
starting with the AddBiomechanics reviewed core. Counts and access conditions
were checked against primary sources on 2026-10-25.

.. list-table:: Candidate validation data
   :header-rows: 1
   :widths: 23 42 35

   * - Source
     - Added evidence
     - Access and reuse
   * - `AddBiomechanics core <https://addbiomechanics.org/download_data.html>`_
     - 273 subjects, over 70 hours, scaled skeletons, optical markers and
       measured ground forces/moments; model-derived joint torques.
     - CC BY 4.0. Credit original sources listed in the release. Prefer reviewed
       core over unreviewed public uploads. Fits and inferred torques are not
       direct measurements of muscle forces.
   * - `OpenCap data <https://mobl.mech.utah.edu/resources/>`_
     - The lab lists 110 subjects with video, mocap, forces and EMG. Useful for
       assessing capture errors and muscle-activation timing.
     - SimTK release-specific access/terms must be checked. The software's
       Apache licence does not establish the data licence.
   * - `SKEL / BioAmass <https://skel.is.tue.mpg.de/>`_
     - BSM skeleton fits paired to SMPL/AMASS sequences, directly relevant to
       skin-to-skeleton correspondence and issue 166.
     - Registration; `noncommercial, no-redistribution terms
       <https://skel.is.tue.mpg.de/license.html>`_. Keep user-provided assets
       outside the repository. Verify BSM-to-target frames and segment mapping.
   * - `EMDB <https://eth-ait.github.io/emdb/>`_
     - Ten subjects, 81 sequences, scanned reference shapes and global body
       trajectories; useful for real-world pose and root-trajectory errors.
     - Application/approval required. The website's CC BY-SA licence does not
       establish data reuse rights. No measured ground forces are supplied.

`AMASS <https://amass.is.tue.mpg.de/>`_ and
`BEDLAM <https://bedlam.is.tue.mpg.de/>`_ add broad motion and body-shape stress
tests, but require registration and restrict redistribution. BEDLAM is
synthetic SMPL-X data, so it does not independently measure internal bone shape
or muscle attachments. `OSSO <https://osso.is.tue.mpg.de/>`_ is especially
relevant to population anatomy: it learned skin/skeleton relationships from
2,000 DXA subjects, but the paired data require UK Biobank access.

Validation should first compare observed marker residuals and segment lengths,
then evaluate contacts and root motion, then measured force and EMG agreement
where available. Preserve subject-disjoint evaluation: motion frames from a
person used to tune marker correspondences must not count as a new test subject.
Do not transfer a source model's joint torques or generalized coordinates to a
different target by name alone.

Local interoperability verification also read the upstream OpenSim
`Rajagopal static trial
<https://github.com/opensim-org/opensim-models/blob/d9b05d470b1a481c222372c85b75772faf8f7792/Pipelines/Rajagopal/ExpData/static_walk_2.trc>`_:
153 frames, 41 markers, 100 Hz, millimetres, SHA-256
``88ca6ba0388c11fbb8424c27f34f9775856dc28b0738ea63e29120957e603088``.
This verifies file parsing, not
anatomical calibration or real-subject retargeting accuracy. The source trial
is not bundled, and CI uses independently authored synthetic inputs.

Relationship to issue 166 and SKEL
----------------------------------

`Issue 166 <https://github.com/MyoHub/myosuite/issues/166>`_ asks about porting
SMPL through SKEL/OpenSim into MyoSuite. The `SKEL project
<https://skel.is.tue.mpg.de/>`_ provides a biomechanical body representation;
its separate `SMPL2AddBiomechanics project
<https://github.com/MarilynKeller/SMPL2AddBiomechanics>`_ exports virtual markers
and fits an OpenSim model through AddBiomechanics, producing scaled ``.osim``
and ``.mot`` outputs.

An externally fitted OpenSim model can also enter through
``read_opensim_scales(scaled_osim, reference=unscaled_osim)``. Verify source
segment names, local axes and baseline geometry scales against the target.
The default BSM skeleton in that workflow is not Rajagopal/MyoLeg: its scale
vectors and motion coordinates cannot be transferred by renaming alone.
Prefer fitting a compatible target reference, or calibrate an explicit
source-to-target mapping and retarget Cartesian landmarks on the built target.

The adapters here are an incremental implementation toward issue 166. A full
port still needs real-subject correspondence validation and checks of muscle
paths, moment arms, joint constraints, contacts and dynamic feasibility. The
automated vertex pipeline test uses independently authored synthetic geometry;
it does not establish accuracy on a real SMPL fit.
Keep externally licensed models/software and marker definitions in the user's
workflow; :doc:`capture_scaling` distinguishes the relevant licence scopes.
