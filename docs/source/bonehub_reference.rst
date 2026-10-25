BoneHub anatomical references
=============================

Reviewed bone landmarks enter the existing :doc:`capture_scaling` and
:doc:`body_scaling` pipelines. The same subject recipe serves CPU and mjlab.

BoneHub reference data
----------------------

The `Visible Human dataset
<https://huggingface.co/datasets/BoneHub/visible-human-3d-models>`_ provides
individually labelled STL bones and IGES/STEP surfaces from two cadavers. The
`Educational Atlas <https://bonehub.github.io/Educational-Atlas/>`_ is a useful
viewer of the same data. These are anatomical examples, not a population shape
model: interpolating its male and female subjects does not establish realistic
variation in height, sex, age or body composition.

Use the opt-in reference helper to fetch only the bones needed for inspection:

.. code-block:: python

    from pathlib import Path
    from myosuite.integrations.bonehub import BoneHubReference

    reference = BoneHubReference(Path("assets/bonehub"), "01_Male")
    mesh = reference.download_mesh("LOWER_EXTREMITY_RIGHT", "PATELLA_RIGHT")
    # Existing local files: reference.mesh_path(anatomy, bone).
    print(reference.metadata())

Downloads are pinned to dataset commit
``7d520ffe410b3f79d82991b8368b2d51844feb80`` and separated by revision beneath
the chosen directory. One call downloads the requested STL and subject metadata,
not the CT volumes or the whole skeleton. ``mesh_path`` performs no download.
The dataset retains ownership of its CC BY 4.0 geometry; attribution is recorded
for Alavi and Asseln (2026, DOI ``10.57967/hf/10464``) and Andreassen et al.
(2023, DOI ``10.1038/s41597-022-01905-2``).

Load the selected geometry into 3D Slicer and annotate the landmarks required by
your measurement protocol. Save **Fiducial** markups as ``.mrk.json``. Harden any
scene transforms first so the coordinates are in the original patient frame.
Choose landmarks by a reviewed anatomical definition; mesh bounding-box ends
and surface vertices are not automatically joint centres.

.. code-block:: python

    import numpy as np
    from myosuite.integrations.capture import (
        SegmentMeasurement, estimate_marker_scales,
    )

    # Explicit example only: calibrate LPS -> your target's world frame.
    trial = reference.read_markups(
        "reviewed_landmarks.mrk.json",
        units="mm", rotation_to_model=np.eye(3),
        annotation_source="reviewer and anatomical landmark protocol/version",
        definition_license="terms applying to these landmark annotations",
    )
    measurements = [SegmentMeasurement(
        segment="femur_r", marker_pair=("RHIP", "RKNE"),
        reference_length_m=0.42, axis="y",
    )]
    result = estimate_marker_scales(trial, measurements)

The marker names, ``0.42`` m reference distance and local ``y`` scale axis are
illustrative. Derive them from corresponding landmarks on the unscaled target.
An anatomical bone measurement and a skin-marker distance require different
definitions; matching names alone does not make them interchangeable. One
distance constrains only the selected axis; cross-sectional factors remain one.
The existing estimator supports several independent axes with explicit
measurements and rejects overlapping definitions.

BoneHub's exported meshes use LPS patient coordinates (+X left, +Y posterior,
+Z superior). Its Slicer STL header declares ``SPACE=LPS``; the matching STEP
files declare millimetres. STL has no standardized unit field, so confirm units
explicitly when importing. Slicer markups may declare RAS instead: the reader
converts RAS to LPS, then applies the caller's proper LPS-to-model rotation and
metre translation. File-declared units must agree with the supplied units.
Undefined/preview points remain missing; duplicate labels, reflections and
unsupported coordinate systems are rejected.

``result.evidence["trial"]["bonehub"]`` retains subject, pinned revision,
attribution, annotation source/licence, exact annotation file hash and original
coordinate systems. The helper records the caller's subject association; it
cannot prove from coordinates which cadaver was annotated. Source height and
mass are not automatically applied to another person.

The dataset has incomplete elbow-region geometry, grouped digit phalanges and
an additional lumbar vertebra in the female subject. It does not provide a
complete muscle/tendon attachment or joint-constraint specification. For future
bone overlays, register each bone to the corresponding target body and convert
the mesh into that body's local frame before applying segment scaling. Treat
new collision geometry, inertias, muscle attachments and joint frames as
separate, validated changes to the shared CPU/mjlab model recipe.

More subjects from BoneHub VSD
------------------------------

The `VSD lower-extremity dataset
<https://huggingface.co/datasets/BoneHub/vsd-lower-extremities-seg>`_ adds 30
cadaveric cases, ages 19--95, with 63 individual bone labels, STL meshes and
subject metadata. It is derived from Fischer's
`VSDFullBodyBoneModels <https://github.com/RWTHmediTEC/VSDFullBodyBoneModels>`_.
This is a useful cohort for comparing variation beyond the two Visible Human
examples, with different terms: **CC BY-NC-SA 4.0**, including noncommercial
and share-alike conditions. The Python adapter does not bundle those meshes.

.. code-block:: python

    from pathlib import Path
    from myosuite.integrations.bonehub import BoneHubVsdReference

    vsd = BoneHubVsdReference(Path("assets/bonehub"), "001")
    mesh = vsd.download_mesh("Patella_R")
    metadata = vsd.metadata()
    print(metadata["source_subject"])  # Original VSD ID: 002.
    print(metadata["quality_flags"])
    trial = vsd.read_markups(
        "reviewed_vsd_landmarks.mrk.json", units="mm",
        rotation_to_model=np.eye(3),
        annotation_source="reviewer and anatomical landmark protocol/version",
        definition_license="terms applying to these landmark annotations",
    )
    result = estimate_marker_scales(trial, measurements)

This fetches one registered bone, the case metadata and small label dictionary
at commit ``85214b5a182b980a7869b57efba89a59a2445875``. Files are separated from
Visible Human under ``root/vsd-lower-extremities-seg/<revision>``. Subject IDs
``001``--``030`` are BoneHub case numbers; ``source_subject`` preserves the
original VSD ID. Source height/mass are recorded without becoming scale factors
or target mass automatically. The Slicer landmark reader is shared with the
Visible Human adapter and requires source metadata before importing annotations.

Fischer's source paper selects 20 complete subjects and flags ten additional
subjects with incomplete anatomy, implants, missing measurements or inconsistent
metadata. ``source_in_complete_cohort`` and ``quality_flags`` retain this
distinction in scale-estimation evidence. Even complete cases are cadaveric
examples, not a representative population distribution. BoneHub case 026 lacks
some right distal toe phalanges; case 030 lacks all phalanges. Missing bones are
not synthesized. ``vsd-feet-seg`` contains crops of these same 30 subjects and
must not be counted as independent anatomy.

VSD STL headers identify the MATLAB exporter without a standardized coordinate
or unit declaration. Its creation code uses the mesh coordinates in the source
CT physical space. Verify the mesh import and millimetre scale in Slicer; the
Visible Human ``SPACE=LPS`` header is not evidence for VSD files. Export reviewed
fiducials in their declared LPS/RAS patient frame after hardening scene transforms,
then calibrate the proper LPS-to-model rotation as above.

Other useful anatomy data
-------------------------

Primary sources reviewed on 2026-10-25 provide complementary evidence:

.. list-table:: Data for improving anatomy and scaling
   :header-rows: 1
   :widths: 24 42 34

   * - Source
     - What it adds
     - Terms and practical limits
   * - `University of Denver Visible Human
       <https://digitalcommons.du.edu/visiblehuman/>`_
     - 76 muscles, 28 bones, 16 cartilage, 8 ligament and 2 fat geometries per
       subject, with multiple processing stages and volumetric segmentations.
     - CC BY 4.0. Same two Visible Human donors; use for tissue geometry checks,
       not as additional population subjects or ready-made muscle parameters.
   * - `Keast, Bonacci and Fox tibia-fibula SSM
       <https://doi.org/10.6084/m9.figshare.20454462.v4>`_
     - Right tibia/fibula surfaces from 30 cadavers, registered points, statistical
       shape models and examples of shape reconstruction from palpable landmarks.
     - CC BY 4.0 dataset. Young adults, right side only; CT access has separate
       approval. Size, thickness and torsion modes extend beyond affine scaling.
   * - `TotalSegmentator CT v2
       <https://zenodo.org/records/8367088>`_
     - 1,228 clinical CT images with 117 labelled structures; larger variability
       in supported bones and some muscles, with a 102-case exploration subset.
     - CC BY 4.0 dataset. Anatomical coverage varies by scan; it is not a complete
       1,228-person lower-limb mesh or landmark collection.
   * - `OSSO paired body/skeleton data
       <https://osso.is.tue.mpg.de/Dataset.html>`_
     - Paired outer-body and skeleton meshes plus STAR parameters for 2,400
       UK Biobank subjects, relevant to validating surface-to-skeleton inference.
     - UK Biobank application and Tier 3 access are required. Model/software
       licences are separate; these are not freely redistributable fixtures.

For the immediate reference workflow, use the pinned BoneHub VSD cases and
reviewed landmarks to assess segment-length variation. The open tibia-fibula SSM
is a stronger next candidate for studying nonuniform shape changes. Fitting its
shape modes, changing joint frames or replacing muscle attachments needs its
own validation rather than treating arbitrary mesh deformation as body scaling.
