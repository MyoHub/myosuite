Body scaling
============

For user-provided SMPL-family geometry and external marker trajectories, see
:doc:`smpl_geometry`; for labelled capture, see :doc:`capture_scaling`.

:func:`myosuite.core.body_scaling.scale_bodies` scales a musculoskeletal model to a
subject with the rules of OpenSim's Scale Tool. Each body gets per-axis factors
``(sx, sy, sz)`` in its own frame, or one factor for all three axes.

* Everything a body carries scales with that body: geoms, meshes, sites (muscle
  path points), joint anchors, wrapping geoms, cameras, lights and the inertial
  frame. A child body's position scales with its parent.
* Slide-joint translations, their ranges and their polynomial couplings (the
  knee's) scale with the parent body along the joint axis.
* Every muscle's optimal fibre length and tendon slack length scale by the change
  of its muscle-tendon length in the default pose, as in OpenSim. MuJoCo derives
  both from ``lengthrange``, so that is what changes.
* Mass scales with volume (``sx * sy * sz``) or is kept, and can be rescaled to a
  total. Peak muscle force is kept, as OpenSim does, unless ``force_scale`` is
  given.

Joints, actuators, sites and tendons stay the same, so an env's observation and
action spaces do not change.

.. code-block:: python

    from myosuite.core.body_scaling import scale_bodies
    from myosuite.core.model_builder import ModelBuilder

    # One factor per body, or per axis in the body's frame.
    scale_bodies(spec, {"femur_r": (1.0, 1.08, 1.0), "tibia_r": 1.05})
    model = spec.compile()

    # Or in a ModelBuilder chain.
    model, spec = (
        ModelBuilder.from_spec(spec)
        .scale_bodies({"femur_r": 1.08}, mass="keep", total_mass=72.0)
        .build()
    )

Scale factors from OpenSim
--------------------------

:func:`~myosuite.core.body_scaling.read_opensim_scales` reads the factors of an
OpenSim Scale Tool result: a ScaleSet file, or a scaled ``.osim`` model, whose
bodies carry the applied factors on their geometry. Its segment names are those of
the OpenSim model. For the Rajagopal model, which MyoLeg was built from, two
mappings assign every MyoSuite body to a segment:

.. code-block:: python

    from myosuite.core.body_scaling import (
        RAJAGOPAL_MYOFULLBODY_SEGMENTS,
        read_opensim_scales,
        scale_bodies,
    )

    scales = read_opensim_scales("subject_scaled.osim")
    scale_bodies(spec, scales, segments=RAJAGOPAL_MYOFULLBODY_SEGMENTS)

With ``segments``, each body takes the factors of its nearest ancestor that starts
a segment, so the lumbar spine, thorax and shoulder girdle of the MuscleMimic
MyoFullBody follow ``torso`` and the 27 hand bodies follow ``hand``. A segment
missing from the scale set stays unscaled, as in OpenSim.
``RAJAGOPAL_MYOLEG_SEGMENTS`` does the same for MyoLeg.

Subject construction and provenance
-----------------------------------

Use the source-neutral subject builder when factors come from an external tool.
It copies an existing ``MjSpec`` or builds a fresh ``ModelBuilder`` recipe; the
source is unchanged. This construction API is shared model infrastructure, not
a replacement for the registered CPU/mjlab environment factory.

.. code-block:: python

    from myosuite.core.subject_model import (
        ScaleEstimate, SubjectModelConfig, build_subject_model,
    )

    estimate = ScaleEstimate(
        factors=read_opensim_scales("subject_scales.xml"),
        height_m=1.78, mass_kg=72.0, source="subject_scales.xml",
        confidence=0.9, warnings=("No hand markers were available",),
    )
    config = SubjectModelConfig(
        estimate, segments=RAJAGOPAL_MYOFULLBODY_SEGMENTS, mass="keep",
    )
    subject = build_subject_model(spec, config)
    model, scaled_spec = subject.model, subject.spec

``height_m`` records evidence; it does not infer factors. ``mass_kg`` sets total
mass. BVH, C3D and shape-fitting adapters can produce the same ``ScaleEstimate``
without introducing their dependencies into the scaling core. Factors and
segment mappings are copied to prevent later input mutation changing a recipe.

``config.digest`` is a canonical SHA-256 of the versioned recipe, independent of
mapping order and scalar versus three-axis factor syntax. ``subject.metadata``
is JSON-serializable and records the complete recipe, its digest, resolved body
factors, input and output compiled-model digests, MuJoCo version and root offset.
Cache by input-model digest and config digest together. Compiled-model hashes
include assets and are conservatively specific to the MuJoCo version. Rebuild
metadata after any subsequent model edits.

Root and ground alignment
-------------------------

Scaling leaves the free root's world position unchanged. For a standing neutral
pose, name the root and sole collision geoms explicitly:

.. code-block:: python

    from dataclasses import replace
    from myosuite.core.subject_model import GroundAlignment, align_root_to_ground

    # Replace these names with the root/sole names in your model.
    ground = GroundAlignment("root_free", ("sole_right", "sole_left"), clearance=0.002)
    subject = build_subject_model(spec, replace(config, ground=ground))

    # For a different initial pose, align it explicitly without mutating the model.
    qpos, root_offset = align_root_to_ground(subject.model, ground, initial_qpos)

The builder aligns the lowest selected collision point using primitive or mesh
geometry in the neutral ``qpos0`` pose. It changes the free root's initial world-z
translation and shifts all keyframes by the same offset. Keyframes with other
leg poses and frames in a retargeted clip may need their own alignment. Ground
height and clearance are world metres; all sole geoms must belong to that root's
tree. Planes, heightfields and SDF sole geoms are refused. This is kinematic
placement, not a guarantee of balance or a contact-settling simulation.

Collision margins, contact detection tolerances and fall/height thresholds are
task parameters in metres. Review them against the new neutral pose; no universal
factor can update them correctly for arbitrary segment scales. A custom task
must feed the same scaled recipe, initial pose and thresholds to its CPU and
mjlab implementations. Registered tasks retain their nominal morphology.

Skins and retargeted motion
---------------------------

Construct and align the target first, then retarget the source motion against
that target. Named-joint ``MotionClip`` columns can be mapped with
``expand_motion_clip_to_model(clip, subject.model)`` and replayed through
``mj_forward``. Matching joint names and widths only verifies topology; replaying
a nominal clip is not retargeting. Recompute root translations, slide coordinates,
marker trajectories and contact labels for the scaled target. Store the subject
metadata alongside the resulting clip.

Bind the bundled skin with explicit resolved body factors. Moving bones alone
does not scale the offsets of their attached vertices:

.. code-block:: python

    from myosuite.viz.skin import SkinPose, load_skn

    pose = SkinPose.bind(load_skn("fullbody"), subject.model,
                         body_scales=subject.body_scales)
    # pose.vertices(data) follows the scaled model in any pose.

The poser scales each bone influence in its own bind frame, including
anisotropic factors. Standard ``.skn`` and rigged GLB store one common bind vertex
per vertex. When differently scaled influences disagree, those formats cannot
represent the exact deformation in every pose. ``pose.bake(subject.model)`` and
``write_skin_glb(pose, subject.model, path)`` reject such exports by default.
Explicit ``allow_approximate=True`` bakes the rest shape and warns that deformation
may differ. Even the bundled skin has small existing bind inconsistencies, so it
may require this opt-in. Compatible uniform synthetic skins round-trip exactly.

Checkpoint compatibility
------------------------

Unchanged spaces do not establish policy validity after changing muscle paths,
inertia or contacts. Record and check morphology as well as the policy contract:

.. code-block:: python

    from myosuite.utils.checkpoint_manifest import check_contract, write_manifest

    write_manifest(checkpoint_dir, env_id, subject_model=subject)
    check_contract(env_id, checkpoint_dir, subject_model=subject, strict=True)

A nominal or unrecorded checkpoint used with a subject, a different subject, or
a subject checkpoint used with nominal morphology warns and returns ``False``;
``strict=True`` raises. A matching digest identifies construction; it does not
establish biomechanical accuracy or successful policy transfer.

Registered subject environments and policies
--------------------------------------------

``myoLegWalk-v0`` accepts the same typed subject recipe on CPU and mjlab. Other
tasks reject subject recipes until their paired task settings are implemented.
Choose the fall threshold, reset keyframe, sole collision margin and reset
offsets explicitly; these task settings are not inferred from body proportions.

.. code-block:: python

    import json
    from pathlib import Path
    from myosuite import make_env
    from myosuite.core.config import EnvConfig
    from myosuite.core.subject_env import SubjectEnvConfig

    recipe = SubjectEnvConfig(
        model=subject_config,  # SubjectModelConfig with explicit GroundAlignment
        min_height=0.8,
        reset_keyframe=2,
        contact_margin=0.001,
        joint_offsets={"hip_rotation_r": (-0.01, 0.01)},
    )
    Path("subject.json").write_text(json.dumps(recipe.to_dict(), indent=2))
    cpu = make_env(EnvConfig("myoLegWalk-v0", subject=recipe))
    gpu = make_env(EnvConfig("myoLegWalk-v0", subject=recipe),
                   backend="mjlab", device="cuda:0", num_envs=2)

``reset_qpos`` and ``reset_qvel`` optionally select a retargeted initial state
in the built model's coordinate order. The reset pose is aligned using the
named sole geoms after fitting and after any randomized joint offsets. Both
backends expose construction provenance in reset information as
``info["subject_model"]``; partial resets leave other environments' state intact.

Training and evaluation accept ``--subject subject.json``. Training records the
actual instance contract, recipe and feature configuration in its manifest.
Resume and evaluation check those identities before loading weights. The same
automatic check is performed by ``load_policy(env, checkpoint, strict=True)``.
Exported ONNX files carry the manifest, including when native weights are bundled
and subsequently extracted. Custom trainers should call
``write_manifest(checkpoint_dir, env_id, env=env)`` with their actual instance.
The checks include morphology, reset/task settings, feature parameters,
observations, actions and control timing.

Where subject scale factors come from:

* **Markers and the OpenSim Scale Tool**, from motion capture or
  `OpenCap <https://www.opencap.ai>`__ (Apache-2.0).
* **AddBiomechanics** publishes scaled Rajagopal models and motions of 273
  subjects (over 70 hours) under CC BY 4.0
  (`dataset <https://addbiomechanics.org/download_data.html>`__).
* **User-provided virtual markers**, including independently licensed body-fit
  outputs, through the optional labelled-marker adapter in :doc:`capture_scaling`.
  SMPL-Model, ``smplx`` and AMASS have restrictive licences; the separate
  SMPL-Body subset has a CC BY licence. Those permissions have different scopes.
  MyoSuite ships independently authored synthetic markers and no third-party
  model assets or marker-index definitions.

Limits
------

* Per-axis factors assume MyoSuite body frames aligned with the OpenSim ones,
  which holds for the converted MyoLeg and MyoFullBody bodies. One factor per
  body is always safe.
* Anisotropic scaling of a rotated primitive can require shear which MuJoCo's
  primitive shapes cannot represent. Their dimensions use axis-scale
  approximations (spheres use the mean body factor; cylinders average transverse
  factors). Inertias use an affine mass-distribution transform. OpenSim uses
  its own inertia approximations; external validation is still required.
* The world body, and so the free root's position and keyframe root heights, are
  not scaled; a task starting from a standing pose must set the scaled height.
* A motion clip fits one body size: retarget clips against the scaled model.
* The model's ``boundmass`` and ``boundinertia`` floors still apply to tiny bodies.
* Meshes shared by bodies scaled differently, connect/weld equalities on scaled
  bodies and height fields are refused with an error.

Tests
-----

``myosuite/tests/test_body_scaling.py`` checks the rules on a synthetic leg (nested
rotated frames, every orientation type, a wrapping tendon, a coupled slide joint,
an automatic peak force), MyoLeg and the MuscleMimic MyoFullBody:

* one factor ``s`` for every body is an exact similarity: in a matched pose every
  site, tendon and muscle length grows by ``s``, every muscle force is unchanged,
  mass grows by ``s**3`` and inertia by ``s**5``;
* random per-axis factors move every site, joint anchor, geom and child body with
  its body's frame;
* muscle forces in the default pose do not change under any factors (the
  OpenSim length rule);
* factors of one leave the compiled model bit-identical.

``test_subject_model.py`` additionally checks cross-process digest stability,
source immutability, subject mass, primitive ground support, keyframe alignment,
checkpoint rejection and named-joint replay on a scaled MyoFullBody.
``test_body_scaling_skin.py`` checks segment-local skin scaling and explicit
``.skn``/GLB export fidelity on synthetic and bundled skins.

Integration evidence and follow-up review are tracked explicitly:
`OpenSim biomechanical validation <https://github.com/MyoHub/myosuite/issues/560>`__,
`registered CPU/mjlab and policy plumbing <https://github.com/MyoHub/myosuite/issues/561>`__,
and `optional estimation/retargeting adapters <https://github.com/MyoHub/myosuite/issues/562>`__.
