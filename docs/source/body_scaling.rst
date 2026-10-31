Body scaling
============

Two calls scale a musculoskeletal model to a subject.
:func:`myosuite.core.body_scaling.scale_model` returns a scaled copy of a model,
using the rules of OpenSim's Scale Tool. ``make_env(env_id, scales=...,
total_mass=...)`` builds a task for that subject, on the CPU and on mjlab.

.. code-block:: python

    import mujoco
    from myosuite import make_env
    from myosuite.core.body_scaling import scale_model

    spec = mujoco.MjSpec.from_file("myolegs_with_torso.xml")

    # One factor per body, or three along the body's own axes.
    model = scale_model(spec, {"femur_r": 1.08, "tibia_r": (1.0, 0.96, 1.0)})
    # An OpenSim ScaleSet, or a model scaled by the Scale Tool or AddBiomechanics.
    model = scale_model(spec, "subject_scale.xml", total_mass=72.0)
    # Points on a body (in its frame) and the same points measured on the subject.
    model = scale_model(spec, {"femur_r": (model_points, measured_points)})

    env = make_env("myoLegWalk-v0", scales={"femur_r": 1.08}, total_mass=72.0)

``scale_model`` never changes the spec it is given. Joints, actuators, sites and
tendons stay the same, so an env's observation and action spaces do not change.

The rules
---------

Each body gets factors ``(sx, sy, sz)`` along its own axes, as in OpenSim:

* Everything a body carries scales with that body: geoms, meshes, sites (muscle
  path points), joint anchors, wrapping geoms, cameras, lights and the inertial
  frame. A child body's position scales with its parent.
* Slide-joint translations, their ranges and their polynomial couplings (the
  knee's) scale with the parent body along the joint axis.
* Every muscle's optimal fibre length and tendon slack length scale by the change
  of its muscle-tendon length in the default pose. That pose is ``qpos0`` with
  every joint coupling satisfied, as OpenSim assembles it; MyoLeg's ``qpos0``
  leaves the patella 5 cm off its knee coupling. MuJoCo derives both lengths from
  ``lengthrange``, so that is what changes.
* Mass scales with volume (``sx * sy * sz``). Peak muscle force is kept, as in
  OpenSim. ``total_mass`` then rescales the subject's masses to a total.
* A free-floating subject is raised or lowered separately in ``qpos0`` and in
  each keyframe, so that its lowest collision point keeps the height it has in
  the unscaled model in that pose. A keyframe that stood on the floor still
  does, and one that floated above it floats as high.

The three forms of ``scales``
-----------------------------

**Factors per body.** One factor, or three along the body's own axes. A body
not named keeps its size; a body named with a factor of one counts as part of
the subject for ``total_mass``.

**An OpenSim scale file.** A ScaleSet (``<Scale><segment>...<scales>``), or a
scaled ``.osim`` model whose geometry carries the applied ``scale_factors``.
The segments are those of the Rajagopal model, which MyoLeg was built from.
They are mapped onto MyoLeg or the MuscleMimic MyoFullBody, whichever the spec
is. Each body takes the factors of its segment: the lumbar spine, thorax,
shoulder girdle and head of MyoFullBody follow ``torso``, and its 27 hand bodies
follow ``hand``. A segment missing from the file stays unscaled, as in OpenSim.

OpenSim frames are y-up (x anterior, z to the right). MyoLeg's ``root`` and
``torso`` and MyoFullBody's ``Full Body`` are z-up, so each segment's factors
are permuted onto every body's own axes: a 20% taller torso is 20% taller, not
wider.

**Point correspondences.** ``(model_points, measured_points)`` are two ``(n, 3)``
arrays. The first holds points on the body in its own frame in the unscaled
model: markers, named landmarks or mesh vertices. The second holds the same
points measured on the subject, in metres, in any frame (a lab frame, say).
Each body is fitted on its own:

* Points that span all three directions (four or more) give three factors.
  They are fitted by least squares, with a rotation between the frames.
* Two points (a marker pair) give one factor, the ratio of their distances, as
  OpenSim's marker pairs do. Points on a plane give one factor, the ratio of
  their spreads.

Rows with NaN (missing markers) are skipped. A fitted factor outside 0.5 to 2 is
refused, since it usually means millimetres or mismatched points; pass the
factors directly if they are meant.

A subject in an environment
---------------------------

``scales`` and ``total_mass`` are ordinary task kwargs. Pass them to ``make_env``
or in ``EnvConfig.task_kwargs``. The task builds its model from them, and its
mjlab twin builds the same model from the same registration:

.. code-block:: python

    from myosuite import make_env
    from myosuite.core.config import EnvConfig

    cpu = make_env("myoLegWalk-v0", scales="subject_scale.xml", total_mass=72.0)
    gpu = make_env(
        EnvConfig("myoLegWalk-v0", task_kwargs={"scales": "subject_scale.xml"}),
        backend="mjlab",
        num_envs=4096,
    )

* ``myoLegWalk-v0`` takes a subject, and so do its muscle-condition variants
  ``myoSarcLegWalk-v0`` and ``myoFatiLegWalk-v0``. Any other task raises
  ``NotImplementedError`` from its own constructor, on both backends. The
  terrain walks fix a knee threshold for the nominal body, so they refuse too.
* Without ``scales`` and ``total_mass``, every env is built exactly as before.
* The reset keyframe keeps its nominal ground clearance, and both backends
  reset to the same pose.
* Thresholds in metres stay task parameters, for example the fall height
  ``min_height`` (0.8 m). On the CPU, pass it with the scales. The mjlab
  backend takes only ``scales`` and ``total_mass`` as task kwargs; any other
  needs its own registration.

Mass
----

``total_mass`` rescales the subject's bodies, keeping their mass distribution
after the volume scaling. The subject is every kinematic tree that holds a body
named in ``scales``, or one of the mapped segment bodies of an OpenSim file.
With no named body it is the model's single articulated tree. Static scenery and
other agents in a composed scene keep their mass. Several articulated trees and
no named body is an error: name a body of the subject, with a factor of one if
need be.

Skins
-----

Bind the bundled skin with the factors each body was scaled by. Moving bones
alone does not scale the offsets of their attached vertices:

.. code-block:: python

    from myosuite.viz.skin import SkinPose, load_skn

    # fullbody_spec: the MuscleMimic MyoFullBody, which the bundled skin fits.
    factors = {"femur_r": (1.0, 1.08, 1.0), "tibia_r": 1.05}
    model = scale_model(fullbody_spec, factors)
    pose = SkinPose.bind(load_skn("fullbody"), model, body_scales=factors)
    # pose.vertices(data) follows the scaled model in any pose.

The poser scales each bone influence in its own bind frame, anisotropic factors
included. Standard ``.skn`` and rigged GLB files store one bind vertex per
vertex. When differently scaled influences disagree, those formats cannot
represent the exact deformation in every pose, so ``pose.bake(model)`` and
``write_skin_glb(pose, model, path)`` refuse such exports. With
``allow_approximate=True`` they bake the rest shape and warn that the
deformation may differ. The bundled skin has small bind inconsistencies of its
own, so it may need this. Uniformly scaled skins round-trip exactly.

Checkpoints
-----------

A policy trained on a subject records it with its checkpoint. Check it before
loading the policy on another body:

.. code-block:: python

    from myosuite.utils.checkpoint_manifest import check_contract, write_manifest

    write_manifest(run_dir, "myoLegWalk-v0", scales=scales, total_mass=72.0)
    check_contract("myoLegWalk-v0", run_dir, scales=scales, total_mass=72.0, strict=True)

The manifest records the ``scales``, ``total_mass`` and a digest of the compiled
model. The digest hashes the model's numbers but not its file paths, so it
matches across folders and machines. A checkpoint whose subject's digest differs
from the env's raises with ``strict=True``. This includes a subject checkpoint
on the nominal body, and the reverse. Without ``strict``, it warns and returns
``False``. Nominal envs and checkpoints are checked as before. A matching digest
identifies the body, not biomechanical accuracy or successful transfer.

Where scale factors come from
-----------------------------

* **Markers and the OpenSim Scale Tool**, from motion capture or
  `OpenCap <https://www.opencap.ai>`__ (Apache-2.0).
* **AddBiomechanics** publishes scaled Rajagopal models and motions of 273
  subjects (over 70 hours) under CC BY 4.0
  (`dataset <https://addbiomechanics.org/download_data.html>`__).
* **An SMPL body fit**, through virtual markers placed on SMPL vertices (e.g.
  Bittner et al., Sensors 2022). Pass the vertices and the model points as
  point correspondences. SMPL, SMPL-X, the ``smplx`` package code and AMASS are
  licensed for non-commercial use without redistribution, so MyoSuite does not
  ship them or anything derived from them.

Limits
------

* Factors given per body are in that body's frame. Factors from an OpenSim file
  are permuted onto each body's axes; a body rotated from its segment by
  something other than an axis permutation takes the scale along each of its
  axes. One factor per body is always safe.
* Anisotropic scaling of a rotated primitive can require shear, which MuJoCo's
  primitive shapes cannot represent. Their dimensions use axis-scale
  approximations: spheres use the mean body factor, and cylinders average the
  transverse factors. Inertias use an affine mass-distribution transform.
  OpenSim uses its own inertia approximations, so external validation is still
  needed.
* OpenSim scales a joint's offset in the child body with the child's factors.
  Here a child body's position, which includes that offset, scales with the
  parent's. They agree when the offset is zero or the two factors are equal; at
  MyoLeg's knee, femur and tibia factors of 1.05 and 1.0 differ by 0.32 mm.
* The world body is not scaled: a free root keeps its horizontal position, and
  only the root height of each pose changes (see the rules).
* A motion clip fits one body size: retarget clips against the scaled model.
* The model's ``boundmass`` and ``boundinertia`` floors still apply to tiny bodies.
* Meshes shared by bodies scaled differently, connect/weld equalities on scaled
  bodies, height fields and plugin meshes are refused before the model is
  edited.

Tests
-----

``myosuite/tests/test_body_scaling.py`` checks the rules on a synthetic leg (nested
rotated frames, every orientation type, a wrapping tendon, a coupled slide joint,
an automatic peak force), MyoLeg and the MuscleMimic MyoFullBody:

* One factor ``s`` for every body is an exact similarity. In a matched pose
  every site, tendon and muscle length grows by ``s`` and every muscle force is
  unchanged. Mass grows by ``s**3`` and inertia by ``s**5``.
* Random per-axis factors move every site, joint anchor, geom and child body with
  its body's frame.
* Muscle forces in the assembled default pose do not change under any factors
  (the OpenSim length rule).
* Each OpenSim segment's axis map matches the model frames.
* Every pose keeps its ground clearance, measured with MuJoCo's own geom distance.
* The model digest ignores asset paths.
* Point correspondences recover known factors.
* A refused scaling leaves the spec untouched, and factors of one leave the
  compiled model bit-identical.

``test_scaled_env.py`` and ``test_scaled_env_mjlab.py`` check the subject envs:
nominal envs stay bit-identical, the twin builds the CPU model, both backends
reset and step alike, other tasks refuse ``scales``, and manifests record and
check the subject. ``test_body_scaling_skin.py`` checks segment-local skin
scaling and the fidelity of ``.skn``/GLB exports.

Remaining work is tracked in issues:
`OpenSim biomechanical validation <https://github.com/MyoHub/myosuite/issues/560>`__
and `optional estimation/retargeting adapters <https://github.com/MyoHub/myosuite/issues/562>`__.
