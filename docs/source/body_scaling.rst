Body scaling
============

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

Where subject scale factors come from:

* **Markers and the OpenSim Scale Tool**, from motion capture or
  `OpenCap <https://www.opencap.ai>`__ (Apache-2.0).
* **AddBiomechanics** publishes scaled Rajagopal models and motions of 273
  subjects (over 70 hours) under CC BY 4.0
  (`dataset <https://addbiomechanics.org/download_data.html>`__).
* **An SMPL body fit**, through virtual markers placed on SMPL vertices and the
  Scale Tool (e.g. Bittner et al., Sensors 2022). SMPL, SMPL-X, the ``smplx``
  package code and AMASS are licensed for non-commercial use without
  redistribution, so MyoSuite does not ship them or anything derived from them.

Limits
------

* Per-axis factors assume MyoSuite body frames aligned with the OpenSim ones,
  which holds for the converted MyoLeg and MyoFullBody bodies. One factor per
  body is always safe.
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
