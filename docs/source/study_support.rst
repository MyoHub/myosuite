Participant model and motion preparation
========================================

These biomechanics modules operate on models and motion using NumPy, SciPy and
MuJoCo. Tracker protocols, device adapters, live session control, experiment logs
and human validation workflows belong in ``myosuite-shadow``.

Strength, mass and fatigue calibration
--------------------------------------

``strength_from_mvc`` fits muscles active in the measured direction while retaining
unscaled antagonist passive forces and joint springs. Apply the scale only to its
``agonists``. ``fit_strength_from_mvcs`` fits one group scale across postures and
returns residuals; it cannot identify independent muscle strengths. Use signed
gravity-corrected torque in N m, or N for a slide joint, with matching posture and
passive contribution.

.. code-block:: python

   from myosuite.biomechanics.calibration import (
       posture_at, strength_from_mvc, scale_strength,
   )

   qpos = posture_at(model, {"r_elbow_flex": 1.5707963267948966})
   calibration = strength_from_mvc(model, qpos, "r_elbow_flex", measured_torque)
   subject = scale_strength(model, calibration.scale, calibration.agonists)

``max_joint_torque(..., max_activation=...)`` predicts net torque under per-muscle
fatigue caps at the probe posture, including passive torque and coupled virtual
work. An unweighted fatigue average does not equal joint MVC capacity.
MVC calculations use steady activation, including for ``actearly`` actuators.
They reject actuator/joint force clamps, multi-state actuators and body gravity
compensation rather than silently fitting a non-affine or assisted torque model.
Enabled control clamps on stateless muscle actuators must contain the full
``[0, 1]`` activation interval; otherwise MVC routines reject the model.

``scale_body_mass`` accepts explicit ``represented_mass`` or a
``participant_mass / reference_mass`` ratio. ``bodies`` selects the subject rather
than furniture/another actor. Never set an arm model's mass to the entire person's
mass. Derived constants refresh while effective automatic peak forces remain
fixed. Strength scaling returns a copy and supports automatic and explicit forces.

``check_segment_lengths`` compares body/site endpoint distances with measurements
on an externally scaled subject model. Registration and mass scaling do not resize
geometry, muscle paths or moment arms. ``endurance_time`` and
``fit_fatigue_parameters`` fit 3CC-r coefficients on calibration holds; independent
trials are still required for validation.
Fitting first increases the starting fatigue rate, within a bounded search, until
the model predicts finite failure times inside the fitting horizon. This avoids a
flat censored loss when the initial model fatigues much more slowly than the
participant. Optimization uses at most 400 objective evaluations and reports a
failure if initialization or convergence fails.
``endurance_time`` stops integration at ``max_time`` even when the check interval
is longer than the remaining time. A hold that has not failed at that horizon
returns infinity.

External loading
----------------

``force_at_body`` and ``force_at_site`` map measured forces/torques through MuJoCo's
wrench mapping. ``held_mass_qfrc`` applies gravity at a body point or site and
omits the inertia of an accelerating object. Use measured loads or model the
object for that case. Locations, frames and units require calibration. Event labels
do not apply loads.

Calibrated camera reconstruction
--------------------------------

``PinholeCamera`` stores undistorted pixel intrinsics and world-to-camera
extrinsics, with translations in metres. ``triangulate_point`` checks positive
depth, parallax and reprojection. ``lift_depth`` needs aligned metric camera-Z
depth. Relative depth, surface pointmaps and ray distance are different quantities.
Crops/resizing require matching intrinsics. Sapiens2 decoding and person/time
assignment live outside MyoSuite. Synthetic reprojection tests do not establish
physical calibration or participant accuracy.

Streaming
---------

``StreamingKinematics`` fits timestamped poses with configurable lag/window.
Coupled positions, velocities and accelerations follow the model constraints.
Reset on reference-space changes or tracking discontinuities; otherwise an origin
jump can be misinterpreted as acceleration and muscle loading.
