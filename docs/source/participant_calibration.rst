Participant strength calibration
================================

The group fit estimates one common scale for selected muscles across signed,
gravity-corrected MVC postures. It retains antagonist and joint passive torques;
net joint torques cannot identify individual synergist strengths. This differs
from the regularized per-muscle fit in :mod:`myosuite.core.muscle_strength`.

Probes are isometric and preserve the source model, including early-activation
settings. Force clamps, multi-state actuators and gravity compensation are
rejected; stateless muscle control limits must permit the full [0, 1] range.
Fatigued caps reduce active force while retaining passive force.

``max_joint_torque`` and the selected-muscle ``scale_strength`` delegate to
``myo_sim.physics.muscle_strength`` (MyoHub/myo_sim#211); they raise an
``ImportError`` with an installation hint when the installed myo-sim lacks them.
The participant fits (``strength_from_mvc``, ``fit_strength_from_mvcs``) stay here.

.. automodule:: myosuite.biomechanics.calibration
   :members:
