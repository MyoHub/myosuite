Participant model calibration
=============================

The group fit estimates one common scale for selected muscles across signed,
gravity-corrected MVC postures. It retains antagonist and joint passive torques;
net joint torques cannot identify individual synergist strengths. This differs
from the regularized per-muscle fit in :mod:`myosuite.core.muscle_strength`.

Probes are isometric and preserve the source model, including early-activation
settings. Force clamps, multi-state actuators and gravity compensation are
rejected; stateless muscle control limits must permit the full [0, 1] range.
Fatigued caps reduce active force while retaining passive force.

.. automodule:: myosuite.biomechanics.calibration
   :members:

Mass and geometry preparation
-----------------------------

``scale_body_mass`` returns an independent model with refreshed MuJoCo constants
and unchanged muscle strength. For a partial model, ``represented_mass`` is the
mass of its selected segments. Participant whole-body mass requires an explicit
reference participant mass; it scales selected segments by their mass ratio.
Held objects can be excluded by naming anatomical bodies explicitly.

``check_segment_lengths`` compares measured landmark distances with a prepared
subject model at the calibration posture. It reports signed errors and explicit
absolute/relative acceptance tolerances. It verifies geometry; changing segment
lengths, muscle paths and attachment sites requires a separate model-scaling
procedure and validation of the resulting moment arms.

Fatigue parameter calibration
------------------------------

``endurance_time`` predicts sustained-hold failure from residual 3CC-r capacity.
Failure is interpolated within each integration check interval. A hold that has
not failed by ``max_time`` returns infinity, including when the requested horizon
falls partway through a check interval.

``fit_fatigue_parameters`` fits F, and optionally R, against log endurance times.
It initializes the search with finite predicted failures when the supplied start
is censored. Fitting R requires at least three distinct sustained loads. The rest
recovery multiplier r is fixed because sustained holds do not identify it.
These effective fits require validation for the participant and study protocol;
they do not establish individual muscle fatigue or clinical accuracy.
