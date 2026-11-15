Biomechanics
============

Musculoskeletal analysis of motions in :mod:`myosuite.biomechanics`.

Inverse kinematics
------------------

.. automodule:: myosuite.biomechanics.inverse_kinematics
   :members:

OpenSim motions
---------------

.. automodule:: myosuite.biomechanics.opensim_motion
   :members:

Inverse dynamics
----------------

.. automodule:: myosuite.biomechanics.inverse_dynamics
   :members:

Static optimization
-------------------

.. automodule:: myosuite.biomechanics.static_optimization
   :members:

Uncertainty bands
-----------------

Every nominal and sampled frame must converge before quantiles are reported.
``UncertaintySolveError`` retains the failed result, sample index (``None`` for
the nominal model) and rejected frame indices. ``allow_inaccurate=True`` explicitly
accepts OSQP's ``solved inaccurate`` status; iteration failures and nonfinite
values still raise. Failed draws are not discarded or resampled.
Returned bands include every sample's status, iteration count and QP residuals;
the nominal diagnostics are on ``bands.nominal``.

.. automodule:: myosuite.biomechanics.uncertainty
   :members:
