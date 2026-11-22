Biomechanics
============

Musculoskeletal kinematics, retargeting, inverse dynamics, muscle redundancy solvers, and physiological movement metrics in :mod:`myosuite.biomechanics`.

Inverse kinematics & landmark observability
-------------------------------------------

.. automodule:: myosuite.biomechanics.inverse_kinematics
   :members:

.. automodule:: myosuite.biomechanics.observability
   :members:

OpenSim motions & force-plate loads
-----------------------------------

.. automodule:: myosuite.biomechanics.opensim_motion
   :members:

.. automodule:: myosuite.biomechanics.opensim_grf
   :members:

Body, video & hand retargeting
------------------------------

.. automodule:: myosuite.biomechanics.retarget
   :members:

.. automodule:: myosuite.biomechanics.smplh
   :members:

.. automodule:: myosuite.biomechanics.gvhmr
   :members:

.. automodule:: myosuite.biomechanics.retarget_smplh
   :members:

.. automodule:: myosuite.biomechanics.retarget_instanthmr
   :members:

.. automodule:: myosuite.biomechanics.mano
   :members:

.. automodule:: myosuite.biomechanics.hand_retarget
   :members:

.. automodule:: myosuite.biomechanics.hoi_retarget
   :members:

Inverse dynamics & muscle redundancy
------------------------------------

.. automodule:: myosuite.biomechanics.inverse_dynamics
   :members:

.. automodule:: myosuite.biomechanics.static_optimization
   :members:

.. automodule:: myosuite.biomechanics.computed_muscle_control
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

External loads
--------------

Measured body/site wrenches and held-mass gravity loads mapped to generalized forces.

.. automodule:: myosuite.biomechanics.external_loads
   :members:

Camera
------

Calibrated metric camera reconstruction with depth lifting, triangulation and rejection diagnostics.

.. automodule:: myosuite.biomechanics.camera
   :members:

Muscle effort, EMG & movement quality
-------------------------------------

.. automodule:: myosuite.biomechanics.effort
   :members:

.. automodule:: myosuite.biomechanics.emg
   :members:

.. automodule:: myosuite.biomechanics.kinematic_metrics
   :members:

Muscle mesh reconstruction
--------------------------

.. automodule:: myosuite.viz.muscle_meshes
   :members:
