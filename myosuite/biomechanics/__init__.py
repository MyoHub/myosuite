# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Biomechanics tools: musculoskeletal analysis of motions, motion data and recorded rollouts.

- :mod:`.inverse_kinematics`: inverse kinematics of site poses.
- :mod:`.inverse_dynamics`: inverse dynamics of motion clips.
- :mod:`.static_optimization`: muscle activations that reproduce a motion's inverse dynamics.
- :mod:`.computed_muscle_control`: excitations that make a forward simulation track a motion.
- :mod:`.uncertainty`: Monte Carlo uncertainty bands for muscle estimates.
- :mod:`.opensim_motion`: OpenSim ``.mot`` / ``.sto`` motions on MyoSuite models.
- :mod:`.opensim_grf`: OpenSim 4 state-path normalization and force-plate ``ExternalLoads`` conversion.
- :mod:`.retarget`: landmark IK of motion onto a body model, on its couplings.
- :mod:`.observability`: local landmark geometry, rank and motion ambiguity.
- :mod:`.smplh`: SMPL+H forward kinematics (AMASS motions) in NumPy.
- :mod:`.retarget_smplh`: AMASS SMPL+H motions straight onto a body model (a refit source).
- :mod:`.retarget_instanthmr`: InstantHMR MHR70 subject scaling, inverse kinematics, retargeting and 3D muscle mesh reconstruction.
- :mod:`.effort`: muscle effort metrics of recorded rollouts (NumPy or torch arrays).
- :mod:`.kinematic_metrics`: smoothness, velocity symmetry, curvature-velocity power law, Fitts throughput and cohort distances of recorded motions.
- :mod:`.external_loads`: Measured body/site wrenches and held-mass gravity loads mapped to generalized forces.
- :mod:`.camera`: Calibrated metric camera reconstruction with depth lifting, triangulation and rejection diagnostics.
- :mod:`.foveation`: Calibrated eccentricity-dependent retinal acuity blur and foveal patch sampling.
- :mod:`.chumpy_pickle`: Chumpy-free ``.pkl`` unpickler for legacy SMPL, SMPL+H, SMPL-X and MANO body models.
- :mod:`.emg`: EMG filtering, muscle activation dynamics and tendon-node projection.
- :mod:`.hand_retarget`: MANO-to-wrist frame calibration, presets, and DexPilot SLSQP hand retargeting.
- :mod:`.hoi_retarget`: Hand-object interaction clip transforms and contact-aware MuJoCo IK.
"""
