# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Biomechanics tools: musculoskeletal analysis of motions, motion data and recorded rollouts.

- :mod:`.inverse_kinematics`: inverse kinematics of site poses.
- :mod:`.inverse_dynamics`: inverse dynamics of motion clips.
- :mod:`.static_optimization`: muscle activations that reproduce a motion's inverse dynamics.
- :mod:`.opensim_motion`: OpenSim ``.mot`` / ``.sto`` motions on MyoSuite models.
- :mod:`.retarget`: landmark IK of motion onto a body model, on its couplings.
- :mod:`.smplh`: SMPL+H forward kinematics (AMASS motions) in NumPy.
- :mod:`.retarget_smplh`: AMASS SMPL+H motions straight onto a body model (a refit source).
- :mod:`.effort`: muscle effort metrics of recorded rollouts (NumPy or torch arrays).
- :mod:`.external_loads`: Measured body/site wrenches and held-mass gravity loads mapped to generalized forces.
"""
