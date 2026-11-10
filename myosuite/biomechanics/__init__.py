# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Biomechanics tools for motion data and recorded rollouts.

- :mod:`.retarget`: landmark IK of motion onto a body model, on its couplings.
- :mod:`.smplh`: SMPL+H forward kinematics (AMASS motions) in NumPy.
- :mod:`.retarget_smplh`: AMASS SMPL+H motions straight onto a body model (a refit source).
- :mod:`.effort`: muscle effort metrics of recorded rollouts (NumPy or torch arrays).
"""
