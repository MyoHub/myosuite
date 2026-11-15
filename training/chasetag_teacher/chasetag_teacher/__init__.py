# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""A ChaseTag FBVs player from a myoSuperMimic tracker, and its distillation into a 546 -> 354 student.

* :mod:`~chasetag_teacher.library` / :mod:`~chasetag_teacher.reference`: the KIT locomotion clip library and
  the steered reference motion it plays;
* :mod:`~chasetag_teacher.tracker` / :mod:`~chasetag_teacher.player`: the tracker on one player of the CPU
  scene, steered toward or away from the opponent;
* :mod:`~chasetag_teacher.match`: bouts of ``myoChallengeChaseTagFBVs-v0`` between players;
* :mod:`~chasetag_teacher.distill` / :mod:`~chasetag_teacher.augment`: DAgger, the yaw-canonicalised student
  and its ONNX export.
"""
