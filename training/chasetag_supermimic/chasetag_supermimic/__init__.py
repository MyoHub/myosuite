# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""A ChaseTag FBVs player from a myoSuperMimic tracker, and its distillation into a 546 -> 354 student.

* :mod:`~chasetag_supermimic.library` / :mod:`~chasetag_supermimic.reference`: the KIT locomotion clip library and
  the steered reference motion it plays;
* :mod:`~chasetag_supermimic.tracker` / :mod:`~chasetag_supermimic.player`: the tracker on one player of the CPU
  scene, steered toward or away from the opponent;
* :mod:`~chasetag_supermimic.match`: bouts of ``myoChallengeChaseTagFBVs-v0`` between players;
* :mod:`~chasetag_supermimic.distill` / :mod:`~chasetag_supermimic.augment`: DAgger, the yaw-canonicalised student
  and its ONNX export.
"""
