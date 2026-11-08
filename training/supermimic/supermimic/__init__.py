# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""myoSuperMimic: SuperMimic on MyoSuite's multi-clip full-body tracking task (``myoMimicFullbodyMulticlip-v0``).

A port of muscle-supermimic (github.com/kaibohex/muscle-supermimic, Apache-2.0, commit 8a6244b), which ports its
SuperMimic runner from supermimic_vibed (MIT) and its fast recipe from BlinkMimic and InstantTrack (MIT): see
``NOTICE-muscle-supermimic.txt`` and the ``LICENSE-*.txt`` files of this project. SuperMimic is Liao et al.,
arXiv:2610.28480. Every ported file names its source.

The task (observation, reward, termination, the differentiable kinematics, muscle geometry and activation dynamics
they are computed from) is MyoSuite's; this project consumes it through ``make_env`` and evaluates the task's own
term functions on imagined states. MyoSuite never imports this project.
"""
