# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""SuperMimic for muscle models: a differentiable world model plus BPTT actor-critic.

Adapted from SuperMimic (Liao et al., arXiv:2610.28480) and its MIT-licensed
reproduction (github.com/CuboiLeo/SuperMimic-Reproduced, Copyright (c) 2026 Yuxi Liu)
to the MuscleMimic tasks on the mjlab backend. The world model integrates muscle
activation dynamics analytically and learns only the mechanics; observations,
rewards and terminations are analytic heads shared with the env's term functions.

The adapted parts keep the reproduction's MIT notice: see
``LICENSE-SuperMimic-Reproduced.txt`` in this package.
"""
