# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Gymnasium re-export for ``from myosuite.utils import gym``.

Prefer ``import gymnasium as gym`` plus ``import myosuite`` in new code.
"""

from __future__ import annotations

import gymnasium as gym

import sys

sys.modules[__name__] = gym
