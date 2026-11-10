# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Half-turn symmetry of the FBVs scene: the two start positions are each other's 180 deg rotation about the vertical
axis through the origin, so an observation rotated by it is a valid observation with the same excitations."""

from __future__ import annotations

import numpy as np

NEG = np.r_[0, 1, 89, 90, 531, 532, 534, 535, 537, 538, 540, 541]
"""World-frame x/y entries: root position, root linear velocity, own pelvis position and velocity, opponent-relative
position and velocity (the root angular velocity is body-frame: unchanged)."""


def half_turn(obs: np.ndarray) -> np.ndarray:
    """(N, 546) observations rotated by 180 deg about z."""
    out = obs.copy()
    out[:, NEG] = -out[:, NEG]
    w, x, y, z = (obs[:, 3 + k].copy() for k in range(4))
    out[:, 3], out[:, 4], out[:, 5], out[:, 6] = -z, -y, x, w  # q_z(pi) * q
    return out
