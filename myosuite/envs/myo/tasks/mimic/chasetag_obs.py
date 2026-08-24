# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Additive 537-dim chase-tag BC observation composer.

**Not the same contract as** :class:`~myosuite.integrations.musclemimic.bc_directional_collector`
/ ``MuscleMimicFullbodyDirectionalEnv``'s pure-locomotion obs. This module
exists for a **future** chase-tag BC data-collection/fine-tuning pass and is
not wired into any live training loop by this change — it's a post-hoc
adapter callable from wrapper/collector code around the new full-body
scripted-opponent env or FBVs, not a change to either env's own obs contract.

The design is strictly additive: the existing 528-dim ``_directional_obs``
block (``qpos_local(82) + qvel_local(82) + act(354) + root_vel_body(2) +
heading_cmd(2) + orientation(6)``) is reproduced byte-identically as a
leading prefix — unmodified, so the pretrained ``bc_directional_v2``
checkpoint's first-layer weights can be warm-started into a larger network
via :meth:`~myosuite.envs.myo.tasks.mimic.policy.ActorCritic.load_expanded`
(see that module) without touching the 528 shared columns. Opponent-relative
pose (7 dims, via :func:`relative_pose_obs`) and a chaser/runner role
one-hot (2 dims) are appended after it, for 537 dims total.

Chase-tag has no external heading command (unlike the directional-locomotion
task), so ``heading_theta`` is always fixed at ``0.0`` internally — the
``heading_cmd`` sub-block becomes the constant ``[cos(0), sin(0)] = [1, 0]``
for every call, rather than reflecting a live command. This keeps the
byte-identical guarantee simple to verify (call ``_directional_obs(model,
data, 0.0)`` directly to reproduce the same 528-dim prefix) while carrying no
task-relevant heading information, since none exists in this context.
"""

from __future__ import annotations

import mujoco
import numpy as np

from myosuite.integrations.musclemimic.bc_directional_collector import (
    _directional_obs,
)
from myosuite.terms.opponent_relative_obs import relative_pose_obs

#: Fixed heading passed to ``_directional_obs`` — chase-tag has no external
#: heading command, so the 2-dim heading_cmd sub-block is always [1, 0].
_NO_HEADING_COMMAND_THETA = 0.0

CHASETAG_OBS_DIM = 537


def chasetag_obs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    opponent_pos: np.ndarray,
    opponent_vel: np.ndarray,
    role_onehot: np.ndarray,
) -> np.ndarray:
    """Assemble the 537-dim additive chase-tag BC observation.

    Args:
        model: Compiled full-body MuJoCo model.
        data: Current ``MjData`` state.
        opponent_pos: Opponent position relative to the world, shape ``(3,)``
            (already resolved by the caller — mocap body xpos for the
            scripted-opponent env, or the opponent agent's pelvis site xpos
            for FBVs).
        opponent_vel: Opponent linear velocity, shape ``(3,)``, resolved the
            same way as ``opponent_pos``.
        role_onehot: Chaser/runner one-hot, shape ``(2,)`` (see
            :func:`myosuite.terms.multiplayer.chase_tag_vs_obs.role_obs`).

    Returns:
        Float32 array of shape ``(537,)``: the 528-dim ``_directional_obs``
        prefix (byte-identical to a direct call with the same fixed
        ``heading_theta``), followed by the 7-dim opponent-relative block and
        the 2-dim role one-hot.
    """
    directional_block = _directional_obs(model, data, _NO_HEADING_COMMAND_THETA)

    self_pos = data.qpos[:3].astype(np.float32)
    self_vel = data.qvel[:3].astype(np.float32)
    opponent_block = relative_pose_obs(self_pos, self_vel, opponent_pos, opponent_vel)

    role_block = np.asarray(role_onehot, dtype=np.float32)

    obs = np.concatenate([directional_block, opponent_block, role_block]).astype(
        np.float32
    )
    assert obs.shape == (
        CHASETAG_OBS_DIM,
    ), f"chasetag_obs produced shape {obs.shape}, expected ({CHASETAG_OBS_DIM},)"
    return obs


__all__ = ["chasetag_obs", "CHASETAG_OBS_DIM"]
