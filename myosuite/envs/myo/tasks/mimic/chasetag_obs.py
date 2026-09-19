# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Additive 537-dim chase-tag BC observation composer, and a simpler,
empirically better-performing alternative that reuses the directional
policy's own heading channel instead.

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

**Correction (superseded design choice, kept for reference/comparison):**
this module originally hardcoded ``heading_theta=0.0`` on the reasoning that
"chase-tag has no external heading command". That reasoning was wrong — the
opponent's position *is* a natural heading command (point toward it to
chase, away from it to evade), and forcing the appended 9 columns to be
inferred zero-shot (never seen during BC training, which only ever sees the
528-dim locomotion obs) turned out to produce **large, unpredictable
run-to-run variance**: across 4 independently BC-trained checkpoints with
near-identical training/validation loss, this ``chasetag_obs`` approach gave
chase-tag survival times spanning 66-147 steps (std ~35). Simply computing
``heading_theta`` as the angle toward (CHASE) or away from (EVADE) the
opponent and feeding it through the *already-trained* ``_directional_obs``
heading channel (see :func:`chasetag_heading_directional_obs` below) cut
that spread to 80-111 steps (std ~10-26) at N=50 seeds per checkpoint,
because it keeps the policy entirely within its trained input distribution
instead of hoping for zero-shot generalization. Velocity-led "intercept"
heading (aiming ahead of the opponent's predicted position) was also tested
and made no measurable difference against this repo's scripted opponent
(mostly stationary/slow, see ``ChallengeOpponent.opponent_probabilities``),
so plain angle-to-opponent is the recommended default.
:func:`chasetag_heading_directional_obs` is the recommended inference path
for CHASE/EVADE with a warm-started-but-not-fine-tuned directional
checkpoint; ``chasetag_obs`` remains useful only as the observation contract
for an actual chase-tag-aware BC/fine-tuning pass, where the model would get
to *learn* what those extra 9 columns mean rather than needing them to work
zero-shot.
"""

from __future__ import annotations

import math

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


def chasetag_heading_directional_obs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    opponent_pos_xy: np.ndarray,
    evade: bool,
) -> np.ndarray:
    """Native 528-dim directional obs, heading pointed at (or away from) the
    opponent -- the recommended chase-tag inference path (see module
    docstring for why this outperforms and is far more consistent than
    :func:`chasetag_obs`'s zero-shot additive approach).

    Uses a pretrained ``bc_directional_v2``-style checkpoint exactly as
    trained: no obs-space expansion, no ``ActorCritic.load_expanded``, no
    fine-tuning. Just ``ActorCritic.load(ckpt, obs_dim=528, ...)``.

    Args:
        model: Compiled full-body MuJoCo model.
        data: Current ``MjData`` state.
        opponent_pos_xy: Opponent world-frame XY position, shape ``(2,)``.
        evade: If ``True``, heading points away from the opponent (EVADE
            role); if ``False``, toward it (CHASE role).

    Returns:
        Float32 array of shape ``(528,)`` -- identical layout to
        :func:`~myosuite.integrations.musclemimic.bc_directional_collector._directional_obs`.
    """
    pelvis_xy = data.qpos[:2]
    delta = np.asarray(opponent_pos_xy, dtype=np.float64) - pelvis_xy
    if evade:
        delta = -delta
    heading_theta = float(math.atan2(delta[1], delta[0]))
    return _directional_obs(model, data, heading_theta)


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
