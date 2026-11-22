# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Shared term functions for MyoSuite environments.

These pure functions work across all three backend paths (CPU/numpy,
MJX/jax.Array, mjlab/torch.Tensor) via the EnvAccessor protocol.

Reward dict contract
--------------------
Every :func:`get_reward_dict` implementation **must** return a ``dict`` containing
the following mandatory keys:

``dense`` (*float*)
    Scalar reward signal passed to the RL agent each step.

``solved`` (*bool*)
    ``True`` when the task goal is achieved on this step. Used for logging and
    curriculum advancement; does not automatically end the episode.

``done`` (*bool*)
    ``True`` when the episode should terminate for task-internal reasons
    (e.g. object dropped, health depleted). The environment's ``step()`` uses
    this to set ``terminated=True``.

Additional task-specific keys (e.g. ``"vel_reward"``, ``"total_hits"``) are
encouraged for debugging and curriculum design, but the three keys above are
the only ones enforced at runtime by
:func:`~myosuite.envs.gymnasium_env._validate_reward_dict`.

Example (minimal)::

    def get_reward_dict(self, obs_dict):
        dist = float(obs_dict["pose_err"].mean())
        return {
            "dense": -dist,
            "solved": dist < 0.02,
            "done": False,
        }
"""

from myosuite.terms import (
    base_obs,
    base_reward,
    base_termination,
    base_event,
    base_action,
)

__all__ = [
    "base_obs",
    "base_reward",
    "base_termination",
    "base_event",
    "base_action",
]
