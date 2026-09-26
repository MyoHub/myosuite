# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""
Path utility functions for MyoSuite environments.

Extracted from env_base.py. Provides vectorised reward computation over
rollout paths (MJRL-compatible format) and success evaluation helpers.

Env contract (for compute_path_rewards and evaluate_success):
  - env must expose: get_reward_dict(obs_dict), and for compute_path_rewards
    also obsvec2obsdict(obs_vec) and rwd_mode (string key into rwd_dict).
  - Legacy BaseV0/MujocoEnv and MyoGymnasiumEnv (with compatibility shims)
    satisfy this. When legacy shims are removed, callers must use envs that
    provide obsvec2obsdict (or equivalent) and rwd_mode.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_path_rewards(env: Any, paths: dict) -> dict:
    """Compute vectorised rewards for a batch of rollout paths.

    Calls ``env.get_reward_dict`` over the full observation trajectory stored
    in *paths*, then time-aligns the rewards (last step is redundant).

    Args:
        env: A ``MujocoEnv`` instance (must expose ``obsvec2obsdict``,
            ``get_reward_dict``, and ``rwd_mode``).
        paths: Dict with key ``"observations"`` of shape
            ``(num_traj, horizon, obs_dim)``.  Rewards and done flags are
            written back in-place.

    Returns:
        The same *paths* dict with ``"rewards"`` and ``"done"`` populated.
    """
    obs_dict = env.obsvec2obsdict(paths["observations"])
    rwd_dict = env.get_reward_dict(obs_dict)

    rewards = rwd_dict[env.rwd_mode]
    done = rwd_dict["done"]
    # time-align rewards (last step is redundant)
    done[..., :-1] = done[..., 1:]
    rewards[..., :-1] = rewards[..., 1:]
    paths["done"] = done if done.shape[0] > 1 else done.ravel()
    paths["rewards"] = rewards if rewards.shape[0] > 1 else rewards.ravel()
    return paths


def truncate_paths(paths: list[dict]) -> list[dict]:
    """Truncate rollout paths at the first terminal transition.

    Args:
        paths: List of path dicts, each containing a ``"rewards"`` array and
            a ``"done"`` array of the same length.

    Returns:
        The same list with each path truncated in-place at termination.
    """
    paths[0]["rewards"].shape[0]
    for path in paths:
        if path["done"][-1] == False:  # noqa: E712
            path["terminated"] = False
        elif path["done"][0] == False:  # noqa: E712
            terminated_idx = sum(~path["done"]) + 1
            for key in path.keys():
                path[key] = path[key][: terminated_idx + 1, ...]
            path["terminated"] = True
    return paths


def evaluate_success(
    env: Any,
    paths: list[dict],
    logger: Any = None,
    successful_steps: int = 5,
) -> float:
    """Evaluate rollout paths and return the success percentage.

    A path is considered successful when the ``"solved"`` flag is ``True`` for
    more than *successful_steps* time steps.

    Args:
        env: A ``MujocoEnv`` instance (used only for ``env.horizon``).
        paths: List of path dicts, each with ``env_infos["solved"]``.
        logger: Optional logger exposing ``log_kv(key, value)``; metrics are
            written if provided.
        successful_steps: Minimum number of solved steps to count as success.

    Returns:
        Percentage (0–100) of successful paths.
    """
    num_success = 0
    num_paths = len(paths)

    for path in paths:
        if np.sum(path["env_infos"]["solved"] * 1.0) > successful_steps:
            num_success += 1
    success_percentage = num_success * 100.0 / num_paths

    if logger:
        rwd_sparse = np.mean([np.mean(p["env_infos"]["rwd_sparse"]) for p in paths])
        rwd_dense = np.mean(
            [np.sum(p["env_infos"]["rwd_dense"]) / env.horizon for p in paths]
        )
        logger.log_kv("rwd_sparse", rwd_sparse)
        logger.log_kv("rwd_dense", rwd_dense)
        logger.log_kv("success_percentage", success_percentage)

    return success_percentage
