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


def _truncate_time_arrays(data: dict, length: int, end: int) -> None:
    for key, value in data.items():
        if isinstance(value, dict):
            _truncate_time_arrays(value, length, end)
        elif isinstance(value, np.ndarray) and value.ndim and len(value) == length:
            data[key] = value[:end]


def truncate_paths(paths: list[dict]) -> list[dict]:
    """Truncate rollout paths at the first terminal transition.

    Args:
        paths: List of path dicts, each containing a ``"rewards"`` array and
            a one-dimensional ``"done"`` array of the same length. NumPy arrays
            with that leading length are time-aligned, including leaves of nested
            ``env_infos`` and ``agent_infos`` dictionaries. Other values, such as
            scalar metadata and arrays of other lengths, are preserved.

    Returns:
        The same list, retaining the first terminal transition and setting each
        path's ``"terminated"`` flag. Empty paths and paths with no terminal
        transition keep all their frames and get ``terminated=False``.

    Raises:
        ValueError: If a path's done flags are not one-dimensional or have a
            different length from its rewards.
    """
    for path in paths:
        done = np.asarray(path["done"])
        length = len(path["rewards"])
        if done.ndim != 1 or len(done) != length:
            raise ValueError(
                "done must be one-dimensional and have the same length as rewards"
            )
        terminal = np.flatnonzero(done)
        if terminal.size:
            _truncate_time_arrays(path, length, int(terminal[0]) + 1)
        path["terminated"] = bool(terminal.size)
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
        successful_steps: A path succeeds when it is solved on more than this
            many time steps (strict, as in upstream MyoSuite / RoboHive).

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
