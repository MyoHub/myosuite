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

from collections.abc import Iterable, Mapping
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


def path_obs_series(path: Mapping, keys: Iterable[str]) -> dict[str, np.ndarray]:
    """Time series of observation-dict entries of one rollout path, from the reset state.

    :func:`~myosuite.utils.policy_utils.examine_policy` records are time-aligned:
    record ``t`` holds the obs dict of state ``s_t``, from the reset state ``s_0`` to
    the final state, so sample ``i`` is the state ``i * dt`` after the reset.

    Paths in the older post-step layout (``env_infos[t]`` after step ``t + 1``, the
    terminal record possibly repeated) are realigned: a repeated record is dropped and
    ``s_0`` is recovered from ``path["observations"][0]`` by splitting the flat vector
    in the key order and sizes of the obs dict (the order ``MyoGymnasiumEnv``
    concatenates them in). That layout is recognised only when
    ``path["observations"][1]`` equals the flattened first obs dict, so a normalised or
    otherwise transformed observation vector is never split.

    Args:
        path: A rollout path / trace group with ``env_infos.obs_dict`` (time-stacked
            arrays) and optionally ``observations``.
        keys: Obs-dict keys to return; keys missing from the obs dict are skipped.

    Returns:
        Key to ``(T + 1, ...)`` array (``(T, ...)`` for an old-layout path without a
        usable reset observation).
    """
    series = {
        k: np.asarray(v, dtype=float) for k, v in path["env_infos"]["obs_dict"].items()
    }
    observations = path.get("observations")
    if observations is not None and len(observations) >= 2:
        observations = np.asarray(observations, dtype=float)
        flat = [v.reshape(len(v), -1) for v in series.values()]
        first = np.concatenate([f[0] for f in flat])
        old_layout = (
            observations.shape[-1] == first.size
            and len(observations) in (len(flat[0]), len(flat[0]) + 1)
            and not np.allclose(observations[0], first, rtol=1e-6, atol=1e-6)
            and np.allclose(observations[1], first, rtol=1e-6, atol=1e-6)
        )
        if old_layout:
            # Drop the repeated terminal record if present, then prepend s_0.
            n_post = len(observations) - 1
            splits = np.cumsum([f.shape[1] for f in flat])[:-1]
            reset = np.split(observations[0], splits)
            series = {
                k: np.concatenate([r.reshape(1, *v.shape[1:]), v[:n_post]])
                for (k, v), r in zip(series.items(), reset)
            }
    return {k: series[k] for k in keys if k in series}
