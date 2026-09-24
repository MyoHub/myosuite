# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Register mjlab twins of CPU env ids, including muscle-condition variants."""

from __future__ import annotations

import logging
from collections.abc import Callable

import gymnasium as gym
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import register_mjlab_task

_log = logging.getLogger(__name__)

# Prefixes the CPU registry uses for muscle-condition variants of ``myo*`` ids.
_CONDITION_PREFIXES = ("myoSarc", "myoFati", "myoReaf")


def condition_variants(env_id: str) -> list[str]:
    """CPU-registered muscle-condition variants of *env_id* (base id excluded)."""
    if not env_id.startswith("myo"):
        return []
    candidates = (prefix + env_id[3:] for prefix in _CONDITION_PREFIXES)
    return [cid for cid in candidates if cid in gym.registry]


def register_cpu_twins(
    env_ids: tuple[str, ...],
    env_cfg_fn: Callable[..., ManagerBasedRlEnvCfg],
    rl_cfg_fn: Callable[[], RslRlOnPolicyRunnerCfg],
) -> None:
    """Register each id in *env_ids* and its muscle-condition variants.

    A task whose config cannot be built (e.g. a model asset missing from the
    installed ``myo_sim``) is skipped with a warning so the others stay usable.

    Args:
        env_ids: Base CPU env ids.
        env_cfg_fn: ``env_cfg_fn(env_id, play=False)`` building the env config
            from the CPU registration of ``env_id``.
        rl_cfg_fn: Builds the PPO runner config.
    """
    for base_id in env_ids:
        for env_id in (base_id, *condition_variants(base_id)):
            try:
                env_cfg = env_cfg_fn(env_id)
                play_cfg = env_cfg_fn(env_id, play=True)
            except Exception:  # noqa: BLE001  (any model/config failure)
                _log.warning("mjlab: skipping %s (config failed)", env_id, exc_info=True)
                continue
            register_mjlab_task(
                task_id=env_id, env_cfg=env_cfg, play_env_cfg=play_cfg, rl_cfg=rl_cfg_fn()
            )
