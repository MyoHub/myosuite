# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Register mjlab twins of CPU env ids, including muscle-condition variants."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any

import gymnasium as gym
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import register_mjlab_task

from myosuite.envs.myo.backends.mjlab.tasks.cpu_reference import feature_overrides

_log = logging.getLogger(__name__)

# Prefixes the CPU registry uses for muscle-condition variants of ``myo*`` ids.
_CONDITION_PREFIXES = ("myoSarc", "myoFati", "myoReaf")


def condition_variants(env_id: str) -> list[str]:
    """CPU-registered muscle-condition variants of *env_id* (base id excluded)."""
    if not env_id.startswith("myo"):
        return []
    candidates = (prefix + env_id[3:] for prefix in _CONDITION_PREFIXES)
    return [cid for cid in candidates if cid in gym.registry]


_CFG_FACTORIES: dict[str, Callable[..., ManagerBasedRlEnvCfg]] = {}


def rebuild_twin_cfg(
    env_id: str,
    features: Iterable[Any] = (),
    task_kwargs: dict[str, Any] | None = None,
) -> ManagerBasedRlEnvCfg:
    """Rebuild the twin config of *env_id* from a modified CPU registration.

    Args:
        env_id: A CPU-twin id registered through :func:`register_cpu_twins`.
        features: ``EnvConfig.features`` (wrapper specs) added to the CPU registration.
        task_kwargs: CPU constructor kwargs (``frame_skip``) replacing the registered ones.

    Returns:
        A fresh env config.

    Raises:
        NotImplementedError: If the twin is not built from the CPU registration.
    """
    factory = _CFG_FACTORIES.get(env_id)
    if factory is None:
        raise NotImplementedError(
            f"{env_id} is not a CPU twin: its mjlab config cannot be rebuilt from "
            "EnvConfig.features or ctrl_dt."
        )
    with feature_overrides(features, task_kwargs):
        return factory(env_id)


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
                _log.warning(
                    "mjlab: skipping %s (config failed)", env_id, exc_info=True
                )
                continue
            _CFG_FACTORIES[env_id] = env_cfg_fn
            register_mjlab_task(
                task_id=env_id,
                env_cfg=env_cfg,
                play_env_cfg=play_cfg,
                rl_cfg=rl_cfg_fn(),
            )
