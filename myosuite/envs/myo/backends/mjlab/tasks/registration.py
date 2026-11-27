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

# Suffixes the CPU registry uses for muscle-condition variants (see _register_task_variants).
_CONDITION_SUFFIXES = ("Sarc", "Fati", "Reaf")


def _variant_env_id(base_env_id: str, suffix: str) -> str:
    """Derive a condition-variant env id matching ``registry._register_task_variants``."""
    if base_env_id.startswith("myo"):
        return f"myo{suffix}{base_env_id[3:]}"
    if "-" in base_env_id:
        base, ver = base_env_id.rsplit("-", 1)
        return f"{base}{suffix}-{ver}"
    return f"{base_env_id}{suffix}"


def condition_variants(env_id: str) -> list[str]:
    """CPU-registered muscle-condition variants of *env_id* (base id excluded)."""
    candidates = (_variant_env_id(env_id, sfx) for sfx in _CONDITION_SUFFIXES)
    return [cid for cid in candidates if cid in gym.registry]


_CFG_FACTORIES: dict[str, Callable[..., ManagerBasedRlEnvCfg]] = {}


def rebuild_twin_cfg(
    env_id: str,
    features: Iterable[Any] = (),
    task_kwargs: dict[str, Any] | None = None,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Rebuild the twin config of *env_id* from a modified CPU registration.

    Args:
        env_id: A CPU-twin id registered through :func:`register_cpu_twins`.
        features: ``EnvConfig.features`` (wrapper specs) added to the CPU registration.
        task_kwargs: CPU constructor kwargs (``frame_skip``, the subject's
            ``scales`` and ``total_mass``) replacing the registered ones.
        play: Build the play (evaluation) variant of the config.

    Returns:
        A fresh env config.

    Raises:
        NotImplementedError: If the twin is not built from the CPU registration.
    """
    factory = _CFG_FACTORIES.get(env_id)
    if factory is None:
        raise NotImplementedError(
            f"{env_id} is not a CPU twin: its mjlab config cannot be rebuilt from "
            "EnvConfig.features, ctrl_dt, scales or total_mass."
        )
    with feature_overrides(features, task_kwargs):
        return factory(env_id, play=play)


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
