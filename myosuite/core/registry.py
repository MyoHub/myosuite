# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""
Unified environment registry for all three backend paths.

Wraps gymnasium.register() for CPU envs and provides make_env() as a
backend-aware factory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gymnasium as gym
from gymnasium.envs.registration import WrapperSpec, registry as _gym_registry

from myosuite.core.config import EnvConfig, TaskConfig
from myosuite.core.multi_agent_config import MultiAgentTaskConfig

if TYPE_CHECKING:
    from myosuite.core.specs import EnvSpec

# Internal store: env_id -> registration metadata
_ENV_REGISTRY: dict[str, dict[str, Any]] = {}
_MJ_INSTABILITY_WRAPPER_ENTRY_POINT = (
    "myosuite.envs.wrappers:MjInstabilityTerminationWrapper"
)


def _append_default_wrappers(
    additional_wrappers: tuple[WrapperSpec, ...] | tuple[Any, ...],
    wrap_mj_instability_termination: bool,
) -> tuple[Any, ...]:
    """Append default wrapper specs while preserving caller-provided wrappers."""
    wrappers = tuple(additional_wrappers or ())
    if not wrap_mj_instability_termination:
        return wrappers
    if any(
        getattr(spec, "entry_point", None) == _MJ_INSTABILITY_WRAPPER_ENTRY_POINT
        for spec in wrappers
    ):
        return wrappers
    return wrappers + (
        WrapperSpec(
            name="MjInstabilityTerminationWrapper",
            entry_point=_MJ_INSTABILITY_WRAPPER_ENTRY_POINT,
            kwargs={},
        ),
    )


def register(env_spec: EnvSpec, **kwargs: Any) -> str:
    """Register an environment from an :class:`EnvSpec`.

    This is a thin compatibility adapter that routes declarative env specs
    through the existing :func:`register_task` path.
    """
    task_config = env_spec.task_spec.build_task_config()
    return register_task(
        task_config=task_config,
        env_id=env_spec.env_id,
        backends=env_spec.task_spec.backends,
        **kwargs,
    )


def register_env(
    env_id: str,
    entry_point: str,
    max_episode_steps: int = 200,
    config: EnvConfig | None = None,
    backend_configs: dict[str, Any] | None = None,
    wrap_mj_instability_termination: bool = True,
    **kwargs: Any,
) -> None:
    """Register an environment for the CPU (Gymnasium) path.

    Registration is deterministic across import order. If *env_id* already
    exists with identical spec, this is a no-op. If the existing spec differs
    (entry point / max steps / kwargs), the stale spec is replaced in-place.

    Args:
        env_id: Gymnasium env id (e.g. "myoElbowPose1D6MRandom-v0").
        entry_point: Python dotted path to the env class.
        max_episode_steps: Episode step limit.
        config: Optional EnvConfig with model/scene/backend settings.
        backend_configs: Optional per-backend override dicts.
        wrap_mj_instability_termination: Append the default instability wrapper
            so loaded environments convert MuJoCo instability into termination.
        **kwargs: Additional keyword arguments forwarded to gym.register().

    Example:
        >>> register_env(
        ...     "myoElbowPose1D6MRandom-v0",
        ...     "myosuite.envs.myo.tasks.basic.arm.pose:PoseEnvV0",
        ...     max_episode_steps=200,
        ... )
    """
    kwargs = dict(kwargs)
    kwargs["additional_wrappers"] = _append_default_wrappers(
        kwargs.get("additional_wrappers", ()),
        wrap_mj_instability_termination=wrap_mj_instability_termination,
    )
    _ENV_REGISTRY[env_id] = {
        "entry_point": entry_point,
        "max_episode_steps": max_episode_steps,
        "config": config,
        "backend_configs": backend_configs or {},
        "additional_wrappers": kwargs["additional_wrappers"],
    }
    if env_id in _gym_registry:
        existing = _gym_registry[env_id]
        new_kwargs = kwargs.get("kwargs", {})
        same_spec = (
            existing.entry_point == entry_point
            and int(existing.max_episode_steps or 0) == int(max_episode_steps)
            and dict(existing.kwargs or {}) == dict(new_kwargs or {})
            and tuple(existing.additional_wrappers or ())
            == tuple(kwargs["additional_wrappers"] or ())
        )
        if same_spec:
            return
        # Replace stale registration deterministically (avoids import-order drift).
        del _gym_registry[env_id]
    gym.register(
        id=env_id,
        entry_point=entry_point,
        max_episode_steps=max_episode_steps,
        **kwargs,
    )


def register_task(
    task_config: TaskConfig | MultiAgentTaskConfig,
    env_id: str = "",
    backends: set[str] | None = None,
    _expand_variants: bool = True,
    **kwargs: Any,
) -> str:
    """Register a modular task environment from a :class:`TaskConfig`.

    Derives a Gymnasium ``env_id`` from the task config class name when
    *env_id* is not provided (e.g. ``ElbowPoseTask`` → ``"ElbowPoseTask-v0"``).
    Calls :func:`register_env` internally, so registration is idempotent.

    When *backends* includes ``"mjx"``, the environment is also registered
    with the MJX backend via :class:`~myosuite.envs.myo.backends.mjx.mjx_modular_env.MjxModularTaskEnv`.

    Any ``VariantSpec`` entries declared on ``type(task_config).variants``
    are expanded automatically: each variant is registered as a separate
    environment with *config_delta* merged into the base config.

    Args:
        task_config: Data-driven task specification.
        env_id: Override for the Gymnasium env id.  If empty, a name is
            derived from ``type(task_config).__name__``.
        backends: Set of backends to register for.  Defaults to ``{"cpu"}``.
            Supported values: ``"cpu"``, ``"mjx"``.
        **kwargs: Additional keyword arguments forwarded to :func:`register_env`.

    Returns:
        The resolved Gymnasium env id.

    Example:
        >>> from dataclasses import dataclass
        >>> from myosuite.core.config import TaskConfig
        >>> @dataclass
        ... class ElbowTask(TaskConfig):
        ...     model: str = "elbow_standard"
        >>> env_id = register_task(ElbowTask(), backends={"cpu", "mjx"})
        >>> print(env_id)
        ElbowTask-v0
    """
    if backends is None:
        backends = {"cpu"}

    if not env_id:
        env_id = f"{type(task_config).__name__}-v0"

    # --- Multi-agent path ---
    if isinstance(task_config, MultiAgentTaskConfig):
        if "cpu" in backends:
            entry_point = (
                "myosuite.envs.multi_agent_modular_env:ModularMultiAgentTaskEnv"
            )
            register_env(
                env_id=env_id,
                entry_point=entry_point,
                max_episode_steps=task_config.max_episode_steps,
                kwargs={"task_config": task_config},
                **kwargs,
            )
        return env_id

    # --- Single-agent CPU registration ---
    if "cpu" in backends:
        entry_point = "myosuite.envs.modular_env:ModularTaskEnv"
        env_cfg = task_config.to_env_config(env_id=env_id)
        register_env(
            env_id=env_id,
            entry_point=entry_point,
            max_episode_steps=task_config.max_episode_steps,
            config=env_cfg,
            kwargs={"task_config": task_config},
            **kwargs,
        )

    # --- MJX registration ---
    if "mjx" in backends:
        _register_task_mjx(env_id, task_config)

    # --- Variant expansion (only for base tasks, not their variants) ---
    if _expand_variants:
        _register_task_variants(task_config, env_id, backends, **kwargs)

    return env_id


def _register_task_mjx(env_id: str, task_config: TaskConfig) -> None:
    """Register a TaskConfig-based env on the MJX backend.

    Args:
        env_id: Gymnasium env id to use for MJX registration.
        task_config: Task specification.
    """
    try:
        from mujoco_playground import registry as pg_registry
        from myosuite.envs.myo.backends.mjx.mjx_modular_env import (
            MjxModularTaskEnv,
            modular_task_config,
        )
    except ImportError:
        return  # MJX extras not installed; skip silently

    cfg = modular_task_config(task_config)

    def _cfg_fn() -> Any:
        return cfg

    def _env_cls(config: Any) -> MjxModularTaskEnv:
        return MjxModularTaskEnv(task_config, config_overrides=None)

    if env_id not in pg_registry._envs:  # type: ignore[attr-defined]
        pg_registry.register_environment(env_id, _env_cls, _cfg_fn)


def _register_task_variants(
    task_config: TaskConfig,
    base_env_id: str,
    backends: set[str],
    **kwargs: Any,
) -> None:
    """Expand and register VariantSpec entries declared on the task config class.

    Args:
        task_config: Base task config whose class declares ``variants``.
        base_env_id: Gymnasium env id for the base (non-variant) task.
        backends: Backend set forwarded to each variant registration.
        **kwargs: Forwarded to :func:`register_task`.
    """
    import copy
    import dataclasses

    from myosuite.core.config import VariantSpec

    variants: list[VariantSpec] = type(task_config).variants
    if not variants:
        return

    # Derive the variant env_id: insert suffix after the first "myo" prefix.
    # e.g. "myoElbowPoseRandom-v0" + suffix "Sarc" → "myoSarcElbowPoseRandom-v0"
    # For non-"myo" ids, suffix is appended before the version tag.
    for vspec in variants:
        if base_env_id.startswith("myo"):
            variant_id = "myo" + vspec.suffix + base_env_id[3:]
        else:
            # e.g. "ElbowPoseTask-v0" → "ElbowPoseTaskSarc-v0"
            base, ver = base_env_id.rsplit("-", 1)
            variant_id = f"{base}{vspec.suffix}-{ver}"

        # Build variant config by replacing fields from config_delta
        variant_cfg = copy.copy(task_config)
        for field_name, value in vspec.config_delta.items():
            if dataclasses.is_dataclass(variant_cfg):
                object.__setattr__(variant_cfg, field_name, value)

        register_task(
            task_config=variant_cfg,
            env_id=variant_id,
            backends=backends,
            _expand_variants=False,
            **kwargs,
        )


def make_env(env_id: str, backend: str = "cpu", **overrides: Any) -> Any:
    """Create an environment on the specified backend.

    Args:
        env_id: Registered environment identifier.
        backend: One of "cpu", "mjx", or "mjlab".
        **overrides: Keyword arguments forwarded to the env constructor.

    Returns:
        An environment instance appropriate for the backend.

    Raises:
        ValueError: If the backend is not recognised.

    Example:
        >>> env = make_env("myoElbowPose1D6MRandom-v0")
        >>> env = make_env("myoElbowPose1D6MRandom-v0", backend="mjx", num_envs=4096)
    """
    if backend == "cpu":
        return gym.make(env_id, **overrides)
    elif backend == "mjx":
        try:
            from mujoco_playground import registry as pg_registry

            return pg_registry.load(env_id)
        except ImportError as e:
            raise ImportError(
                "MJX backend requires mujoco_playground. "
                "Install with: pip install myosuite[mjx]"
            ) from e
    elif backend == "mjlab":
        try:
            import mjlab.envs
        except ImportError as e:
            raise ImportError(
                "mjlab backend requires mjlab. "
                "Install with: pip install myosuite[mjlab]"
            ) from e

        if hasattr(mjlab.envs, "make"):
            return mjlab.envs.make(env_id, **overrides)

        # Fallback: mjlab 1.x uses tasks.registry (no envs.make).
        try:
            import mjlab.tasks  # noqa: F401 — ensure task packages loaded
            from mjlab.envs import ManagerBasedRlEnv
            from mjlab.tasks.registry import load_env_cfg, list_tasks
        except ImportError as e:
            raise ImportError(
                "mjlab 1.x fallback requires mjlab.tasks.registry; "
                "install a mjlab build that provides mjlab.envs.make or "
                "ensure MyoSuite tasks are registered with mjlab.tasks.registry"
            ) from e

        if env_id not in list_tasks():
            raise ValueError(
                f"env_id {env_id!r} not in mjlab task registry (list_tasks()). "
                "MyoSuite mjlab tasks must be registered via register_mjlab_task()."
            )
        cfg = load_env_cfg(env_id)

        # Allow vectorized env count override for mjlab 1.x fallback path.
        num_envs = overrides.pop("num_envs", None)
        if num_envs is not None and hasattr(cfg, "scene"):
            cfg.scene.num_envs = int(num_envs)

        device = overrides.pop("device", None)
        if device is None:
            try:
                import torch

                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        env_cls = getattr(cfg, "env_cls", ManagerBasedRlEnv)
        return env_cls(cfg, device=device, **overrides)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Choose from: cpu, mjx, mjlab")
