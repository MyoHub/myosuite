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

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import WrapperSpec, registry as _gym_registry

from myosuite.core.config import EnvConfig, TaskConfig, check_control_step
from myosuite.core.multi_agent_config import MultiAgentTaskConfig

if TYPE_CHECKING:
    from myosuite.core.specs import EnvSpec

# Internal store: env_id -> registration metadata
_ENV_REGISTRY: dict[str, dict[str, Any]] = {}
_MJ_INSTABILITY_WRAPPER_ENTRY_POINT = (
    "myosuite.envs.wrappers:MjInstabilityTerminationWrapper"
)
_ATTRIBUTE_FORWARDING_WRAPPER_ENTRY_POINT = (
    "myosuite.envs.wrappers:AttributeForwardingWrapper"
)


def _append_default_wrappers(
    additional_wrappers: tuple[WrapperSpec, ...] | tuple[Any, ...],
    wrap_mj_instability_termination: bool,
) -> tuple[Any, ...]:
    """Append default wrapper specs while preserving caller-provided wrappers."""
    wrappers = tuple(additional_wrappers or ())
    # The instability wrapper is the outermost one and forwards public attributes;
    # without it, a plain forwarding wrapper takes that place.
    entry_point, name = (
        (_MJ_INSTABILITY_WRAPPER_ENTRY_POINT, "MjInstabilityTerminationWrapper")
        if wrap_mj_instability_termination
        else (_ATTRIBUTE_FORWARDING_WRAPPER_ENTRY_POINT, "AttributeForwardingWrapper")
    )
    if any(getattr(spec, "entry_point", None) == entry_point for spec in wrappers):
        return wrappers
    return wrappers + (WrapperSpec(name=name, entry_point=entry_point, kwargs={}),)


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


def _deep_equal(a: Any, b: Any) -> bool:
    """Equality that also works for dicts, sequences and arrays (``==`` raises on arrays)."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_deep_equal(x, y) for x, y in zip(a, b))
    return bool(a == b)


def register_env(
    env_id: str,
    entry_point: str,
    max_episode_steps: int = 200,
    backend_configs: dict[str, Any] | None = None,
    wrap_mj_instability_termination: bool = True,
    time_limit: bool = True,
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
        backend_configs: Optional per-backend override dicts.
        wrap_mj_instability_termination: Append the default instability wrapper
            so loaded environments convert MuJoCo instability into termination.
        time_limit: Let ``gym.make`` enforce *max_episode_steps* with
            gymnasium's ``TimeLimit``. ``False`` for an env that truncates
            itself, such as the multi-agent env, whose per-agent ``truncated``
            dict ``TimeLimit`` would replace with a bare ``True``; the limit is
            then only recorded here.
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
        "backend_configs": backend_configs or {},
        "additional_wrappers": kwargs["additional_wrappers"],
    }
    gym_max_episode_steps = max_episode_steps if time_limit else None
    if env_id in _gym_registry:
        existing = _gym_registry[env_id]
        new_kwargs = kwargs.get("kwargs", {})
        same_spec = (
            existing.entry_point == entry_point
            and existing.max_episode_steps == gym_max_episode_steps
            and _deep_equal(dict(existing.kwargs or {}), dict(new_kwargs or {}))
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
        max_episode_steps=gym_max_episode_steps,
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
                # The env truncates itself with per-agent dicts.
                time_limit=False,
                kwargs={"task_config": task_config},
                **kwargs,
            )
        return env_id

    # --- Single-agent CPU registration ---
    if "cpu" in backends:
        entry_point = "myosuite.envs.modular_env:ModularTaskEnv"
        register_env(
            env_id=env_id,
            entry_point=entry_point,
            max_episode_steps=task_config.max_episode_steps,
            kwargs={"task_config": task_config},
            **kwargs,
        )

    # --- MJX registration ---
    if "mjx" in backends:
        _register_task_mjx(env_id, task_config, kwargs.get("additional_wrappers", ()))

    # --- Variant expansion (only for base tasks, not their variants) ---
    if _expand_variants:
        _register_task_variants(task_config, env_id, backends, **kwargs)

    return env_id


def mjx_feature_overrides(features: Iterable[Any]) -> dict[str, Any] | None:
    """MJX config overrides for the *features* the experimental MJX route supports.

    Only sarcopenia (a model edit) is supported.

    Args:
        features: ``EnvConfig.features`` (wrapper specs).

    Returns:
        The config overrides, or ``None`` if a feature is not supported on MJX.
    """
    overrides: dict[str, Any] = {}
    for spec in features:
        if spec.name != "SarcopeniaWrapper":
            return None
        overrides["sarcopenia_force_scale"] = (spec.kwargs or {}).get(
            "force_scale", 0.5
        )
    return overrides


def _register_task_mjx(
    env_id: str, task_config: TaskConfig, features: Iterable[Any] = ()
) -> None:
    """Register a TaskConfig-based env on the MJX backend.

    Args:
        env_id: Gymnasium env id to use for MJX registration.
        task_config: Task specification.
        features: Wrapper specs of the registration (see :func:`mjx_feature_overrides`).
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
    defaults = mjx_feature_overrides(features) or {}

    def _cfg_fn() -> Any:
        return cfg

    def _env_cls(
        config: Any, config_overrides: dict[str, Any] | None = None
    ) -> MjxModularTaskEnv:
        # The env rebuilds its config from task_config; overrides apply on top.
        return MjxModularTaskEnv(
            task_config, config_overrides={**defaults, **(config_overrides or {})}
        )

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

        # The experimental mjx route supports sarcopenia only (a model edit).
        variant_backends = (
            backends
            if mjx_feature_overrides(vspec.features) is not None
            else backends & {"cpu"}
        )
        variant_kwargs = dict(kwargs)
        if vspec.features:
            variant_kwargs["additional_wrappers"] = (
                tuple(variant_kwargs.get("additional_wrappers", ())) + vspec.features
            )
        register_task(
            task_config=variant_cfg,
            env_id=variant_id,
            backends=variant_backends,
            _expand_variants=False,
            **variant_kwargs,
        )


def make_env(
    env_id: str | EnvConfig, backend: str | None = None, **overrides: Any
) -> Any:
    """Create an environment on the specified backend.

    One call builds the same env on every backend: the registration of the env id
    gives the defaults, an :class:`~myosuite.core.config.EnvConfig` overrides them
    (episode length, ``features`` such as noise or fatigue, task kwargs).

    Args:
        env_id: Registered environment identifier, or an :class:`EnvConfig`.
        backend: One of "cpu", "mjx", or "mjlab"; default: ``EnvConfig.backend``
            ("cpu" for a plain id).
        **overrides: Backend-specific options, merged over ``EnvConfig.backend_options``
            and (CPU) ``task_kwargs``: env constructor kwargs on the CPU, ``device``
            etc. on mjlab, ``config_overrides`` on "mjx".

    Returns:
        An environment instance appropriate for the backend.

    Raises:
        ValueError: If the backend is not recognised, or a config field does not
            apply to it (``num_envs > 1`` on the CPU).
        NotImplementedError: If ``features`` are given on the experimental "mjx" backend.

    Example:
        >>> env = make_env("myoElbowPose1D6MRandom-v0")
        >>> cfg = EnvConfig(
        ...     "myoElbowPose1D6MRandom-v0",
        ...     features=(wrapper_spec(MotorNoiseWrapper, motor_noise=0.05),),
        ... )
        >>> env = make_env(cfg)  # CPU
        >>> envs = make_env(cfg, backend="mjlab", num_envs=4096)
    """
    config = env_id if isinstance(env_id, EnvConfig) else EnvConfig(env_id=env_id)
    backend = backend or config.backend
    if backend == "cpu":
        return _make_cpu(config, overrides)
    elif backend == "mjx":
        return _make_mjx(config, overrides)
    elif backend == "mjlab":
        return _make_mjlab(config, overrides)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Choose from: cpu, mjx, mjlab")


def _substeps(ctrl_dt: float, timestep: float) -> int:
    """Physics steps per control step of *ctrl_dt* (a whole multiple of *timestep*)."""
    n_substeps = max(1, round(ctrl_dt / timestep))
    check_control_step(n_substeps, timestep, ctrl_dt)
    return n_substeps


def _make_cpu(config: EnvConfig, overrides: dict[str, Any]) -> Any:
    from myosuite.envs.wrappers import apply_features

    if config.num_envs not in (None, 1):
        raise ValueError(
            f"The CPU backend builds one env, got num_envs={config.num_envs}; use "
            "backend='mjlab' for parallel envs."
        )
    kwargs = {**config.task_kwargs, **config.backend_options, **overrides}
    if config.max_episode_steps is not None:
        kwargs["max_episode_steps"] = config.max_episode_steps
    env = gym.make(config.env_id, **kwargs)
    if config.ctrl_dt is not None:
        host = env.unwrapped
        n_substeps = _substeps(config.ctrl_dt, host.model.opt.timestep)
        if n_substeps != host.frame_skip:
            env.close()
            try:
                env = gym.make(config.env_id, **{**kwargs, "frame_skip": n_substeps})
            except TypeError as e:
                raise ValueError(
                    f"{config.env_id} does not take frame_skip, so ctrl_dt cannot change."
                ) from e
    return apply_features(env, config.features)


def _make_mjx(config: EnvConfig, overrides: dict[str, Any]) -> Any:
    feature_options = mjx_feature_overrides(config.features)
    if feature_options is None:
        raise NotImplementedError(
            "The experimental mjx backend supports only SarcopeniaWrapper in "
            "EnvConfig.features."
        )
    try:
        from mujoco_playground import registry as pg_registry
    except ImportError as e:
        raise ImportError(
            "MJX backend requires mujoco_playground. "
            "Install with: pip install myosuite[mjx]"
        ) from e
    options = {**feature_options, **config.backend_options, **overrides}
    if config.num_envs is not None:
        options["num_envs"] = config.num_envs
    if config.ctrl_dt is not None:
        options["ctrl_dt"] = config.ctrl_dt
    return pg_registry.load(config.env_id, config_overrides=options or None)


def _make_mjlab(config: EnvConfig, overrides: dict[str, Any]) -> Any:
    try:
        import mjlab.envs
    except ImportError as e:
        raise ImportError(
            "mjlab backend requires mjlab. Install with: pip install myosuite[mjlab]"
        ) from e

    # Register the MyoSuite twins with mjlab's task registry (idempotent). Without
    # this, they exist only when myosuite is pip-installed (mjlab's entry point) or
    # something else imported the backend first.
    import myosuite.envs.myo.backends.mjlab  # noqa: F401, PLC0415
    from myosuite.envs.myo.backends.mjlab.tasks import cpu_reference
    from myosuite.envs.myo.backends.mjlab.tasks.registration import rebuild_twin_cfg

    if config.task_kwargs:
        raise NotImplementedError(
            "EnvConfig.task_kwargs are CPU env constructor kwargs; the mjlab twin reads "
            "them from the registration."
        )
    options = {**config.backend_options, **overrides}
    if config.num_envs is not None:
        options["num_envs"] = config.num_envs
    env_id = config.env_id

    if hasattr(mjlab.envs, "make") and not (
        config.features
        or config.max_episode_steps is not None
        or config.ctrl_dt is not None
    ):
        return mjlab.envs.make(env_id, **options)

    # mjlab 1.x has no envs.make: build the cfg from tasks.registry.
    try:
        import mjlab.tasks  # noqa: F401 — ensure task packages loaded
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.tasks.registry import list_tasks, load_env_cfg
    except ImportError as e:
        raise ImportError(
            "mjlab 1.x fallback requires mjlab.tasks.registry; "
            "install a mjlab build that provides mjlab.envs.make or "
            "ensure MyoSuite tasks are registered with mjlab.tasks.registry"
        ) from e

    if env_id not in list_tasks():
        raise ValueError(
            f"env_id {env_id!r} not in mjlab task registry (list_tasks()): "
            "it has no mjlab twin (register one with register_mjlab_task())."
        )
    # The registered cfg is built once at import; features and ctrl_dt rebuild it from
    # the CPU registration (extra wrappers, ``frame_skip``).
    cfg = load_env_cfg(env_id)
    task_kwargs: dict[str, Any] = {}
    if config.ctrl_dt is not None:
        n_substeps = _substeps(config.ctrl_dt, cfg.sim.mujoco.timestep)
        if n_substeps != cfg.decimation:
            task_kwargs["frame_skip"] = n_substeps
    if config.features or task_kwargs:
        cfg = rebuild_twin_cfg(env_id, config.features, task_kwargs)

    num_envs = options.pop("num_envs", None)
    if num_envs is not None and hasattr(cfg, "scene"):
        cfg.scene.num_envs = int(num_envs)
    if config.max_episode_steps is not None:
        step_dt = cfg.decimation * cfg.sim.mujoco.timestep
        cfg.episode_length_s = cpu_reference.episode_length_s(
            config.max_episode_steps, step_dt
        )

    device = options.pop("device", None)
    if device is None:
        try:
            import torch

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    env_cls = getattr(cfg, "env_cls", ManagerBasedRlEnv)
    return env_cls(cfg, device=device, **options)
