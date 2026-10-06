# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Ordered muscle-command stages that wrappers install in a CPU env.

An env maps its action to a muscle excitation (sigmoid, or as-is for the walk
envs) and then runs the installed stages before writing ``ctrl``:

``map (10) -> noise (20) -> fatigue (30) -> reroute (40) -> custom stages -> ctrl (100)``

No stage is installed by default: a plain env runs only its own map. The
``myoFati*`` / ``myoReaf*`` ids and the wrappers install them; noise additionally
needs a nonzero level. The three built-in stages have a fixed order
(:data:`STAGE_ORDER`), a property of the stage and not of the wrapper nesting, so a stack
built in any order behaves the same. The wrappers in
:mod:`myosuite.envs.wrappers` (``MotorNoiseWrapper``, ``FatigueWrapper``,
``ReafferentationWrapper``) install one stage each through
:meth:`CtrlStageHost.add_ctrl_stage`.

**Custom stages** run after the built-in ones, in the order they were installed (the
wrapping order, or the order of ``EnvConfig.features``): nothing to configure. Only to
insert one earlier, give it an explicit ``order`` between :data:`MAP_ORDER` and
:data:`WRITE_ORDER`; two stages with the same explicit order run in installation order and
trigger a :class:`StageOrderWarning`.

Two kinds of custom stages exist:

* a **portable** :class:`ExcitationStage` (``ExcitationStageWrapper``) works on the
  muscle excitations only, written for numpy and torch, and runs on the CPU env and
  on its mjlab twin;
* an **env-aware** stage (``CtrlStageWrapper``) gets the host env and runs on the CPU
  only.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import mujoco
import numpy as np

MAP_ORDER = 10
"""Priority of the env's own action-to-excitation map; stages run after it."""

WRITE_ORDER = 100
"""Priority of the ``ctrl`` write; stages run before it."""

STAGE_ORDER: dict[str, int] = {"noise": 20, "fatigue": 30, "reroute": 40}
"""Priorities of the built-in stages (fixed, ascending)."""

LATE_ORDER = 50
"""Priority of a custom stage without an explicit order: after the built-ins, in the
order the stages were installed."""

CTRL_STAGE_ORDER: tuple[str, ...] = tuple(STAGE_ORDER)
"""Names of the built-in stages in the order they run."""


class StageOrderWarning(UserWarning):
    """Two stages have the same order, so their relative order is only the name order."""


warnings.simplefilter("always", StageOrderWarning)  # shown on every occurrence


def check_custom_order(name: str, order: float | None) -> None:
    """Raise if an explicit order is not strictly between map and ctrl write (``None`` is fine)."""
    if order is None:
        return
    if not MAP_ORDER < order < WRITE_ORDER:
        raise ValueError(
            f"The order of the custom stage {name!r} must be between {MAP_ORDER} and "
            f"{WRITE_ORDER} (exclusive), got {order}."
        )


def warn_order_clash(
    name: str, order: float, installed: dict[str, float], stacklevel: int = 3
) -> None:
    """Warn prominently if *name* has the same *order* as an installed stage.

    Args:
        name: The stage being installed.
        order: Its order.
        installed: ``{stage name: order}`` of the stages already installed.
        stacklevel: Frame the warning is attributed to.
    """
    clash = [*(n for n, o in installed.items() if o == order), name]
    if len(clash) > 1:
        warnings.warn(
            f"STAGE ORDER CLASH: the stages {clash} all have order {order}. They run in the "
            f"order they were installed ({' -> '.join(clash)}); give them distinct orders "
            f"(built-in orders: {STAGE_ORDER}; custom stages without an order run after them).",
            StageOrderWarning,
            stacklevel=stacklevel,
        )


class ExcitationStage:
    """Base class of a portable custom stage on the muscle excitations.

    Subclass it and implement ``__call__(u, xp)``: *u* holds the muscle excitations
    (``(n_muscles,)`` numpy on the CPU env, ``(n_envs, n_muscles)`` torch on mjlab),
    *xp* is the matching array module (``numpy`` or ``torch``); return the new
    excitations. Use only operations both modules share (``xp.clip``, ``xp.where``,
    ``xp.zeros_like``, arithmetic, ...) and no randomness unless it comes from *xp*.
    Keep state in arrays created from *u* (``xp.zeros_like(u)``) so that it has the right
    shape and device, and clear it in :meth:`reset`.

    Attributes:
        name: Unique stage name (not a built-in name).
        order: ``None`` (default): runs after the built-in stages, in installation order;
            or a priority strictly between :data:`MAP_ORDER` and :data:`WRITE_ORDER` to
            insert it earlier.
    """

    name: str = "stage"
    order: float | None = None

    def __call__(self, u: Any, xp: Any) -> Any:  # pragma: no cover - interface
        raise NotImplementedError

    def reset(self, env_ids: Any = None) -> None:
        """Clear the state of the given envs (``None`` or ``slice(None)``: all)."""


class LowPassStage(ExcitationStage):
    """First-order low-pass filter on the excitations (a portable example stage).

    ``y <- y + alpha (u - y)``; the first step after a reset passes *u* through.

    Args:
        alpha: Smoothing factor in ``(0, 1]``; ``1`` is no filtering.
        name: Stage name.
        order: Stage order.
    """

    def __init__(
        self, alpha: float = 0.3, name: str = "lowpass", order: float | None = None
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}.")
        self.alpha, self.name, self.order = alpha, name, order
        self._y: Any = None
        self._fresh: Any = None

    def __call__(self, u: Any, xp: Any) -> Any:
        if self._y is None:
            # one flag per env, on u's device (0-d on the CPU env)
            self._y, self._fresh = xp.zeros_like(u), xp.ones_like(u[..., 0], dtype=bool)
        fresh = self._fresh[..., None]
        self._y = xp.where(fresh, u, self._y + self.alpha * (u - self._y))
        self._fresh = xp.zeros_like(self._fresh)
        return self._y.clone() if hasattr(self._y, "clone") else self._y.copy()

    def reset(self, env_ids: Any = None) -> None:
        if self._fresh is None:
            return
        if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
            self._fresh[...] = True
        else:
            self._fresh[env_ids] = True


CtrlStage = Callable[[Any, np.ndarray], np.ndarray]
"""``stage(env, ctrl) -> ctrl``: edits (a copy of) the control vector."""

ResetStage = Callable[[Any], None]
"""``reset(env)``: per-episode reset, called where the env reset its muscle state."""


_REMOVED_KWARGS = {
    "muscle_condition": 'use a variant id ("myoFati*", "myoSarc*", "myoReaf*") or wrap the env '
    "with FatigueWrapper / SarcopeniaWrapper / ReafferentationWrapper",
    "fatigue_reset_vec": "pass it to FatigueWrapper",
    "fatigue_reset_random": "pass it to FatigueWrapper (or call env.set_fatigue_reset_random)",
    "motor_noise": "wrap the env with MotorNoiseWrapper",
}


def reject_removed_kwargs(env_cls: type, kwargs: dict[str, Any]) -> None:
    """Raise ``TypeError`` for the constructor kwargs that wrappers replaced."""
    for name, hint in _REMOVED_KWARGS.items():
        if name in kwargs:
            raise TypeError(
                f"{env_cls.__name__}() no longer takes {name!r}: {hint} "
                "(myosuite.envs.wrappers)."
            )


class CtrlStageHost:
    """Mixin of the CPU envs whose action pipeline runs wrapper-installed stages.

    The host env provides ``model``, ``np_random``, ``_muscle_act_ind`` (the
    muscle entries of ``ctrl``) and calls :meth:`_run_ctrl_stages` on the
    excitation vector and :meth:`_run_reset_stages` in ``reset()``.
    """

    supports_ctrl_stages = True

    def _stage_store(self) -> dict[str, tuple[float, CtrlStage, ResetStage | None]]:
        return self.__dict__.setdefault("_ctrl_stage_store", {})

    def _stages_in_order(
        self,
    ) -> list[tuple[str, tuple[float, CtrlStage, ResetStage | None]]]:
        # Stable: stages of equal order keep their installation order.
        return sorted(self._stage_store().items(), key=lambda item: item[1][0])

    def add_ctrl_stage(
        self,
        name: str,
        apply: CtrlStage,
        reset: ResetStage | None = None,
        order: float | None = None,
    ) -> None:
        """Install a stage; it runs at its priority, after the env's map.

        Args:
            name: A built-in stage (:data:`STAGE_ORDER`) or a new unique name.
            apply: ``apply(env, ctrl) -> ctrl``.
            reset: Optional per-episode reset of the stage's state.
            order: ``None`` (default) for a custom stage: after the built-in stages, in
                installation order. Or a priority strictly between :data:`MAP_ORDER`
                and :data:`WRITE_ORDER` to insert it earlier. Built-in stages have a
                fixed priority (omit it). The same explicit order as another installed
                stage triggers a :class:`StageOrderWarning`.

        Raises:
            ValueError: If *name* is already installed, an explicit *order* is out of
                range, or a built-in stage gets another order.
        """
        store = self._stage_store()
        if name in store:
            raise ValueError(
                f"The {name!r} stage is already installed on this env; wrap it once."
            )
        if name in STAGE_ORDER:
            if order is not None and order != STAGE_ORDER[name]:
                raise ValueError(
                    f"The built-in {name!r} stage has the fixed order {STAGE_ORDER[name]}."
                )
            order = STAGE_ORDER[name]
        else:
            check_custom_order(name, order)
        if order is None:
            order = LATE_ORDER
        else:
            warn_order_clash(
                name, order, {n: o for n, (o, _, _) in store.items()}, stacklevel=4
            )
        store[name] = (order, apply, reset)

    def remove_ctrl_stage(self, name: str) -> None:
        """Uninstall a stage (no-op if it is not installed)."""
        self._stage_store().pop(name, None)

    @property
    def ctrl_stages(self) -> tuple[str, ...]:
        """Names of the installed stages, in the order they run."""
        return tuple(name for name, _ in self._stages_in_order())

    @property
    def muscle_fatigue(self) -> Any:
        """Fatigue model of the installed ``FatigueWrapper``.

        Raises:
            AttributeError: If no fatigue stage is installed.
        """
        stage = self._stage_store().get("fatigue")
        if stage is None:
            raise AttributeError("muscle_fatigue: no FatigueWrapper on this env")
        return stage[1].fatigue

    def _stage_muscle_index(self) -> Any:
        """Entries of ``ctrl`` that are muscle excitations."""
        return self._muscle_act_ind

    def _stage_actuator_suffix(self) -> str:
        """Suffix of the EPL/EIP actuator names (``"_r"`` on recipe-built hand models)."""
        if hasattr(self, "_name_sfx"):
            return self._name_sfx
        return "_r" if hasattr(self, "_model_recipe") else ""

    def _run_ctrl_stages(self, ctrl: np.ndarray) -> np.ndarray:
        """Apply the installed stages to the excitation vector *ctrl*."""
        for _, (_, apply, _) in self._stages_in_order():
            ctrl = apply(self, ctrl)
        return ctrl

    def _run_reset_stages(self) -> None:
        """Reset the installed stages for a new episode."""
        for _, (_, _, reset) in self._stages_in_order():
            if reset is not None:
                reset(self)


def noise_stage(get_cfg: Callable[[], Any]) -> CtrlStage:
    """Signal-dependent + constant Gaussian noise on the muscle excitations.

    Args:
        get_cfg: Returns the current :class:`~myosuite.terms.base_action.MotorNoiseCfg`
            (read every step, so the wrapper's config can change). A disabled
            config draws nothing, so the env's random stream is untouched.
    """
    from myosuite.terms.base_action import sample_motor_noise  # noqa: PLC0415

    def apply(env: Any, ctrl: np.ndarray) -> np.ndarray:
        idx = env._stage_muscle_index()
        ctrl[idx] = sample_motor_noise(
            ctrl[idx], get_cfg(), env.np_random.standard_normal, np
        )
        return ctrl

    return apply


def fatigue_stage(
    fatigue: Any, get_reset: Callable[[], tuple[Any, bool]]
) -> tuple[CtrlStage, ResetStage]:
    """3CC-r fatigue on the muscle excitations and its per-episode reset.

    Args:
        fatigue: A :class:`~myosuite.physics.fatigue.CumulativeFatigue`.
        get_reset: Returns ``(fatigue_reset_vec, fatigue_reset_random)``.
    """

    def apply(env: Any, ctrl: np.ndarray) -> np.ndarray:
        idx = env._stage_muscle_index()
        ctrl[idx], _, _ = fatigue.compute_act(ctrl[idx])
        return ctrl

    def reset(env: Any) -> None:
        vec, random = get_reset()
        fatigue.reset(
            fatigue_reset_vec=vec, fatigue_reset_random=random, np_random=env.np_random
        )

    apply.fatigue = fatigue  # type: ignore[attr-defined]
    return apply, reset


def reroute_stage(model: Any, suffix: str) -> CtrlStage:
    """Reafferentation: EIP's command drives EPL and EIP is silenced."""
    epl = model.actuator(f"EPL{suffix}").id
    eip = model.actuator(f"EIP{suffix}").id

    def apply(env: Any, ctrl: np.ndarray) -> np.ndarray:
        ctrl[epl] = ctrl[eip].copy()
        ctrl[eip] = 0.0
        return ctrl

    return apply


def muscle_mask(model: mujoco.MjModel) -> np.ndarray:
    """Boolean mask of the muscle actuators of *model*."""
    return model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
