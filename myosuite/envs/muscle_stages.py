# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Ordered muscle-command stages that wrappers install in a CPU env.

An env maps its action to a muscle excitation (sigmoid, or as-is for the walk
envs) and then runs the installed stages in the fixed order of
:data:`CTRL_STAGE_ORDER` before writing ``ctrl``:

``map (env) -> noise -> fatigue -> reroute (reafferentation) -> ctrl``

The order is a property of the stage, not of the wrapper nesting, so a stack
built in any order behaves the same. The wrappers in
:mod:`myosuite.envs.wrappers` (``MotorNoiseWrapper``, ``FatigueWrapper``,
``ReafferentationWrapper``) install one stage each through
:meth:`CtrlStageHost.add_ctrl_stage`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import mujoco
import numpy as np

CTRL_STAGE_ORDER: tuple[str, ...] = ("noise", "fatigue", "reroute")
"""Stages applied after the env's action-to-excitation map, in this order."""

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

    def _stage_store(self) -> dict[str, tuple[CtrlStage, ResetStage | None]]:
        return self.__dict__.setdefault("_ctrl_stage_store", {})

    def add_ctrl_stage(
        self, name: str, apply: CtrlStage, reset: ResetStage | None = None
    ) -> None:
        """Install a stage; it runs at its position in :data:`CTRL_STAGE_ORDER`.

        Args:
            name: One of :data:`CTRL_STAGE_ORDER`.
            apply: ``apply(env, ctrl) -> ctrl``.
            reset: Optional per-episode reset of the stage's state.

        Raises:
            ValueError: If *name* is unknown or already installed.
        """
        if name not in CTRL_STAGE_ORDER:
            raise ValueError(
                f"Unknown stage {name!r}; expected one of {CTRL_STAGE_ORDER}."
            )
        store = self._stage_store()
        if name in store:
            raise ValueError(
                f"The {name!r} stage is already installed on this env; wrap it once."
            )
        store[name] = (apply, reset)

    def remove_ctrl_stage(self, name: str) -> None:
        """Uninstall a stage (no-op if it is not installed)."""
        self._stage_store().pop(name, None)

    @property
    def ctrl_stages(self) -> tuple[str, ...]:
        """Names of the installed stages, in the order they run."""
        store = self._stage_store()
        return tuple(name for name in CTRL_STAGE_ORDER if name in store)

    @property
    def muscle_fatigue(self) -> Any:
        """Fatigue model of the installed ``FatigueWrapper``.

        Raises:
            AttributeError: If no fatigue stage is installed.
        """
        stage = self._stage_store().get("fatigue")
        if stage is None:
            raise AttributeError("muscle_fatigue: no FatigueWrapper on this env")
        return stage[0].fatigue

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
        store = self._stage_store()
        for name in CTRL_STAGE_ORDER:
            if name in store:
                ctrl = store[name][0](self, ctrl)
        return ctrl

    def _run_reset_stages(self) -> None:
        """Reset the installed stages for a new episode."""
        store = self._stage_store()
        for name in CTRL_STAGE_ORDER:
            if name in store and store[name][1] is not None:
                store[name][1](self)


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
