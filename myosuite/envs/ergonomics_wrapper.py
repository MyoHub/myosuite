# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Gymnasium wrapper recording per-step musculoskeletal ergonomics telemetry."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium.utils import RecordConstructorArgs

from myosuite.biomechanics.effort import (
    muscle_saturation,
    muscle_saturation_bounds,
    squared_activation,
)
from myosuite.envs.wrappers import _ForwardPublicAttributes
from myosuite.physics.fatigue import CumulativeFatigue


@dataclass(frozen=True)
class ErgonomicsConfig:
    """Configuration for :class:`ErgonomicsWrapper`.

    Attributes:
        effort_ema_alpha: Exponential moving-average factor for ``smoothed_effort`` in ``(0, 1]``.
        rom_margin_fraction: Fractional joint-range border zone where ``joint_limit_strain`` rises.
        passive_torque_scale: Reference passive joint torque norm [N*m] for strain scaling.
        saturation_margin: Excitation distance from ``0`` or ``1`` counted as saturated.
        reward_penalty_weight: Non-negative scalar subtracting
            ``reward_penalty_weight * 0.5 * (smoothed_effort + joint_limit_strain)``
            from scalar step rewards (``0.0`` keeps rewards unchanged).
    """

    effort_ema_alpha: float = 0.25
    rom_margin_fraction: float = 0.15
    passive_torque_scale: float = 18.0
    saturation_margin: float = 0.05
    reward_penalty_weight: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 < self.effort_ema_alpha <= 1.0):
            raise ValueError(
                f"effort_ema_alpha must be in (0, 1], got {self.effort_ema_alpha}"
            )
        if not (0.0 < self.rom_margin_fraction <= 0.5):
            raise ValueError(
                f"rom_margin_fraction must be in (0, 0.5], got {self.rom_margin_fraction}"
            )
        if not (
            np.isfinite(self.passive_torque_scale) and self.passive_torque_scale > 0.0
        ):
            raise ValueError(
                f"passive_torque_scale must be positive and finite, got {self.passive_torque_scale}"
            )
        if not (0.0 < self.saturation_margin < 0.5):
            raise ValueError(
                f"saturation_margin must be in (0, 0.5), got {self.saturation_margin}"
            )
        if not (
            np.isfinite(self.reward_penalty_weight)
            and self.reward_penalty_weight >= 0.0
        ):
            raise ValueError(
                f"reward_penalty_weight must be non-negative and finite, got {self.reward_penalty_weight}"
            )


class ErgonomicsWrapper(_ForwardPublicAttributes, RecordConstructorArgs, gym.Wrapper):
    """Gymnasium wrapper populating ``info["ergonomics"]`` on ``reset()`` and ``step()``.

    Tracks six normalized musculoskeletal quantities in ``[0, 1]``:
    - ``instantaneous_effort``: mean squared muscle activation ``mean_m a_m^2``.
    - ``smoothed_effort``: exponential moving average of ``instantaneous_effort``.
    - ``cumulative_fatigue``: mean 3CC-r fatigued compartment fraction ``MF`` across muscles.
    - ``peak_muscle_fatigue``: maximum 3CC-r ``MF`` across muscles.
    - ``muscle_saturation``: fraction of muscle commands saturated per
      :func:`~myosuite.biomechanics.effort.muscle_saturation`.
    - ``joint_limit_strain``: combined 1-DOF joint-limit proximity and passive torque load.

    When the wrapped environment already runs a :class:`~myosuite.envs.wrappers.FatigueWrapper`,
    its ``muscle_fatigue`` state is read directly without double-stepping; otherwise a
    read-only :class:`~myosuite.physics.fatigue.CumulativeFatigue` observer tracks 3CC-r state.
    """

    def __init__(
        self,
        env: gym.Env,
        config: ErgonomicsConfig | Mapping[str, Any] | None = None,
        *,
        reward_penalty_weight: float | None = None,
    ) -> None:
        RecordConstructorArgs.__init__(
            self, config=config, reward_penalty_weight=reward_penalty_weight
        )
        gym.Wrapper.__init__(self, env)

        if config is None:
            cfg = ErgonomicsConfig()
        elif isinstance(config, ErgonomicsConfig):
            cfg = config
        elif isinstance(config, Mapping):
            valid = {f.name for f in dataclasses.fields(ErgonomicsConfig)}
            unknown = set(config) - valid
            if unknown:
                raise ValueError(f"Unknown ErgonomicsConfig keys: {sorted(unknown)}")
            cfg = ErgonomicsConfig(**dict(config))
        else:
            raise TypeError(
                f"Expected ErgonomicsConfig, Mapping, or None; got {type(config).__name__}"
            )
        if reward_penalty_weight is not None:
            cfg = dataclasses.replace(
                cfg, reward_penalty_weight=float(reward_penalty_weight)
            )
        self.config = cfg

        base = self.unwrapped
        model = base.model
        frame_skip = int(getattr(base, "frame_skip", 10))

        self._fatigue_observer: CumulativeFatigue | None = None
        if self._host_fatigue() is None and model.nu > 0:
            self._fatigue_observer = CumulativeFatigue(model, frame_skip, seed=None)

        idx = getattr(base, "_muscle_act_ind", None)
        if idx is None:
            idx = np.flatnonzero(
                model.actuator_gaintype == mujoco.mjtGain.mjGAIN_MUSCLE
            )
        self._muscle_idx = np.asarray(idx, dtype=np.intp)

        clip: tuple[float, float] | None = None
        if (
            isinstance(self.action_space, gym.spaces.Box)
            and self.action_space.shape == (model.nu,)
            and self._muscle_idx.size > 0
        ):
            lo_c = float(
                np.max(
                    np.asarray(self.action_space.low, dtype=np.float64)[
                        self._muscle_idx
                    ]
                )
            )
            hi_c = float(
                np.min(
                    np.asarray(self.action_space.high, dtype=np.float64)[
                        self._muscle_idx
                    ]
                )
            )
            if np.isfinite(lo_c) and np.isfinite(hi_c) and lo_c < hi_c:
                clip = (lo_c, hi_c)
        self._saturation_bounds = muscle_saturation_bounds(
            sigmoid=bool(getattr(base, "normalize_act", False)),
            clip=clip,
            margin=self.config.saturation_margin,
        )

        jnt_limited = np.asarray(model.jnt_limited, dtype=bool)
        qpos_adrs: list[int] = []
        lows: list[float] = []
        highs: list[float] = []
        for jid in range(model.njnt):
            if not jnt_limited[jid]:
                continue
            jtype = int(model.jnt_type[jid])
            if jtype not in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            ):
                continue
            lo, hi = float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1])
            if hi - lo > 1e-6:
                qpos_adrs.append(int(model.jnt_qposadr[jid]))
                lows.append(lo)
                highs.append(hi)
        self._limited_qpos_adrs = np.asarray(qpos_adrs, dtype=np.intp)
        self._jnt_lows = np.asarray(lows, dtype=np.float64)
        self._jnt_highs = np.asarray(highs, dtype=np.float64)

        self._smoothed_effort = 0.0
        self._last_ergonomics: dict[str, float] = {}

    def _host_fatigue(self) -> Any | None:
        try:
            return self.env.get_wrapper_attr("muscle_fatigue")
        except AttributeError:
            return None

    def _muscle_excitations_and_activations(self) -> tuple[np.ndarray, np.ndarray]:
        base = self.unwrapped
        model, data = base.model, base.data
        ctrl = (
            np.clip(np.asarray(data.ctrl[self._muscle_idx], dtype=np.float64), 0.0, 1.0)
            if model.nu > 0 and self._muscle_idx.size > 0
            else np.empty(0, dtype=np.float64)
        )
        if model.na > 0:
            act = np.clip(np.asarray(data.act, dtype=np.float64), 0.0, 1.0)
        else:
            act = ctrl
        return ctrl, act

    def _compute_effort_and_saturation(
        self, action: np.ndarray | None = None
    ) -> tuple[float, float]:
        ctrl, act = self._muscle_excitations_and_activations()
        effort = (
            float(
                np.clip(
                    squared_activation(act[None, :]) / float(max(act.size, 1)),
                    0.0,
                    1.0,
                )
            )
            if act.size > 0
            else 0.0
        )
        if (
            action is not None
            and self._muscle_idx.size > 0
            and action.ndim == 1
            and action.size == self.unwrapped.model.nu
        ):
            m_act = np.asarray(action[self._muscle_idx], dtype=np.float64)
            sat = float(
                muscle_saturation(m_act[None, :], bounds=self._saturation_bounds)
            )
        elif ctrl.size > 0:
            sat = float(
                muscle_saturation(ctrl[None, :], margin=self.config.saturation_margin)
            )
        else:
            sat = 0.0
        return effort, sat

    def _step_and_read_fatigue(self, *, step_observer: bool) -> tuple[float, float]:
        host_fat = self._host_fatigue()
        if host_fat is not None:
            mf = np.asarray(host_fat.MF, dtype=np.float64)
        elif self._fatigue_observer is not None:
            if step_observer:
                ctrl, act = self._muscle_excitations_and_activations()
                drive = act if act.size == self._fatigue_observer.MF.size else ctrl
                if drive.size == self._fatigue_observer.MF.size and drive.size > 0:
                    self._fatigue_observer.compute_act(drive)
            mf = np.asarray(self._fatigue_observer.MF, dtype=np.float64)
        else:
            return 0.0, 0.0

        if mf.size == 0:
            return 0.0, 0.0
        return (
            float(np.clip(np.mean(mf), 0.0, 1.0)),
            float(np.clip(np.max(mf), 0.0, 1.0)),
        )

    def _compute_joint_limit_strain(self) -> float:
        base = self.unwrapped
        data = base.data

        rom_strain = 0.0
        if self._limited_qpos_adrs.size > 0:
            q = data.qpos[self._limited_qpos_adrs]
            spans = self._jnt_highs - self._jnt_lows
            margin = np.minimum(q - self._jnt_lows, self._jnt_highs - q) / (0.5 * spans)
            zone = self.config.rom_margin_fraction
            violation = np.clip((zone - margin) / zone, 0.0, 1.5)
            sq = np.square(violation)
            top_k = min(3, sq.size)
            top_mean = float(np.mean(np.partition(sq, -top_k)[-top_k:]))
            rom_strain = float(
                np.clip(0.65 * top_mean + 0.35 * float(np.mean(sq)), 0.0, 1.0)
            )

        passive_norm = float(np.linalg.norm(data.qfrc_passive))
        torque_strain = float(np.tanh(passive_norm / self.config.passive_torque_scale))
        return float(np.clip(0.60 * rom_strain + 0.40 * torque_strain, 0.0, 1.0))

    def _update_ergonomics(
        self,
        info: dict[str, Any],
        *,
        is_reset: bool,
        action: np.ndarray | None = None,
    ) -> dict[str, float]:
        inst_effort, sat = self._compute_effort_and_saturation(action=action)
        mean_fat, peak_fat = self._step_and_read_fatigue(step_observer=not is_reset)
        strain = self._compute_joint_limit_strain()

        if is_reset:
            self._smoothed_effort = inst_effort
        else:
            alpha = self.config.effort_ema_alpha
            self._smoothed_effort = float(
                np.clip(
                    (1.0 - alpha) * self._smoothed_effort + alpha * inst_effort,
                    0.0,
                    1.0,
                )
            )

        ergo = {
            "instantaneous_effort": inst_effort,
            "smoothed_effort": self._smoothed_effort,
            "cumulative_fatigue": mean_fat,
            "peak_muscle_fatigue": peak_fat,
            "muscle_saturation": sat,
            "joint_limit_strain": strain,
        }
        self._last_ergonomics = ergo
        info["ergonomics"] = dict(ergo)
        return ergo

    @property
    def ergonomics_state(self) -> dict[str, float]:
        """Most recent ergonomics telemetry dictionary."""
        return dict(self._last_ergonomics)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        if self._fatigue_observer is not None:
            self._fatigue_observer.reset(
                fatigue_reset_vec=None,
                fatigue_reset_random=False,
                np_random=self.unwrapped.np_random,
            )
        self._smoothed_effort = 0.0
        info = dict(info)
        self._update_ergonomics(info, is_reset=True)
        return obs, info

    def step(self, action: Any) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        act_arr = (
            np.asarray(action)
            if isinstance(self.action_space, gym.spaces.Box)
            else None
        )
        ergo = self._update_ergonomics(info, is_reset=False, action=act_arr)
        if self.config.reward_penalty_weight > 0.0 and not isinstance(reward, Mapping):
            penalty = (
                0.5
                * self.config.reward_penalty_weight
                * (ergo["smoothed_effort"] + ergo["joint_limit_strain"])
            )
            reward = float(reward) - penalty
        return obs, reward, bool(terminated), bool(truncated), info


__all__ = ["ErgonomicsConfig", "ErgonomicsWrapper"]
