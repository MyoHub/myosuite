# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Sensorimotor delay and sensory noise, shared by the CPU and mjlab backends.

A :class:`SensorimotorCfg` passed as the ``sensorimotor`` kwarg of a CPU
registration (or ``gym.make``) configures both twins of an env id; the mjlab
twin reads it through ``cpu_reference``. Every count is in control steps
(``ctrl_dt = frame_skip * timestep``). Semantics (cross-backend contract):

* observation delay ``k``: the policy observation at step ``t`` is the one
  computed at step ``max(0, t - k)``; a reset fills the history with the reset
  observation;
* action delay ``k``: the raw policy action of step ``t`` is applied at step
  ``t + k``, before the action -> excitation mapping (clip, sigmoid, fatigue,
  reafferentation); the first ``k`` steps after a reset apply the raw action 0;
* observation noise ``sigma``: i.i.d. ``N(0, sigma^2)`` added to every element
  of the (delayed) policy observation.
"""

from __future__ import annotations

import math
import numbers
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SensorimotorCfg:
    """Sensorimotor delay and observation noise of a task (default: off).

    Attributes:
        obs_delay_steps: Control steps between computing an observation and the
            policy receiving it.
        action_delay_steps: Control steps between the policy issuing an action
            and the env applying it.
        obs_noise_std: Standard deviation of the additive Gaussian noise on the
            policy observation (observation units).
    """

    obs_delay_steps: int = 0
    action_delay_steps: int = 0
    obs_noise_std: float = 0.0

    def __post_init__(self) -> None:
        for name in ("obs_delay_steps", "action_delay_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise TypeError(f"{name} must be an int, got {value!r}")
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
            object.__setattr__(self, name, int(value))
        std = float(self.obs_noise_std)
        if not math.isfinite(std) or std < 0.0:
            raise ValueError(f"obs_noise_std must be finite and >= 0, got {std}")
        object.__setattr__(self, "obs_noise_std", std)

    @property
    def enabled(self) -> bool:
        """Whether any delay or noise is configured."""
        return bool(
            self.obs_delay_steps or self.action_delay_steps or self.obs_noise_std
        )

    @classmethod
    def coerce(cls, value: SensorimotorCfg | None) -> SensorimotorCfg:
        """The config of a ``sensorimotor`` kwarg (``None`` means off).

        Args:
            value: Registration / constructor kwarg value.

        Returns:
            The config.

        Raises:
            TypeError: If *value* is neither ``None`` nor a ``SensorimotorCfg``.
        """
        if value is None:
            return cls()
        if not isinstance(value, cls):
            raise TypeError(f"sensorimotor must be a SensorimotorCfg, got {value!r}")
        return value


def _copy(frame: Any) -> Any:
    """Copy of a numpy array or torch tensor."""
    return frame.clone() if hasattr(frame, "clone") else np.array(frame, copy=True)


class FixedLagBuffer:
    """Frames delayed by a fixed number of pushes (numpy or torch, no RNG).

    ``push`` returns the frame pushed ``lag`` pushes earlier; until ``lag``
    frames have been pushed since the last ``refill`` it returns the fill.
    Frames keep their own dtype, so the delayed stream is the input stream,
    shifted.

    Args:
        lag: Number of pushes between storing and returning a frame.
        fill: Frame returned by the first ``lag`` pushes.
    """

    def __init__(self, lag: int, fill: Any) -> None:
        self.lag = int(lag)
        self._frames: deque[Any] = deque(_copy(fill) for _ in range(self.lag))

    def push(self, frame: Any) -> Any:
        """Store *frame* and return the frame pushed ``lag`` pushes earlier.

        Args:
            frame: The newest frame (copied).

        Returns:
            The delayed frame (owned by the caller).
        """
        self._frames.append(_copy(frame))
        return self._frames.popleft()

    def refill(self, fill: Any, rows: Any = None) -> None:
        """Make the next ``lag`` pushes return *fill*.

        Args:
            fill: Fill frame.
            rows: Rows (indices, slice or mask over the leading axis) to refill;
                ``None`` refills whole frames.
        """
        if rows is None:
            self._frames = deque(_copy(fill) for _ in range(self.lag))
            return
        for frame in self._frames:
            frame[rows] = fill[rows]


class CpuSensorimotor:
    """Delay and noise state of one CPU env (driven by ``MyoGymnasiumEnv``).

    Args:
        cfg: The env's sensorimotor config.
    """

    def __init__(self, cfg: SensorimotorCfg) -> None:
        self.cfg = cfg
        self._obs: FixedLagBuffer | None = None
        self._actions: FixedLagBuffer | None = None

    def reset(
        self, obs: np.ndarray, action_fill: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Start an episode: fill the histories, return the policy observation.

        Args:
            obs: Reset observation vector.
            action_fill: Raw action applied during the first delayed steps.
            rng: Env RNG for the observation noise.

        Returns:
            The (noisy) reset observation.
        """
        if not isinstance(obs, np.ndarray) or obs.ndim != 1:
            raise TypeError("sensorimotor supports flat ndarray observations only")
        self._obs = FixedLagBuffer(self.cfg.obs_delay_steps, obs)
        self._actions = FixedLagBuffer(self.cfg.action_delay_steps, action_fill)
        return self._add_noise(_copy(obs), rng)

    def action(self, action: Any) -> Any:
        """The raw action to apply this step (issued ``action_delay_steps`` ago)."""
        if self._actions is None:
            raise RuntimeError("reset() must be called before step()")
        return self._actions.push(np.asarray(action))

    def observe(self, obs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """The policy observation for the post-step observation *obs*."""
        if self._obs is None:
            raise RuntimeError("reset() must be called before step()")
        return self._add_noise(self._obs.push(obs), rng)

    def _add_noise(self, obs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        std = self.cfg.obs_noise_std
        if std == 0.0:  # no draw: the env RNG stream is untouched
            return obs
        return (obs + rng.normal(0.0, std, size=obs.shape)).astype(obs.dtype)
