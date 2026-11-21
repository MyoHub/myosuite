# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Non-inertial base-excitation profiles and D'Alembert body-wrench mechanics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from myosuite.utils.colored_noise import ColoredNoiseProcess


def _cross3(a: Any, b: Any, xp: Any) -> Any:
    """Backend-agnostic cross product along the trailing 3-vector axis."""
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    return xp.stack(
        [ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx],
        axis=-1,
    )


def _as_vec3(value: Any, xp: Any, ref: Any) -> Any:
    """Convert *value* (or ``None`` -> zeros) to the dtype/device of *ref*."""
    if getattr(xp, "__name__", "") == "torch":
        if value is None:
            return xp.zeros(3, dtype=ref.dtype, device=ref.device)
        return xp.as_tensor(value, dtype=ref.dtype, device=ref.device)
    if value is None:
        return xp.zeros(3, dtype=ref.dtype)
    return xp.asarray(value, dtype=ref.dtype)


@dataclass(frozen=True)
class BaseKinematics:
    """Instantaneous 6-DOF kinematics of an accelerating/rotating support frame.

    All vectors are expressed in the support-frame (simulation world) axes:
    ``lin_acc`` [m/s^2], ``ang_vel`` [rad/s], and ``ang_acc`` [rad/s^2].
    """

    lin_acc: np.ndarray
    ang_vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    ang_acc: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def __post_init__(self) -> None:
        for name in ("lin_acc", "ang_vel", "ang_acc"):
            arr = np.asarray(getattr(self, name), dtype=np.float64)
            if arr.shape != (3,) or not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must be a finite 3-vector, got {arr.shape}")
            object.__setattr__(self, name, arr)


@runtime_checkable
class BaseExcitationProfile(Protocol):
    """Protocol for time-varying non-inertial base-excitation generators."""

    def reset(self, rng: np.random.Generator, dt: float) -> Any:
        """Initialize per-episode state from a seeded generator and control ``dt``."""
        ...

    def sample(self, t: float, state: Any) -> BaseKinematics:
        """Evaluate support-frame kinematics at episode time ``t`` [s]."""
        ...


@dataclass(frozen=True)
class HarmonicBaseProfile:
    """Multi-sine translational and rotational base excitation.

    Args:
        lin_amp: Linear acceleration amplitudes [m/s^2], shape ``(K, 3)`` or ``(3,)``.
        freq_hz: Frequencies [Hz], shape ``(K,)`` or scalar.
        phase_rad: Phase offsets [rad], shape ``(K, 3)`` or ``(3,)``.
        ang_amp: Angular velocity amplitudes [rad/s], shape ``(K, 3)`` or ``(3,)``.
        random_phase: Randomize phases uniformly in ``[0, 2*pi)`` on ``reset``.
    """

    lin_amp: tuple[float, ...] | np.ndarray = (0.0, 0.0, 0.0)
    freq_hz: float | tuple[float, ...] | np.ndarray = 1.0
    phase_rad: tuple[float, ...] | np.ndarray = (0.0, 0.0, 0.0)
    ang_amp: tuple[float, ...] | np.ndarray = (0.0, 0.0, 0.0)
    random_phase: bool = False

    def _normalized_params(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        freqs = np.atleast_1d(np.asarray(self.freq_hz, dtype=np.float64))
        if freqs.ndim != 1 or freqs.size == 0 or np.any(freqs < 0.0):
            raise ValueError("freq_hz must be non-empty and non-negative")
        k = freqs.size
        lin = np.broadcast_to(np.asarray(self.lin_amp, dtype=np.float64), (k, 3)).copy()
        ang = np.broadcast_to(np.asarray(self.ang_amp, dtype=np.float64), (k, 3)).copy()
        phase = np.broadcast_to(
            np.asarray(self.phase_rad, dtype=np.float64), (k, 3)
        ).copy()
        return freqs, lin, ang, phase

    def reset(self, rng: np.random.Generator, dt: float) -> dict[str, np.ndarray]:
        del dt
        freqs, lin, ang, phase = self._normalized_params()
        ang_phase = phase.copy()
        if self.random_phase:
            phase = rng.uniform(0.0, 2.0 * math.pi, size=phase.shape)
            ang_phase = rng.uniform(0.0, 2.0 * math.pi, size=phase.shape)
        return {
            "freqs": freqs,
            "lin": lin,
            "ang": ang,
            "phase": phase,
            "ang_phase": ang_phase,
        }

    def sample(self, t: float, state: dict[str, np.ndarray]) -> BaseKinematics:
        omega_f = 2.0 * math.pi * state["freqs"][:, None]
        arg_lin = omega_f * float(t) + state["phase"]
        arg_ang = omega_f * float(t) + state["ang_phase"]
        lin_acc = np.sum(state["lin"] * np.sin(arg_lin), axis=0)
        ang_vel = np.sum(state["ang"] * np.sin(arg_ang), axis=0)
        ang_acc = np.sum(state["ang"] * omega_f * np.cos(arg_ang), axis=0)
        return BaseKinematics(lin_acc=lin_acc, ang_vel=ang_vel, ang_acc=ang_acc)


@dataclass(frozen=True)
class StochasticBaseProfile:
    """Power-law colored-noise base excitation drawn via ``ColoredNoiseProcess``.

    Args:
        lin_std: Per-axis RMS linear acceleration [m/s^2], shape ``(3,)``.
        ang_std: Per-axis RMS angular velocity [rad/s], shape ``(3,)``.
        beta: Power-law PSD exponent (0: white, 1: pink, 2: Brownian).
        buffer_steps: Buffered steps per colored-noise draw.
    """

    lin_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    ang_std: tuple[float, float, float] = (0.0, 0.0, 0.0)
    beta: float = 1.0
    buffer_steps: int = 512

    def reset(self, rng: np.random.Generator, dt: float) -> dict[str, Any]:
        if dt <= 0.0 or not math.isfinite(dt):
            raise ValueError(f"dt must be positive and finite, got {dt}")
        lin_scale = np.asarray(self.lin_std, dtype=np.float64)
        ang_scale = np.asarray(self.ang_std, dtype=np.float64)
        if lin_scale.shape != (3,) or ang_scale.shape != (3,):
            raise ValueError("lin_std and ang_std must have shape (3,)")
        n_buf = max(8, int(self.buffer_steps))
        proc = ColoredNoiseProcess(beta=float(self.beta), size=(6, n_buf), rng=rng)
        return {
            "proc": proc,
            "dt": float(dt),
            "lin_scale": lin_scale,
            "ang_scale": ang_scale,
            "prev_ang_vel": np.zeros(3, dtype=np.float64),
        }

    def sample(self, t: float, state: dict[str, Any]) -> BaseKinematics:
        del t
        raw = np.asarray(state["proc"].sample(1), dtype=np.float64)
        lin_acc = state["lin_scale"] * raw[:3]
        ang_vel = state["ang_scale"] * raw[3:]
        ang_acc = (ang_vel - state["prev_ang_vel"]) / state["dt"]
        state["prev_ang_vel"] = ang_vel
        return BaseKinematics(lin_acc=lin_acc, ang_vel=ang_vel, ang_acc=ang_acc)


@dataclass(frozen=True)
class PulseBaseProfile:
    """Smooth half-sine transient base acceleration/rotation pulse.

    Args:
        start_s: Pulse onset time [s].
        duration_s: Positive pulse width [s].
        peak_lin_acc: Peak linear acceleration vector [m/s^2].
        peak_ang_vel: Peak angular velocity vector [rad/s].
    """

    start_s: float = 0.5
    duration_s: float = 0.25
    peak_lin_acc: tuple[float, float, float] = (2.0, 0.0, 0.0)
    peak_ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.duration_s <= 0.0 or not math.isfinite(self.duration_s):
            raise ValueError(f"duration_s must be positive, got {self.duration_s}")

    def reset(self, rng: np.random.Generator, dt: float) -> None:
        del rng, dt
        return None

    def sample(self, t: float, state: None) -> BaseKinematics:
        del state
        tau = (float(t) - self.start_s) / self.duration_s
        if tau < 0.0 or tau > 1.0:
            return BaseKinematics(lin_acc=np.zeros(3, dtype=np.float64))
        env = math.sin(math.pi * tau)
        denv_dt = (math.pi / self.duration_s) * math.cos(math.pi * tau)
        lin = np.asarray(self.peak_lin_acc, dtype=np.float64) * env
        ang = np.asarray(self.peak_ang_vel, dtype=np.float64) * env
        ang_acc = np.asarray(self.peak_ang_vel, dtype=np.float64) * denv_dt
        return BaseKinematics(lin_acc=lin, ang_vel=ang, ang_acc=ang_acc)


@dataclass(frozen=True)
class RecordedBaseProfile:
    """Interpolated empirical 6-DOF base acceleration / IMU time series.

    Args:
        times: Strictly increasing timestamps [s], shape ``(T,)``.
        lin_acc: Linear accelerations [m/s^2], shape ``(T, 3)``.
        ang_vel: Optional angular velocities [rad/s], shape ``(T, 3)``.
        ang_acc: Optional angular accelerations [rad/s^2], shape ``(T, 3)``;
            derived from ``ang_vel`` via ``np.gradient`` when omitted.
        loop: Wrap ``t`` modulo ``times[-1] - times[0]`` when ``True``.
    """

    times: np.ndarray
    lin_acc: np.ndarray
    ang_vel: np.ndarray | None = None
    ang_acc: np.ndarray | None = None
    loop: bool = False

    def __post_init__(self) -> None:
        t = np.asarray(self.times, dtype=np.float64)
        lin = np.asarray(self.lin_acc, dtype=np.float64)
        if t.ndim != 1 or t.size < 2 or np.any(np.diff(t) <= 0.0):
            raise ValueError("times must be 1D, length >= 2, and strictly increasing")
        if lin.shape != (t.size, 3):
            raise ValueError(f"lin_acc must have shape ({t.size}, 3), got {lin.shape}")
        ang = (
            np.zeros_like(lin)
            if self.ang_vel is None
            else np.asarray(self.ang_vel, dtype=np.float64)
        )
        if ang.shape != (t.size, 3):
            raise ValueError(f"ang_vel must have shape ({t.size}, 3), got {ang.shape}")
        alpha = (
            np.gradient(ang, t, axis=0)
            if self.ang_acc is None
            else np.asarray(self.ang_acc, dtype=np.float64)
        )
        if alpha.shape != (t.size, 3):
            raise ValueError(
                f"ang_acc must have shape ({t.size}, 3), got {alpha.shape}"
            )
        object.__setattr__(self, "times", t)
        object.__setattr__(self, "lin_acc", lin)
        object.__setattr__(self, "ang_vel", ang)
        object.__setattr__(self, "ang_acc", alpha)

    def reset(self, rng: np.random.Generator, dt: float) -> None:
        del rng, dt
        return None

    def sample(self, t: float, state: None) -> BaseKinematics:
        del state
        t_arr = self.times
        query = float(t)
        if self.loop:
            span = float(t_arr[-1] - t_arr[0])
            query = float(t_arr[0]) + math.fmod(max(0.0, query - float(t_arr[0])), span)
        lin = np.array([np.interp(query, t_arr, self.lin_acc[:, i]) for i in range(3)])
        ang = np.array([np.interp(query, t_arr, self.ang_vel[:, i]) for i in range(3)])
        alpha = np.array(
            [np.interp(query, t_arr, self.ang_acc[:, i]) for i in range(3)]
        )
        return BaseKinematics(lin_acc=lin, ang_vel=ang, ang_acc=alpha)


@dataclass(frozen=True)
class CompositeBaseProfile:
    """Additive superposition of multiple :class:`BaseExcitationProfile` components."""

    profiles: tuple[BaseExcitationProfile, ...]

    def reset(self, rng: np.random.Generator, dt: float) -> list[Any]:
        return [p.reset(rng, dt) for p in self.profiles]

    def sample(self, t: float, state: list[Any]) -> BaseKinematics:
        lin = np.zeros(3, dtype=np.float64)
        ang = np.zeros(3, dtype=np.float64)
        alpha = np.zeros(3, dtype=np.float64)
        for prof, sub_state in zip(self.profiles, state):
            kin = prof.sample(t, sub_state)
            lin += kin.lin_acc
            ang += kin.ang_vel
            alpha += kin.ang_acc
        return BaseKinematics(lin_acc=lin, ang_vel=ang, ang_acc=alpha)


def inertial_body_wrenches(
    xp: Any,
    body_mass: Any,
    body_inertia: Any,
    xipos: Any,
    ximat: Any,
    cvel: Any,
    root_com: Any,
    lin_acc: Any,
    *,
    ang_vel: Any | None = None,
    ang_acc: Any | None = None,
    pivot: Any | None = None,
    dynamic_mask: Any | None = None,
) -> Any:
    """Compute D'Alembert non-inertial COM wrenches ``(..., nbody, 6)`` in NumPy or Torch.

    Applies translational acceleration, Euler tangential acceleration, centrifugal,
    Coriolis, and rotational Euler/gyroscopic torques to every dynamic body segment
    in an accelerating/rotating support frame.

    Args:
        xp: Array module (``numpy`` or ``torch``).
        body_mass: Body masses ``(nbody,)`` or ``(..., nbody)``.
        body_inertia: Principal inertias ``(nbody, 3)`` or ``(..., nbody, 3)``.
        xipos: Body COM world positions ``(..., nbody, 3)``.
        ximat: Body inertial-frame world rotations ``(..., nbody, 9)`` or ``(..., nbody, 3, 3)``.
        cvel: MuJoCo body spatial velocities ``[ang(3), lin_subtree(3)]``, shape ``(..., nbody, 6)``.
        root_com: Subtree COM of each body's kinematic root, shape ``(..., nbody, 3)``.
        lin_acc: Support-frame linear acceleration ``(..., 3)`` [m/s^2].
        ang_vel: Support-frame angular velocity ``(..., 3)`` [rad/s].
        ang_acc: Support-frame angular acceleration ``(..., 3)`` [rad/s^2].
        pivot: Rotation pivot in world coordinates ``(..., 3)`` [m].
        dynamic_mask: Boolean mask ``(nbody,)`` selecting articulated/free bodies.

    Returns:
        World-frame COM wrenches ``[force(3), torque(3)]`` of shape ``(..., nbody, 6)``.
    """
    a0 = _as_vec3(lin_acc, xp, xipos)[..., None, :]
    omega = _as_vec3(ang_vel, xp, xipos)[..., None, :]
    alpha = _as_vec3(ang_acc, xp, xipos)[..., None, :]
    p0 = _as_vec3(pivot, xp, xipos)[..., None, :]

    r_i = xipos - p0
    omega_rel = cvel[..., :3]
    v_rel = cvel[..., 3:] + _cross3(omega_rel, xipos - root_com, xp)

    fict_acc = -(
        a0
        + _cross3(alpha, r_i, xp)
        + _cross3(omega, _cross3(omega, r_i, xp), xp)
        + 2.0 * _cross3(omega, v_rel, xp)
    )
    force = body_mass[..., None] * fict_acc

    rot = ximat.reshape(*xipos.shape[:-1], 3, 3)

    def _apply_world_inertia(vec: Any) -> Any:
        # R @ diag(I) @ R^T @ vec without forming full 3x3 inertia tensors.
        local = xp.sum(rot * vec[..., :, None], axis=-2)
        scaled = body_inertia * local
        return xp.sum(rot * scaled[..., None, :], axis=-1)

    i_omega = _apply_world_inertia(omega)
    i_omega_rel = _apply_world_inertia(omega_rel)
    i_alpha_eff = _apply_world_inertia(alpha + _cross3(omega, omega_rel, xp))
    torque = (
        -i_alpha_eff
        - _cross3(omega, i_omega + i_omega_rel, xp)
        - _cross3(omega_rel, i_omega, xp)
    )

    wrench = xp.concatenate([force, torque], axis=-1)
    if dynamic_mask is not None:
        mask = (
            xp.as_tensor(dynamic_mask, dtype=xp.bool, device=xipos.device)
            if getattr(xp, "__name__", "") == "torch"
            else xp.asarray(dynamic_mask, dtype=bool)
        )
        wrench = wrench * mask[..., None]
    return wrench


__all__ = [
    "BaseExcitationProfile",
    "BaseKinematics",
    "CompositeBaseProfile",
    "HarmonicBaseProfile",
    "PulseBaseProfile",
    "RecordedBaseProfile",
    "StochasticBaseProfile",
    "inertial_body_wrenches",
]
