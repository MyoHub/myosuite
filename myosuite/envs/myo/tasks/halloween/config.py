"""Shared CPU/MJLab contract for the experimental muscle-action task."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HalloweenConfig:
    """Timing and projectile distribution; curriculum is opt-in on both backends."""

    sim_dt: float = 0.002
    ctrl_dt: float = 0.01
    episode_s: float = 6.0
    throw_interval_s: tuple[float, float] = (1.5, 3.0)
    pumpkin_radius: float = 0.09
    pumpkin_mass: float = 0.35
    park_pos: tuple[float, float, float] = (6.5, 6.5, 0.15)
    fall_pelvis_z: float = 0.6
    fall_head_z: float = 0.9
    easy_start: float = 1.6
    ramp_steps: int = 48000


HALLOWEEN = HalloweenConfig()


@dataclass(frozen=True)
class PumpkinThrow:
    """A frontal ballistic throw, in metres and seconds relative to home."""

    target_x: float
    target_z: float
    flight_s: float
    speed: float


@dataclass(frozen=True)
class AvoidanceConfig:
    """Reproducible TERRA showcase curriculum; no controller or policy weights."""

    throws: tuple[PumpkinThrow, ...] = (
        PumpkinThrow(-0.11537105180914775, 1.4608824024981797, 1.2563822360825192, 8.0),
        PumpkinThrow(0.13093328164781803, 1.4949661043026456, 1.2074422533634466, 8.0),
        PumpkinThrow(0.02295215779072947, 1.7203338380650184, 1.0910301086419278, 12.0),
    )
    first_wait_steps: tuple[int, int] = (40, 81)
    between_wait_steps: int = 50
    recovery_steps: int = 750
