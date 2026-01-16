import numpy as np
from typing import Sequence, Tuple, Optional

_TIME_PRECISION = 4

def make_sinusoid_reference(
    ranges: Sequence[Tuple[float, float]],
    freqs: Sequence[float],
    fs: float,
    duration: float,
    phases: Optional[Sequence[float]] = None,
    include_velocity: bool = True,
):
    """
    ranges: [(min_i, max_i),...] per DoF
    freqs:  [f_i,...] in Hz, same length as ranges
    fs:     sampling frequency (Hz)
    duration: total duration (seconds)
    phases: optional phase per DoF (radians), default = 0
    include_velocity: if True, also return analytic dq/dt
    """
    ranges = np.asarray(ranges, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    d = ranges.shape[0]  # num dofs
    phases = np.zeros(d) if phases is None else np.asarray(phases, dtype=float)
    assert phases.shape[0] == freqs.shape[0] == d, "phases and freqs must match number of DoFs in ranges"

    T = int(np.ceil(duration * fs))
    t = np.arange(0, duration, 1/fs)
    t = np.around(t, _TIME_PRECISION)
    mins, maxs = ranges[:, 0], ranges[:, 1]
    amplitudes = 0.5 * (maxs - mins)
    offsets = 0.5 * (maxs + mins)
    omega = 2.0 * np.pi * freqs

    robot = offsets + amplitudes * np.sin(omega * t[:, None] + phases[None, :])

    # Build reference dictionary for Track mode
    ref = {
        "time": t,
        "robot": robot,
        "robot_init": robot[0]
    }

    if include_velocity:
      robot_vel = (omega * amplitudes) * np.cos(omega * t[:, None] + phases[None, :])
      ref["robot_vel"] = robot_vel

    return ref


if __name__ == "__main__":
    ref = make_sinusoid_reference(
        ranges=[(0.05, np.pi/2)],
        freqs=[0.25],
        fs=50,
        duration=5.0,
    )
    np.savez("sinusoid_ref.npz", **ref)
