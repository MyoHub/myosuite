# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""6-axis IMU trace loader for non-inertial support-base excitation profiles.

Explicit data contracts:
- **Sensor & support axes:** ``acc_cols`` and ``gyro_cols`` are expressed in the moving
  support-base local right-handed frame ``(x, y, z)``, assuming the IMU axes are rigidly
  aligned with the support frame.
- **Units:** Accelerometer channels after multiplying by ``acc_scale`` are in ``m/s^2``;
  gyroscope channels are in ``rad/s`` (or ``deg/s`` when ``gyro_in_degrees=True``).
- **Quaternion convention:** Optional ``quat_cols`` follow MuJoCo scalar-first ``(w, x, y, z)``
  order representing ``R_world_from_support`` (rotating vectors from the local support
  frame into the inertial world frame: ``v_world = R(q_wxyz) @ v_support``).
- **Specific force & gravity:** A strapdown accelerometer measures proper specific force
  ``f_support = a_kin_support - R_world_from_support^T @ g_world`` (e.g. ``[0, 0, +9.81]``
  at rest when ``g_world = [0, 0, -9.81]`` and ``q = [1, 0, 0, 0]``). When ``quat_cols``
  is provided, ``R(q)^T @ world_gravity`` is added to recover pure kinematic support
  acceleration ``a_kin_support``. Otherwise, ``subtract_initial_samples > 0`` subtracts
  the mean of the initial stationary window under constant support attitude.
- **Timestamps:** ``time_col`` must contain at least 2 finite, strictly increasing
  timestamps in seconds [s]; timestamps are shifted so playback starts at ``t = 0``.
  Non-finite values in time, acceleration, angular velocity, or quaternion columns raise
  ``ValueError``.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from myosuite.physics.base_excitation import RecordedBaseProfile


def _quat_rotate_inverse(quats_wxyz: np.ndarray, world_vec: np.ndarray) -> np.ndarray:
    """Rotate inertial ``world_vec`` into the support frame via ``q_wxyz`` (support->world)."""
    q = np.asarray(quats_wxyz, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4 or not np.all(np.isfinite(q)):
        raise ValueError("Quaternion array must have finite shape (T, 4) in wxyz order")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise ValueError("Quaternion entries must be non-zero")
    q = q / norm
    w = q[:, 0:1]
    xyz = -q[:, 1:4]
    v = np.broadcast_to(np.asarray(world_vec, dtype=np.float64), (q.shape[0], 3))
    uv = np.cross(xyz, v)
    uuv = np.cross(xyz, uv)
    return v + 2.0 * (w * uv + uuv)


def load_imu_base_profile(
    path: str | Path,
    *,
    time_col: str = "time",
    acc_cols: tuple[str, str, str] = ("ax", "ay", "az"),
    gyro_cols: tuple[str, str, str] = ("gx", "gy", "gz"),
    quat_cols: tuple[str, str, str, str] | None = None,
    acc_scale: float = 1.0,
    gyro_in_degrees: bool = False,
    world_gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    subtract_initial_samples: int = 0,
    loop: bool = False,
) -> RecordedBaseProfile:
    """Load a 6-axis IMU recording (``.npz`` or ``.csv``) into a :class:`RecordedBaseProfile`."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"IMU trace file not found: {file_path}")
    if not (math.isfinite(acc_scale) and acc_scale > 0.0):
        raise ValueError("acc_scale must be finite and positive")
    if subtract_initial_samples < 0:
        raise ValueError("subtract_initial_samples must be >= 0")
    g_world = np.asarray(world_gravity, dtype=np.float64)
    if g_world.shape != (3,) or not np.all(np.isfinite(g_world)):
        raise ValueError("world_gravity must be a finite 3-vector")

    suffix = file_path.suffix.lower()
    if suffix == ".npz":
        with np.load(file_path, allow_pickle=False) as archive:
            times = np.asarray(archive[time_col], dtype=np.float64)
            if "lin_acc" in archive and all(c not in archive for c in acc_cols):
                lin_acc = np.asarray(archive["lin_acc"], dtype=np.float64)
            else:
                lin_acc = np.column_stack(
                    [np.asarray(archive[c], dtype=np.float64) for c in acc_cols]
                )
            if "ang_vel" in archive and all(c not in archive for c in gyro_cols):
                ang_vel = np.asarray(archive["ang_vel"], dtype=np.float64)
            else:
                ang_vel = np.column_stack(
                    [np.asarray(archive[c], dtype=np.float64) for c in gyro_cols]
                )
            quats = (
                np.column_stack(
                    [np.asarray(archive[c], dtype=np.float64) for c in quat_cols]
                )
                if quat_cols is not None
                else None
            )
    elif suffix == ".csv":
        with file_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        if len(rows) < 2:
            raise ValueError("CSV IMU recording must contain at least 2 rows")
        times = np.array([float(r[time_col]) for r in rows], dtype=np.float64)
        lin_acc = np.array(
            [[float(r[c]) for c in acc_cols] for r in rows], dtype=np.float64
        )
        ang_vel = np.array(
            [[float(r[c]) for c in gyro_cols] for r in rows], dtype=np.float64
        )
        quats = (
            np.array([[float(r[c]) for c in quat_cols] for r in rows], dtype=np.float64)
            if quat_cols is not None
            else None
        )
    else:
        raise ValueError(
            f"Unsupported IMU file extension {suffix!r}; expected '.npz' or '.csv'."
        )

    if times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times)):
        raise ValueError("IMU timestamps must be a 1D finite array of length >= 2")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("IMU timestamps must be strictly increasing")
    if lin_acc.shape != (len(times), 3) or not np.all(np.isfinite(lin_acc)):
        raise ValueError("IMU linear accelerations must be finite with shape (T, 3)")
    if ang_vel.shape != (len(times), 3) or not np.all(np.isfinite(ang_vel)):
        raise ValueError("IMU angular velocities must be finite with shape (T, 3)")

    times = times - float(times[0])
    lin_acc = lin_acc * float(acc_scale)
    if gyro_in_degrees:
        ang_vel = np.radians(ang_vel)

    if quats is not None:
        g_body = _quat_rotate_inverse(quats, g_world)
        lin_acc = lin_acc + g_body
    elif subtract_initial_samples > 0:
        n_bias = min(int(subtract_initial_samples), lin_acc.shape[0])
        lin_acc = lin_acc - np.mean(lin_acc[:n_bias], axis=0, keepdims=True)

    ang_acc = np.asarray(np.gradient(ang_vel, times, axis=0), dtype=np.float64)
    return RecordedBaseProfile(
        times=times,
        lin_acc=lin_acc,
        ang_vel=ang_vel,
        ang_acc=ang_acc,
        loop=loop,
    )


__all__ = ["load_imu_base_profile"]
