# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Motion-capture trial I/O: TRC read/write, gap filling, and C3D analogs/force plates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline

from myosuite.biomechanics.external_loads import force_at_body
from myosuite.integrations.capture.markers import _open_c3d, _read_markers

_LEN_TO_M = {"m": 1.0, "cm": 0.01, "mm": 0.001}
_FORCE_TO_N = {"n": 1.0, "kn": 1000.0}
_MOMENT_TO_NM = {"nm": 1.0, "nmm": 0.001, "ncm": 0.01, "n*m": 1.0, "n*mm": 0.001}


def _check_proper_rotation(rot: np.ndarray | None) -> np.ndarray:
    if rot is None:
        return np.eye(3)
    r = np.asarray(rot, dtype=float)
    if (
        r.shape != (3, 3)
        or not np.allclose(r @ r.T, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(r), 1.0, atol=1e-6)
    ):
        raise ValueError("point_to_world must be a proper 3x3 rotation matrix")
    return r


@dataclass(frozen=True)
class MarkerClip:
    """Labelled 3D marker trajectories in metres."""

    names: tuple[str, ...]
    positions: np.ndarray
    rate: float | None


def read_markers(
    path: str | Path, *, point_to_world: np.ndarray | None = None
) -> MarkerClip:
    """Read a ``.trc``, ``.c3d`` or landmark ``.csv`` file into metres."""
    raw = _read_markers(path)
    rot = _check_proper_rotation(point_to_world)
    return MarkerClip(raw.names, raw.positions @ rot.T, raw.rate)


def write_trc(
    path: str | Path,
    names: Sequence[str],
    positions: np.ndarray,
    rate: float,
    *,
    units: str = "mm",
) -> Path:
    """Write ``(T, M, 3)`` metre positions to an OpenSim/Pose2Sim ``.trc`` file."""
    unit_key = units.strip().lower()
    if unit_key not in _LEN_TO_M:
        raise ValueError(f"units must be one of {sorted(_LEN_TO_M)}, got {units!r}")
    pts = np.asarray(positions, dtype=float)
    labels = tuple(str(n).strip() for n in names)
    if not labels or len(set(labels)) != len(labels) or not all(labels):
        raise ValueError("names must be unique and non-empty")
    if pts.ndim != 3 or pts.shape[1:] != (len(labels), 3) or len(pts) < 1:
        raise ValueError(f"positions must have shape (T >= 1, {len(labels)}, 3)")
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("rate must be finite and positive")

    scale = 1.0 / _LEN_TO_M[unit_key]
    scaled = pts * scale
    n_frames, n_markers, _ = scaled.shape
    out = Path(path)

    h1 = f"PathFileType\t4\t(X/Y/Z)\t{out.name}"
    h2 = "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames"
    h3 = f"{rate:g}\t{rate:g}\t{n_frames}\t{n_markers}\t{unit_key}\t{rate:g}\t1\t{n_frames}"
    col_names = ["Frame#", "Time"]
    sub_names = ["", ""]
    for idx, label in enumerate(labels, start=1):
        col_names.extend([label, "", ""])
        sub_names.extend([f"X{idx}", f"Y{idx}", f"Z{idx}"])

    lines = [h1, h2, h3, "\t".join(col_names), "\t".join(sub_names)]
    for t in range(n_frames):
        row = [str(t + 1), f"{t / rate:.8g}"]
        for m in range(n_markers):
            xyz = scaled[t, m]
            if np.isfinite(xyz).all():
                row.extend(f"{v:.8g}" for v in xyz)
            else:
                row.extend(["", "", ""])
        lines.append("\t".join(row))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def fill_marker_gaps(
    positions: np.ndarray,
    rate: float,
    *,
    max_gap_s: float = 0.20,
    method: str = "cubic",
) -> np.ndarray:
    """Interpolate internal ``NaN`` marker dropouts no longer than ``max_gap_s``."""
    pts = np.asarray(positions, dtype=float)
    if pts.ndim != 3 or pts.shape[2] != 3:
        raise ValueError("positions must have shape (T, M, 3)")
    if not np.isfinite(rate) or rate <= 0 or max_gap_s < 0:
        raise ValueError("rate must be positive and max_gap_s non-negative")
    if method not in ("cubic", "linear"):
        raise ValueError("method must be 'cubic' or 'linear'")

    max_frames = int(np.floor(max_gap_s * rate + 1e-9))
    if max_frames == 0 or len(pts) < 3:
        return pts.copy()

    out = pts.copy()
    t_idx = np.arange(len(pts), dtype=float)
    for m in range(pts.shape[1]):
        valid = np.isfinite(pts[:, m]).all(axis=1)
        ok = np.flatnonzero(valid)
        if len(ok) < 2 or len(ok) == len(pts):
            continue
        interp = (
            CubicSpline(t_idx[ok], pts[ok, m], axis=0)(t_idx)
            if method == "cubic" and len(ok) >= 4
            else np.column_stack(
                [np.interp(t_idx, t_idx[ok], pts[ok, m, c]) for c in range(3)]
            )
        )
        for start, end in zip(ok[:-1], ok[1:]):
            gap = end - start - 1
            if 0 < gap <= max_frames:
                out[start + 1 : end, m] = interp[start + 1 : end]
    return out


@dataclass(frozen=True)
class ForcePlateData:
    """Force-plate wrench resampled to marker frames in world SI units (N, m, N m)."""

    corners: np.ndarray
    force: np.ndarray
    cop: np.ndarray
    free_torque: np.ndarray


@dataclass(frozen=True)
class C3DTrial:
    """Synchronized 3D markers, raw analog channels (e.g. EMG), and force plates."""

    markers: MarkerClip
    analog_names: tuple[str, ...]
    analog_units: tuple[str, ...]
    analogs: np.ndarray
    analog_rate: float | None
    force_plates: tuple[ForcePlateData, ...]


def _downsample_plate(
    force_a: np.ndarray,
    cop_a: np.ndarray,
    tz_a: np.ndarray,
    corners: np.ndarray,
    n_frames: int,
    min_force_n: float,
) -> ForcePlateData:
    n_analog = len(force_a)
    center = corners.mean(axis=0)
    if n_analog == 0 or n_frames == 0:
        z = np.zeros((n_frames, 3))
        return ForcePlateData(corners, z, np.tile(center, (n_frames, 1)), z.copy())
    if n_analog % n_frames == 0:
        sub = n_analog // n_frames
        f_blk = force_a.reshape(n_frames, sub, 3)
        tz_blk = tz_a.reshape(n_frames, sub, 3)
        cop_blk = np.where(np.isfinite(cop_a), cop_a, center).reshape(n_frames, sub, 3)
        mag = np.linalg.norm(f_blk, axis=-1, keepdims=True)
        w = np.where(mag >= min_force_n, mag, 0.0)
        w_sum = w.sum(axis=1)
        force = f_blk.mean(axis=1)
        free_torque = tz_blk.mean(axis=1)
        cop = np.where(
            w_sum > 0,
            (cop_blk * w).sum(axis=1) / np.where(w_sum > 0, w_sum, 1.0),
            center,
        )
    else:
        t_dst = np.linspace(0.0, 1.0, n_frames)
        t_src = np.linspace(0.0, 1.0, n_analog)
        clean_cop = np.where(np.isfinite(cop_a), cop_a, center)
        force = np.column_stack(
            [np.interp(t_dst, t_src, force_a[:, i]) for i in range(3)]
        )
        cop = np.column_stack(
            [np.interp(t_dst, t_src, clean_cop[:, i]) for i in range(3)]
        )
        free_torque = np.column_stack(
            [np.interp(t_dst, t_src, tz_a[:, i]) for i in range(3)]
        )

    unloaded = np.linalg.norm(force, axis=1) < min_force_n
    force[unloaded] = 0.0
    free_torque[unloaded] = 0.0
    cop[unloaded] = center
    return ForcePlateData(corners, force, cop, free_torque)


def read_c3d_trial(
    path: str | Path,
    *,
    point_to_world: np.ndarray | None = None,
    reaction: bool = True,
    min_force_n: float = 5.0,
) -> C3DTrial:
    """Read markers, analog channels (EMG/sensors) and calibrated force plates from C3D."""
    rot = _check_proper_rotation(point_to_world)
    markers = read_markers(path, point_to_world=rot)
    capture = _open_c3d(path, extract_forceplat_data=True)

    analog_grp = capture["parameters"].get("ANALOG", {})
    a_labels = list(analog_grp.get("LABELS", {}).get("value", []))
    if "LABELS2" in analog_grp:
        a_labels += list(analog_grp["LABELS2"]["value"])
    a_units = list(analog_grp.get("UNITS", {}).get("value", []))
    if "UNITS2" in analog_grp:
        a_units += list(analog_grp["UNITS2"]["value"])
    raw_analogs = np.asarray(
        capture["data"].get("analogs", np.zeros((1, 0, 0))), dtype=float
    )
    analogs = (
        raw_analogs[0].T.copy()
        if raw_analogs.ndim == 3 and raw_analogs.shape[1]
        else np.zeros((0, 0))
    )
    n_ch = analogs.shape[1] if analogs.ndim == 2 else 0
    a_names = tuple(str(lbl).strip() for lbl in a_labels[:n_ch])
    a_unit_tup = tuple(
        str(a_units[i]).strip() if i < len(a_units) else "" for i in range(n_ch)
    )
    a_rate_vals = analog_grp.get("RATE", {}).get("value", [])
    a_rate = (
        float(a_rate_vals[0]) if a_rate_vals and float(a_rate_vals[0]) > 0 else None
    )

    sign = -1.0 if reaction else 1.0
    plates: list[ForcePlateData] = []
    for pf in capture["data"].get("platform", []):
        u_pos = _LEN_TO_M.get(str(pf["unit_position"]).strip().lower(), 0.001)
        u_frc = _FORCE_TO_N.get(str(pf["unit_force"]).strip().lower(), 1.0)
        u_mom = _MOMENT_TO_NM.get(str(pf["unit_moment"]).strip().lower(), 0.001)
        corners = (np.asarray(pf["corners"], dtype=float).T * u_pos) @ rot.T
        f_a = (sign * np.asarray(pf["force"], dtype=float).T * u_frc) @ rot.T
        cop_a = (np.asarray(pf["center_of_pressure"], dtype=float).T * u_pos) @ rot.T
        tz_a = (sign * np.asarray(pf["Tz"], dtype=float).T * u_mom) @ rot.T
        plates.append(
            _downsample_plate(
                f_a, cop_a, tz_a, corners, len(markers.positions), min_force_n
            )
        )

    return C3DTrial(markers, a_names, a_unit_tup, analogs, a_rate, tuple(plates))


def force_plates_to_qfrc(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    plates: Sequence[ForcePlateData],
    *,
    contact_bodies: Sequence[str] = ("calcn_r", "calcn_l"),
    plate_bodies: Mapping[int, str] | None = None,
    min_force_n: float = 5.0,
    max_distance_m: float = 0.35,
) -> np.ndarray:
    """Map C3D force-plate wrenches onto contacting model bodies as ``(T, nv)`` ``qfrc``."""
    poses = np.asarray(qpos, dtype=float)
    single = poses.ndim == 1
    if single:
        poses = poses[None, :]
    if poses.ndim != 2 or poses.shape[1] != model.nq:
        raise ValueError(f"qpos must have shape (T, {model.nq}) or ({model.nq},)")

    cand_ids = [model.body(name).id for name in contact_bodies]
    data = mujoco.MjData(model)
    out = np.zeros((len(poses), model.nv))

    for t in range(len(poses)):
        data.qpos[:] = poses[t]
        mujoco.mj_fwdPosition(model, data)
        for p_idx, plate in enumerate(plates):
            f = plate.force[t]
            if np.linalg.norm(f) < min_force_n or not np.isfinite(plate.cop[t]).all():
                continue
            if plate_bodies is not None and p_idx in plate_bodies:
                body_name = plate_bodies[p_idx]
            else:
                dists = [
                    np.linalg.norm(data.xpos[bid] - plate.cop[t]) for bid in cand_ids
                ]
                best = int(np.argmin(dists))
                if dists[best] > max_distance_m:
                    continue
                body_name = contact_bodies[best]
            out[t] += force_at_body(
                model,
                poses[t],
                body_name,
                f,
                point=plate.cop[t],
                torque=plate.free_torque[t],
                frame="world",
            )
    return out[0] if single else out
