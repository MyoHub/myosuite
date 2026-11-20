# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""OpenSim 4 state-path normalization and force-plate ``ExternalLoads`` conversion."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path

import mujoco
import numpy as np

from myosuite.biomechanics.opensim_motion import (
    GAIT2392_SIGNS,
    OpenSimMotion,
    _level_ground,
)


def normalize_opensim_motion(
    motion: OpenSimMotion,
) -> tuple[OpenSimMotion, Mapping[str, float]]:
    """Strip OpenSim 4 state paths and infer knee-flexion sign conventions.

    Args:
        motion: Parsed OpenSim storage columns.

    Returns:
        ``(cleaned_motion, signs)`` with coordinate names stripped of
        ``/jointset/.../value`` prefixes, ``/speed`` columns removed, and
        knee-flexion sign map empty for Rajagopal ``walker_knee`` paths or
        :data:`GAIT2392_SIGNS` otherwise.
    """
    keep_cols: list[int] = []
    clean_labels: list[str] = []
    seen: set[str] = set()
    has_walker_knee = False

    for idx, raw_label in enumerate(motion.labels):
        label = raw_label.strip()
        if label.endswith("/speed"):
            continue
        if "/walker_knee_" in label:
            has_walker_knee = True
        if label.startswith("/"):
            parts = [p for p in label.split("/") if p]
            if parts and parts[-1] == "value" and len(parts) >= 2:
                label = parts[-2]
            elif parts:
                label = parts[-1]
        if label not in seen:
            seen.add(label)
            keep_cols.append(idx)
            clean_labels.append(label)

    signs: Mapping[str, float] = {} if has_walker_knee else GAIT2392_SIGNS
    return (
        OpenSimMotion(
            time=motion.time.copy(),
            labels=tuple(clean_labels),
            values=motion.values[:, keep_cols].copy(),
            in_degrees=motion.in_degrees,
        ),
        signs,
    )


def _xyz_columns(motion: OpenSimMotion, prefix: str) -> np.ndarray | None:
    if not prefix:
        return None
    cols = [f"{prefix}{axis}" for axis in "xyz"]
    if not all(c in motion.labels for c in cols):
        return None
    return np.stack([motion.column(c) for c in cols], axis=1)


def opensim_grf_to_qfrc(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    time: np.ndarray,
    grf_motion: OpenSimMotion,
    external_loads_xml: str | Path,
    *,
    root_body: str = "pelvis",
    ground_quat: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Map OpenSim force-plate ``ExternalLoads`` onto ``(T, nv)`` generalized forces.

    Args:
        model: Compiled MuJoCo model.
        qpos: ``(T, nq)`` generalized positions aligned with *time*.
        time: ``(T,)`` sample timestamps [s].
        grf_motion: Parsed OpenSim GRF ``.mot`` / ``.sto`` columns.
        external_loads_xml: Path to an OpenSim ``ExternalLoads`` XML file or inline XML string.
        root_body: Root body used to orient OpenSim's Y-up ground when *ground_quat* is ``None``.
        ground_quat: Optional ``(w, x, y, z)`` quaternion placing OpenSim ground in world frame.

    Returns:
        ``(T, nv)`` generalized forces matching the external wrenches.
    """
    qpos = np.asarray(qpos, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    if qpos.shape != (len(time), model.nq):
        raise ValueError(
            f"qpos must have shape ({len(time)}, {model.nq}), got {qpos.shape}"
        )

    xml_str = str(external_loads_xml)
    root = (
        ET.fromstring(xml_str)
        if xml_str.lstrip().startswith("<")
        else ET.parse(Path(external_loads_xml)).getroot()
    )

    if ground_quat is None:
        g_quat = _level_ground(model, model.body(root_body).id)
    else:
        g_quat = np.asarray(ground_quat, dtype=np.float64)
        g_quat = g_quat / np.linalg.norm(g_quat)
    rot_mat = np.zeros(9)
    mujoco.mju_quat2Mat(rot_mat, g_quat)
    r_ground = rot_mat.reshape(3, 3)

    body_names = {model.body(b).name: b for b in range(model.nbody)}
    loads: list[tuple[int, np.ndarray, np.ndarray]] = []
    t_src = grf_motion.time

    for ef in root.findall(".//ExternalForce"):
        body_name = (ef.findtext("applied_to_body") or "").strip()
        if body_name not in body_names:
            continue
        f_expr = (ef.findtext("force_expressed_in_body") or "ground").strip()
        p_expr = (ef.findtext("point_expressed_in_body") or "ground").strip()
        if f_expr != "ground" or p_expr != "ground":
            raise ValueError(
                f"ExternalForce {ef.attrib.get('name', body_name)!r} uses non-ground frame "
                f"({f_expr!r}, {p_expr!r}); only ground-expressed forces are supported"
            )
        f_raw = _xyz_columns(
            grf_motion, (ef.findtext("force_identifier") or "").strip()
        )
        if f_raw is None:
            continue
        p_raw = _xyz_columns(
            grf_motion, (ef.findtext("point_identifier") or "").strip()
        )
        tau_raw = _xyz_columns(
            grf_motion, (ef.findtext("torque_identifier") or "").strip()
        )
        if tau_raw is None:
            tau_raw = np.zeros_like(f_raw)
        if p_raw is not None:
            active = np.linalg.norm(f_raw, axis=1, keepdims=True) > 1e-6
            p_safe = np.where(active & np.isfinite(p_raw), p_raw, 0.0)
            tau0_raw = tau_raw + np.cross(p_safe, f_raw)
        else:
            tau0_raw = tau_raw

        f_world_src = f_raw @ r_ground.T
        tau0_world_src = tau0_raw @ r_ground.T
        f_interp = np.column_stack(
            [np.interp(time, t_src, f_world_src[:, k]) for k in range(3)]
        )
        tau0_interp = np.column_stack(
            [np.interp(time, t_src, tau0_world_src[:, k]) for k in range(3)]
        )
        loads.append((body_names[body_name], f_interp, tau0_interp))

    if not loads:
        raise ValueError(
            "no ExternalForce entries matched model bodies and GRF columns"
        )

    data = mujoco.MjData(model)
    origin = np.zeros(3)
    out = np.zeros((len(time), model.nv), dtype=np.float64)
    for t in range(len(time)):
        data.qpos[:] = qpos[t]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        qfrc = out[t]
        for body_id, f_arr, tau0_arr in loads:
            if np.any(f_arr[t]) or np.any(tau0_arr[t]):
                mujoco.mj_applyFT(
                    model, data, f_arr[t], tau0_arr[t], origin, body_id, qfrc
                )
    return out


__all__ = [
    "normalize_opensim_motion",
    "opensim_grf_to_qfrc",
]
