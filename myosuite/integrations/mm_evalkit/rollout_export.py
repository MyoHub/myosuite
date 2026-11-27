# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Rollout ``.npz`` archive and ``policies.yaml`` exporters for ``amathislab/mm-evalkit``."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np

from myosuite.biomechanics.computed_muscle_control import (
    ComputedMuscleControlConfig,
    computed_muscle_control,
)
from myosuite.biomechanics.emg import project_tendon_nodes
from myosuite.biomechanics.emg_validation import EMGTrial
from myosuite.biomechanics.inverse_dynamics import clip_kinematics
from myosuite.biomechanics.static_optimization import (
    StaticOptimizationConfig,
    static_optimization,
)
from myosuite.integrations.mm_evalkit.stats_and_config import dump_policies_yaml


def _resample_trial_for_rollout(
    model: mujoco.MjModel,
    trial: EMGTrial,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Resample ``trial.qpos`` (and optional ``external_qfrc``) onto a uniform ``dt`` grid."""
    if trial.qpos is None:
        raise ValueError(
            f"trial {trial.trial_id!r} requires qpos for export_mm_evalkit_rollout"
        )
    src_t = trial.resolved_kinematics_time()
    t0, t1 = float(src_t[0]), float(src_t[-1])
    n_frames = max(4, round((t1 - t0) / dt) + 1)
    t_grid = t0 + np.arange(n_frames, dtype=np.float64) * dt
    t_clip = np.clip(t_grid, t0, t1)

    q_src = np.asarray(trial.qpos, dtype=np.float64)
    q_out = np.column_stack(
        [np.interp(t_clip, src_t, q_src[:, col]) for col in range(model.nq)]
    )
    for jnt in range(model.njnt):
        jtype = int(model.jnt_type[jnt])
        adr = int(model.jnt_qposadr[jnt])
        if jtype == int(mujoco.mjtJoint.mjJNT_FREE):
            q_quat = q_out[:, adr + 3 : adr + 7]
            norms = np.linalg.norm(q_quat, axis=1, keepdims=True)
            q_out[:, adr + 3 : adr + 7] = np.where(
                norms > 1e-12, q_quat / norms, [1, 0, 0, 0]
            )
        elif jtype == int(mujoco.mjtJoint.mjJNT_BALL):
            q_quat = q_out[:, adr : adr + 4]
            norms = np.linalg.norm(q_quat, axis=1, keepdims=True)
            q_out[:, adr : adr + 4] = np.where(
                norms > 1e-12, q_quat / norms, [1, 0, 0, 0]
            )
    q_out = project_tendon_nodes(model, q_out)

    ext_out: np.ndarray | None = None
    if trial.external_qfrc is not None:
        ext_src = np.asarray(trial.external_qfrc, dtype=np.float64)
        ext_out = np.column_stack(
            [np.interp(t_clip, src_t, ext_src[:, col]) for col in range(model.nv)]
        )
    return t_grid, q_out, ext_out


def _estimate_touch_observations(
    model: mujoco.MjModel,
    qpos_seq: np.ndarray,
    trial: EMGTrial,
    t_grid: np.ndarray,
) -> np.ndarray:
    """Compute 4-column ``[r_heel, r_toe, l_heel, l_toe]`` foot-contact observations."""
    n_steps = len(qpos_seq)
    touch = np.zeros((n_steps, 4), dtype=np.float64)

    sensor_names = ("r_foot", "r_toes", "l_foot", "l_toes")
    sensor_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, s) for s in sensor_names
    ]
    data = mujoco.MjData(model)
    if all(sid >= 0 for sid in sensor_ids):
        adrs = [int(model.sensor_adr[sid]) for sid in sensor_ids]
        for t in range(n_steps):
            data.qpos[:] = qpos_seq[t]
            mujoco.mj_forward(model, data)
            for col, adr in enumerate(adrs):
                touch[t, col] = max(0.0, float(data.sensordata[adr]))
        if np.any(touch > 0.0):
            return touch

    foot_geom_cols: dict[int, int] = {}
    for gid in range(model.ngeom):
        gname = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").lower()
        if "foot" not in gname and "calcn" not in gname and "toes" not in gname:
            continue
        is_left = gname.endswith("_l") or gname.startswith("l_")
        is_fore = "bofoot" in gname or "toe" in gname
        base = 2 if is_left else 0
        foot_geom_cols[gid] = base + (1 if is_fore else 0)

    if foot_geom_cols:
        frc = np.zeros(6, dtype=np.float64)
        for t in range(n_steps):
            data.qpos[:] = qpos_seq[t]
            mujoco.mj_forward(model, data)
            for c_idx in range(data.ncon):
                con = data.contact[c_idx]
                c1 = foot_geom_cols.get(int(con.geom[0]))
                c2 = foot_geom_cols.get(int(con.geom[1]))
                foot_col = c1 if c1 is not None else c2
                if foot_col is not None:
                    mujoco.mj_contactForce(model, data, c_idx, frc)
                    touch[t, foot_col] += max(0.0, float(frc[0]))
        if np.any(touch > 0.0):
            return touch

    if trial.cycle_events is not None and len(trial.cycle_events) > 0:
        for ev_start, ev_end in np.asarray(trial.cycle_events, dtype=np.float64):
            dur = float(ev_end - ev_start)
            if dur <= 1e-6:
                continue
            phase = (t_grid - float(ev_start)) / dur
            in_cycle = (phase >= 0.0) & (phase <= 1.0)
            r_stance = in_cycle & (phase < 0.60)
            l_stance = in_cycle & ((phase < 0.10) | (phase >= 0.50))
            touch[r_stance, 0] = 1.0
            touch[r_stance, 1] = 1.0
            touch[l_stance, 2] = 1.0
            touch[l_stance, 3] = 1.0
    return touch


def export_mm_evalkit_rollout(
    model: mujoco.MjModel,
    trials: Sequence[EMGTrial],
    out_path: str | Path,
    *,
    dt: float = 0.01,
    solver: Literal["static_optimization", "cmc"] = "static_optimization",
    traj_ids: Sequence[int] | None = None,
    motion_group: str | None = None,
    kinematics_lowpass_hz: float | None = 6.0,
    optimization: StaticOptimizationConfig | None = None,
    cmc_config: ComputedMuscleControlConfig | None = None,
) -> Path:
    """Export policy-free solved trajectories in ``mm-evalkit``'s rollout ``.npz`` contract.

    The exported archive satisfies both ``mm_evalkit.workflows.bimanual.rollout.validate_rollout``
    (fixed-base models) and ``mm_evalkit.workflows.gait.rollout_validation.validate_rollout_archive``
    (free-root full-body models) at ``dt=0.01`` s (100 Hz) with ``activation_sample_phase='pre_step'``.
    """
    if not trials:
        raise ValueError("trials must contain at least one EMGTrial")
    if solver not in ("static_optimization", "cmc"):
        raise ValueError(
            f"solver must be 'static_optimization' or 'cmc', got {solver!r}"
        )
    if traj_ids is not None and len(traj_ids) != len(trials):
        raise ValueError(
            f"traj_ids length {len(traj_ids)} must match trials length {len(trials)}"
        )

    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
        for j in range(model.njnt)
    ]
    archive: dict[str, Any] = {
        "n_episodes": np.int64(len(trials)),
        "dt": np.float64(dt),
        "joint_names": np.asarray(joint_names, dtype=str),
        "activation_sample_phase": np.asarray("pre_step"),
    }
    if motion_group is not None and motion_group.strip():
        archive["motion_group"] = np.asarray(motion_group.strip())

    for ep_idx, trial in enumerate(trials):
        tid = int(ep_idx if traj_ids is None else traj_ids[ep_idx])
        if tid < 0:
            raise ValueError(f"traj_id must be non-negative, got {tid}")
        t_grid, q_res, ext_res = _resample_trial_for_rollout(model, trial, dt)
        kin = clip_kinematics(model, q_res, dt, lowpass_hz=kinematics_lowpass_hz)
        if solver == "static_optimization":
            so_res = static_optimization(
                model,
                kin.qpos,
                kin.qvel,
                kin.qacc,
                external_qfrc=ext_res,
                config=optimization,
            )
            policy_qpos = kin.qpos
            activations = np.clip(so_res.activation, 0.0, 1.0)
        else:
            cfg = cmc_config or ComputedMuscleControlConfig(
                optimization=optimization or StaticOptimizationConfig()
            )
            cmc_res = computed_muscle_control(
                model,
                kin.qpos,
                kin.qvel,
                kin.qacc,
                dt,
                external_qfrc=ext_res,
                config=cfg,
            )
            policy_qpos = np.vstack([kin.qpos[:1], cmc_res.qpos])
            activations = np.clip(
                np.vstack([cmc_res.optimization.activation[:1], cmc_res.activation]),
                0.0,
                1.0,
            )

        touch_obs = _estimate_touch_observations(model, policy_qpos, trial, t_grid)
        prefix = f"episode_{ep_idx}"
        archive[f"{prefix}_traj_id"] = np.int64(tid)
        archive[f"{prefix}_traj_length"] = np.int64(len(kin.qpos))
        archive[f"{prefix}_joint_positions"] = np.asarray(policy_qpos, dtype=np.float64)
        archive[f"{prefix}_muscle_activations"] = np.asarray(
            activations, dtype=np.float64
        )
        archive[f"{prefix}_traj_qpos"] = np.asarray(kin.qpos, dtype=np.float64)
        archive[f"{prefix}_touch_observations"] = np.asarray(
            touch_obs, dtype=np.float64
        )

    dst = Path(out_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, allow_pickle=False, **archive)
    return dst


def _slugify_policy_id(raw: str, index: int) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "-", raw.strip().casefold()).strip("-_")
    if not slug or not slug[0].isalnum():
        slug = f"model-{index}"
    return slug


def write_mm_evalkit_policy_config(
    path: str | Path,
    policies: Sequence[Mapping[str, Any]],
) -> Path:
    """Write a ``policies.yaml`` file conforming to ``mm_evalkit.policies.load_policy_specs``."""
    if not policies:
        raise ValueError("policies must be non-empty")
    seen_ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for idx, entry in enumerate(policies):
        label = str(entry.get("label") or entry.get("id") or f"model-{idx}").strip()
        raw_id = str(entry.get("id") or label)
        pid = _slugify_policy_id(raw_id, idx)
        if pid in seen_ids:
            pid = f"{pid}-{idx}"
        seen_ids.add(pid)
        traj_path = str(Path(entry["trajectory_data_path"]).as_posix())
        item: dict[str, Any] = {
            "id": pid,
            "label": label,
            "trajectory_data_path": traj_path,
        }
        for opt_key in ("color", "linestyle"):
            if entry.get(opt_key):
                item[opt_key] = str(entry[opt_key])
        normalized.append(item)

    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = dump_policies_yaml(normalized)
    except ImportError:
        lines = ["policies:"]
        for item in normalized:
            lines.append(f"  - id: {json.dumps(item['id'])}")
            lines.append(f"    label: {json.dumps(item['label'])}")
            lines.append(
                f"    trajectory_data_path: {json.dumps(item['trajectory_data_path'])}"
            )
            for opt_key in ("color", "linestyle"):
                if opt_key in item:
                    lines.append(f"    {opt_key}: {json.dumps(item[opt_key])}")
        text = "\n".join(lines) + "\n"
    dst.write_text(text, encoding="utf-8")
    return dst


__all__ = [
    "export_mm_evalkit_rollout",
    "write_mm_evalkit_policy_config",
]
