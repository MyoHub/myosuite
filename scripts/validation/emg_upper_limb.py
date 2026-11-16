# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Upper-limb shoulder, elbow and hand dataset adapters for EMG validation."""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import scipy.io as sio
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from myosuite.biomechanics.emg import EMGFilterConfig, project_tendon_nodes
from myosuite.biomechanics.emg_validation import (
    ChannelBinding,
    EMGChannelMap,
    EMGSource,
    EMGTrial,
)
from myosuite.biomechanics.inverse_dynamics import clip_kinematics, required_qfrc
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings

EMG2POSE_JOINT_NAMES: tuple[str, ...] = (
    "cmc_abduction",
    "cmc_flexion",
    "mp_flexion",
    "ip_flexion",
    "mcp2_flexion",
    "mcp2_abduction",
    "pm2_flexion",
    "md2_flexion",
    "mcp3_flexion",
    "mcp3_abduction",
    "pm3_flexion",
    "md3_flexion",
    "mcp4_flexion",
    "mcp4_abduction",
    "pm4_flexion",
    "md4_flexion",
    "mcp5_flexion",
    "mcp5_abduction",
    "pm5_flexion",
    "md5_flexion",
)

# Fallback right-girdle rhythm keyframes from Seth et al. (2019) CMC trials ``abd21``
# (coronal abduction) and ``flx01`` (sagittal flexion) over [0, 30, 60, 90, 120, 150] deg
# humerothoracic elevation. Used only when ``myo_sim/build/data/scapulothoracic.json``
# (added in MyoHub/myo_sim#189, section ``rhythm.frames``) is not yet installed.
_SETH_ELEV_RAD = np.radians([0.0, 30.0, 60.0, 90.0, 120.0, 150.0])
_SETH_ABD_SC = np.array(
    [
        [-0.310669, -0.4906, -0.5855, -0.6383, -0.6733, -0.7716],
        [-0.073304, -0.0362, 0.0525, 0.1275, 0.1883, 0.1755],
        [0.000000, 0.0345, 0.1386, 0.2640, 0.4341, 0.5146],
    ]
)
_SETH_FLX_SC = np.array(
    [
        [-0.310669, -0.2808, -0.2690, -0.3511, -0.5082, -0.8003],
        [-0.073304, -0.0996, -0.0194, 0.1096, 0.1522, 0.1914],
        [0.000000, 0.0538, 0.1779, 0.3257, 0.5348, 0.5409],
    ]
)
_SETH_ABD_AC = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.9979, -0.0357, -0.0537, -0.0008],
        [0.9955, -0.0743, -0.0564, 0.0172],
        [0.9931, -0.0986, -0.0239, 0.0589],
        [0.9904, -0.1005, 0.0137, 0.0934],
        [0.9527, -0.1712, 0.1300, 0.2147],
    ]
)
_SETH_FLX_AC = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.9997, -0.0020, 0.0229, -0.0043],
        [0.9961, -0.0344, 0.0652, 0.0487],
        [0.9880, -0.0533, 0.1001, 0.1048],
        [0.9807, -0.0618, 0.1239, 0.1382],
        [0.9509, -0.1752, 0.1177, 0.2263],
    ]
)


def _load_seth_rhythm_keyframes(
    sc_rest: np.ndarray,
    ac_rest: np.ndarray,
    json_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load ``abd21`` and ``flx01`` girdle keyframes from ``scapulothoracic.json`` if available."""
    if json_path is None:
        try:
            import myo_sim

            cand = (
                Path(myo_sim.__file__).resolve().parent
                / "build"
                / "data"
                / "scapulothoracic.json"
            )
            if cand.is_file():
                json_path = cand
        except ImportError:
            json_path = None

    if json_path is None or not json_path.is_file():
        return _SETH_ELEV_RAD, _SETH_ABD_SC, _SETH_FLX_SC, _SETH_ABD_AC, _SETH_FLX_AC

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    frames = payload.get("rhythm", {}).get("frames", {})
    if "abduction" not in frames or "flexion" not in frames:
        return _SETH_ELEV_RAD, _SETH_ABD_SC, _SETH_FLX_SC, _SETH_ABD_AC, _SETH_FLX_AC

    bins = ("30", "60", "90", "120", "max")
    sc_names = (
        "sternoclavicular_r2_r",
        "sternoclavicular_r3_r",
        "sternoclavicular_r1_r",
    )

    def _extract_mode(
        mode_frames: dict[str, Any], preferred_trial: str
    ) -> tuple[np.ndarray, np.ndarray]:
        trials = list(mode_frames["30"]["trial"])
        idx = trials.index(preferred_trial) if preferred_trial in trials else 0
        sc_mat = np.empty((3, 1 + len(bins)), dtype=np.float64)
        ac_mat = np.empty((1 + len(bins), 4), dtype=np.float64)
        sc_mat[:, 0] = sc_rest
        ac_mat[0] = ac_rest
        for col, b in enumerate(bins, start=1):
            entry = mode_frames[b]
            for k, sname in enumerate(sc_names):
                sc_mat[k, col] = float(entry[sname][idx])
            q = np.asarray(entry["acromioclavicular_r"][idx], dtype=np.float64)
            ac_mat[col] = q / np.linalg.norm(q)
        return sc_mat, ac_mat

    abd_sc, abd_ac = _extract_mode(frames["abduction"], "abd21")
    flx_sc, flx_ac = _extract_mode(frames["flexion"], "flx01")
    return _SETH_ELEV_RAD, abd_sc, flx_sc, abd_ac, flx_ac


def _axis_rot(axis: np.ndarray, angle: float) -> np.ndarray:
    return Rotation.from_rotvec(axis * float(angle)).as_matrix()


def map_upper_limb_to_qpos(
    model: mujoco.MjModel,
    coords: dict[str, np.ndarray],
    *,
    scapulothoracic_json: Path | None = None,
) -> tuple[np.ndarray, list[int], bool]:
    """Map MoBL-ARMS upper-limb coordinates [rad] onto *model* ``qpos``.

    Handles both MoBL-ARMS coupled shoulders and Seth et al. (2019) scapulothoracic
    shoulders (free sternoclavicular hinges + ``acromioclavicular_r`` ball joint).

    On scapulothoracic models, sternoclavicular and acromioclavicular kinematics are
    interpolated over non-negative humerothoracic elevation (extension ``shoulder_elv < 0``
    clamps to 0 so the girdle stays at rest) between Seth et al. (2019)'s ``abd21``
    (coronal abduction, ``elv_angle = 0``) and ``flx01`` (sagittal flexion,
    ``elv_angle = pi/2``) trajectories using the smooth blend assumption
    ``w_flx = sin^2(clip(elv_angle, 0, pi/2))``: planes behind the coronal plane
    (``elv_angle < 0``) clamp to ``abd21`` and cross-body planes past sagittal
    (``elv_angle > pi/2``) clamp to ``flx01``.
    """
    n_frames = len(next(iter(coords.values())))
    jnt_lookup = {model.joint(j).name: j for j in range(model.njnt)}
    qpos = np.tile(model.qpos0, (n_frames, 1))
    ctrl_qadr: list[int] = []

    def _find_jnt(base: str) -> int | None:
        return next(
            (jnt_lookup[c] for c in (f"{base}_r", base) if c in jnt_lookup), None
        )

    ac_jid = jnt_lookup.get("acromioclavicular_r")
    is_scapulothoracic = ac_jid is not None and int(model.jnt_type[ac_jid]) == int(
        mujoco.mjtJoint.mjJNT_BALL
    )

    for base in ("elbow_flexion", "pro_sup", "deviation", "flexion"):
        if base not in coords:
            continue
        jid = _find_jnt(base)
        if jid is None:
            continue
        qadr = int(model.jnt_qposadr[jid])
        vals = np.asarray(coords[base], dtype=np.float64)
        if model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]
            vals = np.clip(vals, lo, hi)
        qpos[:, qadr] = vals
        ctrl_qadr.append(qadr)

    has_shoulder = any(
        k in coords for k in ("elv_angle", "shoulder_elv", "shoulder_rot")
    )
    if has_shoulder and not is_scapulothoracic:
        for base in ("elv_angle", "shoulder_elv", "shoulder_rot"):
            if base not in coords:
                continue
            jid = _find_jnt(base)
            if jid is None:
                continue
            qadr = int(model.jnt_qposadr[jid])
            vals = np.asarray(coords[base], dtype=np.float64)
            if model.jnt_limited[jid]:
                lo, hi = model.jnt_range[jid]
                vals = np.clip(vals, lo, hi)
            qpos[:, qadr] = vals
            ctrl_qadr.append(qadr)
    elif has_shoulder and is_scapulothoracic:
        assert ac_jid is not None
        p_arr = np.asarray(
            coords.get("elv_angle", np.zeros(n_frames)), dtype=np.float64
        )
        e_arr = np.asarray(
            coords.get("shoulder_elv", np.zeros(n_frames)), dtype=np.float64
        )
        r_arr = np.asarray(
            coords.get("shoulder_rot", np.zeros(n_frames)), dtype=np.float64
        )

        j_elv = model.joint("elv_angle_r").id
        j_selv = model.joint("shoulder_elv_r").id
        j_s12 = model.joint("shoulder1_r2_r").id
        j_srot = model.joint("shoulder_rot_r").id
        ay = model.jnt_axis[j_elv].copy()
        ax = model.jnt_axis[j_selv].copy()
        ar = model.jnt_axis[j_srot].copy()

        sc_adrs = [
            int(model.jnt_qposadr[model.joint(n).id])
            for n in (
                "sternoclavicular_r2_r",
                "sternoclavicular_r3_r",
                "sternoclavicular_r1_r",
            )
        ]
        ac_adr = int(model.jnt_qposadr[ac_jid])
        gh_adrs = [int(model.jnt_qposadr[j]) for j in (j_elv, j_selv, j_s12, j_srot)]
        ctrl_qadr.extend([gh_adrs[0], gh_adrs[1], gh_adrs[3], *sc_adrs])

        chest_id = model.body("chest_r").id
        hum_id = model.body("humerus_r").id
        data = mujoco.MjData(model)
        elev_grid, abd_sc, flx_sc, abd_ac, flx_ac = _load_seth_rhythm_keyframes(
            model.qpos0[sc_adrs],
            model.qpos0[ac_adr : ac_adr + 4],
            json_path=scapulothoracic_json,
        )

        def _gh_residual(x: np.ndarray, tgt: np.ndarray) -> np.ndarray:
            return Rotation.from_matrix(
                (
                    _axis_rot(ay, x[0])
                    @ _axis_rot(ax, x[1])
                    @ _axis_rot(ay, -x[0])
                    @ _axis_rot(ar, x[2])
                ).T
                @ tgt
            ).as_rotvec()

        # Frame-0 seed uses ~60% glenohumeral share of humerothoracic elevation
        # (cf. 2:1 = 0.67 Inman et al., 2.4:1 = 0.70 Braman et al. 2009,
        # 5:4 = 0.56 Poppen & Walker 1976); subsequent frames warm-start from prev_gh.
        prev_gh = np.array([float(p_arr[0]), 0.6 * float(e_arr[0]), float(r_arr[0])])

        for t in range(n_frames):
            p_ht, e_ht, r_ht = float(p_arr[t]), float(e_arr[t]), float(r_arr[t])
            # Smooth interpolation assumption between coronal abduction (0) and sagittal flexion (pi/2).
            w_flx = float(np.sin(np.clip(p_ht, 0.0, 0.5 * np.pi)) ** 2)
            # Clip negative elevation (arm extension) to 0 so the girdle stays at its rest posture.
            e_clamped = float(np.clip(e_ht, 0.0, float(elev_grid[-1])))

            for k, sadr in enumerate(sc_adrs):
                v_abd = float(np.interp(e_clamped, elev_grid, abd_sc[k]))
                v_flx = float(np.interp(e_clamped, elev_grid, flx_sc[k]))
                qpos[t, sadr] = (1.0 - w_flx) * v_abd + w_flx * v_flx

            q_abd = np.array(
                [np.interp(e_clamped, elev_grid, abd_ac[:, c]) for c in range(4)]
            )
            q_flx = np.array(
                [np.interp(e_clamped, elev_grid, flx_ac[:, c]) for c in range(4)]
            )
            if float(np.dot(q_abd, q_flx)) < 0.0:
                q_flx = -q_flx
            # Component-wise linear blend + renormalization (nlerp); accurate for AC rotations (<35 deg).
            ac_q = (1.0 - w_flx) * q_abd + w_flx * q_flx
            ac_q /= np.linalg.norm(ac_q)
            qpos[t, ac_adr : ac_adr + 4] = ac_q

            for gadr in gh_adrs:
                qpos[t, gadr] = 0.0
            data.qpos[:] = qpos[t]
            mujoco.mj_kinematics(model, data)
            r_chest = data.xmat[chest_id].reshape(3, 3)
            r_hum0 = data.xmat[hum_id].reshape(3, 3)
            r_ht_mat = (
                _axis_rot(ay, p_ht)
                @ _axis_rot(ax, e_ht)
                @ _axis_rot(ay, -p_ht)
                @ _axis_rot(ar, r_ht)
            )
            r_gh_target = r_hum0.T @ (r_chest @ r_ht_mat)

            sol = least_squares(
                _gh_residual, prev_gh, args=(r_gh_target,), xtol=1e-10, ftol=1e-10
            )
            if t == 0 or sol.cost > 1e-6:
                alt_seeds = (
                    np.array([p_ht, abs(0.6 * e_ht), r_ht]),
                    np.array([p_ht + np.pi, -0.6 * e_ht, r_ht]),
                )
                for seed in alt_seeds:
                    cand = least_squares(
                        _gh_residual, seed, args=(r_gh_target,), xtol=1e-10, ftol=1e-10
                    )
                    if cand.cost + 1e-10 < sol.cost:
                        sol = cand
            if e_ht >= 0.0 and sol.x[1] < 0.0:
                # (p + pi, -e, r) represents the same 3D rotation; keep GH elevation non-negative
                # so frame-0 warm starts cannot lock onto the inverted plane+pi branch.
                flipped = least_squares(
                    _gh_residual,
                    np.array([sol.x[0] + np.pi, -sol.x[1], sol.x[2]]),
                    args=(r_gh_target,),
                    xtol=1e-10,
                    ftol=1e-10,
                )
                if flipped.cost <= sol.cost + 1e-8 and flipped.x[1] >= 0.0:
                    sol = flipped
            p_gh = (float(sol.x[0]) + np.pi) % (2.0 * np.pi) - np.pi
            e_gh = float(sol.x[1])
            r_gh = float(sol.x[2])
            prev_gh = np.array([p_gh, e_gh, r_gh])
            qpos[t, gh_adrs[0]] = p_gh
            qpos[t, gh_adrs[1]] = e_gh
            qpos[t, gh_adrs[2]] = -p_gh
            qpos[t, gh_adrs[3]] = r_gh

    project_joint_couplings(JointCouplings.from_model(model), qpos)
    qpos = project_tendon_nodes(model, qpos)
    has_st_contact = (
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "scapulothoracic_ellipsoid_r"
        )
        >= 0
    )
    return qpos, sorted(set(ctrl_qadr)), has_st_contact


def map_upper_limb_torques_to_qfrc(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    dt: float,
    coords: dict[str, np.ndarray],
    torques: dict[str, np.ndarray],
) -> np.ndarray:
    """Map measured human upper-limb coordinate torques [N m] into ``external_qfrc``."""
    kin = clip_kinematics(model, qpos, dt, lowpass_hz=6.0)
    data = mujoco.MjData(model)
    j_lookup = {model.joint(j).name: j for j in range(model.njnt)}
    body_lookup = {model.body(b).name: b for b in range(model.nbody)}
    coupled_slaves = [
        int(model.jnt_dofadr[int(model.eq_obj1id[e])])
        for e in range(model.neq)
        if model.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT and model.eq_active0[e]
    ]
    ac_jid = j_lookup.get("acromioclavicular_r")
    is_st = ac_jid is not None and int(model.jnt_type[ac_jid]) == int(
        mujoco.mjtJoint.mjJNT_BALL
    )

    j_elv = j_lookup.get("elv_angle_r", j_lookup.get("elv_angle"))
    j_selv = j_lookup.get("shoulder_elv_r", j_lookup.get("shoulder_elv"))
    j_srot = j_lookup.get("shoulder_rot_r", j_lookup.get("shoulder_rot"))
    chest_id = body_lookup.get("chest_r", 0)
    hum_id = body_lookup.get("humerus_r", body_lookup.get("humerus", -1))
    has_sh = (
        all(k in torques for k in ("elv_angle", "shoulder_elv", "shoulder_rot"))
        and j_elv is not None
        and j_selv is not None
        and j_srot is not None
    )
    ay = (
        model.jnt_axis[j_elv].copy() if j_elv is not None else np.array([0.0, 1.0, 0.0])
    )
    ax = (
        model.jnt_axis[j_selv].copy()
        if j_selv is not None
        else np.array([1.0, 0.0, 0.0])
    )
    ar = (
        model.jnt_axis[j_srot].copy()
        if j_srot is not None
        else np.array([0.0, 1.0, 0.0])
    )

    sh_dofs = [
        int(model.jnt_dofadr[j_lookup[n]])
        for n in (
            "sternoclavicular_r2_r",
            "sternoclavicular_r3_r",
            "sternoclavicular_r1_r",
            "elv_angle_r",
            "shoulder_elv_r",
            "shoulder1_r2_r",
            "shoulder_rot_r",
        )
        if n in j_lookup
    ]
    if ac_jid is not None:
        ac_d = int(model.jnt_dofadr[ac_jid])
        sh_dofs.extend([ac_d, ac_d + 1, ac_d + 2])

    ext = np.zeros((len(qpos), model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    for t in range(len(qpos)):
        data.qpos[:], data.qvel[:] = kin.qpos[t], kin.qvel[t]
        mujoco.mj_fwdPosition(model, data)
        mujoco.mj_fwdVelocity(model, data)
        tau_id = required_qfrc(model, data, kin.qacc[t])

        for cname in ("elbow_flexion", "pro_sup", "deviation", "flexion"):
            if cname in torques:
                jid = j_lookup.get(f"{cname}_r", j_lookup.get(cname))
                if jid is not None:
                    dadr = int(model.jnt_dofadr[jid])
                    ext[t, dadr] = tau_id[dadr] - float(torques[cname][t])

        if has_sh:
            assert j_elv is not None and j_selv is not None and j_srot is not None
            if not is_st:
                ext[t, coupled_slaves] = tau_id[coupled_slaves]
                for cname, jid in (
                    ("elv_angle", j_elv),
                    ("shoulder_elv", j_selv),
                    ("shoulder_rot", j_srot),
                ):
                    dadr = int(model.jnt_dofadr[jid])
                    ext[t, dadr] = tau_id[dadr] - float(torques[cname][t])
            elif hum_id >= 0:
                p_ht = float(coords["elv_angle"][t])
                e_ht = float(coords["shoulder_elv"][t])
                r_chest = (
                    data.xmat[chest_id].reshape(3, 3) if chest_id > 0 else np.eye(3)
                )
                r1 = _axis_rot(ay, p_ht)
                r2 = r1 @ _axis_rot(ax, e_ht)
                r3 = r2 @ _axis_rot(ay, -p_ht)
                j_ht_world = r_chest @ np.column_stack([ay - r2 @ ay, r1 @ ax, r3 @ ar])
                tau_ht = np.array(
                    [
                        torques["elv_angle"][t],
                        torques["shoulder_elv"][t],
                        torques["shoulder_rot"][t],
                    ]
                )
                m_human_world = np.linalg.solve(
                    j_ht_world @ j_ht_world.T + 1e-3 * np.eye(3), j_ht_world @ tau_ht
                )
                mujoco.mj_jacBody(model, data, None, jacr, hum_id)
                tau_sh_target = jacr.T @ m_human_world
                ext[t, sh_dofs] = tau_id[sh_dofs] - tau_sh_target[sh_dofs]
    return ext


def thoracohumeral_to_upper_limb_coords(
    model: mujoco.MjModel, r_ht: np.ndarray
) -> dict[str, np.ndarray]:
    """Decompose humerus-in-thorax rotations into MoBL-ARMS shoulder coordinates.

    Args:
        model: Target model; its ``elv_angle``, ``shoulder_elv`` and
            ``shoulder_rot`` joint axes define the Y-X-Y sequence (ideal OpenSim
            axes are used when a joint is missing).
        r_ht: ``(T, 3, 3)`` humerus rotations in OpenSim thorax axes (X anterior,
            Y up, Z right), identity with the arm hanging at the side.

    Returns:
        ``elv_angle``, ``shoulder_elv`` and ``shoulder_rot`` [rad] for
        :func:`map_upper_limb_to_qpos`.
    """
    jnt_lookup = {model.joint(j).name: j for j in range(model.njnt)}

    def _axis(base: str, default: tuple[float, float, float]) -> np.ndarray:
        jid = next(
            (jnt_lookup[c] for c in (f"{base}_r", base) if c in jnt_lookup), None
        )
        return (
            np.asarray(default, dtype=np.float64)
            if jid is None
            else model.jnt_axis[jid].copy()
        )

    ay = _axis("elv_angle", (0.0, 1.0, 0.0))
    ax = _axis("shoulder_elv", (-1.0, 0.0, 0.0))
    ar = _axis("shoulder_rot", (0.0, 1.0, 0.0))

    def _residual(x: np.ndarray, tgt: np.ndarray) -> np.ndarray:
        r_fit = (
            _axis_rot(ay, x[0])
            @ _axis_rot(ax, x[1])
            @ _axis_rot(ay, -x[0])
            @ _axis_rot(ar, x[2])
        )
        return Rotation.from_matrix(r_fit.T @ tgt).as_rotvec()

    out = np.zeros((len(r_ht), 3), dtype=np.float64)
    starts = [np.array([p, 0.5, 0.0]) for p in (-0.5, 0.5, 1.5)]
    prev = min(
        (least_squares(_residual, s, args=(r_ht[0],)) for s in starts),
        key=lambda s: s.cost,
    ).x
    for t in range(len(r_ht)):
        prev = least_squares(_residual, prev, args=(r_ht[t],), xtol=1e-10, ftol=1e-10).x
        if prev[1] < 0.0:
            # (p + pi, -e, r) is the same rotation; keep elevation non-negative.
            prev = least_squares(
                _residual,
                np.array([prev[0] + np.pi, -prev[1], prev[2]]),
                args=(r_ht[t],),
                xtol=1e-10,
                ftol=1e-10,
            ).x
        prev[0] = (prev[0] + np.pi) % (2.0 * np.pi) - np.pi
        out[t] = prev
    return {
        "elv_angle": out[:, 0],
        "shoulder_elv": out[:, 1],
        "shoulder_rot": out[:, 2],
    }


class Quesada2024UpperLimbSource(EMGSource):
    """Quesada et al. (2024) 12-ch MVC-normalized shoulder and elbow EMG adapter."""

    name = "quesada2024"

    def __init__(
        self,
        data_root: str | Path = "downloads/emg/quesada2024",
        *,
        subjects: Sequence[str] | None = None,
        pattern: str | None = None,
        window_duration_s: float = 8.0,
        kinematics_rate_hz: float = 100.0,
    ) -> None:
        self.data_root = Path(data_root)
        self.zip_path = (
            self.data_root
            if self.data_root.suffix.lower() == ".zip"
            else self.data_root / "data.zip"
        )
        if not self.zip_path.exists() and not self.data_root.exists():
            raise FileNotFoundError(f"Quesada 2024 data not found at {self.data_root}")
        self.window_duration_s = window_duration_s
        self.kinematics_rate_hz = kinematics_rate_hz
        self.pattern = re.compile(pattern) if pattern else None
        self._wanted_subjects = (
            {s.upper() for s in subjects} if subjects is not None else None
        )
        self._ids_cache: tuple[str, ...] | None = None

    def _read_member_text(self, rel_path: str) -> str:
        if self.zip_path.exists():
            with zipfile.ZipFile(self.zip_path) as zf:
                return zf.read(rel_path).decode("utf-8", errors="replace")
        return (self.data_root / rel_path).read_text(encoding="utf-8")

    def trial_ids(self) -> Sequence[str]:
        if self._ids_cache is not None:
            return self._ids_cache
        if self.zip_path.exists():
            with zipfile.ZipFile(self.zip_path) as zf:
                entries = [
                    n[: -len("/emgFilt.csv")]
                    for n in zf.namelist()
                    if n.endswith("/emgFilt.csv") and not n.startswith("__MACOSX")
                ]
        else:
            entries = [
                p.parent.relative_to(self.data_root).as_posix()
                for p in self.data_root.glob("S*/*/*/emgFilt.csv")
            ]

        def _sort_key(tid: str) -> tuple[int, int, int, str]:
            subj, cond, t_tok = tid.split("/")
            s_num = int(re.sub(r"\D", "", subj) or "0")
            t_num = int(re.sub(r"\D", "", t_tok) or "0")
            cond_rank = 0 if cond == "MJ" else 1
            return (0 if t_num == 0 else 1, cond_rank, t_num * 100 + s_num, tid)

        filtered = [
            e
            for e in sorted(entries, key=_sort_key)
            if (
                self._wanted_subjects is None
                or e.split("/")[0].upper() in self._wanted_subjects
            )
            and (self.pattern is None or self.pattern.search(e))
        ]
        self._ids_cache = tuple(filtered)
        return self._ids_cache

    def default_channel_map(self, model: mujoco.MjModel) -> EMGChannelMap:
        del model
        return EMGChannelMap(
            name="quesada2024_upper_limb_12ch",
            version=1,
            bindings=(
                ChannelBinding("DELTAnt", ("DELT1_r",)),
                ChannelBinding("DELTMed", ("DELT2_r",)),
                ChannelBinding("DELTPost", ("DELT3_r",)),
                ChannelBinding("PECT", ("PECM1_r", "PECM2_r", "PECM3_r")),
                ChannelBinding("LATI", ("LAT1_r", "LAT2_r", "LAT3_r")),
                ChannelBinding("TRILong", ("TRIlong_r",)),
                ChannelBinding("TRILat", ("TRIlat_r",)),
                ChannelBinding("TRIMed", ("TRImed_r",)),
                ChannelBinding("BICLong", ("BIClong_r",)),
                ChannelBinding("BICShort", ("BICshort_r",)),
                ChannelBinding("BRA", ("BRA_r",)),
                ChannelBinding("BRD", ("BRD_r",)),
            ),
        )

    def get_trial(self, trial_id: str, model: mujoco.MjModel | None = None) -> EMGTrial:
        if trial_id not in self.trial_ids():
            raise KeyError(f"unknown Quesada 2024 trial {trial_id!r}")
        emg_rows = list(
            csv.reader(io.StringIO(self._read_member_text(f"{trial_id}/emgFilt.csv")))
        )
        pos_rows = list(
            csv.reader(
                io.StringIO(self._read_member_text(f"{trial_id}/humanPositions.csv"))
            )
        )
        trq_rows = list(
            csv.reader(
                io.StringIO(self._read_member_text(f"{trial_id}/muscleTorque.csv"))
            )
        )
        emg_hdr = [h.strip() for h in emg_rows[0][1:]]
        pos_hdr = [h.strip() for h in pos_rows[0][1:]]
        trq_hdr = [h.strip() for h in trq_rows[0][1:]]
        emg_arr = np.asarray(
            [[float(x) for x in r] for r in emg_rows[1:] if any(r)], dtype=np.float64
        )
        pos_arr = np.asarray(
            [[float(x) for x in r] for r in pos_rows[1:] if any(r)], dtype=np.float64
        )
        trq_arr = np.asarray(
            [[float(x) for x in r] for r in trq_rows[1:] if any(r)], dtype=np.float64
        )

        n_min = min(len(emg_arr), len(pos_arr), len(trq_arr))
        emg_arr, pos_arr, trq_arr = emg_arr[:n_min], pos_arr[:n_min], trq_arr[:n_min]
        fs = 1.0 / float(np.median(np.diff(emg_arr[:, 0])))

        if self.window_duration_s > 0:
            win_len = min(n_min, max(64, round(self.window_duration_s * fs)))
            if win_len < n_min:
                speed = np.linalg.norm(np.diff(pos_arr[:, 1:], axis=0), axis=1)
                kernel = np.ones(win_len) / win_len
                start_idx = min(
                    int(np.argmax(np.convolve(speed, kernel, mode="valid"))),
                    n_min - win_len,
                )
            else:
                start_idx = 0
            emg_arr = emg_arr[start_idx : start_idx + win_len]
            pos_arr = pos_arr[start_idx : start_idx + win_len]
            trq_arr = trq_arr[start_idx : start_idx + win_len]
        else:
            start_idx = 0

        rel_time = emg_arr[:, 0] - float(emg_arr[0, 0])
        signals = np.clip(emg_arr[:, 1:], 0.0, 2.0)

        qpos = None
        kin_time = None
        ext_qfrc = None
        ctrl_qadr: list[int] = []
        use_st_contact = False
        if model is not None:
            stride = max(1, round(fs / self.kinematics_rate_hz))
            kin_time = rel_time[::stride].copy()
            coords = {
                col_name: pos_arr[::stride, idx + 1]
                for idx, col_name in enumerate(pos_hdr)
            }
            torques = {
                col_name: trq_arr[::stride, idx + 1]
                for idx, col_name in enumerate(trq_hdr)
            }
            qpos, ctrl_qadr, use_st_contact = map_upper_limb_to_qpos(model, coords)
            dt_kin = float(np.median(np.diff(kin_time)))
            ext_qfrc = map_upper_limb_torques_to_qfrc(
                model, qpos, dt_kin, coords, torques
            )

        subj, cond, _ = trial_id.split("/")
        return EMGTrial(
            trial_id=trial_id,
            time=rel_time,
            signals=signals,
            channel_names=tuple(emg_hdr),
            signal_kind="mvc_normalized",
            qpos=qpos,
            kinematics_time=kin_time,
            external_qfrc=ext_qfrc,
            meta={
                "dataset": self.name,
                "subject": subj,
                "condition": cond,
                "window_start_s": float(start_idx / fs),
                "controlled_qpos_adr": ctrl_qadr,
                "contact_forces_fallback": use_st_contact,
            },
        )


def map_emg2pose_to_hand_qpos(
    model: mujoco.MjModel, joint_angles_20: np.ndarray
) -> tuple[np.ndarray, list[int]]:
    """Map ``(T, 20)`` ``emg2pose`` UmeTrack joint angles [rad] onto *model* ``qpos``."""
    angles = np.asarray(joint_angles_20, dtype=np.float64)
    if angles.ndim != 2 or angles.shape[1] != len(EMG2POSE_JOINT_NAMES):
        raise ValueError(f"expected (T, 20) joint_angles, got {angles.shape}")
    jnt_lookup = {model.joint(j).name: j for j in range(model.njnt)}
    qpos = np.tile(model.qpos0, (len(angles), 1))
    controlled_qadr: list[int] = []

    for col, base_name in enumerate(EMG2POSE_JOINT_NAMES):
        jid = next(
            (
                jnt_lookup[cand]
                for cand in (f"{base_name}_r", base_name, f"{base_name}_l")
                if cand in jnt_lookup
            ),
            None,
        )
        if jid is None:
            continue
        qadr = int(model.jnt_qposadr[jid])
        vals = angles[:, col]
        if model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]
            vals = np.clip(vals, lo, hi)
        qpos[:, qadr] = vals
        controlled_qadr.append(qadr)

    project_joint_couplings(JointCouplings.from_model(model), qpos)
    qpos = project_tendon_nodes(model, qpos)
    return qpos, sorted(controlled_qadr)


class Emg2PoseHandSource(EMGSource):
    """Meta ``emg2pose`` 2 kHz 16-channel wrist sEMG and 20-DOF hand pose adapter."""

    name = "emg2pose"

    def __init__(
        self,
        data_root: str | Path = "downloads/emg/emg2pose",
        *,
        window_duration_s: float = 4.0,
        kinematics_rate_hz: float = 100.0,
        prefer_side: str | None = "right",
    ) -> None:
        root = Path(data_root)
        sub = root / "emg2pose_dataset_mini"
        self.hdf5_dir = sub if sub.exists() else root
        files = sorted(self.hdf5_dir.glob("*.hdf5"))
        if not files:
            raise FileNotFoundError(f"no .hdf5 recordings found under {self.hdf5_dir}")
        if prefer_side in ("right", "left"):
            preferred = [f for f in files if f.stem.endswith(f"_{prefer_side}")]
            others = [f for f in files if not f.stem.endswith(f"_{prefer_side}")]
            files = preferred + others
        self._files = {f.stem: f for f in files}
        self.window_duration_s = window_duration_s
        self.kinematics_rate_hz = kinematics_rate_hz

    def trial_ids(self) -> Sequence[str]:
        return tuple(self._files.keys())

    def default_channel_map(self, model: mujoco.MjModel) -> EMGChannelMap | None:
        del model
        return None

    def get_trial(self, trial_id: str, model: mujoco.MjModel | None = None) -> EMGTrial:
        if trial_id not in self._files:
            raise KeyError(f"unknown emg2pose recording {trial_id!r}")
        import h5py

        path = self._files[trial_id]
        with h5py.File(path, "r", locking=False) as hf:
            grp = hf["emg2pose"]
            attrs = {
                k: (v.item() if hasattr(v, "item") else v) for k, v in grp.attrs.items()
            }
            ts = grp["timeseries"]
            fs = float(attrs.get("sample_rate", 2000.0))
            n_total = len(ts)
            win_len = min(n_total, max(64, round(self.window_duration_s * fs)))

            if win_len < n_total:
                stride_scan = max(1, round(fs / 50.0))
                coarse_pose = np.asarray(
                    ts["joint_angles"][::stride_scan], dtype=np.float64
                )
                speed = np.linalg.norm(np.diff(coarse_pose, axis=0), axis=1)
                win_coarse = max(1, win_len // stride_scan)
                if len(speed) > win_coarse:
                    kernel = np.ones(win_coarse) / win_coarse
                    best_c = int(np.argmax(np.convolve(speed, kernel, mode="valid")))
                    start_idx = min(best_c * stride_scan, n_total - win_len)
                else:
                    start_idx = 0
            else:
                start_idx = 0

            chunk = ts[start_idx : start_idx + win_len]
            raw_time = np.asarray(chunk["time"], dtype=np.float64)
            rel_time = raw_time - float(raw_time[0])
            if np.any(np.diff(rel_time) <= 0):
                rel_time = np.arange(len(chunk), dtype=np.float64) / fs
            emg_raw = np.asarray(chunk["emg"], dtype=np.float64)
            pose_20 = np.asarray(chunk["joint_angles"], dtype=np.float64)

        qpos = None
        kin_time = None
        ctrl_qadr: list[int] = []
        if model is not None:
            stride = max(1, round(fs / self.kinematics_rate_hz))
            kin_time = rel_time[::stride].copy()
            qpos, ctrl_qadr = map_emg2pose_to_hand_qpos(model, pose_20[::stride])

        return EMGTrial(
            trial_id=trial_id,
            time=rel_time,
            signals=emg_raw,
            channel_names=tuple(f"emg_{c:02d}" for c in range(emg_raw.shape[1])),
            signal_kind="raw",
            qpos=qpos,
            kinematics_time=kin_time,
            filter_config=EMGFilterConfig(
                bandpass_hz=(20.0, 450.0),
                lowpass_hz=10.0,
                normalize="peak",
            ),
            meta={
                "dataset": self.name,
                "stage": str(attrs.get("stage", "")),
                "side": str(attrs.get("side", "")),
                "user": str(attrs.get("user", "")),
                "split": str(attrs.get("split", "")),
                "window_start_s": float(start_idx / fs),
                "controlled_qpos_adr": ctrl_qadr,
            },
        )


# CyberGlove II 22-sensor column -> anatomical hand/wrist joint (Atzori et al., Sci. Data 2014).
_CYBERGLOVE_JOINT_MAP: tuple[tuple[int, str, float], ...] = (
    (0, "cmc_abduction", 1.0),
    (3, "cmc_flexion", 1.0),
    (1, "mp_flexion", 1.0),
    (2, "ip_flexion", 1.0),
    (4, "mcp2_flexion", 1.0),
    (5, "pm2_flexion", 1.0),
    (6, "md2_flexion", 1.0),
    (9, "mcp2_abduction", 0.4),
    (7, "mcp3_flexion", 1.0),
    (8, "pm3_flexion", 1.0),
    (10, "md3_flexion", 1.0),
    (13, "mcp3_abduction", 0.4),
    (11, "mcp4_flexion", 1.0),
    (12, "pm4_flexion", 1.0),
    (17, "mcp4_abduction", 0.4),
    (14, "mcp5_flexion", 1.0),
    (15, "pm5_flexion", 1.0),
    (16, "md5_flexion", 1.0),
    (19, "flexion", 0.6),
    (20, "deviation", 0.6),
    (21, "pro_sup", 0.6),
)


def map_cyberglove_to_hand_qpos(
    model: mujoco.MjModel,
    glove_window: np.ndarray,
    rest_median: np.ndarray,
    sensor_span: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    """Map uncalibrated 22-sensor CyberGlove II counts onto *model* ``qpos``."""
    jnt_lookup = {model.joint(j).name: j for j in range(model.njnt)}
    qpos = np.tile(model.qpos0, (len(glove_window), 1))
    ctrl_qadr: list[int] = []

    for col, base_name, gain in _CYBERGLOVE_JOINT_MAP:
        jid = next(
            (jnt_lookup[c] for c in (f"{base_name}_r", base_name) if c in jnt_lookup),
            None,
        )
        if jid is None:
            continue
        qadr = int(model.jnt_qposadr[jid])
        span = float(sensor_span[col])
        if col == 10 and span > 180.0:
            # Sensor 11 wraps on several NinaPro DB5 recordings; couple DIP3 to PIP3.
            norm_delta = (glove_window[:, 8] - rest_median[8]) / max(
                float(sensor_span[8]), 25.0
            )
            gain_eff = 0.67
        else:
            norm_delta = (glove_window[:, col] - rest_median[col]) / max(span, 25.0)
            gain_eff = gain

        if model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]
            q0 = float(model.qpos0[qadr])
            pos_room = max(float(hi) - q0, 0.2)
            neg_room = max(q0 - float(lo), 0.1)
            vals = q0 + np.where(
                norm_delta >= 0.0,
                gain_eff * 0.85 * pos_room * norm_delta,
                gain_eff * 0.85 * neg_room * norm_delta,
            )
            vals = np.clip(vals, lo, hi)
        else:
            vals = model.qpos0[qadr] + gain_eff * norm_delta
        qpos[:, qadr] = vals
        ctrl_qadr.append(qadr)

    for dip_base, pip_base in (("md4_flexion", "pm4_flexion"),):
        jid_dip = next(
            (jnt_lookup[c] for c in (f"{dip_base}_r", dip_base) if c in jnt_lookup),
            None,
        )
        jid_pip = next(
            (jnt_lookup[c] for c in (f"{pip_base}_r", pip_base) if c in jnt_lookup),
            None,
        )
        if jid_dip is not None and jid_pip is not None:
            q_dip = int(model.jnt_qposadr[jid_dip])
            q_pip = int(model.jnt_qposadr[jid_pip])
            vals = 0.67 * qpos[:, q_pip]
            if model.jnt_limited[jid_dip]:
                vals = np.clip(vals, *model.jnt_range[jid_dip])
            qpos[:, q_dip] = vals
            ctrl_qadr.append(q_dip)

    project_joint_couplings(JointCouplings.from_model(model), qpos)
    qpos = project_tendon_nodes(model, qpos)
    return qpos, sorted(set(ctrl_qadr))


class NinaProDB5Source(EMGSource):
    """NinaPro DB5 16-ch dual-Myo forearm sEMG and 22-sensor CyberGlove II adapter."""

    name = "ninapro_db5"

    def __init__(
        self,
        data_root: str | Path = "downloads/emg/ninapro_db5",
        *,
        subjects: Sequence[str] | None = None,
        pattern: str | None = None,
        kinematics_rate_hz: float = 100.0,
    ) -> None:
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(
                f"NinaPro DB5 data_root not found: {self.data_root}"
            )
        self.kinematics_rate_hz = kinematics_rate_hz
        self.pattern = re.compile(pattern) if pattern else None

        zips = sorted(
            self.data_root.glob("s*.zip"),
            key=lambda p: int(re.sub(r"\D", "", p.stem) or "0"),
        )
        if not zips:
            raise FileNotFoundError(
                f"no s1.zip..s10.zip archives found under {self.data_root}"
            )
        if subjects is not None:
            wanted = {s.lower() for s in subjects}
            zips = [z for z in zips if z.stem.lower() in wanted]
        if not zips:
            raise FileNotFoundError(
                f"requested subjects {subjects} not found in {self.data_root}"
            )
        self.subjects = tuple(z.stem for z in zips)
        self._cache: dict[tuple[str, int], dict[str, Any]] = {}
        self._ids_cache: tuple[str, ...] | None = None

    def _load_exercise(self, subj: str, ex: int) -> dict[str, Any]:
        key = (subj, ex)
        if key not in self._cache:
            if len(self._cache) >= 3:
                self._cache.clear()
            num = int(re.sub(r"\D", "", subj))
            zip_path = self.data_root / f"{subj}.zip"
            member = f"{subj}/S{num}_E{ex}_A1.mat"
            with zipfile.ZipFile(zip_path) as zf:
                mat = sio.loadmat(io.BytesIO(zf.read(member)), simplify_cells=True)
            glove = np.asarray(mat["glove"], dtype=np.float64)
            restim = np.asarray(mat["restimulus"]).ravel()
            rest_med = np.median(glove[restim == 0], axis=0)
            span = np.percentile(glove, 98, axis=0) - np.percentile(glove, 2, axis=0)
            self._cache[key] = {
                "emg": np.asarray(mat["emg"], dtype=np.float64),
                "glove": glove,
                "restim": restim,
                "rerep": np.asarray(mat["rerepetition"]).ravel(),
                "fs": float(mat.get("frequency", 200.0)),
                "rest_med": rest_med,
                "span": span,
            }
        return self._cache[key]

    def trial_ids(self) -> Sequence[str]:
        if self._ids_cache is not None:
            return self._ids_cache
        prio: list[str] = []
        rest: list[str] = []
        # Representative grasps (E3), isometric/wrist postures (E2), and finger flexions (E1).
        featured = ((3, 1), (3, 6), (2, 2), (2, 6), (1, 1), (1, 5))
        for subj in self.subjects:
            for ex, stim in featured:
                prio.append(f"{subj}/E{ex}/stim_{stim:02d}_rep_01")
            for ex in (3, 2, 1):
                n_stim = 23 if ex == 3 else (17 if ex == 2 else 12)
                for stim in range(1, n_stim + 1):
                    tid = f"{subj}/E{ex}/stim_{stim:02d}_rep_01"
                    if tid not in prio:
                        rest.append(tid)
        all_ids = prio + rest
        if self.pattern is not None:
            all_ids = [t for t in all_ids if self.pattern.search(t)]
        self._ids_cache = tuple(all_ids)
        return self._ids_cache

    def default_channel_map(self, model: mujoco.MjModel) -> EMGChannelMap | None:
        del model
        return None

    def get_trial(self, trial_id: str, model: mujoco.MjModel | None = None) -> EMGTrial:
        parts = trial_id.split("/")
        if len(parts) != 3:
            raise KeyError(f"invalid NinaPro trial_id {trial_id!r}")
        subj, ex_tok, sr_tok = parts
        ex = int(ex_tok[1:])
        m_sr = re.match(r"stim_(\d+)_rep_(\d+)", sr_tok)
        if m_sr is None:
            raise KeyError(f"invalid NinaPro stimulus token {sr_tok!r}")
        stim, rep = int(m_sr.group(1)), int(m_sr.group(2))

        rec = self._load_exercise(subj, ex)
        fs = float(rec["fs"])
        mask = (rec["restim"] == stim) & (rec["rerep"] == rep)
        if not np.any(mask):
            mask = rec["restim"] == stim
        idx = np.flatnonzero(mask)
        if len(idx) < 16:
            raise KeyError(f"stimulus {stim} rep {rep} absent in {subj} E{ex}")
        pad = round(0.35 * fs)
        i0 = max(0, int(idx[0]) - pad)
        i1 = min(len(rec["emg"]), int(idx[-1]) + pad)

        emg_win = rec["emg"][i0:i1]
        glove_win = rec["glove"][i0:i1]
        rel_time = np.arange(len(emg_win), dtype=np.float64) / fs

        qpos = None
        kin_time = None
        ctrl_qadr: list[int] = []
        if model is not None:
            stride = max(1, round(fs / self.kinematics_rate_hz))
            kin_time = rel_time[::stride].copy()
            qpos, ctrl_qadr = map_cyberglove_to_hand_qpos(
                model, glove_win[::stride], rec["rest_med"], rec["span"]
            )

        return EMGTrial(
            trial_id=trial_id,
            time=rel_time,
            signals=emg_win,
            channel_names=tuple(f"myo_{c:02d}" for c in range(emg_win.shape[1])),
            signal_kind="raw",
            qpos=qpos,
            kinematics_time=kin_time,
            filter_config=EMGFilterConfig(
                bandpass_hz=(15.0, 90.0),
                lowpass_hz=5.0,
                normalize="peak",
            ),
            meta={
                "dataset": self.name,
                "subject": subj,
                "exercise": ex,
                "stimulus": stim,
                "repetition": rep,
                "controlled_qpos_adr": ctrl_qadr,
            },
        )


_WFDB_GAIN_RE = re.compile(r"^([+-]?[0-9.eE+-]+)(?:\(([+-]?[0-9.eE+-]+)\))?")

_GRABMYO_SEQUENCES: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("wrist_forearm", (11, 12, 13, 14)),
    ("grasp_opposition", (15, 16, 1, 4, 3, 2)),
    ("finger_extension", (9, 7, 8, 10, 5)),
)

_GRABMYO_GESTURE_TARGETS: dict[int, dict[str, float]] = {
    1: {
        "cmc_abduction": -0.35,
        "cmc_flexion": -0.45,
        "mp_flexion": -0.50,
        "ip_flexion": -0.40,
        "mcp2_flexion": 0.55,
        "pm2_flexion": 0.60,
        "md2_flexion": 0.45,
        "mcp3_flexion": 0.55,
        "pm3_flexion": 0.60,
        "mcp4_flexion": 0.55,
        "pm4_flexion": 0.60,
        "mcp5_flexion": 0.55,
        "pm5_flexion": 0.60,
    },
    2: {"cmc_abduction": -0.55, "cmc_flexion": -0.30},
    3: {
        "cmc_abduction": 0.75,
        "cmc_flexion": -0.75,
        "mp_flexion": -0.45,
        "cmc5_flexion": 0.75,
        "mcp5_flexion": 0.65,
        "pm5_flexion": 0.55,
    },
    4: {
        "cmc_abduction": 0.70,
        "cmc_flexion": -0.60,
        "mp_flexion": -0.45,
        "ip_flexion": -0.35,
        "mcp2_flexion": 0.60,
        "pm2_flexion": 0.55,
        "md2_flexion": 0.40,
    },
    5: {
        "cmc_abduction": -0.60,
        "mp_flexion": 0.70,
        "ip_flexion": 0.70,
        "mcp2_flexion": -0.65,
        "mcp3_flexion": 0.35,
        "pm3_flexion": 0.40,
        "mcp4_flexion": 0.35,
        "pm4_flexion": 0.40,
        "mcp5_flexion": 0.35,
        "pm5_flexion": 0.40,
    },
    6: {
        "cmc_abduction": -0.60,
        "mp_flexion": 0.70,
        "mcp5_flexion": -0.65,
        "mcp2_flexion": 0.35,
        "pm2_flexion": 0.40,
        "mcp3_flexion": 0.35,
        "pm3_flexion": 0.40,
        "mcp4_flexion": 0.35,
        "pm4_flexion": 0.40,
    },
    7: {
        "mcp2_flexion": -0.65,
        "mcp3_flexion": -0.65,
        "mcp4_flexion": 0.40,
        "pm4_flexion": 0.45,
        "mcp5_flexion": 0.40,
        "pm5_flexion": 0.45,
    },
    8: {
        "mcp5_flexion": -0.70,
        "mcp2_flexion": 0.35,
        "mcp3_flexion": 0.35,
        "mcp4_flexion": 0.35,
    },
    9: {
        "mcp2_flexion": -0.70,
        "mcp3_flexion": 0.35,
        "mcp4_flexion": 0.35,
        "mcp5_flexion": 0.35,
    },
    10: {"cmc_abduction": -0.65, "mp_flexion": 0.75, "ip_flexion": 0.75},
    11: {"flexion": -0.65},
    12: {"flexion": 0.65},
    13: {"pro_sup": -0.65},
    14: {"pro_sup": 0.65},
    15: {
        "cmc_abduction": -0.55,
        "mp_flexion": 0.65,
        "ip_flexion": 0.65,
        "mcp2_flexion": -0.60,
        "mcp3_flexion": -0.60,
        "mcp4_flexion": -0.60,
        "mcp5_flexion": -0.60,
        "mcp2_abduction": 0.50,
        "mcp5_abduction": -0.50,
    },
    16: {
        "cmc_abduction": 0.45,
        "cmc_flexion": -0.55,
        "mp_flexion": -0.65,
        "ip_flexion": -0.65,
        "mcp2_flexion": 0.75,
        "pm2_flexion": 0.75,
        "md2_flexion": 0.65,
        "mcp3_flexion": 0.75,
        "pm3_flexion": 0.75,
        "md3_flexion": 0.65,
        "mcp4_flexion": 0.75,
        "pm4_flexion": 0.75,
        "md4_flexion": 0.65,
        "mcp5_flexion": 0.75,
        "pm5_flexion": 0.75,
        "md5_flexion": 0.65,
    },
}


def read_wfdb_record(hea_path: str | Path) -> tuple[float, tuple[str, ...], np.ndarray]:
    """Read a format-16 PhysioNet WFDB ``.hea`` / ``.dat`` pair in physical units."""
    hp = Path(hea_path)
    lines = [
        ln.strip()
        for ln in hp.read_text(encoding="utf-8", errors="replace").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    hdr = lines[0].split()
    n_sig, fs, n_samples = int(hdr[1]), float(hdr[2]), int(hdr[3])
    gains = np.ones(n_sig, dtype=np.float64)
    baselines = np.zeros(n_sig, dtype=np.float64)
    names: list[str] = []
    dat_name = hp.with_suffix(".dat").name

    for idx in range(n_sig):
        tok = lines[idx + 1].split()
        dat_name = tok[0]
        if int(tok[1]) != 16:
            raise ValueError(f"{hp}: unsupported WFDB signal format {tok[1]!r}")
        m_gain = _WFDB_GAIN_RE.match(tok[2])
        if m_gain is not None:
            g_val = float(m_gain.group(1))
            gains[idx] = g_val if g_val != 0.0 else 1.0
            if m_gain.group(2) is not None:
                baselines[idx] = float(m_gain.group(2))
        names.append(tok[-1])

    raw = np.frombuffer((hp.parent / dat_name).read_bytes(), dtype="<i2")
    raw_2d = raw[: n_samples * n_sig].reshape(n_samples, n_sig)
    signals = (raw_2d.astype(np.float64) - baselines) / gains
    return fs, tuple(names), signals


def map_grabmyo_gestures_to_qpos(
    model: mujoco.MjModel,
    kin_time: np.ndarray,
    active_segments: Sequence[tuple[int, float, float]],
) -> tuple[np.ndarray, list[int]]:
    """Synthesize smooth anatomical joint trajectories aligned to GRABMyo gesture windows."""
    jnt_lookup = {model.joint(j).name: j for j in range(model.njnt)}
    qpos = np.tile(model.qpos0, (len(kin_time), 1))
    ctrl_qadr: set[int] = set()

    for gest_id, t_start, t_end in active_segments:
        targets = _GRABMYO_GESTURE_TARGETS.get(gest_id, {})
        dur = max(t_end - t_start, 1e-6)
        ramp = min(0.22, 0.35 * dur)
        env = np.zeros_like(kin_time)
        in_win = (kin_time >= t_start) & (kin_time <= t_end)
        tau = kin_time[in_win] - t_start
        w = np.ones_like(tau)
        rise = tau < ramp
        fall = tau > dur - ramp
        w[rise] = 0.5 * (1.0 - np.cos(np.pi * tau[rise] / ramp))
        w[fall] = 0.5 * (1.0 - np.cos(np.pi * (dur - tau[fall]) / ramp))
        env[in_win] = w

        for base_name, norm_exc in targets.items():
            jid = next(
                (
                    jnt_lookup[c]
                    for c in (f"{base_name}_r", base_name)
                    if c in jnt_lookup
                ),
                None,
            )
            if jid is None:
                continue
            qadr = int(model.jnt_qposadr[jid])
            q0 = float(model.qpos0[qadr])
            if model.jnt_limited[jid]:
                lo, hi = (float(v) for v in model.jnt_range[jid])
                span = (hi - q0) if norm_exc >= 0.0 else (q0 - lo)
                delta = norm_exc * max(span, 0.0)
                qpos[:, qadr] = np.clip(qpos[:, qadr] + delta * env, lo, hi)
            else:
                qpos[:, qadr] = qpos[:, qadr] + norm_exc * env
            ctrl_qadr.add(qadr)

    project_joint_couplings(JointCouplings.from_model(model), qpos)
    qpos = project_tendon_nodes(model, qpos)
    return qpos, sorted(ctrl_qadr)


class GrabMyoSource(EMGSource):
    """PhysioNet GRABMyo 28-ch 2048 Hz forearm+wrist sEMG multi-gesture adapter."""

    name = "grabmyo"

    def __init__(
        self,
        data_root: str | Path = "downloads/emg/grabmyo",
        *,
        subjects: Sequence[str] | None = None,
        pattern: str | None = None,
        kinematics_rate_hz: float = 100.0,
        rest_slice_s: float = 0.45,
        gesture_slice_s: float = 0.90,
    ) -> None:
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"GRABMyo data_root not found: {self.data_root}")
        self.kinematics_rate_hz = kinematics_rate_hz
        self.rest_slice_s = rest_slice_s
        self.gesture_slice_s = gesture_slice_s
        self.pattern = re.compile(pattern) if pattern else None

        wanted: set[str] | None = None
        if subjects is not None:
            wanted = set()
            for s in subjects:
                sl = s.strip().lower()
                digits = re.sub(r"\D", "", sl)
                wanted.add(
                    sl if sl.startswith("participant") else f"participant{digits}"
                )
        rec_re = re.compile(
            r"^session(\d+)_(participant\d+)_gesture(\d+)_trial(\d+)$", re.IGNORECASE
        )
        self._records: dict[tuple[str, str, int], dict[int, Path]] = {}
        for hea in sorted(self.data_root.rglob("*.hea")):
            m = rec_re.match(hea.stem)
            if m is None or not hea.with_suffix(".dat").exists():
                continue
            sess = f"Session{int(m.group(1))}"
            part = m.group(2).lower()
            if wanted is not None and part not in wanted:
                continue
            gest, rep = int(m.group(3)), int(m.group(4))
            self._records.setdefault((sess, part, rep), {})[gest] = hea

        if not self._records:
            raise FileNotFoundError(
                f"no WFDB gesture records found under {self.data_root}"
            )
        self._ids_cache: tuple[str, ...] | None = None

    def trial_ids(self) -> Sequence[str]:
        if self._ids_cache is not None:
            return self._ids_cache
        keys = sorted(
            self._records.keys(),
            key=lambda k: (
                int(re.sub(r"\D", "", k[0]) or "0"),
                k[2],
                int(re.sub(r"\D", "", k[1]) or "0"),
            ),
        )
        prio: list[str] = []
        rest: list[str] = []
        for seq_name, gest_ids in _GRABMYO_SEQUENCES:
            for sess, part, rep in keys:
                avail = self._records[(sess, part, rep)]
                if 17 in avail and sum(g in avail for g in gest_ids) >= 2:
                    tid = f"{sess}/{part}/trial{rep}_{seq_name}"
                    if rep == 1:
                        prio.append(tid)
                    else:
                        rest.append(tid)
        all_ids = prio + rest
        if self.pattern is not None:
            all_ids = [t for t in all_ids if self.pattern.search(t)]
        self._ids_cache = tuple(all_ids)
        return self._ids_cache

    def default_channel_map(self, model: mujoco.MjModel) -> EMGChannelMap | None:
        del model
        return None

    @staticmethod
    def _concat_with_crossfade(
        chunks: Sequence[np.ndarray], fade: int = 32
    ) -> np.ndarray:
        if len(chunks) == 1:
            return chunks[0].copy()
        out = chunks[0].copy()
        for nxt in chunks[1:]:
            k = min(fade, len(out) // 4, len(nxt) // 4)
            if k > 1:
                w = np.linspace(0.0, 1.0, k)[:, None]
                blend = (1.0 - w) * out[-k:] + w * nxt[:k]
                out = np.vstack([out[:-k], blend, nxt[k:]])
            else:
                out = np.vstack([out, nxt])
        return out

    def get_trial(self, trial_id: str, model: mujoco.MjModel | None = None) -> EMGTrial:
        parts = trial_id.split("/")
        if len(parts) != 3 or "_" not in parts[2]:
            raise KeyError(f"invalid GRABMyo trial_id {trial_id!r}")
        sess, part, rep_seq = parts
        rep_tok, seq_name = rep_seq.split("_", 1)
        rep = int(re.sub(r"\D", "", rep_tok) or "0")
        gest_map = self._records.get((sess, part, rep))
        seq_spec = dict(_GRABMYO_SEQUENCES).get(seq_name)
        if gest_map is None or seq_spec is None or 17 not in gest_map:
            raise KeyError(f"unknown GRABMyo trial {trial_id!r}")

        fs, raw_names, rest_sig = read_wfdb_record(gest_map[17])
        keep_cols = [
            i for i, nm in enumerate(raw_names) if not nm.upper().startswith("U")
        ]
        ch_names = tuple(raw_names[i] for i in keep_cols)
        rest_emg = rest_sig[:, keep_cols]

        n_rest = max(32, min(len(rest_emg), round(self.rest_slice_s * fs)))
        r_start = max(0, (len(rest_emg) - n_rest) // 2)
        rest_chunk = rest_emg[r_start : r_start + n_rest]

        fade = min(32, n_rest // 4)
        chunks: list[np.ndarray] = [rest_chunk]
        cur_samples = len(rest_chunk)
        active_segments: list[tuple[int, float, float]] = []

        for gid in seq_spec:
            if gid not in gest_map:
                continue
            _, _, g_sig = read_wfdb_record(gest_map[gid])
            g_emg = g_sig[:, keep_cols]
            n_act = max(64, min(len(g_emg), round(self.gesture_slice_s * fs)))
            g_start = max(0, (len(g_emg) - n_act) // 2)
            act_chunk = g_emg[g_start : g_start + n_act]

            seg_i0 = cur_samples - fade
            seg_i1 = seg_i0 + len(act_chunk)
            active_segments.append((gid, seg_i0 / fs, seg_i1 / fs))
            chunks.append(act_chunk)
            cur_samples = seg_i1

            chunks.append(rest_chunk)
            cur_samples = cur_samples - fade + len(rest_chunk)

        emg_full = self._concat_with_crossfade(chunks, fade=fade)
        rel_time = np.arange(len(emg_full), dtype=np.float64) / fs

        qpos = None
        kin_time = None
        ctrl_qadr: list[int] = []
        if model is not None:
            stride = max(1, round(fs / self.kinematics_rate_hz))
            kin_time = rel_time[::stride].copy()
            qpos, ctrl_qadr = map_grabmyo_gestures_to_qpos(
                model, kin_time, active_segments
            )

        return EMGTrial(
            trial_id=trial_id,
            time=rel_time,
            signals=emg_full,
            channel_names=ch_names,
            signal_kind="raw",
            qpos=qpos,
            kinematics_time=kin_time,
            filter_config=EMGFilterConfig(
                bandpass_hz=(20.0, 450.0),
                notch_hz=60.0,
                lowpass_hz=10.0,
                normalize="peak",
            ),
            meta={
                "dataset": self.name,
                "session": sess,
                "participant": part,
                "repetition": rep,
                "sequence": seq_name,
                "gestures": [g for g, _, _ in active_segments],
                "controlled_qpos_adr": ctrl_qadr,
            },
        )


# Delsys Trigno sensors S4-S6 on the right arm (Fritsche et al., 2026, Table 2).
ULTRAMOCAP_CHANNELS: dict[str, str] = {
    "IM EMG4": "biceps_brachii",
    "IM EMG5": "triceps_long",
    "IM EMG6": "deltoid_middle",
}

# Bilateral upper-extremity and trunk model (Chen et al.): oblique right SC_z axis.
_BUET_SC_Z_AXIS = np.array([0.0873, 0.0, 0.57446378]) / np.linalg.norm(
    [0.0873, 0.0, 0.57446378]
)


def _mean_speed_window(
    angles: np.ndarray, rate_hz: float, duration_s: float
) -> tuple[int, int]:
    """Start and length [frames] of the window with the fastest joint motion."""
    n = len(angles)
    win = min(n, max(8, round(duration_s * rate_hz)))
    if win >= n:
        return 0, n
    speed = np.linalg.norm(np.diff(angles, axis=0), axis=1)
    start = int(np.argmax(np.convolve(speed, np.ones(win) / win, mode="valid")))
    return min(start, n - win), win


def buet_arm_to_upper_limb_coords(
    model: mujoco.MjModel, ik_deg: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Map right-arm coordinates of the bilateral upper-extremity/trunk model [deg].

    The thoracohumeral rotation is rebuilt from the sternoclavicular
    (``SC_y``, ``SC_x``, ``SC_z``) and glenohumeral (``arm_flex_r``,
    ``arm_add_r``, ``arm_rot_r``) chain and decomposed into MoBL-ARMS angles.
    ``pro_sup_r`` is zero in full supination there, so it is shifted by -90 deg
    onto the MyoSuite neutral-zero, pronation-positive convention.
    """
    rad = {k: np.radians(np.asarray(v, dtype=np.float64)) for k, v in ik_deg.items()}
    n = len(rad["arm_flex_r"])
    r_ht = np.empty((n, 3, 3), dtype=np.float64)
    e_x, e_y, e_z = np.eye(3)
    for t in range(n):
        r_ht[t] = (
            _axis_rot(e_y, rad["SC_y"][t])
            @ _axis_rot(e_x, rad["SC_x"][t])
            @ _axis_rot(_BUET_SC_Z_AXIS, rad["SC_z"][t])
            @ _axis_rot(e_z, rad["arm_flex_r"][t])
            @ _axis_rot(e_x, rad["arm_add_r"][t])
            @ _axis_rot(e_y, rad["arm_rot_r"][t])
        )
    coords = thoracohumeral_to_upper_limb_coords(model, r_ht)
    coords["elbow_flexion"] = rad["elbow_flex_r"]
    coords["pro_sup"] = rad["pro_sup_r"] - 0.5 * np.pi
    return coords


class UltraMoCap2025Source(EMGSource):
    """ULTRA-MoCap right-arm biceps, triceps and middle deltoid sEMG with IK.

    Fritsche, O., Camacho, S., Hossain, M. S. B., Halfpenny, T., Arciniegas, C.,
    Dranetz, J., Hadley, D., Guo, Z. & Choi, H. (2026). ULTRA-MoCap: A
    multimodal IMU and sEMG dataset for upper body joint kinematics analysis.
    Scientific Data 13, 622. https://doi.org/10.1038/s41597-026-06687-5.
    Data: https://doi.org/10.6084/m9.figshare.28741943 (CC BY 4.0).

    13 participants, five arm tasks at several speeds, 2 kHz Delsys Trigno sEMG
    synchronised with 100 Hz Vicon OpenSim IK (``Processed IK/*.csv``).
    """

    name = "ultramocap2025"

    def __init__(
        self,
        data_root: str | Path = "downloads/emg/ultramocap2025",
        *,
        subjects: Sequence[str] | None = None,
        pattern: str | None = None,
        window_duration_s: float = 8.0,
        kinematics_rate_hz: float = 100.0,
    ) -> None:
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(
                f"ULTRA-MoCap data_root not found: {self.data_root}"
            )
        self.window_duration_s = window_duration_s
        self.kinematics_rate_hz = kinematics_rate_hz
        self.pattern = re.compile(pattern) if pattern else None
        wanted = {s.upper() for s in subjects} if subjects is not None else None
        stem_re = re.compile(r"^(P\d+)_([A-Za-z]+)_([A-Za-z0-9]+)$")
        self._files: dict[str, tuple[Path, Path]] = {}
        for emg in sorted(self.data_root.glob("P*/*/EMG/*.csv")):
            m = stem_re.match(emg.stem)
            ik = emg.parent.parent / "Processed IK" / emg.name
            if m is None or not ik.exists():
                continue
            if wanted is not None and m.group(1).upper() not in wanted:
                continue
            self._files["/".join(m.groups())] = (emg, ik)
        if not self._files:
            raise FileNotFoundError(
                f"no P*/<task>/EMG/*.csv with Processed IK under {self.data_root}"
            )
        self._ids_cache: tuple[str, ...] | None = None

    def trial_ids(self) -> Sequence[str]:
        if self._ids_cache is not None:
            return self._ids_cache

        def _key(tid: str) -> tuple[int, int, str]:
            subj, _, cond = tid.split("/")
            return (0 if cond == "Normal" else 1, int(subj[1:]), tid)

        ids = sorted(self._files, key=_key)
        if self.pattern is not None:
            ids = [t for t in ids if self.pattern.search(t)]
        self._ids_cache = tuple(ids)
        return self._ids_cache

    def default_channel_map(self, model: mujoco.MjModel) -> EMGChannelMap:
        del model
        return EMGChannelMap(
            name="ultramocap2025_right_arm_3ch",
            version=1,
            bindings=(
                ChannelBinding("biceps_brachii", ("BIClong_r", "BICshort_r")),
                ChannelBinding("triceps_long", ("TRIlong_r",)),
                ChannelBinding("deltoid_middle", ("DELT2_r",)),
            ),
        )

    def get_trial(self, trial_id: str, model: mujoco.MjModel | None = None) -> EMGTrial:
        if trial_id not in self._files:
            raise KeyError(f"unknown ULTRA-MoCap trial {trial_id!r}")
        emg_path, ik_path = self._files[trial_id]
        with emg_path.open(encoding="utf-8") as fh:
            hdr = [h.strip() for h in fh.readline().split(",")]
            emg = np.loadtxt(fh, delimiter=",", ndmin=2)
        keep = [i for i, h in enumerate(hdr) if h in ULTRAMOCAP_CHANNELS]
        # Frame numbers count 100 Hz mocap frames from 1; sub-frames are 2 kHz samples.
        emg_time = (emg[:, 0] - 1.0) / 100.0 + emg[:, 1] / 2000.0
        with ik_path.open(encoding="utf-8") as fh:
            ik_hdr = [h.strip() for h in fh.readline().split(",")]
            ik = np.loadtxt(fh, delimiter=",", ndmin=2)
        ik_time = ik[:, 0]

        t_lo = max(float(emg_time[0]), float(ik_time[0]))
        t_hi = min(float(emg_time[-1]), float(ik_time[-1]))
        in_ik = (ik_time >= t_lo) & (ik_time <= t_hi)
        ik, ik_time = ik[in_ik], ik_time[in_ik]
        arm_cols = [
            ik_hdr.index(c)
            for c in ("arm_flex_r", "arm_add_r", "arm_rot_r", "elbow_flex_r")
        ]
        ik_rate = 1.0 / float(np.median(np.diff(ik_time)))
        start, win = (
            _mean_speed_window(ik[:, arm_cols], ik_rate, self.window_duration_s)
            if self.window_duration_s > 0
            else (0, len(ik))
        )
        ik, ik_time = ik[start : start + win], ik_time[start : start + win]
        t0, t1 = float(ik_time[0]), float(ik_time[-1])
        in_emg = (emg_time >= t0 - 1e-9) & (emg_time <= t1 + 1e-9)

        qpos = None
        kin_time = None
        ctrl_qadr: list[int] = []
        if model is not None:
            kin_time = np.arange(0.0, t1 - t0 + 1e-9, 1.0 / self.kinematics_rate_hz)
            ik_deg = {
                c: np.interp(kin_time, ik_time - t0, ik[:, ik_hdr.index(c)])
                for c in (
                    "SC_y",
                    "SC_x",
                    "SC_z",
                    "arm_flex_r",
                    "arm_add_r",
                    "arm_rot_r",
                    "elbow_flex_r",
                    "pro_sup_r",
                )
            }
            qpos, ctrl_qadr, _ = map_upper_limb_to_qpos(
                model, buet_arm_to_upper_limb_coords(model, ik_deg)
            )

        subj, task, cond = trial_id.split("/")
        return EMGTrial(
            trial_id=trial_id,
            time=emg_time[in_emg] - t0,
            signals=emg[in_emg][:, keep],
            channel_names=tuple(ULTRAMOCAP_CHANNELS[hdr[i]] for i in keep),
            signal_kind="raw",
            qpos=qpos,
            kinematics_time=kin_time,
            filter_config=EMGFilterConfig(
                bandpass_hz=(20.0, 450.0),
                notch_hz=60.0,
                lowpass_hz=6.0,
                normalize="peak",
            ),
            meta={
                "dataset": self.name,
                "subject": subj,
                "task": task,
                "condition": cond,
                "window_start_s": t0,
                "controlled_qpos_adr": ctrl_qadr,
            },
        )


# Xsens-style 48-DoF URDF (Latella et al.): every segment frame is the T8 frame
# in the T-pose (x anterior, y left, z up); OpenSim thorax axes are x, z, -y.
_XSENS_TO_OPENSIM = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
_XSENS_LUMBAR_JOINTS: tuple[str, ...] = ("jL5S1", "jL4L3", "jL1T12")


def xsens_urdf_to_upper_limb_coords(
    model: mujoco.MjModel, angles: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Map right-arm and lumbar angles of the Xsens-style 48-DoF URDF [rad].

    ``angles`` holds ``angle_<joint>_<axis>`` columns. Lumbar rotations sum into
    ``flex_extension`` (extension positive) and ``lat_bending``; ``jT9T8``
    joins the rigid thorax, so the shoulder chain is rebuilt from T12 and
    referenced to the arm-hanging pose. The URDF turns the forearm about the
    upper-arm axis before elbow flexion, so ``jRightElbow_y`` is folded into
    humeral rotation; this keeps both forearm direction and hand orientation,
    with ``pro_sup`` left neutral.
    """
    n = len(angles["angle_jRightShoulder_x"])
    e_x, e_y, e_z = np.eye(3)
    seq = (
        ("jT9T8_x", e_x),
        ("jT9T8_y", e_y),
        ("jT9T8_z", e_z),
        ("jRightC7Shoulder_x", e_x),
        ("jRightShoulder_x", e_x),
        ("jRightShoulder_y", e_y),
        ("jRightShoulder_z", e_z),
        ("jRightElbow_y", e_y),
    )
    # The URDF's T-pose arm hangs after a 90 deg rotation about the anterior axis.
    n_pose_t = _axis_rot(e_x, 0.5 * np.pi).T
    r_ht = np.empty((n, 3, 3), dtype=np.float64)
    for t in range(n):
        r_xs = np.eye(3)
        for col, axis in seq:
            r_xs = r_xs @ _axis_rot(axis, angles[f"angle_{col}"][t])
        r_ht[t] = _XSENS_TO_OPENSIM @ (r_xs @ n_pose_t) @ _XSENS_TO_OPENSIM.T
    coords = thoracohumeral_to_upper_limb_coords(model, r_ht)
    coords["elbow_flexion"] = np.asarray(angles["angle_jRightElbow_z"])
    coords["flex_extension"] = -np.sum(
        [np.asarray(angles[f"angle_{j}_y"]) for j in _XSENS_LUMBAR_JOINTS], axis=0
    )
    coords["lat_bending"] = np.sum(
        [np.asarray(angles[f"angle_{j}_x"]) for j in _XSENS_LUMBAR_JOINTS], axis=0
    )
    return coords


class Maurice2020Source(EMGSource):
    """AndyData-lab overhead pointing: right anterior deltoid and erector spinae sEMG.

    Maurice, P., Camernik, J., Gorjan, D., Schirrmeister, B., Bornmann, J.,
    Tagliapietra, L., Latella, C., Pucci, D., Fritzsche, L., Ivaldi, S. & Babic, J.
    (2020). Objective and subjective effects of a passive exoskeleton on overhead
    work. IEEE Trans. Neural Syst. Rehabil. Eng. 28(1), 152-164.
    https://doi.org/10.1109/TNSRE.2019.2945368.
    Data: https://doi.org/10.5281/zenodo.1472214 (CC BY 4.0).

    12 participants point at overhead targets with (``WE``) and without
    (``NE1``/``NE2``) a passive arm-support exoskeleton; 1 kHz sEMG and 60 Hz
    Xsens joint angles share one clock. One trial spans the move to a new start
    position, the reach to the target and the 2 s hold. The joint-angle files
    carry no global pelvis pose, so the pelvis stays upright.
    """

    name = "maurice2020"

    def __init__(
        self,
        data_root: str | Path = "downloads/emg/maurice2020",
        *,
        subjects: Sequence[str] | None = None,
        pattern: str | None = None,
        sessions: Sequence[str] = ("NE1", "NE2", "WE"),
        kinematics_rate_hz: float = 100.0,
    ) -> None:
        root = Path(data_root)
        nested = root / "AndyData-lab-onePersonWithExoskeleton"
        self.data_root = nested if nested.exists() else root
        if not (self.data_root / "emg").exists():
            raise FileNotFoundError(f"AndyData emg/ folder not found under {root}")
        self.kinematics_rate_hz = kinematics_rate_hz
        self.pattern = re.compile(pattern) if pattern else None
        wanted = (
            {int(re.sub(r"\D", "", s) or "0") for s in subjects}
            if subjects is not None
            else None
        )
        stem_re = re.compile(
            r"^Participant_(\d+)_Group_[AB]_Session_(\w+?)_Block_(\d+)\.emg$"
        )
        self._blocks: dict[tuple[int, str, int], tuple[Path, Path, Path]] = {}
        for emg in sorted(self.data_root.glob("emg/emg_csv/Participant_*/*/*.emg.csv")):
            m = stem_re.match(emg.stem)
            if m is None or m.group(2) not in sessions:
                continue
            subj, sess, block = int(m.group(1)), m.group(2), int(m.group(3))
            if wanted is not None and subj not in wanted:
                continue
            rel = emg.relative_to(self.data_root / "emg" / "emg_csv").parent
            base = emg.name[: -len(".emg.csv")]
            ja = (
                self.data_root
                / "jointangles"
                / "jointangles_csv"
                / rel
                / f"{base}.jointangles.csv"
            )
            info = self.data_root / "info" / "info_csv" / rel / f"{base}.info.csv"
            if ja.exists() and info.exists():
                self._blocks[(subj, sess, block)] = (emg, ja, info)
        if not self._blocks:
            raise FileNotFoundError(
                f"no blocks with emg, jointangles and info CSVs under {self.data_root}"
            )
        self._cache: dict[tuple[int, str, int], dict[str, Any]] = {}
        self._ids_cache: tuple[str, ...] | None = None

    @staticmethod
    def _read_csv(path: Path) -> tuple[list[str], np.ndarray]:
        with path.open(encoding="utf-8") as fh:
            hdr = [h.strip() for h in fh.readline().split(",")]
            fh.readline()
            return hdr, np.loadtxt(fh, delimiter=",", ndmin=2)

    def _load_block(self, key: tuple[int, str, int]) -> dict[str, Any]:
        if key not in self._cache:
            if len(self._cache) >= 2:
                self._cache.clear()
            emg_p, ja_p, info_p = self._blocks[key]
            emg_hdr, emg = self._read_csv(emg_p)
            ja_hdr, ja = self._read_csv(ja_p)
            _, info = self._read_csv(info_p)
            # Some joint-angle files end with stale rows from earlier blocks.
            stale = np.flatnonzero(np.diff(ja[:, 0]) <= 0)
            if len(stale):
                ja = ja[: stale[0] + 1]
            self._cache[key] = {
                "emg_hdr": emg_hdr,
                "emg": emg,
                "ja_hdr": ja_hdr,
                "ja": ja,
                "info": info,
            }
        return self._cache[key]

    def _trial_windows(
        self, key: tuple[int, str, int]
    ) -> dict[int, tuple[float, float]]:
        blk = self._load_block(key)
        info = blk["info"]
        t, state, trial = info[:, 0], info[:, 1], info[:, 2].astype(int)
        t_lo = max(float(blk["ja"][0, 0]), float(blk["emg"][0, 0]))
        t_hi = min(float(blk["ja"][-1, 0]), float(blk["emg"][-1, 0]))
        out: dict[int, tuple[float, float]] = {}
        for k in np.unique(trial):
            rows = np.flatnonzero(trial == k)
            onset = rows[state[rows] == 2]
            hold = rows[state[rows] == 4]
            if len(onset) == 0 or len(hold) == 0:
                continue
            # From the move to the new start position (after any break) to the hold's end.
            new_start = rows[(state[rows] == 7) & (rows < onset[0])]
            t_begin = (
                float(t[new_start[-1]]) if len(new_start) else float(t[onset[0]]) - 0.5
            )
            nxt = rows[-1] + 1
            t_end = float(t[nxt]) if nxt < len(t) else float(t[rows[-1]])
            if t_begin >= t_lo and t_end <= t_hi:
                out[int(k)] = (t_begin, t_end)
        return out

    def trial_ids(self) -> Sequence[str]:
        if self._ids_cache is not None:
            return self._ids_cache
        prio: list[str] = []
        rest: list[str] = []
        for key in sorted(self._blocks):
            subj, sess, block = key
            for i, k in enumerate(sorted(self._trial_windows(key))):
                tid = f"P{subj:02d}/{sess}/block{block}/trial{k:02d}"
                (prio if i == 0 else rest).append(tid)
        all_ids = prio + rest
        if self.pattern is not None:
            all_ids = [t for t in all_ids if self.pattern.search(t)]
        self._ids_cache = tuple(all_ids)
        return self._ids_cache

    def default_channel_map(self, model: mujoco.MjModel) -> EMGChannelMap:
        longissimus = tuple(
            model.actuator(i).name
            for i in range(model.nu)
            if model.actuator(i).name.startswith(("LTpL_", "LTpT_"))
            and model.actuator(i).name.endswith("_r")
        )
        bindings = [ChannelBinding("anterior_deltoid", ("DELT1_r",))]
        if longissimus:
            bindings.append(ChannelBinding("erector_spinae_longissimus", longissimus))
        return EMGChannelMap(
            name="maurice2020_right_shoulder_back",
            version=1,
            bindings=tuple(bindings),
        )

    def get_trial(self, trial_id: str, model: mujoco.MjModel | None = None) -> EMGTrial:
        m = re.match(r"^P(\d+)/(\w+)/block(\d+)/trial(\d+)$", trial_id)
        key = (int(m.group(1)), m.group(2), int(m.group(3))) if m is not None else None
        if m is None or key not in self._blocks:
            raise KeyError(f"unknown AndyData trial {trial_id!r}")
        window = self._trial_windows(key).get(int(m.group(4)))
        if window is None:
            raise KeyError(f"trial {trial_id!r} has no target onset and hold")
        t0, t1 = window
        blk = self._load_block(key)
        emg, ja = blk["emg"], blk["ja"]
        in_emg = (emg[:, 0] >= t0) & (emg[:, 0] <= t1)
        # Stored timestamps are float32-rounded; samples are uniform at 1 kHz.
        emg_time = float(emg[in_emg, 0][0]) - t0 + np.arange(int(in_emg.sum())) / 1000.0
        ch_names = tuple(blk["emg_hdr"][1:])

        qpos = None
        kin_time = None
        ctrl_qadr: list[int] = []
        if model is not None:
            ja_hdr = blk["ja_hdr"]
            kin_time = np.arange(0.0, t1 - t0 + 1e-9, 1.0 / self.kinematics_rate_hz)
            angles = {
                c: np.interp(kin_time + t0, ja[:, 0], ja[:, i])
                for i, c in enumerate(ja_hdr)
                if c.startswith("angle_")
            }
            coords = xsens_urdf_to_upper_limb_coords(model, angles)
            qpos, ctrl_qadr, _ = map_upper_limb_to_qpos(model, coords)
            jnt_lookup = {model.joint(j).name: j for j in range(model.njnt)}
            for name in ("flex_extension", "lat_bending"):
                jid = jnt_lookup.get(name)
                if jid is None:
                    continue
                qadr = int(model.jnt_qposadr[jid])
                vals = coords[name]
                if model.jnt_limited[jid]:
                    vals = np.clip(vals, *model.jnt_range[jid])
                qpos[:, qadr] = vals
                ctrl_qadr.append(qadr)
            project_joint_couplings(JointCouplings.from_model(model), qpos)
            qpos = project_tendon_nodes(model, qpos)
            ctrl_qadr = sorted(set(ctrl_qadr))

        subj, sess, block, trial = trial_id.split("/")
        return EMGTrial(
            trial_id=trial_id,
            time=emg_time,
            signals=emg[in_emg, 1:],
            channel_names=ch_names,
            signal_kind="raw",
            qpos=qpos,
            kinematics_time=kin_time,
            filter_config=EMGFilterConfig(
                bandpass_hz=(20.0, 450.0),
                notch_hz=50.0,
                lowpass_hz=6.0,
                normalize="peak",
            ),
            meta={
                "dataset": self.name,
                "subject": subj,
                "session": sess,
                "exoskeleton": sess == "WE",
                "block": block,
                "trial": trial,
                "window_s": [t0, t1],
                "controlled_qpos_adr": ctrl_qadr,
            },
        )


__all__ = [
    "EMG2POSE_JOINT_NAMES",
    "Emg2PoseHandSource",
    "GrabMyoSource",
    "NinaProDB5Source",
    "Quesada2024UpperLimbSource",
    "map_cyberglove_to_hand_qpos",
    "map_emg2pose_to_hand_qpos",
    "map_grabmyo_gestures_to_qpos",
    "map_upper_limb_to_qpos",
    "map_upper_limb_torques_to_qfrc",
    "read_wfdb_record",
]
