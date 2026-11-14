# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Lower-limb gait and custom-manifest dataset adapters for EMG validation."""

from __future__ import annotations

import csv
import importlib
import io
import json
import re
import struct
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import mujoco
import numpy as np
import scipy.io as sio
from scipy.io.matlab._mio5 import MatFile5Reader

from myosuite.biomechanics.emg import EMGFilterConfig, project_tendon_nodes
from myosuite.biomechanics.emg_validation import (
    EMG_TRIAL_META_KEY,
    ChannelBinding,
    EMGChannelMap,
    EMGSource,
    EMGTrial,
    load_emg_trial,
    spatial_site_channel_map,
)
from myosuite.biomechanics.inverse_dynamics import clip_kinematics, required_qfrc
from myosuite.biomechanics.opensim_motion import (
    GAIT2392_SIGNS,
    PELVIS_COORDINATES,
    OpenSimMotion,
    opensim_motion_to_qpos,
    read_opensim_motion,
)


def _opensim_grf_lib() -> ModuleType:
    try:
        return importlib.import_module("myosuite.biomechanics.opensim_grf")
    except ModuleNotFoundError as exc:
        if exc.name != "myosuite.biomechanics.opensim_grf":
            raise
        raise ImportError(
            "OpenSim GRF gait helpers live in myosuite.biomechanics.opensim_grf "
            "(MyoHub/myosuite#845). Merge or rebase onto #845 before loading "
            "OpenSim ExternalLoads GRF trials."
        ) from exc


def normalize_opensim_motion(
    motion: OpenSimMotion,
) -> tuple[OpenSimMotion, Mapping[str, float]]:
    try:
        mod = importlib.import_module("myosuite.biomechanics.opensim_grf")
    except ModuleNotFoundError as exc:
        if exc.name != "myosuite.biomechanics.opensim_grf":
            raise
    else:
        return mod.normalize_opensim_motion(motion)
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
            label = (
                parts[-2]
                if parts and parts[-1] == "value" and len(parts) >= 2
                else (parts[-1] if parts else label)
            )
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


def opensim_grf_to_qfrc(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    time: np.ndarray,
    grf_motion: OpenSimMotion,
    external_loads_xml: str | Path,
    **kwargs: Any,
) -> np.ndarray:
    return _opensim_grf_lib().opensim_grf_to_qfrc(
        model, qpos, time, grf_motion, external_loads_xml, **kwargs
    )


DIMITROV_SHANK_ACTUATORS: tuple[str, ...] = (
    "tibant_r",
    "edl_r",
    "ehl_r",
    "perbrev_r",
    "perlong_r",
    "gasmed_r",
    "gaslat_r",
    "soleus_r",
    "tibpost_r",
    "fdl_r",
    "fhl_r",
)

LENCIONI_CHANNELS: tuple[str, ...] = (
    "tibialis_anterior",
    "soleus",
    "gastrocnemius_medialis",
    "peroneus_longus",
    "rectus_femoris",
    "vastus_medialis",
    "biceps_femoris",
    "gluteus_maximus",
)


def _slice_opensim_motion(
    motion: OpenSimMotion, t_start: float, t_end: float
) -> OpenSimMotion:
    mask = (motion.time >= t_start - 1e-9) & (motion.time <= t_end + 1e-9)
    if np.count_nonzero(mask) < 4:
        raise ValueError(
            f"time window [{t_start}, {t_end}] leaves fewer than 4 frames in OpenSimMotion"
        )
    return OpenSimMotion(
        time=motion.time[mask].copy(),
        labels=motion.labels,
        values=motion.values[mask].copy(),
        in_degrees=motion.in_degrees,
    )


def _parse_opensim_motion_text(
    text: str, *, in_degrees: bool | None = None
) -> OpenSimMotion:
    lines = text.splitlines()
    try:
        end = next(i for i, line in enumerate(lines) if line.strip() == "endheader")
    except StopIteration:
        raise ValueError("OpenSim storage text has no 'endheader' line") from None
    header: dict[str, str] = {}
    for line in lines[:end]:
        key, sep, val = line.partition("=")
        if sep:
            header[key.strip().lower()] = val.strip().lower()
    raw_labels = lines[end + 1].split("\t" if "\t" in lines[end + 1] else None)
    labels = [lb.strip() for lb in raw_labels]
    if labels[0].lower() != "time":
        raise ValueError(f"first column is {labels[0]!r}, expected 'time'")
    data = np.loadtxt(lines[end + 2 :], ndmin=2)
    if in_degrees is None:
        in_degrees = header.get("indegrees", "yes") == "yes"
    return OpenSimMotion(
        time=data[:, 0],
        labels=tuple(labels[1:]),
        values=data[:, 1:],
        in_degrees=bool(in_degrees),
    )


class RajagopalWalkingSource(EMGSource):
    """OpenSim Moco ``example3DWalking`` EMG, IK and force-plate GRF adapter."""

    name = "rajagopal2016"

    def __init__(
        self,
        data_root: str | Path = "downloads/emg/rajagopal2016_opensim",
        *,
        kinematics_rate_hz: float = 100.0,
        window_s: tuple[float, float] = (0.48, 1.61),
    ) -> None:
        root = Path(data_root)
        candidates = [
            root
            / "opensim-core"
            / "OpenSim"
            / "Examples"
            / "Moco"
            / "example3DWalking",
            root / "example3DWalking",
            root,
        ]
        found = next(
            (c for c in candidates if (c / "electromyography.sto").exists()), None
        )
        if found is None:
            raise FileNotFoundError(f"could not find electromyography.sto under {root}")
        self.example_dir = found
        self.kinematics_rate_hz = kinematics_rate_hz
        self.window_s = window_s

    def trial_ids(self) -> Sequence[str]:
        return ("moco_3d_walking",)

    def default_channel_map(self, model: mujoco.MjModel) -> EMGChannelMap:
        del model
        return EMGChannelMap(
            name="rajagopal2016_right_leg",
            version=1,
            bindings=(
                ChannelBinding("gluteus_maximus", ("glmax1_r", "glmax2_r", "glmax3_r")),
                ChannelBinding("gluteus_medius", ("glmed1_r", "glmed2_r", "glmed3_r")),
                ChannelBinding("medial_hamstrings", ("semimem_r", "semiten_r")),
                ChannelBinding("biceps_femoris", ("bflh_r", "bfsh_r")),
                ChannelBinding("rectus_femoris", ("recfem_r",)),
                ChannelBinding("vastus_lateralis", ("vaslat_r",)),
                ChannelBinding("vastus_medius", ("vasmed_r",)),
                ChannelBinding(
                    "gastrocnemius", ("gasmed_r", "gaslat_r"), reference_peak=0.87
                ),
                ChannelBinding("soleus", ("soleus_r",), reference_peak=0.77),
                ChannelBinding("tibialis_anterior", ("tibant_r",), reference_peak=0.37),
            ),
        )

    def get_trial(self, trial_id: str, model: mujoco.MjModel | None = None) -> EMGTrial:
        if trial_id not in self.trial_ids():
            raise KeyError(f"unknown Rajagopal trial {trial_id!r}")
        t0, t1 = self.window_s
        emg_mot = _slice_opensim_motion(
            read_opensim_motion(
                self.example_dir / "electromyography.sto", in_degrees=False
            ),
            t0,
            t1,
        )
        coords_raw = read_opensim_motion(self.example_dir / "coordinates.sto")
        coords_clean, signs = normalize_opensim_motion(coords_raw)
        coords_win = _slice_opensim_motion(coords_clean, t0, t1)

        qpos = None
        kin_time = None
        ext_qfrc = None
        target_actuators: list[str] = []
        if model is not None:
            mapped = opensim_motion_to_qpos(
                model, coords_win, rate=self.kinematics_rate_hz, signs=signs
            )
            qpos = project_tendon_nodes(model, mapped.qpos)
            kin_time = mapped.time
            target_actuators = [
                model.actuator(i).name
                for i in range(model.nu)
                if model.actuator(i).name.endswith("_r")
            ]
            grf_path = self.example_dir / "grf_walk.mot"
            xml_path = self.example_dir / "grf_walk.xml"
            if grf_path.exists() and xml_path.exists():
                grf_mot = read_opensim_motion(grf_path, in_degrees=False)
                ext_qfrc = opensim_grf_to_qfrc(model, qpos, kin_time, grf_mot, xml_path)

        meta: dict[str, Any] = {
            "dataset": self.name,
            "window_s": [t0, t1],
            "source_dir": str(self.example_dir),
        }
        if target_actuators:
            meta["target_actuators"] = target_actuators

        return EMGTrial(
            trial_id=trial_id,
            time=emg_mot.time,
            signals=emg_mot.values,
            channel_names=emg_mot.labels,
            signal_kind="peak_normalized",
            qpos=qpos,
            kinematics_time=kin_time,
            external_qfrc=ext_qfrc,
            cycle_events=np.asarray([[t0, t1]], dtype=np.float64),
            meta=meta,
        )


_LENCIONI_LOADS_XML = """<ExternalLoads>
  <objects>
    <ExternalForce name="right_grf">
      <applied_to_body>calcn_r</applied_to_body>
      <force_expressed_in_body>ground</force_expressed_in_body>
      <point_expressed_in_body>ground</point_expressed_in_body>
      <force_identifier>f_</force_identifier>
      <point_identifier>p_</point_identifier>
      <torque_identifier>t_</torque_identifier>
    </ExternalForce>
  </objects>
</ExternalLoads>"""


class Lencioni2019Source(EMGSource):
    """Lencioni et al. (2019) 8-ch 1 kHz lower-limb EMG across speeds and stairs."""

    name = "lencioni2019"

    def __init__(
        self,
        data_root: str | Path = "downloads/emg/lencioni2019",
        *,
        subjects: Sequence[str] | None = None,
        pattern: str | None = None,
        kinematics_rate_hz: float = 100.0,
    ) -> None:
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"Lencioni data_root not found: {self.data_root}")
        self.kinematics_rate_hz = kinematics_rate_hz
        self.pattern = re.compile(pattern) if pattern else None

        avail = sorted(
            self.data_root.glob("Subject*.mat"),
            key=lambda p: int(re.sub(r"\D", "", p.stem) or "0"),
        )
        if not avail:
            raise FileNotFoundError(
                f"no Subject*.mat files found under {self.data_root}"
            )
        if subjects is None:
            chosen = [p.stem for p in avail[:5]]
        elif len(subjects) == 1 and subjects[0].lower() == "all":
            chosen = [p.stem for p in avail]
        else:
            wanted = {s.lower() for s in subjects}
            chosen = [p.stem for p in avail if p.stem.lower() in wanted]
        if not chosen:
            raise FileNotFoundError(
                f"requested subjects {subjects} not found in {self.data_root}"
            )
        self.subjects = tuple(chosen)
        self._cache: dict[str, Any] = {}
        self._ids_cache: tuple[str, ...] | None = None

    def _load_subject(self, subj: str) -> Any:
        if subj not in self._cache:
            if len(self._cache) >= 4:
                self._cache.clear()
            self._cache[subj] = sio.loadmat(
                self.data_root / f"{subj}.mat", simplify_cells=True
            )["s"]
        return self._cache[subj]

    def trial_ids(self) -> Sequence[str]:
        if self._ids_cache is not None:
            return self._ids_cache
        task_prio: dict[str, str] = {}
        per_subj_first: list[str] = []
        rest: list[str] = []
        for subj in self.subjects:
            s = self._load_subject(subj)
            emg_side = str(s["EMGSide"]).strip().upper()
            seen_subj_tasks: set[str] = set()
            for idx, rec in enumerate(np.atleast_1d(s["Data"])):
                if str(rec["Foot"]).strip().upper() != emg_side:
                    continue
                emg = np.asarray(rec["EMG"])
                if emg.ndim != 2 or emg.shape[0] != 8 or emg.shape[1] < 64:
                    continue
                task = str(rec["Task"]).strip()
                tid = f"{subj}/{task}/trial_{idx:02d}"
                if task not in task_prio:
                    task_prio[task] = tid
                elif task not in seen_subj_tasks:
                    per_subj_first.append(tid)
                else:
                    rest.append(tid)
                seen_subj_tasks.add(task)
        ordered = [
            task_prio[t]
            for t in ("Walking", "ToeWalking", "HeelWalking", "StepUp", "StepDown")
            if t in task_prio
        ]
        for v in task_prio.values():
            if v not in ordered:
                ordered.append(v)
        all_ids = ordered + per_subj_first + rest
        if self.pattern is not None:
            all_ids = [t for t in all_ids if self.pattern.search(t)]
        self._ids_cache = tuple(all_ids)
        return self._ids_cache

    def default_channel_map(self, model: mujoco.MjModel) -> EMGChannelMap:
        del model
        return EMGChannelMap(
            name="lencioni2019_right_leg",
            version=1,
            bindings=(
                ChannelBinding("tibialis_anterior", ("tibant_r",), reference_peak=0.37),
                ChannelBinding("soleus", ("soleus_r",), reference_peak=0.77),
                ChannelBinding(
                    "gastrocnemius_medialis", ("gasmed_r",), reference_peak=0.87
                ),
                ChannelBinding("peroneus_longus", ("perlong_r",)),
                ChannelBinding("rectus_femoris", ("recfem_r",)),
                ChannelBinding("vastus_medialis", ("vasmed_r",)),
                ChannelBinding("biceps_femoris", ("bflh_r", "bfsh_r")),
                ChannelBinding("gluteus_maximus", ("glmax1_r", "glmax2_r", "glmax3_r")),
            ),
        )

    def get_trial(self, trial_id: str, model: mujoco.MjModel | None = None) -> EMGTrial:
        parts = trial_id.split("/")
        if len(parts) != 3 or not parts[2].startswith("trial_"):
            raise KeyError(f"invalid Lencioni trial_id {trial_id!r}")
        subj, task, idx_tok = parts
        idx = int(idx_tok.split("_")[1])
        s = self._load_subject(subj)
        data_arr = np.atleast_1d(s["Data"])
        if idx >= len(data_arr):
            raise KeyError(f"trial {trial_id!r} out of range for {subj}")
        rec = data_arr[idx]

        is_left = str(s["EMGSide"]).strip().upper().startswith("L")
        mirror = -1.0 if is_left else 1.0
        emg_freq = float(s["EMGFreq"])
        emg_raw = np.asarray(rec["EMG"], dtype=np.float64).T
        dur_s = (len(emg_raw) - 1) / emg_freq
        emg_time = np.arange(len(emg_raw), dtype=np.float64) / emg_freq

        ang = np.asarray(rec["Ang"], dtype=np.float64).copy()
        stand = s.get("StandingData")
        if isinstance(stand, Mapping) and "Ang" in stand:
            stand_ang = np.asarray(stand["Ang"], dtype=np.float64)
            bias = (
                stand_ang[:, None]
                if stand_ang.ndim == 1 and len(stand_ang) == 12
                else (
                    np.nanmean(stand_ang, axis=1, keepdims=True)
                    if stand_ang.ndim == 2 and stand_ang.shape[0] == 12
                    else None
                )
            )
            if bias is not None:
                for c_idx in (2, 5, 8, 9, 10, 11):
                    ang[c_idx] -= bias[c_idx]

        t_101 = np.linspace(0.0, dur_s, ang.shape[1])
        pelv_xyz = np.asarray(rec["Com"], dtype=np.float64) * 1e-3
        if "Marker" in rec and "MarkerVarName" in s:
            mk = np.asarray(rec["Marker"], dtype=np.float64)
            mnames = [str(m).strip() for m in s["MarkerVarName"]]
            if (
                "XRxAsis" in mnames
                and "XLxAsis" in mnames
                and mk.ndim == 2
                and mk.shape[1] >= 4
            ):
                ix_ra, ix_la = mnames.index("XRxAsis"), mnames.index("XLxAsis")
                mid_asis = 0.5 * (mk[ix_ra : ix_ra + 3] + mk[ix_la : ix_la + 3])
                if np.isfinite(mid_asis).all():
                    t_mk = np.linspace(0.0, dur_s, mk.shape[1])
                    pelv_xyz = np.vstack(
                        [np.interp(t_101, t_mk, mid_asis[k]) for k in range(3)]
                    )

        deg2rad = np.pi / 180.0
        mot_labels = (
            "pelvis_tx",
            "pelvis_ty",
            "pelvis_tz",
            "pelvis_tilt",
            "pelvis_list",
            "pelvis_rotation",
            "hip_flexion_r",
            "hip_adduction_r",
            "hip_rotation_r",
            "knee_angle_r",
            "ankle_angle_r",
            "subtalar_angle_r",
        )
        mot_vals = np.column_stack(
            [
                pelv_xyz[0],
                pelv_xyz[1],
                mirror * pelv_xyz[2],
                ang[0] * deg2rad,
                mirror * ang[1] * deg2rad,
                mirror * ang[2] * deg2rad,
                ang[3] * deg2rad,
                ang[4] * deg2rad,
                ang[5] * deg2rad,
                ang[6] * deg2rad,
                ang[9] * deg2rad,
                ang[10] * deg2rad,
            ]
        )
        ik_motion = OpenSimMotion(
            time=t_101, labels=mot_labels, values=mot_vals, in_degrees=False
        )

        qpos = None
        kin_time = None
        ext_qfrc = None
        has_grf = False
        if model is not None:
            mapped = opensim_motion_to_qpos(
                model, ik_motion, rate=self.kinematics_rate_hz, signs={}
            )
            qpos = project_tendon_nodes(model, mapped.qpos)
            kin_time = mapped.time
            mom_raw = np.asarray(rec.get("Mom", ()), dtype=np.float64)
            bm = float(s.get("BM", np.nan))
            if (
                mom_raw.ndim == 2
                and mom_raw.shape == (9, len(t_101))
                and np.isfinite(mom_raw).all()
                and np.isfinite(bm)
                and bm > 0
            ):
                mom_nm = mom_raw * bm
                dt = float(np.median(np.diff(kin_time)))
                kin = clip_kinematics(model, qpos, dt, lowpass_hz=6.0)
                data = mujoco.MjData(model)
                ext_qfrc = np.zeros((len(qpos), model.nv), dtype=np.float64)
                j_lookup = {model.joint(j).name: j for j in range(model.njnt)}
                slave_dofs = [
                    int(model.jnt_dofadr[int(model.eq_obj1id[e])])
                    for e in range(model.neq)
                    if model.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT
                    and model.eq_active0[e]
                ]
                root_dofs = [
                    d_idx
                    for j in np.flatnonzero(
                        model.jnt_type == mujoco.mjtJoint.mjJNT_FREE
                    )
                    for d_idx in range(
                        int(model.jnt_dofadr[j]), int(model.jnt_dofadr[j]) + 6
                    )
                ]
                dof_specs = [
                    (int(model.jnt_dofadr[j_lookup[jn]]), m_idx, -1.0)
                    for jn, m_idx in (
                        ("hip_flexion_r", 0),
                        ("hip_adduction_r", 1),
                        ("hip_rotation_r", 2),
                        ("knee_angle_r", 3),
                        ("ankle_angle_r", 6),
                        ("subtalar_angle_r", 7),
                    )
                    if jn in j_lookup
                ]
                for t_i in range(len(qpos)):
                    data.qpos[:], data.qvel[:] = kin.qpos[t_i], kin.qvel[t_i]
                    mujoco.mj_fwdPosition(model, data)
                    mujoco.mj_fwdVelocity(model, data)
                    tau_id = required_qfrc(model, data, kin.qacc[t_i])
                    ext_qfrc[t_i, root_dofs] = tau_id[root_dofs]
                    ext_qfrc[t_i, slave_dofs] = tau_id[slave_dofs]
                    for dadr, m_idx, sgn in dof_specs:
                        tau_tgt = sgn * float(
                            np.interp(kin_time[t_i], t_101, mom_nm[m_idx])
                        )
                        ext_qfrc[t_i, dadr] = tau_id[dadr] - tau_tgt
                has_grf = True
            else:
                grf = np.asarray(rec.get("Grf", ()))
                if grf.ndim == 2 and grf.shape[0] == 9 and grf.shape[1] >= 8:
                    grf_f = grf.astype(np.float64)
                    grf_freq = float(s["GrfFreq"])
                    t_grf = np.arange(grf_f.shape[1], dtype=np.float64) / grf_freq
                    if t_grf[-1] < dur_s - 1e-4:
                        t_pad = np.r_[t_grf, t_grf[-1] + 1e-3, dur_s]
                        grf_f = np.hstack([grf_f, np.zeros((9, 2), dtype=np.float64)])
                    else:
                        t_pad = t_grf
                    grf_cols = np.column_stack(
                        [
                            grf_f[0],
                            grf_f[1],
                            mirror * grf_f[2],
                            grf_f[3] * 1e-3,
                            grf_f[4] * 1e-3,
                            mirror * grf_f[5] * 1e-3,
                            mirror * grf_f[6],
                            mirror * grf_f[7],
                            grf_f[8],
                        ]
                    )
                    grf_mot = OpenSimMotion(
                        time=t_pad,
                        labels=(
                            "f_x",
                            "f_y",
                            "f_z",
                            "p_x",
                            "p_y",
                            "p_z",
                            "t_x",
                            "t_y",
                            "t_z",
                        ),
                        values=grf_cols,
                        in_degrees=False,
                    )
                    ext_qfrc = opensim_grf_to_qfrc(
                        model, qpos, kin_time, grf_mot, _LENCIONI_LOADS_XML
                    )
                    has_grf = True

        return EMGTrial(
            trial_id=trial_id,
            time=emg_time,
            signals=emg_raw,
            channel_names=LENCIONI_CHANNELS,
            signal_kind="raw",
            qpos=qpos,
            kinematics_time=kin_time,
            external_qfrc=ext_qfrc,
            cycle_events=np.asarray([[0.0, dur_s]], dtype=np.float64),
            filter_config=EMGFilterConfig(
                bandpass_hz=(20.0, 400.0),
                notch_hz=50.0,
                lowpass_hz=6.0,
                normalize="peak",
            ),
            meta={
                "dataset": self.name,
                "subject": subj,
                "task": task,
                "speed_m_s": float(rec.get("speed", np.nan)),
                "force_plates_matched": has_grf,
                "contact_forces_fallback": not has_grf,
                "target_actuators": [
                    model.actuator(i).name
                    for i in range(model.nu)
                    if model.actuator(i).name.endswith("_r")
                ]
                if model is not None
                else [],
            },
        )


def _extract_mcos_cells(mat_bytes: bytes) -> tuple[dict[str, Any], Any]:
    mat = sio.loadmat(io.BytesIO(mat_bytes), simplify_cells=False)
    ws = mat["__function_workspace__"].tobytes()
    start = next(
        off
        for off in range(104, min(len(ws) - 8, 256), 8)
        if struct.unpack_from("<II", ws, off)[0] == 14
    )
    fake_mat = b"MATLAB 5.0 MAT-file" + b" " * (124 - 19) + b"\x00\x01IM" + ws[start:]
    rdr = MatFile5Reader(io.BytesIO(fake_mat), simplify_cells=True)
    rdr.initialize_read()
    rdr.mat_stream.seek(128)
    hdr, _ = rdr.read_var_header()
    cells = rdr.read_var_array(hdr)
    return mat, cells


def _decode_mcos_table(
    opaque_obj: Any, cells: Any
) -> tuple[tuple[str, ...], np.ndarray]:
    field = (
        "_ObjectMetadata"
        if hasattr(opaque_obj, "dtype")
        and opaque_obj.dtype.names
        and "_ObjectMetadata" in opaque_obj.dtype.names
        else "arr"
    )
    oid = int(np.frombuffer(opaque_obj[0][field].tobytes(), dtype="<u4")[4])
    base = (oid - 1) * 7 + 2
    cols = [np.asarray(col, dtype=np.float64).squeeze() for col in cells[base]]
    names = tuple(str(v) for v in cells[base + 5])
    return names, np.column_stack(cols)


_DIMITROV_FOLDER_MAP: dict[tuple[str, ...], str] = {
    (
        "Level_Ground",
        "Walking",
        "Self_Selected_Speed",
    ): "RawData/Level Ground/LG_Walk_SelfSelected",
    ("Level_Ground", "Walking", "Slow_Speed"): "RawData/Level Ground/LG_Walk_Slow",
    ("Level_Ground", "Walking", "Fast_Speed"): "RawData/Level Ground/LG_Walk_Fast",
    (
        "Level_Ground",
        "Stand_to_Walk",
        "Right_Foot_Start",
    ): "RawData/Level Ground/LG_Stand_to_Walk/RightFootStart",
    (
        "Level_Ground",
        "Stand_to_Walk",
        "Left_Foot_Start",
    ): "RawData/Level Ground/LG_Stand_to_Walk/LeftFootStart",
    (
        "Level_Ground",
        "Sit_to_Stand_to_Walk",
        "Right_Foot_Start",
    ): "RawData/Level Ground/LG_Sit_Stand_Walk/RightFootStart",
    (
        "Level_Ground",
        "Sit_to_Stand_to_Walk",
        "Left_Foot_Start",
    ): "RawData/Level Ground/LG_Sit_Stand_Walk/LeftFootStart",
    (
        "Level_Ground",
        "SideStepping",
        "ToTheRight",
    ): "RawData/Level Ground/LG_SideStep/ToTheRight",
    (
        "Level_Ground",
        "SideStepping",
        "ToTheLeft",
    ): "RawData/Level Ground/LG_SideStep/ToTheLeft",
    (
        "Stairs_Ambulation",
        "Ascend",
        "Self_Selected_Speed",
    ): "RawData/Stairs and Ramps/Stairs/Ascent/Self_Selected_wSpeed",
    (
        "Stairs_Ambulation",
        "Ascend",
        "Slow_Speed",
    ): "RawData/Stairs and Ramps/Stairs/Ascent/Slow_wSpeed",
    (
        "Stairs_Ambulation",
        "Ascend",
        "Fast_Speed",
    ): "RawData/Stairs and Ramps/Stairs/Ascent/Fast_wSpeed",
    (
        "Stairs_Ambulation",
        "Descend",
        "Self_Selected_Speed",
    ): "RawData/Stairs and Ramps/Stairs/Descent/Self_Selected_wSpeed",
    (
        "Stairs_Ambulation",
        "Descend",
        "Slow_Speed",
    ): "RawData/Stairs and Ramps/Stairs/Descent/Slow_wSpeed",
    (
        "Stairs_Ambulation",
        "Descend",
        "Fast_Speed",
    ): "RawData/Stairs and Ramps/Stairs/Descent/Fast_wSpeed",
    (
        "Ramp_Ambulation",
        "Fifteen_Degree",
        "Ascend",
        "Self_Selected_Speed",
    ): "RawData/Stairs and Ramps/Ramps/15degree_Incline/Ascent/Self_Selected_wSpeed",
    (
        "Ramp_Ambulation",
        "Fifteen_Degree",
        "Ascend",
        "Slow_Speed",
    ): "RawData/Stairs and Ramps/Ramps/15degree_Incline/Ascent/Slow_wSpeed",
    (
        "Ramp_Ambulation",
        "Fifteen_Degree",
        "Ascend",
        "Fast_Speed",
    ): "RawData/Stairs and Ramps/Ramps/15degree_Incline/Ascent/Fast_wSpeed",
    (
        "Ramp_Ambulation",
        "Fifteen_Degree",
        "Descend",
        "Self_Selected_Speed",
    ): "RawData/Stairs and Ramps/Ramps/15degree_Incline/Descent/Self_Selected_wSpeed",
    (
        "Ramp_Ambulation",
        "Fifteen_Degree",
        "Descend",
        "Slow_Speed",
    ): "RawData/Stairs and Ramps/Ramps/15degree_Incline/Descent/Slow_wSpeed",
    (
        "Ramp_Ambulation",
        "Fifteen_Degree",
        "Descend",
        "Fast_Speed",
    ): "RawData/Stairs and Ramps/Ramps/15degree_Incline/Descent/Fast_wSpeed",
    (
        "Ramp_Ambulation",
        "Five_Degree",
        "Ascend",
        "Self_Selected_Speed",
    ): "RawData/Stairs and Ramps/Ramps/5degree_Incline/Ascent/Self_Selected_wSpeed",
    (
        "Ramp_Ambulation",
        "Five_Degree",
        "Ascend",
        "Slow_Speed",
    ): "RawData/Stairs and Ramps/Ramps/5degree_Incline/Ascent/Slow_wSpeed",
    (
        "Ramp_Ambulation",
        "Five_Degree",
        "Ascend",
        "Fast_Speed",
    ): "RawData/Stairs and Ramps/Ramps/5degree_Incline/Ascent/Fast_wSpeed",
    (
        "Ramp_Ambulation",
        "Five_Degree",
        "Descend",
        "Self_Selected_Speed",
    ): "RawData/Stairs and Ramps/Ramps/5degree_Incline/Descent/Self_Selected_wSpeed",
    (
        "Ramp_Ambulation",
        "Five_Degree",
        "Descend",
        "Slow_Speed",
    ): "RawData/Stairs and Ramps/Ramps/5degree_Incline/Descent/Slow_wSpeed",
    (
        "Ramp_Ambulation",
        "Five_Degree",
        "Descend",
        "Fast_Speed",
    ): "RawData/Stairs and Ramps/Ramps/5degree_Incline/Descent/Fast_wSpeed",
}

_DIMITROV_PRIORITY_BRANCHES: tuple[tuple[str, ...], ...] = (
    ("Level_Ground", "Walking", "Self_Selected_Speed"),
    ("Stairs_Ambulation", "Ascend", "Self_Selected_Speed"),
    ("Stairs_Ambulation", "Descend", "Self_Selected_Speed"),
    ("Ramp_Ambulation", "Fifteen_Degree", "Ascend", "Self_Selected_Speed"),
    ("Ramp_Ambulation", "Fifteen_Degree", "Descend", "Self_Selected_Speed"),
    ("Ramp_Ambulation", "Five_Degree", "Ascend", "Self_Selected_Speed"),
)


class Dimitrov2023Source(EMGSource):
    """Streaming zip/MAT loader for Dimitrov et al. (2023) 64-ch below-knee HDsEMG."""

    name = "dimitrov2023"

    def __init__(
        self,
        data_root: str | Path = "downloads/emg/dimitrov2023",
        *,
        subjects: Sequence[str] | None = None,
        pattern: str | None = None,
        kinematics_rate_hz: float = 100.0,
    ) -> None:
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"Dimitrov data_root not found: {self.data_root}")
        self.kinematics_rate_hz = kinematics_rate_hz
        self.pattern = re.compile(pattern) if pattern else None
        avail = sorted(p.stem for p in self.data_root.glob("P*.zip"))
        if not avail:
            avail = sorted(p.parent.name for p in self.data_root.glob("P*/P*.mat"))
        if subjects is not None:
            wanted = {s.upper() for s in subjects}
            avail = [s for s in avail if s.upper() in wanted]
        if not avail:
            raise FileNotFoundError(
                f"no P01..P10 archives found under {self.data_root}"
            )
        self.subjects = tuple(avail)
        self._cache: dict[str, tuple[Any, Any, list[str]]] = {}
        self._ids_cache: tuple[str, ...] | None = None

    def _load_subject(self, subj: str) -> tuple[Any, Any, list[str]]:
        if subj in self._cache:
            return self._cache[subj]
        zip_path = self.data_root / f"{subj}.zip"
        if zip_path.exists():
            with zipfile.ZipFile(zip_path) as zf:
                members = [n for n in zf.namelist() if not n.startswith("__MACOSX")]
                mat_member = next(n for n in members if n.endswith(f"{subj}.mat"))
                mat_bytes = zf.read(mat_member)
        else:
            mat_path = self.data_root / subj / f"{subj}.mat"
            mat_bytes = mat_path.read_bytes()
            members = []
        mat, cells = _extract_mcos_cells(mat_bytes)
        gait_root = mat[subj][0, 0]["RightFoot_GaitCycle_Data"][0, 0]
        if len(self._cache) >= 2:
            self._cache.clear()
        self._cache[subj] = (gait_root, cells, members)
        return gait_root, cells, members

    @staticmethod
    def _iter_branches(
        node: Any, prefix: tuple[str, ...] = ()
    ) -> list[tuple[tuple[str, ...], Any]]:
        out: list[tuple[tuple[str, ...], Any]] = []
        if not hasattr(node, "dtype") or node.dtype.names is None:
            return out
        names = node.dtype.names
        if "IK" in names and ("RightLeg_EMG" in names or "RightFoot_EMG" in names):
            out.append((prefix, np.atleast_1d(node).ravel()))
            return out
        scalar = (
            node.flat[0] if isinstance(node, np.ndarray) and node.ndim > 0 else node
        )
        if not hasattr(scalar, "dtype") or scalar.dtype.names is None:
            return out
        for field_name in scalar.dtype.names:
            sub = scalar[field_name]
            if (
                isinstance(sub, np.ndarray)
                and sub.size > 0
                and sub.dtype.names is not None
            ):
                out.extend(
                    Dimitrov2023Source._iter_branches(sub, (*prefix, field_name))
                )
        return out

    def trial_ids(self) -> Sequence[str]:
        if self._ids_cache is not None:
            return self._ids_cache
        priority: list[str] = []
        rest: list[str] = []
        for subj in self.subjects:
            gait_root, _, _ = self._load_subject(subj)
            branch_map = dict(self._iter_branches(gait_root))
            seen_prio: set[str] = set()
            for br in _DIMITROV_PRIORITY_BRANCHES:
                if br in branch_map and len(branch_map[br]) > 0:
                    tid = f"{subj}/{'/'.join(br)}/trial_00"
                    seen_prio.add(tid)
                    priority.append(tid)
            for br, arr in branch_map.items():
                for idx in range(len(arr)):
                    tid = f"{subj}/{'/'.join(br)}/trial_{idx:02d}"
                    if tid not in seen_prio:
                        rest.append(tid)
        all_ids = priority + rest
        if self.pattern is not None:
            all_ids = [t for t in all_ids if self.pattern.search(t)]
        self._ids_cache = tuple(all_ids)
        return self._ids_cache

    def default_channel_map(self, model: mujoco.MjModel) -> EMGChannelMap | None:
        del model
        return None

    def _match_force_plate_loads(
        self,
        subj: str,
        branch: tuple[str, ...],
        ik_motion: OpenSimMotion,
        members: Sequence[str],
        model: mujoco.MjModel,
        qpos: np.ndarray,
        kin_time: np.ndarray,
    ) -> np.ndarray | None:
        rel_folder = _DIMITROV_FOLDER_MAP.get(branch)
        zip_path = self.data_root / f"{subj}.zip"
        if rel_folder is None or not zip_path.exists():
            return None
        prefix = f"{subj}/{rel_folder}/"
        ik_files = [
            m for m in members if m.startswith(prefix) and m.lower().endswith("_ik.mot")
        ]
        if not ik_files:
            return None

        t0 = float(ik_motion.time[0])
        tx0 = float(ik_motion.column("pelvis_tx")[0])
        best_stem: str | None = None
        best_err = float("inf")

        with zipfile.ZipFile(zip_path) as zf:
            for ik_member in ik_files:
                cand = _parse_opensim_motion_text(
                    zf.read(ik_member).decode("utf-8", errors="replace")
                )
                if t0 < cand.time[0] - 0.02 or t0 > cand.time[-1] + 0.02:
                    continue
                tx_cand = float(np.interp(t0, cand.time, cand.column("pelvis_tx")))
                err = abs(tx_cand - tx0)
                if err < best_err:
                    best_err = err
                    best_stem = ik_member[: -len("_ik.mot")]

            if best_stem is None or best_err > 0.05:
                return None

            mot_member = next(
                (
                    m
                    for m in members
                    if m.lower() == f"{best_stem.lower()}_filtered.mot"
                ),
                None,
            )
            xml_member = next(
                (
                    m
                    for m in members
                    if m.lower()
                    in (
                        f"{best_stem.lower()}_external_loads.xml",
                        f"{best_stem.lower()}_externalloads.xml",
                    )
                ),
                None,
            )
            if mot_member is None or xml_member is None:
                return None

            grf_mot = _parse_opensim_motion_text(
                zf.read(mot_member).decode("utf-8", errors="replace"), in_degrees=False
            )
            xml_text = zf.read(xml_member).decode("utf-8", errors="replace")

        win_mask = (grf_mot.time >= kin_time[0] - 0.02) & (
            grf_mot.time <= kin_time[-1] + 0.02
        )
        if not np.any(win_mask):
            return None
        vy_cols = [i for i, lb in enumerate(grf_mot.labels) if lb.endswith("_vy")]
        if vy_cols and float(np.abs(grf_mot.values[win_mask][:, vy_cols]).max()) < 25.0:
            return None
        return opensim_grf_to_qfrc(model, qpos, kin_time, grf_mot, xml_text)

    def get_trial(self, trial_id: str, model: mujoco.MjModel | None = None) -> EMGTrial:
        parts = trial_id.split("/")
        if len(parts) < 4 or not parts[-1].startswith("trial_"):
            raise KeyError(f"invalid Dimitrov trial_id {trial_id!r}")
        subj = parts[0]
        branch = tuple(parts[1:-1])
        idx = int(parts[-1].split("_")[1])

        gait_root, cells, members = self._load_subject(subj)
        branch_map = dict(self._iter_branches(gait_root))
        if branch not in branch_map or idx >= len(branch_map[branch]):
            raise KeyError(f"trial {trial_id!r} not found in {subj}.mat")
        rec = branch_map[branch][idx]

        emg_field = (
            "RightLeg_EMG" if "RightLeg_EMG" in rec.dtype.names else "RightFoot_EMG"
        )
        emg_raw = np.asarray(rec[emg_field], dtype=np.float64)
        ik_labels, ik_mat = _decode_mcos_table(rec["IK"], cells)
        ik_time = ik_mat[:, 0]
        ik_motion = OpenSimMotion(
            time=ik_time,
            labels=ik_labels[1:],
            values=ik_mat[:, 1:],
            in_degrees=True,
        )
        emg_time = np.linspace(float(ik_time[0]), float(ik_time[-1]), len(emg_raw))
        ch_names = tuple(f"hdsemg_{c:02d}" for c in range(emg_raw.shape[1]))

        qpos = None
        kin_time = None
        ext_qfrc = None
        used_force_plates = False
        if model is not None:
            mapped = opensim_motion_to_qpos(
                model,
                ik_motion,
                rate=self.kinematics_rate_hz,
                signs=GAIT2392_SIGNS,
            )
            qpos = project_tendon_nodes(model, mapped.qpos)
            kin_time = mapped.time
            ext_qfrc = self._match_force_plate_loads(
                subj, branch, ik_motion, members, model, qpos, kin_time
            )
            used_force_plates = ext_qfrc is not None

        return EMGTrial(
            trial_id=trial_id,
            time=emg_time,
            signals=emg_raw,
            channel_names=ch_names,
            signal_kind="raw",
            qpos=qpos,
            kinematics_time=kin_time,
            external_qfrc=ext_qfrc,
            cycle_events=np.asarray(
                [[float(ik_time[0]), float(ik_time[-1])]], dtype=np.float64
            ),
            filter_config=EMGFilterConfig(
                bandpass_hz=(20.0, 450.0),
                notch_hz=50.0,
                lowpass_hz=6.0,
                normalize="peak",
            ),
            meta={
                "dataset": self.name,
                "subject": subj,
                "ambulation": "/".join(branch),
                "force_plates_matched": used_force_plates,
                "contact_forces_fallback": not used_force_plates,
                "target_actuators": list(DIMITROV_SHANK_ACTUATORS),
            },
        )


CAMARGO_CHANNELS: tuple[str, ...] = (
    "gastrocmed",
    "tibialisanterior",
    "soleus",
    "vastusmedialis",
    "vastuslateralis",
    "rectusfemoris",
    "bicepsfemoris",
    "semitendinosus",
    "gracilis",
    "gluteusmedius",
)

_CAMARGO_STEADY_LABELS: frozenset[str] = frozenset(
    {"walk", "stairascent", "stairdescent", "rampascent", "rampdescent"}
)


def _read_camargo_table(path: Path) -> tuple[tuple[str, ...], np.ndarray]:
    raw_bytes = path.read_bytes()
    mat = sio.loadmat(io.BytesIO(raw_bytes), simplify_cells=False)
    if "__function_workspace__" in mat:
        _, cells = _extract_mcos_cells(raw_bytes)
        return _decode_mcos_table(mat["data"], cells)
    cols = tuple(
        str(np.asarray(x).item()).strip() for x in np.atleast_1d(mat["columns"]).ravel()
    )
    return cols, np.asarray(mat["data"], dtype=np.float64)


def _read_camargo_labels(cond_path: Path, n_frames: int) -> list[str]:
    if not cond_path.exists():
        return [""] * n_frames
    raw_bytes = cond_path.read_bytes()
    mat = sio.loadmat(io.BytesIO(raw_bytes), simplify_cells=False)
    if "__function_workspace__" in mat:
        _, cells = _extract_mcos_cells(raw_bytes)
        if len(cells) > 2 and hasattr(cells[2], "__len__") and len(cells[2]) >= 2:
            lbls = [str(x).strip().lower() for x in np.atleast_1d(cells[2][1]).ravel()]
            if len(lbls) == n_frames:
                return lbls
    elif "labels" in mat:
        lbls = [
            str(np.asarray(x).item()).strip().lower()
            for x in np.atleast_1d(mat["labels"]).ravel()
        ]
        if len(lbls) == n_frames:
            return lbls
    return [""] * n_frames


class Camargo2021Source(EMGSource):
    """Camargo et al. (2021) 10-ch 1 kHz multi-mode lower-limb EMG, IK and ID adapter."""

    name = "camargo2021"

    def __init__(
        self,
        data_root: str | Path = "downloads/emg/camargo2021",
        *,
        subjects: Sequence[str] | None = None,
        pattern: str | None = None,
        kinematics_rate_hz: float = 100.0,
    ) -> None:
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(
                f"Camargo 2021 data_root not found: {self.data_root}"
            )
        self.kinematics_rate_hz = kinematics_rate_hz
        self.pattern = re.compile(pattern) if pattern else None

        subj_dirs = sorted(
            (
                p
                for p in self.data_root.glob("AB*")
                if p.is_dir() and any(p.rglob("emg/*.mat"))
            ),
            key=lambda p: int(re.sub(r"\D", "", p.name) or "0"),
        )
        if subjects is not None:
            wanted = {s.upper() for s in subjects}
            subj_dirs = [p for p in subj_dirs if p.name.upper() in wanted]
        if not subj_dirs:
            raise FileNotFoundError(
                f"no AB* subjects with emg/*.mat under {self.data_root}"
            )

        self.subjects = tuple(p.name for p in subj_dirs)
        self._mode_dirs: dict[tuple[str, str], Path] = {}
        self._stems_by_key: dict[tuple[str, str], list[str]] = {}
        for sdir in subj_dirs:
            for emg_dir in sorted(sdir.rglob("emg")):
                if not emg_dir.is_dir():
                    continue
                mode_dir = emg_dir.parent
                stems = sorted(
                    f.stem
                    for f in emg_dir.glob("*.mat")
                    if (mode_dir / "ik" / f.name).exists()
                )
                if stems:
                    key = (sdir.name, mode_dir.name)
                    self._mode_dirs[key] = mode_dir
                    self._stems_by_key[key] = stems
        self._ids_cache: tuple[str, ...] | None = None

    def trial_ids(self) -> Sequence[str]:
        if self._ids_cache is not None:
            return self._ids_cache
        prio: list[str] = []
        rest: list[str] = []
        mode_pref = (
            ("treadmill", ""),
            ("levelground", "normal"),
            ("stair", "_r_"),
            ("ramp", "_l_"),
        )
        for mode, token in mode_pref:
            for subj in self.subjects:
                stems = self._stems_by_key.get((subj, mode), [])
                if not stems:
                    continue
                chosen = next((s for s in stems if token in s), stems[0])
                prio.append(f"{subj}/{mode}/{chosen}")
        seen = set(prio)
        for (subj, mode), stems in sorted(self._stems_by_key.items()):
            for stem in stems:
                tid = f"{subj}/{mode}/{stem}"
                if tid not in seen:
                    seen.add(tid)
                    rest.append(tid)
        all_ids = prio + rest
        if self.pattern is not None:
            all_ids = [t for t in all_ids if self.pattern.search(t)]
        self._ids_cache = tuple(all_ids)
        return self._ids_cache

    def default_channel_map(self, model: mujoco.MjModel) -> EMGChannelMap:
        del model
        return EMGChannelMap(
            name="camargo2021_right_leg_10ch",
            version=1,
            bindings=(
                ChannelBinding("gastrocmed", ("gasmed_r",), reference_peak=0.87),
                ChannelBinding("tibialisanterior", ("tibant_r",), reference_peak=0.37),
                ChannelBinding("soleus", ("soleus_r",), reference_peak=0.77),
                ChannelBinding("vastusmedialis", ("vasmed_r",)),
                ChannelBinding("vastuslateralis", ("vaslat_r",)),
                ChannelBinding("rectusfemoris", ("recfem_r",)),
                ChannelBinding("bicepsfemoris", ("bflh_r", "bfsh_r")),
                ChannelBinding("semitendinosus", ("semiten_r",)),
                ChannelBinding("gracilis", ("grac_r",)),
                ChannelBinding("gluteusmedius", ("glmed1_r", "glmed2_r", "glmed3_r")),
            ),
        )

    @staticmethod
    def _select_stride(
        t_ik: np.ndarray,
        gc_arr: np.ndarray,
        id_cols: tuple[str, ...],
        id_arr: np.ndarray,
        labels: Sequence[str],
    ) -> tuple[int, int, str, bool]:
        hs_idx = (
            np.flatnonzero(np.diff(gc_arr[:, 1]) < -50.0) + 1
            if gc_arr.ndim == 2
            and gc_arr.shape[0] == len(t_ik)
            and gc_arr.shape[1] >= 2
            else np.empty(0, dtype=np.int64)
        )
        req_moments = (
            "hip_flexion_r_moment",
            "knee_angle_r_moment",
            "ankle_angle_r_moment",
        )
        has_id = (
            id_arr.ndim == 2
            and id_arr.shape[0] == len(t_ik)
            and all(c in id_cols for c in req_moments)
        )
        m_idxs = [id_cols.index(c) for c in req_moments] if has_id else []
        ank_col = id_cols.index("ankle_angle_r_moment") if has_id else -1

        best: tuple[tuple[int, int, float], int, int, str, bool] | None = None
        n_hs = len(hs_idx)
        for k in range(n_hs - 1):
            i0, i1 = int(hs_idx[k]), int(hs_idx[k + 1])
            dur = float(t_ik[i1] - t_ik[i0])
            if dur < 0.55 or dur > 2.25 or i1 - i0 < 20:
                continue
            lbl = labels[(i0 + i1) // 2] if len(labels) == len(t_ik) else ""
            if lbl in ("idle", "stand"):
                lbl_tier = 0
            elif (
                lbl in _CAMARGO_STEADY_LABELS
                or lbl == ""
                or lbl.replace(".", "", 1).isdigit()
            ):
                lbl_tier = 3
            elif "turn" not in lbl:
                lbl_tier = 2
            else:
                lbl_tier = 1

            fp_ok = False
            impulse = 0.0
            if has_id:
                sub_m = id_arr[i0 : i1 + 1][:, m_idxs]
                fin_frac = float(np.isfinite(sub_m).all(axis=1).mean())
                ank_min = (
                    float(np.nanmin(id_arr[i0 : i1 + 1, ank_col]))
                    if fin_frac > 0.5
                    else 0.0
                )
                fp_ok = fin_frac >= 0.90 and ank_min <= -35.0
                if fin_frac >= 0.90:
                    impulse = float(np.nanmean(np.abs(sub_m)))
            center_bonus = -abs((k + 0.5) / max(n_hs - 1, 1) - 0.5) * 5.0
            score = (1 if fp_ok else 0, lbl_tier, impulse + center_bonus)
            if best is None or score > best[0]:
                best = (score, i0, i1, lbl, fp_ok)

        if best is not None:
            _, i0, i1, lbl, fp_ok = best
            return i0, i1, lbl, fp_ok

        fp_full = False
        if has_id:
            fin_frac = float(np.isfinite(id_arr[:, m_idxs]).all(axis=1).mean())
            fp_full = fin_frac >= 0.90 and float(np.nanmin(id_arr[:, ank_col])) <= -35.0
        lbl0 = labels[len(t_ik) // 2] if len(labels) == len(t_ik) else ""
        return 0, len(t_ik) - 1, lbl0, fp_full

    def get_trial(self, trial_id: str, model: mujoco.MjModel | None = None) -> EMGTrial:
        parts = trial_id.split("/")
        if len(parts) != 3:
            raise KeyError(f"invalid Camargo 2021 trial_id {trial_id!r}")
        subj, mode, stem = parts
        mode_dir = self._mode_dirs.get((subj, mode))
        if mode_dir is None or not (mode_dir / "emg" / f"{stem}.mat").exists():
            raise KeyError(f"unknown Camargo 2021 trial {trial_id!r}")

        emg_cols, emg_arr = _read_camargo_table(mode_dir / "emg" / f"{stem}.mat")
        ik_cols, ik_arr = _read_camargo_table(mode_dir / "ik" / f"{stem}.mat")
        id_path = mode_dir / "id" / f"{stem}.mat"
        id_cols, id_arr = (
            _read_camargo_table(id_path) if id_path.exists() else ((), np.empty((0, 0)))
        )
        gc_path = mode_dir / "gcRight" / f"{stem}.mat"
        _, gc_arr = (
            _read_camargo_table(gc_path) if gc_path.exists() else ((), np.empty((0, 0)))
        )
        labels = _read_camargo_labels(
            mode_dir / "conditions" / f"{stem}.mat", len(ik_arr)
        )

        t_ik = ik_arr[:, 0]
        i0, i1, stride_lbl, has_fp_id = self._select_stride(
            t_ik, gc_arr, id_cols, id_arr, labels
        )
        t0, t1 = float(t_ik[i0]), float(t_ik[i1])

        emg_mask = (emg_arr[:, 0] >= t0 - 1e-6) & (emg_arr[:, 0] <= t1 + 1e-6)
        emg_time = emg_arr[emg_mask, 0] - t0
        ch_indices = [emg_cols.index(ch) for ch in CAMARGO_CHANNELS if ch in emg_cols]
        ch_names = tuple(emg_cols[c] for c in ch_indices)
        emg_raw = emg_arr[emg_mask][:, ch_indices]

        ik_motion = OpenSimMotion(
            time=t_ik[i0 : i1 + 1] - t0,
            labels=ik_cols[1:],
            values=ik_arr[i0 : i1 + 1, 1:],
            in_degrees=True,
        )

        qpos = None
        kin_time = None
        ext_qfrc = None
        if model is not None:
            mapped = opensim_motion_to_qpos(
                model, ik_motion, rate=self.kinematics_rate_hz, signs=GAIT2392_SIGNS
            )
            qpos = project_tendon_nodes(model, mapped.qpos)
            kin_time = mapped.time
            if has_fp_id:
                dt = float(np.median(np.diff(kin_time)))
                kin = clip_kinematics(model, qpos, dt, lowpass_hz=6.0)
                data = mujoco.MjData(model)
                ext_qfrc = np.zeros((len(qpos), model.nv), dtype=np.float64)
                j_lookup = {model.joint(j).name: j for j in range(model.njnt)}
                slave_dofs = [
                    int(model.jnt_dofadr[int(model.eq_obj1id[e])])
                    for e in range(model.neq)
                    if model.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT
                    and model.eq_active0[e]
                ]
                root_dofs = [
                    d_idx
                    for j in np.flatnonzero(
                        model.jnt_type == mujoco.mjtJoint.mjJNT_FREE
                    )
                    for d_idx in range(
                        int(model.jnt_dofadr[j]), int(model.jnt_dofadr[j]) + 6
                    )
                ]
                dof_specs = [
                    (
                        int(model.jnt_dofadr[j_lookup[jn]]),
                        id_cols.index(f"{jn}_moment"),
                        float(GAIT2392_SIGNS.get(jn, 1.0)),
                    )
                    for jn in (
                        "hip_flexion_r",
                        "hip_adduction_r",
                        "hip_rotation_r",
                        "knee_angle_r",
                        "ankle_angle_r",
                        "subtalar_angle_r",
                        "mtp_angle_r",
                    )
                    if jn in j_lookup and f"{jn}_moment" in id_cols
                ]
                t_id_rel = t_ik[i0 : i1 + 1] - t0
                for ti in range(len(qpos)):
                    data.qpos[:], data.qvel[:] = kin.qpos[ti], kin.qvel[ti]
                    mujoco.mj_fwdPosition(model, data)
                    mujoco.mj_fwdVelocity(model, data)
                    tau_id = required_qfrc(model, data, kin.qacc[ti])
                    ext_qfrc[ti, root_dofs] = tau_id[root_dofs]
                    ext_qfrc[ti, slave_dofs] = tau_id[slave_dofs]
                    for dadr, c_idx, sgn in dof_specs:
                        col_vals = id_arr[i0 : i1 + 1, c_idx]
                        fin = np.isfinite(col_vals)
                        if np.any(fin):
                            tau_tgt = sgn * float(
                                np.interp(kin_time[ti], t_id_rel[fin], col_vals[fin])
                            )
                            ext_qfrc[ti, dadr] = tau_id[dadr] - tau_tgt

        return EMGTrial(
            trial_id=trial_id,
            time=emg_time,
            signals=emg_raw,
            channel_names=ch_names,
            signal_kind="raw",
            qpos=qpos,
            kinematics_time=kin_time,
            external_qfrc=ext_qfrc,
            cycle_events=np.asarray([[0.0, t1 - t0]], dtype=np.float64),
            filter_config=EMGFilterConfig(
                bandpass_hz=(20.0, 400.0),
                notch_hz=60.0,
                lowpass_hz=6.0,
                normalize="peak",
            ),
            meta={
                "dataset": self.name,
                "subject": subj,
                "ambulation": mode,
                "condition_label": stride_lbl,
                "stride_window_s": [t0, t1],
                "force_plates_matched": has_fp_id,
                "contact_forces_fallback": not has_fp_id,
                "target_actuators": [
                    model.actuator(i).name
                    for i in range(model.nu)
                    if model.actuator(i).name.endswith("_r")
                ]
                if model is not None
                else [],
            },
        )


def _read_tabular_file(
    path: Path, default_rate_hz: float | None
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    suffix = path.suffix.lower()
    if suffix in (".sto", ".mot"):
        mot = read_opensim_motion(path, in_degrees=False)
        return mot.time, mot.labels, mot.values
    if suffix in (".csv", ".tsv"):
        delim = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter=delim)
            header = [h.strip() for h in next(reader)]
            rows = [[float(x) for x in row] for row in reader if any(row)]
        arr = np.asarray(rows, dtype=np.float64)
        if header[0].lower() in ("time", "t", "time_s", "seconds"):
            return arr[:, 0], tuple(header[1:]), arr[:, 1:]
        if default_rate_hz is None or default_rate_hz <= 0:
            raise ValueError(
                f"{path}: CSV has no leading 'time' column; set sampling_rate_hz in manifest"
            )
        t = np.arange(len(arr), dtype=np.float64) / float(default_rate_hz)
        return t, tuple(header), arr
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as d:
            sig = d["signals"] if "signals" in d.files else d["emg"]
            if "time" in d.files:
                t = np.asarray(d["time"], dtype=np.float64)
            elif default_rate_hz:
                t = np.arange(len(sig), dtype=np.float64) / float(default_rate_hz)
            else:
                raise ValueError(f"{path}: missing 'time' array and sampling_rate_hz")
            names = (
                tuple(str(x) for x in d["channel_names"].tolist())
                if "channel_names" in d.files
                else tuple(f"ch_{i:02d}" for i in range(sig.shape[1]))
            )
            return t, names, np.asarray(sig, dtype=np.float64)
    if suffix == ".mat":
        mat = sio.loadmat(path, simplify_cells=True)
        sig = np.asarray(mat.get("signals", mat.get("emg")), dtype=np.float64)
        if "time" in mat:
            t = np.asarray(mat["time"], dtype=np.float64).ravel()
        elif default_rate_hz:
            t = np.arange(len(sig), dtype=np.float64) / float(default_rate_hz)
        else:
            raise ValueError(f"{path}: MAT file requires 'time' or sampling_rate_hz")
        raw_names = mat.get("channel_names")
        names = (
            tuple(str(x) for x in raw_names)
            if raw_names is not None
            else tuple(f"ch_{i:02d}" for i in range(sig.shape[1]))
        )
        return t, names, sig
    if suffix in (".hdf5", ".h5"):
        import h5py

        with h5py.File(path, "r", locking=False) as hf:
            sig_key = "signals" if "signals" in hf else "emg"
            sig = np.asarray(hf[sig_key], dtype=np.float64)
            if "time" in hf:
                t = np.asarray(hf["time"], dtype=np.float64).ravel()
            elif default_rate_hz:
                t = np.arange(len(sig), dtype=np.float64) / float(default_rate_hz)
            else:
                raise ValueError(
                    f"{path}: HDF5 requires 'time' dataset or sampling_rate_hz"
                )
            names = tuple(f"ch_{i:02d}" for i in range(sig.shape[1]))
            return t, names, sig
    raise ValueError(f"unsupported EMG file format: {path}")


def _load_manifest_kinematics(
    path: Path,
    model: mujoco.MjModel,
    *,
    rate_hz: float | None,
    in_degrees: bool | None,
    window_s: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix in (".mot", ".sto"):
        raw = read_opensim_motion(path, in_degrees=in_degrees)
        clean, signs = normalize_opensim_motion(raw)
        if window_s is not None:
            clean = _slice_opensim_motion(clean, window_s[0], window_s[1])
        has_pelvis = all(c in clean.labels for c in PELVIS_COORDINATES)
        mapped = opensim_motion_to_qpos(
            model,
            clean,
            rate=rate_hz,
            root_body="pelvis" if has_pelvis else None,
            signs=signs,
        )
        return project_tendon_nodes(model, mapped.qpos), mapped.time

    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as d:
            qpos = np.asarray(d["qpos"], dtype=np.float64)
            if "time" in d.files:
                t = np.asarray(d["time"], dtype=np.float64)
            else:
                freq = (
                    float(d["frequency_hz"])
                    if "frequency_hz" in d.files
                    else (rate_hz or 100.0)
                )
                t = np.arange(len(qpos), dtype=np.float64) / freq
        if window_s is not None:
            mask = (t >= window_s[0] - 1e-9) & (t <= window_s[1] + 1e-9)
            qpos, t = qpos[mask], t[mask]
        return project_tendon_nodes(model, qpos), t

    if suffix in (".csv", ".tsv"):
        t, labels, vals = _read_tabular_file(path, rate_hz)
        mot = OpenSimMotion(
            time=t,
            labels=labels,
            values=vals,
            in_degrees=bool(in_degrees) if in_degrees is not None else False,
        )
        if window_s is not None:
            mot = _slice_opensim_motion(mot, window_s[0], window_s[1])
        has_pelvis = all(c in mot.labels for c in PELVIS_COORDINATES)
        mapped = opensim_motion_to_qpos(
            model,
            mot,
            rate=rate_hz,
            root_body="pelvis" if has_pelvis else None,
            signs={},
        )
        return project_tendon_nodes(model, mapped.qpos), mapped.time

    raise ValueError(f"unsupported kinematics file format: {path}")


class ManifestEMGSource(EMGSource):
    """Declarative JSON/YAML dataset loader for custom lab EMG recordings."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        text = self.manifest_path.read_text(encoding="utf-8")
        if self.manifest_path.suffix.lower() in (".yaml", ".yml"):
            import yaml

            spec = yaml.safe_load(text)
        else:
            spec = json.loads(text)
        if not isinstance(spec, Mapping) or "trials" not in spec:
            raise ValueError(
                f"{self.manifest_path}: manifest must contain a 'trials' list"
            )

        self.spec = dict(spec)
        self.base_dir = self.manifest_path.parent
        self.name = str(spec.get("name", self.manifest_path.stem))
        self._trials = {str(item["trial_id"]): dict(item) for item in spec["trials"]}

    def _resolve_path(self, rel_or_abs: str | Path) -> Path:
        p = Path(rel_or_abs)
        return p if p.is_absolute() else (self.base_dir / p)

    def trial_ids(self) -> Sequence[str]:
        return tuple(self._trials.keys())

    def default_channel_map(self, model: mujoco.MjModel) -> EMGChannelMap | None:
        cmap_spec = self.spec.get("channel_map")
        if cmap_spec is None:
            sites = self.spec.get("electrode_sites")
            if sites:
                return spatial_site_channel_map(
                    model,
                    [str(s) for s in sites],
                    sigma_m=float(self.spec.get("sigma_m", 0.035)),
                    top_k=int(self.spec.get("top_k", 4)),
                    side=self.spec.get("side"),
                    name=f"{self.name}_sites",
                )
            return None
        return EMGChannelMap.from_dict(cmap_spec)

    def get_trial(self, trial_id: str, model: mujoco.MjModel | None = None) -> EMGTrial:
        if trial_id not in self._trials:
            raise KeyError(f"unknown manifest trial {trial_id!r}")
        entry = self._trials[trial_id]
        emg_path = self._resolve_path(entry["emg_path"])

        if emg_path.suffix.lower() == ".npz":
            with np.load(emg_path, allow_pickle=False) as d:
                if EMG_TRIAL_META_KEY in d.files:
                    return load_emg_trial(emg_path)

        rate_hz = entry.get("sampling_rate_hz", self.spec.get("sampling_rate_hz"))
        rate_f = None if rate_hz is None else float(rate_hz)
        time, ch_names, signals = _read_tabular_file(emg_path, rate_f)
        if "channel_names" in entry:
            ch_names = tuple(str(c) for c in entry["channel_names"])
        elif "channel_names" in self.spec:
            ch_names = tuple(str(c) for c in self.spec["channel_names"])

        win_raw = entry.get("window_s", self.spec.get("window_s"))
        win = (float(win_raw[0]), float(win_raw[1])) if win_raw is not None else None
        if win is not None:
            mask = (time >= win[0] - 1e-9) & (time <= win[1] + 1e-9)
            time, signals = time[mask], signals[mask]

        signal_kind = str(
            entry.get("signal_kind", self.spec.get("signal_kind", "peak_normalized"))
        )
        mvc_raw = entry.get("mvc_peaks", self.spec.get("mvc_peaks"))
        mvc_peaks = None if mvc_raw is None else np.asarray(mvc_raw, dtype=np.float64)

        filt_raw = entry.get("filter", self.spec.get("filter"))
        fcfg = None
        if isinstance(filt_raw, Mapping):
            bp = filt_raw.get("bandpass_hz", (20.0, 450.0))
            fcfg = EMGFilterConfig(
                bandpass_hz=None if bp is None else (float(bp[0]), float(bp[1])),
                notch_hz=filt_raw.get("notch_hz"),
                notch_q=float(filt_raw.get("notch_q", 30.0)),
                lowpass_hz=filt_raw.get("lowpass_hz", 6.0),
                order=int(filt_raw.get("order", 2)),
                normalize=str(
                    filt_raw.get(
                        "normalize", "mvc" if mvc_peaks is not None else "peak"
                    )
                ),
            )

        qpos = None
        kin_time = None
        ext_qfrc = None
        kin_rel = entry.get("kinematics_path")
        if model is not None and kin_rel is not None:
            kin_rate = entry.get(
                "kinematics_rate_hz", self.spec.get("kinematics_rate_hz")
            )
            in_deg = entry.get(
                "kinematics_in_degrees", self.spec.get("kinematics_in_degrees")
            )
            qpos, kin_time = _load_manifest_kinematics(
                self._resolve_path(kin_rel),
                model,
                rate_hz=None if kin_rate is None else float(kin_rate),
                in_degrees=None if in_deg is None else bool(in_deg),
                window_s=win,
            )
            grf_rel = entry.get("grf_path")
            loads_rel = entry.get("external_loads_xml")
            if grf_rel is not None and loads_rel is not None:
                grf_mot = read_opensim_motion(
                    self._resolve_path(grf_rel), in_degrees=False
                )
                ext_qfrc = opensim_grf_to_qfrc(
                    model, qpos, kin_time, grf_mot, self._resolve_path(loads_rel)
                )

        events_raw = entry.get("cycle_events")
        events = (
            None if events_raw is None else np.asarray(events_raw, dtype=np.float64)
        )
        return EMGTrial(
            trial_id=trial_id,
            time=time,
            signals=signals,
            channel_names=ch_names,
            signal_kind=signal_kind,
            qpos=qpos,
            kinematics_time=kin_time,
            external_qfrc=ext_qfrc,
            cycle_events=events,
            mvc_peaks=mvc_peaks,
            filter_config=fcfg,
            meta={
                "dataset": self.name,
                "manifest": str(self.manifest_path),
                **dict(entry.get("meta", {})),
            },
        )


__all__ = [
    "CAMARGO_CHANNELS",
    "DIMITROV_SHANK_ACTUATORS",
    "LENCIONI_CHANNELS",
    "Camargo2021Source",
    "Dimitrov2023Source",
    "Lencioni2019Source",
    "ManifestEMGSource",
    "RajagopalWalkingSource",
]
