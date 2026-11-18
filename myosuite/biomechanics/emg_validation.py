# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Model-data EMG comparison and musculoskeletal validation modes."""

from __future__ import annotations

import copy
import importlib
import re
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

import mujoco
import numpy as np

from myosuite.biomechanics.effort import (
    cocontraction_index,
    muscle_saturation,
    squared_activation,
    wasted_torque,
)
from myosuite.biomechanics.emg import (
    EMGFilterConfig,
    activation_to_excitation,
    excitation_to_activation,
    phase_normalize,
    process_emg,
    project_tendon_nodes,
    shift_signal,
)
from myosuite.biomechanics.inverse_dynamics import ClipKinematics, clip_kinematics
from myosuite.biomechanics.static_optimization import (
    StaticOptimizationConfig,
    StaticOptimizationResult,
    StaticOptimizer,
    static_optimization,
)
from myosuite.biomechanics.uncertainty import (
    UncertaintyConfig,
    perturbed_model,
    static_optimization_bands,
)
from myosuite.core.muscle_groups import FUNCTIONAL_MUSCLE_GROUPS, MUSCLE_FMG

_SIGNAL_KINDS = ("raw", "peak_normalized", "mvc_normalized")


@dataclass(frozen=True)
class ChannelBinding:
    """Mapping from one experimental EMG channel to model actuators."""

    channel: str
    actuators: tuple[str, ...] = ()
    weights: tuple[float, ...] = ()
    fmg: str | None = None
    side: str | None = None
    reference_peak: float | None = None


@dataclass(frozen=True)
class ResolvedChannelMap:
    """Compiled projection between model actuators and a trial's EMG channels."""

    channels: tuple[str, ...]
    channel_indices: np.ndarray
    matrix: np.ndarray
    actuator_ids: tuple[tuple[int, ...], ...]
    reference_peaks: np.ndarray
    unmatched_channels: tuple[str, ...]


def _resolve_actuator_names(
    query: str, side: str | None, act_lookup: Mapping[str, int]
) -> list[str]:
    if query in act_lookup:
        return [query]
    inferred_side = side.lower() if side else None
    stem = query
    if query.endswith(("_r", "_l")):
        inferred_side = inferred_side or query[-1]
        stem = query[:-2]
    if stem in act_lookup and (
        inferred_side is None
        or f"{stem}_{'l' if inferred_side == 'r' else 'r'}" not in act_lookup
    ):
        return [stem]
    if inferred_side is not None and f"{stem}_{inferred_side}" in act_lookup:
        return [f"{stem}_{inferred_side}"]
    if inferred_side is None:
        sided = [c for c in (f"{stem}_r", f"{stem}_l") if c in act_lookup]
        if len(sided) == 1:
            return sided
    if not stem or stem[-1].isdigit():
        return []
    suf_pat = rf"(?:_{inferred_side})?" if inferred_side else r"(?:_[rl])?"
    pat = re.compile(rf"^{re.escape(stem)}\d+{suf_pat}$")
    return [
        cand
        for cand in act_lookup
        if pat.match(cand)
        and (
            inferred_side is None
            or not cand.endswith(("_r", "_l"))
            or cand.endswith(f"_{inferred_side}")
        )
    ]


def _expand_fmg_candidates(
    fmg: str, side: str | None, actuator_map: Mapping[str, int]
) -> list[str]:
    if fmg not in FUNCTIONAL_MUSCLE_GROUPS:
        raise KeyError(
            f"unknown functional muscle group {fmg!r}; "
            f"choose from {sorted(FUNCTIONAL_MUSCLE_GROUPS)}"
        )
    matched: list[str] = []
    for base, grp in MUSCLE_FMG.items():
        if grp == fmg:
            for cand in _resolve_actuator_names(base, side, actuator_map):
                if cand not in matched:
                    matched.append(cand)
    return matched


@dataclass(frozen=True)
class EMGChannelMap:
    """Declarative mapping between experimental EMG channels and model actuators."""

    name: str
    version: int = 1
    bindings: tuple[ChannelBinding, ...] = ()

    def resolve(
        self, model: mujoco.MjModel, channel_names: Sequence[str]
    ) -> ResolvedChannelMap:
        """Compile *bindings* against *model* and *channel_names*."""
        act_lookup = {model.actuator(i).name: i for i in range(model.nu)}
        ch_lookup = {name: idx for idx, name in enumerate(channel_names)}
        matched_names: list[str] = []
        matched_idx: list[int] = []
        rows: list[np.ndarray] = []
        act_ids_per_ch: list[tuple[int, ...]] = []
        ref_peaks: list[float] = []
        unmatched: list[str] = []

        for b in self.bindings:
            if b.channel not in ch_lookup:
                unmatched.append(b.channel)
                continue
            explicit = list(b.actuators)
            if b.weights and len(b.weights) != len(explicit):
                raise ValueError(
                    f"channel {b.channel!r}: weights length {len(b.weights)} "
                    f"must match actuators length {len(explicit)}"
                )
            fmg_names = (
                _expand_fmg_candidates(b.fmg, b.side, act_lookup)
                if b.fmg is not None
                else []
            )
            pairs: list[tuple[int, float]] = []
            if explicit and b.weights:
                for a, w in zip(explicit, b.weights, strict=True):
                    if w < 0.0:
                        raise ValueError(f"channel {b.channel!r}: negative weight {w}")
                    if w > 0.0:
                        cands = _resolve_actuator_names(a, b.side, act_lookup)
                        pairs.extend(
                            (act_lookup[c], float(w) / len(cands)) for c in cands
                        )
            else:
                for a in explicit:
                    cands = _resolve_actuator_names(a, b.side, act_lookup)
                    pairs.extend((act_lookup[c], 1.0 / len(cands)) for c in cands)
                pairs.extend((act_lookup[c], 1.0) for c in fmg_names)

            if not pairs:
                unmatched.append(b.channel)
                continue
            row = np.zeros(model.nu, dtype=np.float64)
            ids: list[int] = []
            for aid, w in pairs:
                row[aid] += w
                if aid not in ids:
                    ids.append(aid)
            total = float(row.sum())
            if total <= 0.0:
                unmatched.append(b.channel)
                continue
            row /= total
            matched_names.append(b.channel)
            matched_idx.append(ch_lookup[b.channel])
            rows.append(row)
            act_ids_per_ch.append(tuple(ids))
            ref_peaks.append(
                np.nan if b.reference_peak is None else float(b.reference_peak)
            )

        matrix = (
            np.stack(rows, axis=0)
            if rows
            else np.zeros((0, model.nu), dtype=np.float64)
        )
        return ResolvedChannelMap(
            channels=tuple(matched_names),
            channel_indices=np.asarray(matched_idx, dtype=np.int64),
            matrix=matrix,
            actuator_ids=tuple(act_ids_per_ch),
            reference_peaks=np.asarray(ref_peaks, dtype=np.float64),
            unmatched_channels=tuple(unmatched),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this channel map to a JSON-compatible dictionary."""
        out_bindings = []
        for b in self.bindings:
            item: dict[str, Any] = {"channel": b.channel}
            if b.actuators:
                item["actuators"] = list(b.actuators)
            if b.weights:
                item["weights"] = list(b.weights)
            if b.fmg is not None:
                item["fmg"] = b.fmg
            if b.side is not None:
                item["side"] = b.side
            if b.reference_peak is not None:
                item["reference_peak"] = b.reference_peak
            out_bindings.append(item)
        return {"name": self.name, "version": self.version, "bindings": out_bindings}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EMGChannelMap:
        """Build an :class:`EMGChannelMap` from a parsed JSON/YAML mapping."""
        bindings = tuple(
            ChannelBinding(
                channel=str(item["channel"]),
                actuators=tuple(str(a) for a in item.get("actuators", ())),
                weights=tuple(float(w) for w in item.get("weights", ())),
                fmg=item.get("fmg"),
                side=item.get("side"),
                reference_peak=(
                    None
                    if item.get("reference_peak") is None
                    else float(item["reference_peak"])
                ),
            )
            for item in data.get("bindings", ())
        )
        return cls(
            name=str(data.get("name", "custom")),
            version=int(data.get("version", 1)),
            bindings=bindings,
        )


def _actuator_anchor_points(
    model: mujoco.MjModel, data: mujoco.MjData
) -> list[np.ndarray]:
    wrap_site = int(mujoco.mjtWrap.mjWRAP_SITE)
    wrap_geom = (int(mujoco.mjtWrap.mjWRAP_SPHERE), int(mujoco.mjtWrap.mjWRAP_CYLINDER))
    trn_tendon = int(mujoco.mjtTrn.mjTRN_TENDON)
    trn_joint = int(mujoco.mjtTrn.mjTRN_JOINT)
    trn_site = int(mujoco.mjtTrn.mjTRN_SITE)

    points: list[np.ndarray] = []
    for i in range(model.nu):
        trn_type, trn_id = (
            int(model.actuator_trntype[i]),
            int(model.actuator_trnid[i, 0]),
        )
        pts: list[np.ndarray] = []
        if trn_type == trn_tendon and trn_id >= 0:
            adr, num = int(model.tendon_adr[trn_id]), int(model.tendon_num[trn_id])
            for w in range(adr, adr + num):
                wtype, oid = int(model.wrap_type[w]), int(model.wrap_objid[w])
                if wtype == wrap_site and oid >= 0:
                    pts.append(data.site_xpos[oid].copy())
                elif wtype in wrap_geom and oid >= 0:
                    pts.append(data.geom_xpos[oid].copy())
        elif trn_type == trn_joint and trn_id >= 0:
            pts.append(data.xanchor[trn_id].copy())
        elif trn_type == trn_site and trn_id >= 0:
            pts.append(data.site_xpos[trn_id].copy())
        points.append(np.stack(pts, axis=0) if pts else np.zeros((0, 3)))
    return points


def spatial_site_channel_map(
    model: mujoco.MjModel,
    site_names: Sequence[str],
    *,
    sigma_m: float = 0.035,
    top_k: int = 4,
    side: str | None = None,
    name: str = "spatial_sites",
) -> EMGChannelMap:
    """Build a Gaussian distance-weighted channel map from electrode ``<site>``s."""
    if sigma_m <= 0 or top_k < 1:
        raise ValueError("sigma_m must be > 0 and top_k >= 1")
    data = mujoco.MjData(model)
    data.qpos[:] = project_tendon_nodes(model, model.qpos0)
    mujoco.mj_fwdPosition(model, data)
    anchors = _actuator_anchor_points(model, data)
    suffix = f"_{side.lower()}" if side else None

    bindings: list[ChannelBinding] = []
    for sname in site_names:
        spos = data.site_xpos[model.site(sname).id]
        candidates: list[tuple[float, str]] = []
        for aid, pts in enumerate(anchors):
            aname = model.actuator(aid).name
            if suffix and aname.endswith(("_r", "_l")) and not aname.endswith(suffix):
                continue
            if len(pts):
                candidates.append(
                    (float(np.linalg.norm(pts - spos[None, :], axis=1).min()), aname)
                )
        candidates.sort(key=lambda item: item[0])
        chosen = candidates[:top_k]
        keep = [
            (a, w)
            for d, a in chosen
            if (w := float(np.exp(-0.5 * (d / sigma_m) ** 2))) > 1e-6
        ]
        bindings.append(
            ChannelBinding(
                channel=sname,
                actuators=tuple(a for a, _ in keep),
                weights=tuple(w for _, w in keep),
            )
        )
    return EMGChannelMap(name=name, bindings=tuple(bindings))


@dataclass(frozen=True)
class EMGTrial:
    """Standardized experimental EMG and kinematics trial container."""

    trial_id: str
    time: np.ndarray
    signals: np.ndarray
    channel_names: tuple[str, ...]
    signal_kind: str = "peak_normalized"
    qpos: np.ndarray | None = None
    kinematics_time: np.ndarray | None = None
    external_qfrc: np.ndarray | None = None
    cycle_events: np.ndarray | None = None
    mvc_peaks: np.ndarray | None = None
    filter_config: EMGFilterConfig | None = None
    supported_dofs: tuple[int, ...] = ()
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.signal_kind not in _SIGNAL_KINDS:
            raise ValueError(
                f"signal_kind must be one of {_SIGNAL_KINDS}, got {self.signal_kind!r}"
            )
        t = np.asarray(self.time, dtype=np.float64)
        s = np.asarray(self.signals, dtype=np.float64)
        if t.ndim != 1 or len(t) < 4 or np.any(np.diff(t) <= 0):
            raise ValueError("time must be 1D strictly increasing with length >= 4")
        if s.ndim != 2 or s.shape != (len(t), len(self.channel_names)):
            raise ValueError(
                f"signals shape {s.shape} does not match ({len(t)}, {len(self.channel_names)})"
            )

    def resolved_kinematics_time(self) -> np.ndarray:
        """Sample times [s] aligned with :attr:`qpos`."""
        if self.qpos is None:
            raise ValueError(
                f"trial {self.trial_id!r} has no kinematics (qpos is None)"
            )
        if self.kinematics_time is not None:
            kt = np.asarray(self.kinematics_time, dtype=np.float64)
            if kt.shape != (len(self.qpos),):
                raise ValueError(
                    f"kinematics_time shape {kt.shape} != ({len(self.qpos)},)"
                )
            return kt
        if len(self.qpos) != len(self.time):
            raise ValueError("kinematics_time is required when len(qpos) != len(time)")
        return np.asarray(self.time, dtype=np.float64)

    def envelopes(self, target_time: np.ndarray | None = None) -> np.ndarray:
        """Non-negative linear EMG envelopes, optionally resampled onto *target_time*."""
        t = np.asarray(self.time, dtype=np.float64)
        s = np.asarray(self.signals, dtype=np.float64)
        if self.signal_kind == "raw":
            fs = 1.0 / float(np.median(np.diff(t)))
            cfg = self.filter_config or EMGFilterConfig(
                normalize="mvc" if self.mvc_peaks is not None else "peak"
            )
            env = process_emg(s, fs, cfg, mvc_peaks=self.mvc_peaks)
        else:
            env = np.maximum(s, 0.0)
        if target_time is None:
            return env
        tt = np.asarray(target_time, dtype=np.float64)
        if tt.shape == t.shape and np.allclose(tt, t, atol=1e-9):
            return env
        return np.column_stack(
            [np.interp(tt, t, env[:, c]) for c in range(env.shape[1])]
        )


@runtime_checkable
class EMGSource(Protocol):
    """Dataset adapter supplying :class:`EMGTrial` clips and default channel maps."""

    name: str

    def trial_ids(self) -> Sequence[str]:
        """Available trial identifiers in deterministic order."""

    def get_trial(self, trial_id: str, model: mujoco.MjModel | None = None) -> EMGTrial:
        """Load one :class:`EMGTrial`, optionally mapped onto *model*."""

    def default_channel_map(self, model: mujoco.MjModel) -> EMGChannelMap | None:
        """Default :class:`EMGChannelMap` for *model*, or ``None`` for unmapped arrays."""


def musculoskeletal_fingerprint(model: mujoco.MjModel) -> str:
    """16-hex SHA-256 of kinematics, inertias, equalities, contacts, tendons and muscles.

    Delegates to :func:`myo_sim.fingerprint.musculoskeletal_fingerprint`.
    """
    try:
        from myo_sim.fingerprint import musculoskeletal_fingerprint as fingerprint
    except ModuleNotFoundError as exc:
        if exc.name != "myo_sim.fingerprint":
            raise
        raise ImportError(
            "The musculoskeletal fingerprint requires a myo-sim build containing "
            "MyoHub/myo_sim#210 (myo_sim.fingerprint). Install that "
            "development build or a release containing it; 0.2.5 lacks this API."
        ) from exc
    return fingerprint(model)


def _trial_optimization(
    trial: EMGTrial, optimization: StaticOptimizationConfig | None
) -> StaticOptimizationConfig | None:
    """Fold trial-declared ``supported_dofs`` into *optimization* when unset."""
    dofs = trial.supported_dofs or tuple(
        int(d) for d in trial.meta.get("supported_dofs", ())
    )
    if not dofs:
        return optimization
    if optimization is None:
        return StaticOptimizationConfig(supported_dofs=dofs)
    if optimization.supported_dofs:
        return optimization
    return replace(optimization, supported_dofs=dofs)


def _non_root_dof_mask(
    model: mujoco.MjModel, optimization: StaticOptimizationConfig | None = None
) -> np.ndarray:
    """Dofs the muscles must drive: not free roots, tendon nodes or supported dofs."""
    mask = np.ones(model.nv, dtype=bool)
    if optimization is not None:
        mask[list(optimization.supported_dofs)] = False
    for j in np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE):
        adr = int(model.jnt_dofadr[j])
        mask[adr : adr + 6] = False
    for e in range(model.neq):
        if model.eq_type[e] == mujoco.mjtEq.mjEQ_TENDON and model.eq_active0[e]:
            t1 = int(model.eq_obj1id[e])
            adr = int(model.tendon_adr[t1])
            if (
                model.tendon_num[t1] == 1
                and model.wrap_type[adr] == mujoco.mjtWrap.mjWRAP_JOINT
            ):
                jnt = int(model.wrap_objid[adr])
                if model.jnt_type[jnt] == mujoco.mjtJoint.mjJNT_SLIDE:
                    mask[int(model.jnt_dofadr[jnt])] = False
    return mask


def _clip_moment_arms(model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    data = mujoco.MjData(model)
    out = np.zeros((len(qpos), model.nu, model.nv), dtype=np.float64)
    for t in range(len(qpos)):
        data.qpos[:] = qpos[t]
        mujoco.mj_fwdPosition(model, data)
        mujoco.mju_sparse2dense(
            out[t],
            data.actuator_moment,
            data.moment_rownnz,
            data.moment_rowadr,
            data.moment_colind,
        )
    return out


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    xc, yc = x - x.mean(), y - y.mean()
    denom = float(np.linalg.norm(xc) * np.linalg.norm(yc))
    return 0.0 if denom < 1e-10 else float(np.clip(np.dot(xc, yc) / denom, -1.0, 1.0))


def _cosine_sim(x: np.ndarray, y: np.ndarray) -> float:
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom < 1e-12:
        return 1.0 if float(np.linalg.norm(x) + np.linalg.norm(y)) < 1e-8 else 0.0
    return float(np.clip(np.dot(x, y) / denom, -1.0, 1.0))


def _compute_rsm_and_rsa(
    pred_ch: np.ndarray,
    exp_ch: np.ndarray,
    *,
    rsm_rsa_fn: Callable[[np.ndarray, np.ndarray], tuple[float, float]] | None = None,
) -> tuple[float, float]:
    """Compute RSM coordination similarity (Spearman and Pearson over lower triangle)."""
    if pred_ch.ndim != 2 or exp_ch.shape != pred_ch.shape:
        raise ValueError(
            f"pred_ch {pred_ch.shape} and exp_ch {exp_ch.shape} must have matching 2D shape"
        )
    n_steps, n_channels = pred_ch.shape
    if n_channels < 3 or n_steps < 2:
        return float("nan"), float("nan")
    if rsm_rsa_fn is not None:
        return rsm_rsa_fn(pred_ch, exp_ch)
    from scipy.stats import spearmanr

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rsm_a = np.corrcoef(pred_ch.T)
        rsm_b = np.corrcoef(exp_ch.T)
    idx = np.tril_indices(n_channels, k=-1)
    a, b = rsm_a[idx], rsm_b[idx]
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if a.size < 2 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan"), float("nan")
    return float(spearmanr(a, b).statistic), float(np.corrcoef(a, b)[0, 1])


def _onset_f1(pred: np.ndarray, exp: np.ndarray) -> float:
    p_hi, e_hi = float(pred.max()), float(exp.max())
    if p_hi < 1e-4 and e_hi < 1e-4:
        return 1.0
    p_thr = max(float(pred.min()) + 0.25 * (p_hi - float(pred.min())), 1e-3)
    e_thr = max(float(exp.min()) + 0.25 * (e_hi - float(exp.min())), 1e-3)
    pm, em = pred >= p_thr, exp >= e_thr
    tp = int(np.count_nonzero(pm & em))
    denom = int(np.count_nonzero(pm) + np.count_nonzero(em))
    return 1.0 if denom == 0 else float(2.0 * tp / denom)


def _center_of_activity_pct(sig: np.ndarray) -> float:
    w = np.maximum(sig, 0.0)
    if float(w.sum()) <= 1e-12:
        return 50.0
    angles = np.linspace(0.0, 2.0 * np.pi, len(w), endpoint=False)
    mean_angle = float(np.arctan2(np.dot(w, np.sin(angles)), np.dot(w, np.cos(angles))))
    if mean_angle < 0.0:
        mean_angle += 2.0 * np.pi
    return 100.0 * mean_angle / (2.0 * np.pi)


def _reserve_ratios(
    reserve: np.ndarray,
    joint_torque: np.ndarray,
    dof_mask: np.ndarray,
    actuator_qfrc: np.ndarray | None = None,
) -> np.ndarray:
    res_norm = np.linalg.norm(reserve[:, dof_mask], axis=-1)
    tau_norm = np.linalg.norm(joint_torque[:, dof_mask], axis=-1)
    if actuator_qfrc is not None:
        act_norm = np.linalg.norm(actuator_qfrc[:, dof_mask], axis=-1)
        ref_norm = np.maximum(tau_norm, act_norm + res_norm)
    else:
        ref_norm = np.maximum(tau_norm, res_norm)
    scale = max(float(np.percentile(ref_norm, 90)), 1e-3)
    return res_norm / np.maximum(ref_norm, 0.1 * scale)


def _effort_summary(
    model: mujoco.MjModel,
    qpos_sl: np.ndarray,
    act_sl: np.ndarray,
    frc_sl: np.ndarray,
    res_sl: np.ndarray,
    tau_sl: np.ndarray,
    status_sl: np.ndarray,
    optimization: StaticOptimizationConfig | None = None,
) -> dict[str, float]:
    dof_mask = _non_root_dof_mask(model, optimization)
    moment_arms = _clip_moment_arms(model, qpos_sl)
    act_qfrc = np.einsum("tmn,tm->tn", moment_arms, frc_sl)
    ratios = _reserve_ratios(res_sl, tau_sl, dof_mask, act_qfrc)
    cci = cocontraction_index(moment_arms, frc_sl)
    return {
        "squared_activation": float(squared_activation(act_sl)),
        "muscle_saturation": float(muscle_saturation(act_sl)),
        "cocontraction_mean": float(cci[dof_mask].mean()) if dof_mask.any() else 0.0,
        "wasted_torque_mean_nm": float(wasted_torque(moment_arms, frc_sl).mean()),
        "reserve_max_nm": float(np.abs(res_sl[:, dof_mask]).max())
        if dof_mask.any()
        else 0.0,
        "reserve_to_torque_ratio_max": float(ratios.max()),
        "reserve_to_torque_ratio_p95": float(np.percentile(ratios, 95)),
        "reserve_to_torque_ratio_mean": float(ratios.mean()),
        "solver_converged_fraction": float(np.mean(status_sl == "solved")),
    }


def _uncertainty_activation_quantiles(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    dt: float,
    lowpass_hz: float | None,
    external_qfrc: np.ndarray | None,
    optimization: StaticOptimizationConfig | None,
    uncertainty: UncertaintyConfig,
) -> tuple[StaticOptimizationResult, np.ndarray]:
    if external_qfrc is None:
        bands = static_optimization_bands(
            model,
            qpos,
            dt,
            config=uncertainty,
            lowpass_hz=lowpass_hz,
            optimization=optimization,
        )
        return bands.nominal, bands.activation

    rng = np.random.default_rng(uncertainty.seed)
    hinges = model.jnt_qposadr[model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE]
    opt_cfg = StaticOptimizer(model, optimization).config
    kin0 = clip_kinematics(model, qpos, dt, lowpass_hz=lowpass_hz)
    nominal = static_optimization(
        model,
        kin0.qpos,
        kin0.qvel,
        kin0.qacc,
        external_qfrc=external_qfrc,
        config=opt_cfg,
    )
    acts = []
    for _ in range(uncertainty.samples):
        m_pert = perturbed_model(model, rng, uncertainty)
        q_pert = np.array(qpos, dtype=np.float64)
        q_pert[:, hinges] += np.radians(
            uncertainty.joint_noise_deg
        ) * rng.standard_normal((len(q_pert), len(hinges)))
        q_pert = project_tendon_nodes(m_pert, q_pert)
        kin_s = clip_kinematics(m_pert, q_pert, dt, lowpass_hz=lowpass_hz)
        res_s = static_optimization(
            m_pert,
            kin_s.qpos,
            kin_s.qvel,
            kin_s.qacc,
            external_qfrc=external_qfrc,
            config=opt_cfg,
        )
        acts.append(res_s.activation)
    return nominal, np.quantile(np.stack(acts, axis=0), uncertainty.quantiles, axis=0)


def validate_inverse_emg(
    model: mujoco.MjModel,
    trial: EMGTrial,
    channel_map: EMGChannelMap,
    *,
    kinematics_lowpass_hz: float | None = 6.0,
    max_lag_s: float = 0.10,
    compare_in: str = "excitation",
    trim_s: float = 0.0,
    optimization: StaticOptimizationConfig | None = None,
    uncertainty: UncertaintyConfig | None = None,
    kinematics: ClipKinematics | None = None,
    nominal_so: StaticOptimizationResult | None = None,
) -> dict[str, Any]:
    """Compare static-optimization muscle estimates against experimental EMG channels."""
    if compare_in not in ("excitation", "activation"):
        raise ValueError(
            f"compare_in must be 'excitation' or 'activation', got {compare_in!r}"
        )
    if trial.qpos is None:
        raise ValueError(
            f"trial {trial.trial_id!r} requires qpos for validate_inverse_emg"
        )

    resolved = channel_map.resolve(model, trial.channel_names)
    if not resolved.channels:
        raise ValueError(
            f"channel_map {channel_map.name!r} matched 0 channels on trial {trial.trial_id!r}"
        )

    kin_time = trial.resolved_kinematics_time()
    optimization = _trial_optimization(trial, optimization)
    dt = float(np.median(np.diff(kin_time)))
    q_proj = project_tendon_nodes(model, trial.qpos)
    kin = kinematics or clip_kinematics(
        model, q_proj, dt, lowpass_hz=kinematics_lowpass_hz
    )

    act_quantiles: np.ndarray | None = None
    if uncertainty is not None:
        so_res, act_quantiles = _uncertainty_activation_quantiles(
            model,
            q_proj,
            dt,
            kinematics_lowpass_hz,
            trial.external_qfrc,
            optimization,
            uncertainty,
        )
    elif nominal_so is not None:
        so_res = nominal_so
    else:
        so_res = static_optimization(
            model,
            kin.qpos,
            kin.qvel,
            kin.qacc,
            external_qfrc=trial.external_qfrc,
            config=optimization,
        )

    exp_env = trial.envelopes(kin_time)[:, resolved.channel_indices]
    pred_act = so_res.activation
    if compare_in == "excitation":
        pred_u = activation_to_excitation(pred_act, dt, model=model)
        pred_ch_full = pred_u @ resolved.matrix.T
        exp_ch_full = exp_env.copy()
        band_ch_full = (
            np.stack(
                [
                    activation_to_excitation(act_quantiles[q], dt, model=model)
                    @ resolved.matrix.T
                    for q in range(len(act_quantiles))
                ],
                axis=0,
            )
            if act_quantiles is not None
            else None
        )
    else:
        pred_ch_full = pred_act @ resolved.matrix.T
        exp_cols = []
        for c, aids in enumerate(resolved.actuator_ids):
            aids_arr = np.asarray(aids, dtype=int)
            sub = np.repeat(exp_env[:, [c]], len(aids_arr), axis=1)
            act_sub = excitation_to_activation(
                sub, dt, model=model, actuator_ids=aids_arr
            )
            exp_cols.append(act_sub @ resolved.matrix[c, aids_arr])
        exp_ch_full = np.column_stack(exp_cols)
        band_ch_full = (
            np.einsum("qtm,cm->qtc", act_quantiles, resolved.matrix)
            if act_quantiles is not None
            else None
        )

    n_trim = round(max(0.0, trim_s) / dt)
    if 2 * n_trim >= len(kin_time) - 4:
        n_trim = 0
    sl = slice(n_trim, len(kin_time) - n_trim if n_trim > 0 else None)
    t_eval, pred_ch, exp_ch = kin_time[sl], pred_ch_full[sl], exp_ch_full[sl]
    band_ch = None if band_ch_full is None else band_ch_full[:, sl, :]

    max_steps = max(0, round(max_lag_s / dt))
    lags_s = np.arange(-max_steps, max_steps + 1, dtype=np.float64) * dt

    per_channel: dict[str, dict[str, Any]] = {}
    for c, ch_name in enumerate(resolved.channels):
        p, e = pred_ch[:, c], exp_ch[:, c]
        r0 = _pearson_r(p, e)
        best_r, best_lag = r0, 0.0
        for lag in lags_s:
            r_lag = _pearson_r(p, shift_signal(e, dt, float(lag)))
            if r_lag > best_r:
                best_r, best_lag = r_lag, float(lag)
        e_aligned = shift_signal(e, dt, best_lag)

        if trial.cycle_events is not None:
            valid_ev = trial.cycle_events[
                (trial.cycle_events[:, 0] >= t_eval[0])
                & (trial.cycle_events[:, 1] <= t_eval[-1])
            ]
        else:
            valid_ev = np.zeros((0, 2))
        if len(valid_ev) > 0:
            p_cyc = phase_normalize(p, t_eval, valid_ev).mean(axis=0)
            e_cyc = phase_normalize(e, t_eval, valid_ev).mean(axis=0)
            coa_diff = _center_of_activity_pct(p_cyc) - _center_of_activity_pct(e_cyc)
        else:
            coa_diff = _center_of_activity_pct(p) - _center_of_activity_pct(e)
        coa_diff = ((coa_diff + 50.0) % 100.0) - 50.0

        gain = max(
            0.0, float(np.dot(p, e_aligned) / max(np.dot(e_aligned, e_aligned), 1e-12))
        )
        gi_nrmse = float(np.sqrt(np.mean((p - gain * e_aligned) ** 2))) / max(
            float(np.ptp(p)), 0.05
        )
        is_active = max(float(np.ptp(p)), float(np.ptp(e))) >= 0.05
        ref_peak = resolved.reference_peaks[c]
        calibrated = trial.signal_kind == "mvc_normalized" or bool(
            np.isfinite(ref_peak)
        )
        ch_entry: dict[str, Any] = {
            "actuators": [model.actuator(a).name for a in resolved.actuator_ids[c]],
            "active": is_active,
            "r_zero_lag": r0,
            "r_opt_lag": best_r,
            "opt_lag_ms": 1e3 * best_lag,
            "cosine_similarity": _cosine_sim(p, e_aligned),
            "onset_f1": _onset_f1(p, e_aligned),
            "com_phase_diff_pct": float(coa_diff),
            "gain_invariant_nrmse": float(gi_nrmse),
            "amplitude_calibrated": calibrated,
        }
        if calibrated:
            scale = 1.0 if trial.signal_kind == "mvc_normalized" else float(ref_peak)
            e_cal = e_aligned * scale
            ch_entry["mae"] = float(np.mean(np.abs(p - e_cal)))
            ch_entry["rmse"] = float(np.sqrt(np.mean((p - e_cal) ** 2)))
            ch_entry["peak_pred"] = float(p.max())
            ch_entry["peak_exp"] = float(e_cal.max())
            if band_ch is not None and len(band_ch) >= 2:
                lo_b, hi_b = band_ch[0, :, c], band_ch[-1, :, c]
                ch_entry["band_coverage"] = float(
                    np.mean((e_cal >= lo_b - 0.02) & (e_cal <= hi_b + 0.02))
                )
        per_channel[ch_name] = ch_entry

    effort = _effort_summary(
        model,
        kin.qpos[sl],
        pred_act[sl],
        so_res.actuator_force[sl],
        so_res.reserve[sl],
        so_res.joint_torque[sl],
        so_res.status[sl],
        optimization,
    )

    def _vals(key: str, *, active_only: bool = False) -> np.ndarray:
        items = [
            v[key]
            for v in per_channel.values()
            if key in v and (not active_only or v.get("active", True))
        ]
        if not items and active_only:
            items = [v[key] for v in per_channel.values() if key in v]
        return np.asarray(items, dtype=np.float64)

    summary: dict[str, Any] = {
        "matched_channels": len(resolved.channels),
        "active_channels": int(
            sum(1 for v in per_channel.values() if v.get("active", True))
        ),
        "unmatched_channels": list(resolved.unmatched_channels),
        "compare_in": compare_in,
        "frames_evaluated": len(t_eval),
        "dt_s": dt,
        "r_zero_lag_median": float(np.median(_vals("r_zero_lag", active_only=True))),
        "r_zero_lag_mean": float(np.mean(_vals("r_zero_lag", active_only=True))),
        "r_opt_lag_median": float(np.median(_vals("r_opt_lag", active_only=True))),
        "r_opt_lag_mean": float(np.mean(_vals("r_opt_lag", active_only=True))),
        "opt_lag_ms_median": float(np.median(_vals("opt_lag_ms"))),
        "cosine_similarity_median": float(np.median(_vals("cosine_similarity"))),
        "onset_f1_median": float(np.median(_vals("onset_f1"))),
        "abs_com_phase_diff_pct_median": float(
            np.median(np.abs(_vals("com_phase_diff_pct")))
        ),
        "gain_invariant_nrmse_median": float(np.median(_vals("gain_invariant_nrmse"))),
        "effort": effort,
        "per_channel": per_channel,
    }
    rsa_sp, rsa_pe = _compute_rsm_and_rsa(pred_ch, exp_ch)
    if np.isfinite(rsa_sp):
        summary["muscle_rsa_spearman"] = rsa_sp
        summary["muscle_rsa_pearson"] = rsa_pe
    mae_vals = _vals("mae")
    if len(mae_vals):
        summary["mae_median"] = float(np.median(mae_vals))
        summary["rmse_median"] = float(np.median(_vals("rmse")))
    cov_vals = _vals("band_coverage")
    if len(cov_vals):
        summary["band_coverage_mean"] = float(np.mean(cov_vals))
    return summary


def validate_emg_feasibility(
    model: mujoco.MjModel,
    trial: EMGTrial,
    channel_map: EMGChannelMap,
    *,
    tolerance_band: float = 0.25,
    kinematics_lowpass_hz: float | None = 6.0,
    trim_s: float = 0.0,
    calibrate_unscaled: bool = True,
    optimization: StaticOptimizationConfig | None = None,
    kinematics: ClipKinematics | None = None,
    nominal_so: StaticOptimizationResult | None = None,
) -> dict[str, Any]:
    """Solve EMG-bounded static optimization and audit reserve-torque blow-ups."""
    if tolerance_band < 0.0:
        raise ValueError(f"tolerance_band must be >= 0, got {tolerance_band}")
    if trial.qpos is None:
        raise ValueError(
            f"trial {trial.trial_id!r} requires qpos for validate_emg_feasibility"
        )

    resolved = channel_map.resolve(model, trial.channel_names)
    if not resolved.channels:
        raise ValueError(
            f"channel_map {channel_map.name!r} matched 0 channels on trial {trial.trial_id!r}"
        )

    kin_time = trial.resolved_kinematics_time()
    optimization = _trial_optimization(trial, optimization)
    dt = float(np.median(np.diff(kin_time)))
    q_proj = project_tendon_nodes(model, trial.qpos)
    kin = kinematics or clip_kinematics(
        model, q_proj, dt, lowpass_hz=kinematics_lowpass_hz
    )
    so = StaticOptimizer(model, optimization)
    if nominal_so is None:
        nom_frames = [
            so.solve(
                kin.qpos[t],
                kin.qvel[t],
                kin.qacc[t],
                None if trial.external_qfrc is None else trial.external_qfrc[t],
            )
            for t in range(len(kin.qpos))
        ]
        nominal_so = StaticOptimizationResult(
            **{
                k: np.stack([getattr(fr, k) for fr in nom_frames])
                for k in StaticOptimizationResult.__dataclass_fields__
            }
        )

    exp_env = trial.envelopes(kin_time)[:, resolved.channel_indices]
    nom_ch = nominal_so.activation @ resolved.matrix.T
    ch_act = np.zeros_like(exp_env)
    for c, aids in enumerate(resolved.actuator_ids):
        aids_arr = np.asarray(aids, dtype=int)
        sub = np.repeat(exp_env[:, [c]], len(aids_arr), axis=1)
        act_c = excitation_to_activation(sub, dt, model=model, actuator_ids=aids_arr)
        ch_act[:, c] = act_c @ resolved.matrix[c, aids_arr]
        ref_peak = resolved.reference_peaks[c]
        if trial.signal_kind == "mvc_normalized":
            scale = 1.0
        elif np.isfinite(ref_peak):
            scale = float(ref_peak)
        elif calibrate_unscaled:
            p95_nom = float(np.percentile(nom_ch[:, c], 95))
            p95_emg = float(np.percentile(ch_act[:, c], 95))
            scale = float(np.clip(p95_nom / max(p95_emg, 1e-6), 0.05, 1.0))
        else:
            scale = 1.0
        ch_act[:, c] = np.clip(ch_act[:, c] * scale, 0.0, 1.0)

    col_weights = resolved.matrix.sum(axis=0)
    mapped_act = col_weights > 0.0
    lo_all = np.zeros((len(kin_time), model.nu), dtype=np.float64)
    hi_all = np.ones((len(kin_time), model.nu), dtype=np.float64)
    if np.any(mapped_act):
        target_act = (ch_act @ resolved.matrix[:, mapped_act]) / col_weights[
            None, mapped_act
        ]
        lo_all[:, mapped_act] = np.clip(target_act - tolerance_band, 0.0, 1.0)
        hi_all[:, mapped_act] = np.clip(target_act + tolerance_band, 0.0, 1.0)

    bounded_res = np.empty((len(kin_time), model.nv), dtype=np.float64)
    bounded_frc = np.empty((len(kin_time), model.nu), dtype=np.float64)
    bounded_status = []
    for t in range(len(kin_time)):
        fr = so.solve(
            kin.qpos[t],
            kin.qvel[t],
            kin.qacc[t],
            None if trial.external_qfrc is None else trial.external_qfrc[t],
            max_activation=hi_all[t],
            min_activation=lo_all[t],
        )
        bounded_res[t] = fr.reserve
        bounded_frc[t] = fr.actuator_force
        bounded_status.append(str(fr.status))

    n_trim = round(max(0.0, trim_s) / dt)
    if 2 * n_trim >= len(kin_time) - 4:
        n_trim = 0
    sl = slice(n_trim, len(kin_time) - n_trim if n_trim > 0 else None)

    dof_mask = _non_root_dof_mask(model, optimization)
    dof_indices = np.flatnonzero(dof_mask)
    tau_sl = nominal_so.joint_torque[sl]
    moment_sl = _clip_moment_arms(model, kin.qpos[sl])
    nom_act_qfrc = np.einsum("tmn,tm->tn", moment_sl, nominal_so.actuator_force[sl])
    bnd_act_qfrc = np.einsum("tmn,tm->tn", moment_sl, bounded_frc[sl])
    nom_ratios = _reserve_ratios(nominal_so.reserve[sl], tau_sl, dof_mask, nom_act_qfrc)
    bnd_ratios = _reserve_ratios(bounded_res[sl], tau_sl, dof_mask, bnd_act_qfrc)

    abs_bnd = np.abs(bounded_res[sl][:, dof_indices])
    if abs_bnd.size:
        flat_idx = int(np.argmax(abs_bnd))
        t_rel_arr, d_rel_arr = np.unravel_index(flat_idx, abs_bnd.shape)
        worst_t_rel, worst_d_rel = int(t_rel_arr), int(d_rel_arr)
        worst_dof = int(dof_indices[worst_d_rel])
        worst_dof_name = (
            model.joint(int(model.dof_jntid[worst_dof])).name or f"dof_{worst_dof}"
        )
        worst_dof_nm = float(abs_bnd[worst_t_rel, worst_d_rel])
        worst_time_s = float(kin_time[sl][worst_t_rel])
        p95_per_dof = np.percentile(abs_bnd, 95, axis=0)
        top_dofs = {
            (
                model.joint(int(model.dof_jntid[int(dof_indices[i])])).name
                or f"dof_{dof_indices[i]}"
            ): float(p95_per_dof[i])
            for i in np.argsort(p95_per_dof)[::-1][:5]
        }
    else:
        worst_t_rel, worst_dof_name, worst_dof_nm = 0, "", 0.0
        worst_time_s, top_dofs = float(kin_time[sl][0]), {}

    nom_p95 = float(np.percentile(nom_ratios, 95))
    bnd_p95 = float(np.percentile(bnd_ratios, 95))
    return {
        "tolerance_band": tolerance_band,
        "constrained_actuators": int(np.count_nonzero(mapped_act)),
        "total_actuators": int(model.nu),
        "unconstrained_reserve_ratio_max": float(nom_ratios.max()),
        "unconstrained_reserve_ratio_p95": nom_p95,
        "emg_bounded_reserve_ratio_max": float(bnd_ratios.max()),
        "emg_bounded_reserve_ratio_p95": bnd_p95,
        "emg_bounded_reserve_max_nm": worst_dof_nm,
        "reserve_amplification": float(bnd_p95 / max(nom_p95, 1e-3)),
        "worst_dof_name": worst_dof_name,
        "worst_frame_index": int(worst_t_rel + n_trim),
        "worst_frame_time_s": worst_time_s,
        "top_dof_reserve_p95_nm": top_dofs,
        "solver_converged_fraction": float(
            np.mean(np.asarray(bounded_status)[sl] == "solved")
        ),
    }


def _controlled_hinge_qpos_adr(model: mujoco.MjModel) -> np.ndarray:
    coupled = {
        int(model.eq_obj1id[e])
        for e in range(model.neq)
        if model.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT and model.eq_active0[e]
    }
    adrs = [
        int(model.jnt_qposadr[j])
        for j in range(model.njnt)
        if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE)
        and j not in coupled
    ]
    if not adrs:
        adrs = [
            int(model.jnt_qposadr[j])
            for j in range(model.njnt)
            if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE)
        ]
    return np.asarray(adrs, dtype=np.int64)


def _set_sim_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    qvel: np.ndarray,
    ctrl: np.ndarray,
    act: np.ndarray | None = None,
) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.ctrl[:] = ctrl
    stateful = model.actuator_actadr >= 0
    if np.any(stateful):
        src = ctrl if act is None else act
        data.act[model.actuator_actadr[stateful]] = src[stateful]


def actuator_pose_sensitivity(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    *,
    qvel: np.ndarray | None = None,
    ctrl: np.ndarray | None = None,
    controlled_qpos_adr: Sequence[int] | None = None,
    actuator_ids: Sequence[int] | None = None,
    eps: float = 0.02,
    horizon_s: float = 0.01,
    max_states: int = 32,
    seed: int = 0,
) -> dict[str, Any]:
    """Finite-difference actuator pose-sensitivity audit (deg per unit control)."""
    q_arr = np.asarray(qpos, dtype=np.float64)
    if q_arr.ndim == 1:
        q_arr = q_arr[None, :]
    q_arr = project_tendon_nodes(model, q_arr)
    v_arr = (
        np.zeros((len(q_arr), model.nv), dtype=np.float64)
        if qvel is None
        else np.asarray(qvel, dtype=np.float64)
    )
    c_arr = (
        np.full((len(q_arr), model.nu), 0.2, dtype=np.float64)
        if ctrl is None
        else np.asarray(ctrl, dtype=np.float64)
    )
    q_idx = (
        _controlled_hinge_qpos_adr(model)
        if controlled_qpos_adr is None
        else np.asarray(controlled_qpos_adr, dtype=np.int64)
    )
    if len(q_idx) == 0:
        raise ValueError("model has no hinge joints to measure pose sensitivity")
    eval_act_ids = (
        np.arange(model.nu, dtype=np.int64)
        if actuator_ids is None
        else np.asarray(actuator_ids, dtype=np.int64)
    )

    m = copy.deepcopy(model)
    data = mujoco.MjData(m)
    probe_steps = max(1, round(horizon_s / m.opt.timestep))
    rng = np.random.default_rng(seed)
    n_sel = min(max_states, len(q_arr))
    indices = (
        np.arange(len(q_arr))
        if n_sel == len(q_arr)
        else np.sort(rng.choice(len(q_arr), size=n_sel, replace=False))
    )

    stateful = m.actuator_actadr >= 0
    lo = np.where(
        stateful & (m.actuator_actlimited != 0),
        m.actuator_actrange[:, 0],
        np.where(m.actuator_ctrllimited != 0, m.actuator_ctrlrange[:, 0], 0.0),
    )
    hi = np.where(
        stateful & (m.actuator_actlimited != 0),
        m.actuator_actrange[:, 1],
        np.where(m.actuator_ctrllimited != 0, m.actuator_ctrlrange[:, 1], 1.0),
    )
    muscle = m.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
    lo[muscle], hi[muscle] = 0.0, 1.0

    deg = 180.0 / np.pi
    jac_abs = np.zeros((len(eval_act_ids), len(q_idx)), dtype=np.float64)
    for idx in indices:
        q0, v0, c0 = q_arr[idx], v_arr[idx], c_arr[idx]
        for row_k, j in enumerate(eval_act_ids):
            cp, cm = c0.copy(), c0.copy()
            cp[j] = np.clip(c0[j] + eps, lo[j], hi[j])
            cm[j] = np.clip(c0[j] - eps, lo[j], hi[j])
            h = float(cp[j] - cm[j])
            if h <= 1e-9:
                continue
            _set_sim_state(m, data, q0, v0, cp, cp)
            for _ in range(probe_steps):
                mujoco.mj_step(m, data)
            qp = data.qpos[q_idx].copy()
            _set_sim_state(m, data, q0, v0, cm, cm)
            for _ in range(probe_steps):
                mujoco.mj_step(m, data)
            qm = data.qpos[q_idx].copy()
            jac_abs[row_k] += np.abs((qp - qm) / h) * deg
    jac_abs /= max(len(indices), 1)

    weights = np.sqrt((jac_abs**2).sum(axis=1))
    w_max = float(weights.max(initial=0.0))
    nonzero = weights[weights > 1e-6]
    w_min_nz = float(nonzero.min()) if len(nonzero) else 0.0
    w_med_nz = float(np.median(nonzero)) if len(nonzero) else 0.0
    spread = float(w_max / w_min_nz) if w_min_nz > 0 else float("inf")
    p95 = float(np.percentile(weights, 95)) if len(weights) else 0.0
    p05 = float(np.percentile(weights, 5)) if len(weights) else 0.0
    near_zero_mask = weights < max(1e-4, 1e-3 * w_med_nz)
    act_names = [m.actuator(int(i)).name or f"actuator_{int(i)}" for i in eval_act_ids]
    order = np.argsort(weights)[::-1]

    return {
        "sampled_states": len(indices),
        "controlled_joints": len(q_idx),
        "evaluated_actuators": len(eval_act_ids),
        "mean_deg_per_ctrl": float(weights.mean()),
        "max_deg_per_ctrl": w_max,
        "min_deg_per_ctrl": float(weights.min(initial=0.0)),
        "spread_ratio": spread,
        "p95_to_p05_ratio": float(p95 / max(p05, 1e-9)),
        "n_near_zero": int(np.count_nonzero(near_zero_mask)),
        "dead_actuators": [act_names[i] for i in np.flatnonzero(near_zero_mask)],
        "top_actuators": {act_names[int(i)]: float(weights[int(i)]) for i in order[:5]},
        "bottom_actuators": {
            act_names[int(i)]: float(weights[int(i)]) for i in order[-5:]
        },
        "weights": weights.tolist(),
    }


def open_loop_pose_drift(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    dt: float,
    *,
    ctrl: np.ndarray | None = None,
    activation: np.ndarray | None = None,
    controlled_qpos_adr: Sequence[int] | None = None,
    horizon_s: float = 0.5,
    kinematics_lowpass_hz: float | None = 6.0,
    optimization: StaticOptimizationConfig | None = None,
    kinematics: ClipKinematics | None = None,
    nominal_so: StaticOptimizationResult | None = None,
) -> dict[str, Any]:
    """Compare 1-step teacher-forced tracking against multi-step open-loop rollout drift."""
    q_in = project_tendon_nodes(model, np.asarray(qpos, dtype=np.float64))
    stride = max(1, round(model.opt.timestep / dt)) if dt < model.opt.timestep else 1
    dt_eff = dt * stride
    q_sub = q_in[::stride]
    if len(q_sub) < 4:
        raise ValueError(
            "clip too short after timestep alignment for open_loop_pose_drift"
        )

    kin = (
        kinematics
        if kinematics is not None and stride == 1 and len(kinematics.qpos) == len(q_sub)
        else clip_kinematics(model, q_sub, dt_eff, lowpass_hz=kinematics_lowpass_hz)
    )
    if ctrl is None:
        so_res = (
            nominal_so
            if nominal_so is not None
            and stride == 1
            and len(nominal_so.activation) == len(q_sub)
            else static_optimization(
                model, kin.qpos, kin.qvel, kin.qacc, config=optimization
            )
        )
        act_seq = so_res.activation
        ctrl_seq = activation_to_excitation(act_seq, dt_eff, model=model)
    else:
        ctrl_seq = np.asarray(ctrl, dtype=np.float64)[::stride]
        act_seq = (
            excitation_to_activation(ctrl_seq, dt_eff, model=model)
            if activation is None
            else np.asarray(activation, dtype=np.float64)[::stride]
        )

    q_idx = (
        _controlled_hinge_qpos_adr(model)
        if controlled_qpos_adr is None
        else np.asarray(controlled_qpos_adr, dtype=np.int64)
    )
    substeps = max(1, round(dt_eff / model.opt.timestep))
    m = copy.deepcopy(model)
    data = mujoco.MjData(m)
    deg = 180.0 / np.pi

    n_eval = min(len(kin.qpos) - 1, 250)
    step_errs = np.empty((n_eval, len(q_idx)), dtype=np.float64)
    for t in range(n_eval):
        _set_sim_state(m, data, kin.qpos[t], kin.qvel[t], ctrl_seq[t], act_seq[t])
        for _ in range(substeps):
            mujoco.mj_step(m, data)
        step_errs[t] = np.abs(data.qpos[q_idx] - kin.qpos[t + 1, q_idx]) * deg

    horizon_steps = min(max(1, round(horizon_s / dt_eff)), len(kin.qpos) - 1)
    starts = np.unique(
        np.linspace(
            0,
            max(0, len(kin.qpos) - 1 - horizon_steps),
            num=min(4, len(kin.qpos) - horizon_steps),
            dtype=int,
        )
    )
    rollout_maes: list[float] = []
    final_maes: list[float] = []
    for s0 in starts:
        _set_sim_state(m, data, kin.qpos[s0], kin.qvel[s0], ctrl_seq[s0], act_seq[s0])
        traj = np.empty((horizon_steps, len(q_idx)), dtype=np.float64)
        for k in range(horizon_steps):
            data.ctrl[:] = ctrl_seq[s0 + k]
            for _ in range(substeps):
                mujoco.mj_step(m, data)
            traj[k] = data.qpos[q_idx]
        err = np.abs(traj - kin.qpos[s0 + 1 : s0 + 1 + horizon_steps, q_idx]) * deg
        rollout_maes.append(float(err.mean()))
        final_maes.append(float(err[-1].mean()))

    return {
        "dt_eval_s": dt_eff,
        "substeps_per_frame": substeps,
        "horizon_s": float(horizon_steps * dt_eff),
        "one_step_mae_deg": float(step_errs.mean()),
        "one_step_p90_deg": float(np.percentile(step_errs.mean(axis=1), 90)),
        "open_loop_mae_deg": float(np.mean(rollout_maes)),
        "open_loop_final_mae_deg": float(np.mean(final_maes)),
        "drift_to_step_ratio": float(
            np.mean(rollout_maes) / max(float(step_errs.mean()), 1e-6)
        ),
    }


def _nmf_multiplicative(
    x: np.ndarray, rank: int, max_iter: int = 200, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, float]:
    v = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
    rng = np.random.default_rng(seed)
    u_svd, s_svd, vt_svd = np.linalg.svd(v, full_matrices=False)
    w = np.zeros((v.shape[0], rank), dtype=np.float64)
    h = np.zeros((rank, v.shape[1]), dtype=np.float64)
    for k in range(min(rank, len(s_svd))):
        uk, vk = u_svd[:, k], vt_svd[k, :]
        up, un = np.maximum(uk, 0.0), np.maximum(-uk, 0.0)
        vp, vn = np.maximum(vk, 0.0), np.maximum(-vk, 0.0)
        np_norm = float(np.linalg.norm(up) * np.linalg.norm(vp))
        nn_norm = float(np.linalg.norm(un) * np.linalg.norm(vn))
        if np_norm >= nn_norm and np_norm > 1e-12:
            w[:, k] = (
                np.sqrt(s_svd[k] * np_norm) * up / max(float(np.linalg.norm(up)), 1e-12)
            )
            h[k, :] = (
                np.sqrt(s_svd[k] * np_norm) * vp / max(float(np.linalg.norm(vp)), 1e-12)
            )
        elif nn_norm > 1e-12:
            w[:, k] = (
                np.sqrt(s_svd[k] * nn_norm) * un / max(float(np.linalg.norm(un)), 1e-12)
            )
            h[k, :] = (
                np.sqrt(s_svd[k] * nn_norm) * vn / max(float(np.linalg.norm(vn)), 1e-12)
            )
    jitter = 1e-4 * max(float(np.mean(v)), 1e-4)
    w = np.maximum(w, jitter * np.abs(rng.standard_normal(w.shape)) + 1e-8)
    h = np.maximum(h, jitter * np.abs(rng.standard_normal(h.shape)) + 1e-8)
    eps = 1e-12
    for _ in range(max_iter):
        h *= (w.T @ v) / (w.T @ w @ h + eps)
        w *= (v @ h.T) / (w @ (h @ h.T) + eps)
    recon = w @ h
    denom = float(np.sum(v * v))
    vaf = 1.0 if denom < eps else float(1.0 - np.sum((v - recon) ** 2) / denom)
    return w, h, max(0.0, vaf)


def synergy_subspace_metrics(
    emg: np.ndarray,
    activations: np.ndarray,
    *,
    n_synergies: int = 4,
    ridge_alpha: float = 1.0,
    n_folds: int = 3,
    max_iter: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare unmapped multi-channel EMG arrays against model activations via synergies."""
    x = np.maximum(np.asarray(emg, dtype=np.float64), 0.0)
    y = np.maximum(np.asarray(activations, dtype=np.float64), 0.0)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 8:
        raise ValueError(
            f"emg {x.shape} and activations {y.shape} must be 2D with matching T >= 8"
        )

    active_emg = np.ptp(x, axis=0) > 1e-6
    active_act = np.ptp(y, axis=0) > 1e-4
    x_use = x[:, active_emg] if np.any(active_emg) else x
    y_use = y[:, active_act] if np.any(active_act) else y

    rank = max(1, min(n_synergies, len(x) - 2, x_use.shape[1], y_use.shape[1]))
    w_emg, _, emg_vaf = _nmf_multiplicative(x_use, rank, max_iter=max_iter, seed=seed)
    w_act, _, act_vaf = _nmf_multiplicative(
        y_use, rank, max_iter=max_iter, seed=seed + 1
    )
    q_emg, _ = np.linalg.qr(w_emg)
    q_act, _ = np.linalg.qr(w_act)
    sv = np.linalg.svd(q_emg.T @ q_act, compute_uv=False)
    subspace_sim = float(np.clip(sv.mean(), 0.0, 1.0))

    folds = max(2, min(n_folds, len(x) // 4))
    block = max(1, min(10, len(x) // (2 * folds)))
    fold_ids = (np.arange(len(x)) // block) % folds
    y_pred = np.zeros_like(y_use)
    for k in range(folds):
        te_mask = fold_ids == k
        tr_mask = ~te_mask
        x_tr, y_tr = x_use[tr_mask], y_use[tr_mask]
        mu_x = x_tr.mean(axis=0, keepdims=True)
        sd_x = np.maximum(x_tr.std(axis=0, keepdims=True), 1e-6)
        mu_y = y_tr.mean(axis=0, keepdims=True)
        xz_tr, xz_te = (x_tr - mu_x) / sd_x, (x_use[te_mask] - mu_x) / sd_x
        gram = xz_tr.T @ xz_tr + ridge_alpha * len(xz_tr) * np.eye(xz_tr.shape[1])
        coef = np.linalg.solve(gram, xz_tr.T @ (y_tr - mu_y))
        y_pred[te_mask] = np.clip(mu_y + xz_te @ coef, 0.0, 1.0)

    ss_res = float(np.sum((y_use - y_pred) ** 2))
    ss_tot = float(np.sum((y_use - y_use.mean(axis=0, keepdims=True)) ** 2))
    r2 = 0.0 if ss_tot < 1e-12 else float(np.clip(1.0 - ss_res / ss_tot, -1.0, 1.0))
    corrs = [_pearson_r(y_use[:, j], y_pred[:, j]) for j in range(y_use.shape[1])]
    return {
        "n_synergies": int(rank),
        "active_emg_channels": int(x_use.shape[1]),
        "active_actuators": int(y_use.shape[1]),
        "emg_vaf": emg_vaf,
        "model_vaf": act_vaf,
        "temporal_subspace_similarity": subspace_sim,
        "emg_to_activation_r2": r2,
        "emg_to_activation_corr_mean": float(np.mean(corrs)) if corrs else 0.0,
    }


@dataclass(frozen=True)
class EMGValidationThresholds:
    """Quality-flag thresholds for :func:`validate_trial`."""

    min_r_opt_lag: float = 0.25
    max_unconstrained_reserve_ratio: float = 0.15
    max_emg_bounded_reserve_ratio: float = 0.35
    max_sensitivity_spread_ratio: float = 250.0
    max_dead_actuators: int = 0
    max_open_loop_drift_deg: float = 25.0
    min_synergy_subspace_similarity: float = 0.45


def _resolve_target_actuator_ids(
    model: mujoco.MjModel, meta: Mapping[str, Any]
) -> list[int]:
    target_names = meta.get("target_actuators")
    if not isinstance(target_names, Sequence) or isinstance(target_names, str):
        return []
    act_map = {model.actuator(i).name: i for i in range(model.nu)}
    t_ids: list[int] = []
    for n in target_names:
        for cand in _resolve_actuator_names(str(n), None, act_map):
            aid = act_map[cand]
            if aid not in t_ids:
                t_ids.append(aid)
    return t_ids


def validate_trial(
    model: mujoco.MjModel,
    trial: EMGTrial,
    *,
    channel_map: EMGChannelMap | None = None,
    kinematics_lowpass_hz: float | None = 6.0,
    trim_s: float = 0.0,
    tolerance_band: float = 0.25,
    include_rollout_audit: bool = False,
    include_synergy_metrics: bool = True,
    controlled_qpos_adr: Sequence[int] | None = None,
    optimization: StaticOptimizationConfig | None = None,
    uncertainty: UncertaintyConfig | None = None,
    thresholds: EMGValidationThresholds | None = None,
) -> dict[str, Any]:
    """Run all applicable EMG validation modes on a single :class:`EMGTrial`."""
    thr = thresholds or EMGValidationThresholds()
    flags: list[str] = []
    report: dict[str, Any] = {
        "trial_id": trial.trial_id,
        "signal_kind": trial.signal_kind,
        "emg_channels": len(trial.channel_names),
        "model_fingerprint": musculoskeletal_fingerprint(model),
        "meta": dict(trial.meta),
    }
    if trial.qpos is None:
        report["flags"] = ["missing_kinematics"]
        return report

    kin_time = trial.resolved_kinematics_time()
    optimization = _trial_optimization(trial, optimization)
    dt = float(np.median(np.diff(kin_time)))
    q_proj = project_tendon_nodes(model, trial.qpos)
    kin = clip_kinematics(model, q_proj, dt, lowpass_hz=kinematics_lowpass_hz)
    nom_so = static_optimization(
        model,
        kin.qpos,
        kin.qvel,
        kin.qacc,
        external_qfrc=trial.external_qfrc,
        config=optimization,
    )

    resolved = (
        channel_map.resolve(model, trial.channel_names)
        if channel_map is not None
        else None
    )
    if resolved is not None and len(resolved.channels) > 0:
        assert channel_map is not None
        inv = validate_inverse_emg(
            model,
            trial,
            channel_map,
            kinematics_lowpass_hz=kinematics_lowpass_hz,
            trim_s=trim_s,
            optimization=optimization,
            uncertainty=uncertainty,
            kinematics=kin,
            nominal_so=nom_so,
        )
        feas = validate_emg_feasibility(
            model,
            trial,
            channel_map,
            tolerance_band=tolerance_band,
            kinematics_lowpass_hz=kinematics_lowpass_hz,
            trim_s=trim_s,
            optimization=optimization,
            kinematics=kin,
            nominal_so=nom_so,
        )
        report["inverse_emg"] = inv
        report["feasibility"] = feas
        report["effort"] = inv["effort"]
        if inv["r_opt_lag_median"] < thr.min_r_opt_lag:
            flags.append("poor_waveform_correlation")
        if (
            inv["effort"]["reserve_to_torque_ratio_p95"]
            > thr.max_unconstrained_reserve_ratio
        ):
            flags.append("high_unconstrained_reserves")
        if feas["emg_bounded_reserve_ratio_p95"] > thr.max_emg_bounded_reserve_ratio:
            flags.append("emg_infeasible_reserves")
    else:
        n_trim = round(max(0.0, trim_s) / dt)
        if 2 * n_trim >= len(kin_time) - 4:
            n_trim = 0
        sl = slice(n_trim, len(kin_time) - n_trim if n_trim > 0 else None)
        eff = _effort_summary(
            model,
            kin.qpos[sl],
            nom_so.activation[sl],
            nom_so.actuator_force[sl],
            nom_so.reserve[sl],
            nom_so.joint_torque[sl],
            nom_so.status[sl],
            optimization,
        )
        report["effort"] = eff
        if eff["reserve_to_torque_ratio_p95"] > thr.max_unconstrained_reserve_ratio:
            flags.append("high_unconstrained_reserves")

    target_aids = _resolve_target_actuator_ids(model, trial.meta)
    if include_synergy_metrics:
        env = trial.envelopes(kin_time)
        pred_u = activation_to_excitation(nom_so.activation, dt, model=model)
        if target_aids:
            pred_u = pred_u[:, target_aids]
        syn = synergy_subspace_metrics(env, pred_u)
        report["synergy"] = syn
        if syn["temporal_subspace_similarity"] < thr.min_synergy_subspace_similarity:
            flags.append("low_synergy_subspace_similarity")

    if include_rollout_audit:
        ctrl_u = activation_to_excitation(nom_so.activation, dt, model=model)
        sens = actuator_pose_sensitivity(
            model,
            kin.qpos,
            qvel=kin.qvel,
            ctrl=ctrl_u,
            controlled_qpos_adr=controlled_qpos_adr,
            actuator_ids=target_aids or None,
        )
        drift = open_loop_pose_drift(
            model,
            q_proj,
            dt,
            ctrl=ctrl_u,
            activation=nom_so.activation,
            controlled_qpos_adr=controlled_qpos_adr,
            kinematics_lowpass_hz=kinematics_lowpass_hz,
            optimization=optimization,
            kinematics=kin,
            nominal_so=nom_so,
        )
        report["pose_sensitivity"] = sens
        report["open_loop_drift"] = drift
        if sens["n_near_zero"] > thr.max_dead_actuators:
            flags.append("dead_actuators")
        if sens["spread_ratio"] > thr.max_sensitivity_spread_ratio:
            flags.append("extreme_sensitivity_spread")
        if drift["open_loop_mae_deg"] > thr.max_open_loop_drift_deg:
            flags.append("excessive_open_loop_drift")

    report["flags"] = flags
    return report


def summarize_emg_validation(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-trial validation reports into dataset-level medians and flags."""
    if not reports:
        return {
            "trials": 0,
            "stats": {},
            "per_channel_r_opt_lag_median": {},
            "flagged": [],
        }

    def _collect(*paths: tuple[str, ...]) -> np.ndarray:
        vals = []
        for rep in reports:
            found = None
            for path in paths:
                cur: Any = rep
                for key in path:
                    if not isinstance(cur, Mapping) or key not in cur:
                        cur = None
                        break
                    cur = cur[key]
                if cur is not None and np.isfinite(float(cur)):
                    found = float(cur)
                    break
            if found is not None:
                vals.append(found)
        return np.asarray(vals, dtype=np.float64)

    metric_paths: dict[str, tuple[tuple[str, ...], ...]] = {
        "r_zero_lag_median": (("inverse_emg", "r_zero_lag_median"),),
        "r_opt_lag_median": (("inverse_emg", "r_opt_lag_median"),),
        "muscle_rsa_spearman": (("inverse_emg", "muscle_rsa_spearman"),),
        "muscle_rsa_pearson": (("inverse_emg", "muscle_rsa_pearson"),),
        "cosine_similarity_median": (("inverse_emg", "cosine_similarity_median"),),
        "onset_f1_median": (("inverse_emg", "onset_f1_median"),),
        "gain_invariant_nrmse_median": (
            ("inverse_emg", "gain_invariant_nrmse_median"),
        ),
        "mae_median": (("inverse_emg", "mae_median"),),
        "squared_activation": (
            ("inverse_emg", "effort", "squared_activation"),
            ("effort", "squared_activation"),
        ),
        "cocontraction_mean": (
            ("inverse_emg", "effort", "cocontraction_mean"),
            ("effort", "cocontraction_mean"),
        ),
        "unconstrained_reserve_ratio_p95": (
            ("feasibility", "unconstrained_reserve_ratio_p95"),
            ("effort", "reserve_to_torque_ratio_p95"),
        ),
        "emg_bounded_reserve_ratio_p95": (
            ("feasibility", "emg_bounded_reserve_ratio_p95"),
        ),
        "reserve_amplification": (("feasibility", "reserve_amplification"),),
        "temporal_subspace_similarity": (("synergy", "temporal_subspace_similarity"),),
        "emg_to_activation_r2": (("synergy", "emg_to_activation_r2"),),
        "emg_to_activation_corr_mean": (("synergy", "emg_to_activation_corr_mean"),),
        "sensitivity_spread_ratio": (("pose_sensitivity", "spread_ratio"),),
        "dead_actuators_count": (("pose_sensitivity", "n_near_zero"),),
        "one_step_mae_deg": (("open_loop_drift", "one_step_mae_deg"),),
        "open_loop_mae_deg": (("open_loop_drift", "open_loop_mae_deg"),),
    }
    stats = {
        label: {
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "p95": float(np.percentile(arr, 95)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
        for label, paths in metric_paths.items()
        if len(arr := _collect(*paths))
    }

    ch_buckets: dict[str, list[float]] = {}
    for rep in reports:
        inv = rep.get("inverse_emg")
        if isinstance(inv, Mapping):
            for ch, entry in inv.get("per_channel", {}).items():
                ch_buckets.setdefault(ch, []).append(float(entry["r_opt_lag"]))

    return {
        "trials": len(reports),
        "model_fingerprints": sorted(
            {
                str(rep["model_fingerprint"])
                for rep in reports
                if "model_fingerprint" in rep
            }
        ),
        "stats": stats,
        "per_channel_r_opt_lag_median": {
            ch: float(np.median(vals)) for ch, vals in sorted(ch_buckets.items())
        },
        "flagged": [
            {"trial_id": rep["trial_id"], "flags": list(rep.get("flags", ()))}
            for rep in reports
            if rep.get("flags")
        ],
    }


def _clean_json_value(val: Any) -> Any:
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        f = float(val)
        return f if np.isfinite(f) else None
    return val


def _fallback_holm(p_values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(p_values, dtype=float)
    corrected = np.full(arr.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(arr))
    if finite.size == 0:
        return corrected
    order = finite[np.argsort(arr[finite], kind="stable")]
    m = order.size
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * arr[idx])
        corrected[idx] = min(running, 1.0)
    return corrected


def compare_emg_validation_reports(
    reports_a: Sequence[Mapping[str, Any]],
    reports_b: Sequence[Mapping[str, Any]],
    *,
    label_a: str = "model_a",
    label_b: str = "model_b",
    stats_backend: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Paired Wilcoxon signed-rank + Holm A/B comparison across shared subjects or trials."""
    subjects_a = {
        str(r.get("meta", {}).get("subject"))
        for r in reports_a
        if r.get("meta", {}).get("subject")
    }
    subjects_b = {
        str(r.get("meta", {}).get("subject"))
        for r in reports_b
        if r.get("meta", {}).get("subject")
    }
    use_subject = len(subjects_a & subjects_b) >= 2
    unit_name = "subject" if use_subject else "trial"

    def _unit_key(rep: Mapping[str, Any]) -> str:
        if use_subject:
            subj = rep.get("meta", {}).get("subject")
            if subj:
                return str(subj)
        return str(rep["trial_id"])

    units_a = sorted({_unit_key(r) for r in reports_a})
    units_b = sorted({_unit_key(r) for r in reports_b})
    shared_units = [u for u in units_a if u in set(units_b)]
    if not shared_units:
        return {"unit_name": unit_name, "summary_rows": [], "per_channel_rows": []}

    def _extract_nested(
        rep: Mapping[str, Any], paths: tuple[tuple[str, ...], ...]
    ) -> float:
        for path in paths:
            cur: Any = rep
            for key in path:
                if not isinstance(cur, Mapping) or key not in cur:
                    cur = None
                    break
                cur = cur[key]
            if cur is not None and np.isfinite(float(cur)):
                return float(cur)
        return float("nan")

    def _unit_vector(
        reps: Sequence[Mapping[str, Any]], paths: tuple[tuple[str, ...], ...]
    ) -> list[float]:
        buckets: dict[str, list[float]] = {u: [] for u in shared_units}
        for r in reps:
            uk = _unit_key(r)
            if uk in buckets:
                v = _extract_nested(r, paths)
                if np.isfinite(v):
                    buckets[uk].append(v)
        return [
            float(np.mean(buckets[u])) if buckets[u] else float("nan")
            for u in shared_units
        ]

    channels_all = sorted(
        {
            str(ch)
            for reps in (reports_a, reports_b)
            for r in reps
            if isinstance(r.get("inverse_emg"), Mapping)
            for ch in r["inverse_emg"].get("per_channel", {})
        }
    )

    def _per_channel_dict(reps: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for ch in channels_all:
            buckets: dict[str, list[float]] = {u: [] for u in shared_units}
            for r in reps:
                uk = _unit_key(r)
                inv = r.get("inverse_emg")
                if uk in buckets and isinstance(inv, Mapping):
                    entry = inv.get("per_channel", {}).get(ch)
                    if isinstance(entry, Mapping) and "r_opt_lag" in entry:
                        val = float(entry["r_opt_lag"])
                        if np.isfinite(val):
                            buckets[uk].append(val)
            out[ch] = [
                float(np.mean(buckets[u])) if buckets[u] else float("nan")
                for u in shared_units
            ]
        return out

    scalar_specs: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...] = (
        ("muscle_rsa_spearman", (("inverse_emg", "muscle_rsa_spearman"),)),
        (
            "unconstrained_reserve_ratio_p95",
            (
                ("feasibility", "unconstrained_reserve_ratio_p95"),
                ("effort", "reserve_to_torque_ratio_p95"),
            ),
        ),
        (
            "emg_bounded_reserve_ratio_p95",
            (("feasibility", "emg_bounded_reserve_ratio_p95"),),
        ),
        (
            "temporal_subspace_similarity",
            (("synergy", "temporal_subspace_similarity"),),
        ),
        ("open_loop_mae_deg", (("open_loop_drift", "open_loop_mae_deg"),)),
    )

    if stats_backend is not None:
        scalar_vectors = []
        for metric_name, paths in scalar_specs:
            va = _unit_vector(reports_a, paths)
            vb = _unit_vector(reports_b, paths)
            if any(np.isfinite(va)) and any(np.isfinite(vb)):
                scalar_vectors.append((metric_name, va, vb))
        return stats_backend(
            label_a=label_a,
            label_b=label_b,
            shared_units=shared_units,
            channels_all=channels_all,
            per_channel_a=_per_channel_dict(reports_a) if channels_all else {},
            per_channel_b=_per_channel_dict(reports_b) if channels_all else {},
            scalar_vectors=scalar_vectors,
            unit_name=unit_name,
            clean_json_fn=_clean_json_value,
        )

    from scipy.stats import rankdata, wilcoxon

    def _paired_fallback(
        a_vec: Sequence[float] | np.ndarray,
        b_vec: Sequence[float] | np.ndarray,
    ) -> dict[str, Any]:
        a_arr = np.asarray(a_vec, dtype=float)
        b_arr = np.asarray(b_vec, dtype=float)
        keep = np.isfinite(a_arr) & np.isfinite(b_arr)
        a_arr, b_arr = a_arr[keep], b_arr[keep]
        if a_arr.size == 0:
            return {
                "n": 0,
                "n_nonzero": 0,
                "median_a": None,
                "median_b": None,
                "median_diff": None,
                "p": None,
                "effect_size": None,
            }
        diff = a_arr - b_arr
        nz = diff[diff != 0.0]
        if nz.size == 0:
            return {
                "n": int(a_arr.size),
                "n_nonzero": 0,
                "median_a": float(np.median(a_arr)),
                "median_b": float(np.median(b_arr)),
                "median_diff": 0.0,
                "p": None,
                "effect_size": None,
            }
        ranks = rankdata(np.abs(nz))
        wp = float(ranks[nz > 0].sum())
        wm = float(ranks[nz < 0].sum())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, p_val = wilcoxon(a_arr, b_arr, zero_method="wilcox")
        return {
            "n": int(a_arr.size),
            "n_nonzero": int(nz.size),
            "median_a": float(np.median(a_arr)),
            "median_b": float(np.median(b_arr)),
            "median_diff": float(np.median(diff)),
            "p": float(p_val),
            "effect_size": (wp - wm) / (wp + wm),
        }

    summary_rows = []
    per_channel_rows = []
    if channels_all:
        pca = _per_channel_dict(reports_a)
        pcb = _per_channel_dict(reports_b)
        mat_a = np.column_stack([pca[c] for c in channels_all])
        mat_b = np.column_stack([pcb[c] for c in channels_all])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            res_u = _paired_fallback(
                np.nanmean(mat_a, axis=1), np.nanmean(mat_b, axis=1)
            )
        summary_rows.append(
            {
                "metric": "r_opt_lag",
                "condition_a": label_a,
                "condition_b": label_b,
                **res_u,
            }
        )
        if len(channels_all) >= 2:
            raw_ch = [
                {
                    "metric": "r_opt_lag",
                    "channel": c,
                    **_paired_fallback(pca[c], pcb[c]),
                }
                for c in channels_all
            ]
            corr_p = _fallback_holm(
                [float("nan") if r["p"] is None else float(r["p"]) for r in raw_ch]
            )
            for row, cp in zip(raw_ch, corr_p, strict=True):
                row["p_holm"] = _clean_json_value(cp)
                row["significant"] = bool(np.isfinite(cp) and cp < 0.05)
                per_channel_rows.append(row)
    for metric_name, paths in scalar_specs:
        va = _unit_vector(reports_a, paths)
        vb = _unit_vector(reports_b, paths)
        if any(np.isfinite(va)) and any(np.isfinite(vb)):
            summary_rows.append(
                {
                    "metric": metric_name,
                    "condition_a": label_a,
                    "condition_b": label_b,
                    **_paired_fallback(va, vb),
                }
            )
    for row in summary_rows:
        row["p_holm"] = row["p"]
        row["significant"] = bool(row["p"] is not None and row["p"] < 0.05)
    return {
        "backend": "scipy.stats.wilcoxon",
        "unit_name": unit_name,
        "shared_units": len(shared_units),
        "summary_rows": summary_rows,
        "per_channel_rows": per_channel_rows,
    }


_COMPAT_INTEGRATION_EXPORTS: dict[str, str] = {
    "EMG_TRIAL_META_KEY": "myosuite.integrations.emg.trial_io",
    "load_emg_trial": "myosuite.integrations.emg.trial_io",
    "save_emg_trial": "myosuite.integrations.emg.trial_io",
    "export_mm_evalkit_rollout": "myosuite.integrations.mm_evalkit.rollout_export",
    "write_mm_evalkit_policy_config": "myosuite.integrations.mm_evalkit.rollout_export",
}

__all__ = [
    "ChannelBinding",
    "EMGChannelMap",
    "EMGSource",
    "EMGTrial",
    "EMGValidationThresholds",
    "ResolvedChannelMap",
    "actuator_pose_sensitivity",
    "compare_emg_validation_reports",
    "musculoskeletal_fingerprint",
    "open_loop_pose_drift",
    "spatial_site_channel_map",
    "summarize_emg_validation",
    "synergy_subspace_metrics",
    "validate_emg_feasibility",
    "validate_inverse_emg",
    "validate_trial",
]


def __getattr__(name: str) -> Any:
    if name in _COMPAT_INTEGRATION_EXPORTS:
        mod = importlib.import_module(_COMPAT_INTEGRATION_EXPORTS[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_COMPAT_INTEGRATION_EXPORTS))
