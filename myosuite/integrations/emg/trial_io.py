# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Pickle-free ``.npz`` persistence and directory source adapter for :class:`EMGTrial`."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

from myosuite.biomechanics.emg import EMGFilterConfig
from myosuite.biomechanics.emg_validation import (
    ChannelBinding,
    EMGChannelMap,
    EMGTrial,
)

EMG_TRIAL_META_KEY = "emg_trial_meta_json"
_EMG_TRIAL_FIELDS = {f.name for f in dataclasses.fields(EMGTrial)}


def save_emg_trial(path: str | Path, trial: EMGTrial) -> None:
    """Write an :class:`EMGTrial` to a compressed pickle-free ``.npz`` archive."""
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "time": np.asarray(trial.time, dtype=np.float64),
        "signals": np.asarray(trial.signals, dtype=np.float64),
    }
    for attr in (
        "qpos",
        "kinematics_time",
        "external_qfrc",
        "cycle_events",
        "mvc_peaks",
    ):
        val = getattr(trial, attr)
        if val is not None:
            arrays[attr] = np.asarray(val, dtype=np.float64)
    header: dict[str, object] = {
        "trial_id": trial.trial_id,
        "channel_names": list(trial.channel_names),
        "signal_kind": trial.signal_kind,
        "filter_config": None
        if trial.filter_config is None
        else asdict(trial.filter_config),
        "meta": dict(trial.meta),
    }
    supported = getattr(trial, "supported_dofs", None)
    if supported is not None:
        header["supported_dofs"] = [int(d) for d in supported]
    arrays[EMG_TRIAL_META_KEY] = np.array(json.dumps(header, sort_keys=True))
    np.savez_compressed(dst, allow_pickle=False, **arrays)


def load_emg_trial(path: str | Path) -> EMGTrial:
    """Load an :class:`EMGTrial` saved with :func:`save_emg_trial`."""
    with np.load(Path(path), allow_pickle=False) as d:
        if EMG_TRIAL_META_KEY not in d.files:
            raise ValueError(f"{path}: missing {EMG_TRIAL_META_KEY}")
        header = json.loads(str(d[EMG_TRIAL_META_KEY].item()))
        fcfg_raw = header.get("filter_config")
        fcfg = None
        if fcfg_raw is not None:
            bp = fcfg_raw.get("bandpass_hz")
            fcfg = EMGFilterConfig(
                bandpass_hz=None if bp is None else (float(bp[0]), float(bp[1])),
                notch_hz=fcfg_raw.get("notch_hz"),
                notch_q=float(fcfg_raw.get("notch_q", 30.0)),
                lowpass_hz=fcfg_raw.get("lowpass_hz"),
                order=int(fcfg_raw.get("order", 2)),
                normalize=str(fcfg_raw.get("normalize", "peak")),
            )
        kwargs: dict[str, object] = {
            "trial_id": str(header["trial_id"]),
            "time": d["time"],
            "signals": d["signals"],
            "channel_names": tuple(str(c) for c in header["channel_names"]),
            "signal_kind": str(header.get("signal_kind", "peak_normalized")),
            "qpos": d["qpos"] if "qpos" in d.files else None,
            "kinematics_time": d["kinematics_time"]
            if "kinematics_time" in d.files
            else None,
            "external_qfrc": d["external_qfrc"] if "external_qfrc" in d.files else None,
            "cycle_events": d["cycle_events"] if "cycle_events" in d.files else None,
            "mvc_peaks": d["mvc_peaks"] if "mvc_peaks" in d.files else None,
            "filter_config": fcfg,
            "meta": header.get("meta", {}),
        }
        if "supported_dofs" in _EMG_TRIAL_FIELDS:
            kwargs["supported_dofs"] = tuple(
                int(x) for x in header.get("supported_dofs", ())
            )
        return EMGTrial(**kwargs)  # type: ignore[arg-type]


class NpzDirectoryEMGSource:
    """Dataset source reading standardized :class:`EMGTrial` ``.npz`` files from a directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.name = self.root.name or "npz_emg"
        self._paths = {p.stem: p for p in sorted(self.root.glob("*.npz"))}

    def trial_ids(self) -> Sequence[str]:
        return tuple(self._paths.keys())

    def get_trial(self, trial_id: str, model: mujoco.MjModel | None = None) -> EMGTrial:
        del model
        return load_emg_trial(self._paths[trial_id])

    def default_channel_map(self, model: mujoco.MjModel) -> EMGChannelMap | None:
        if not self._paths:
            return None
        first = load_emg_trial(next(iter(self._paths.values())))
        if isinstance(first.meta.get("channel_map"), dict):
            return EMGChannelMap.from_dict(first.meta["channel_map"])
        act_names = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
            for i in range(model.nu)
        }
        bindings: list[ChannelBinding] = []
        for ch in first.channel_names:
            for cand in (ch, f"{ch}_r", f"{ch}_l"):
                if cand in act_names:
                    bindings.append(ChannelBinding(channel=ch, actuators=(cand,)))
                    break
        return (
            EMGChannelMap(name=f"{self.name}_auto", bindings=tuple(bindings))
            if bindings
            else None
        )


__all__ = [
    "EMG_TRIAL_META_KEY",
    "NpzDirectoryEMGSource",
    "load_emg_trial",
    "save_emg_trial",
]
