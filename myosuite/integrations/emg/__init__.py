# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""EMG trial persistence and external dataset source adapters."""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, str] = {
    "BaseEMGSource": "myosuite.integrations.emg.trial_io",
    "CAMARGO_CHANNELS": "myosuite.integrations.emg.gait_sources",
    "Camargo2021Source": "myosuite.integrations.emg.gait_sources",
    "DIAZ2026_CHANNELS": "myosuite.integrations.emg.upper_limb_sources",
    "DIMITROV_SHANK_ACTUATORS": "myosuite.integrations.emg.gait_sources",
    "Diaz2026HandSource": "myosuite.integrations.emg.upper_limb_sources",
    "Dimitrov2023Source": "myosuite.integrations.emg.gait_sources",
    "EMG2POSE_JOINT_NAMES": "myosuite.integrations.emg.upper_limb_sources",
    "EMG_TRIAL_META_KEY": "myosuite.integrations.emg.trial_io",
    "Emg2PoseHandSource": "myosuite.integrations.emg.upper_limb_sources",
    "Fritsche2026Source": "myosuite.integrations.emg.upper_limb_sources",
    "GrabMyoSource": "myosuite.integrations.emg.upper_limb_sources",
    "LENCIONI_CHANNELS": "myosuite.integrations.emg.gait_sources",
    "Lencioni2019Source": "myosuite.integrations.emg.gait_sources",
    "ManifestEMGSource": "myosuite.integrations.emg.gait_sources",
    "Maurice2020Source": "myosuite.integrations.emg.upper_limb_sources",
    "NinaProDB5Source": "myosuite.integrations.emg.upper_limb_sources",
    "NpzDirectoryEMGSource": "myosuite.integrations.emg.trial_io",
    "Quesada2024UpperLimbSource": "myosuite.integrations.emg.upper_limb_sources",
    "RajagopalWalkingSource": "myosuite.integrations.emg.gait_sources",
    "ScapulothoracicCapabilityError": "myosuite.integrations.emg.upper_limb_sources",
    "has_myosim_scapulothoracic_rhythm": "myosuite.integrations.emg.upper_limb_sources",
    "load_emg_trial": "myosuite.integrations.emg.trial_io",
    "map_upper_limb_to_qpos": "myosuite.integrations.emg.upper_limb_sources",
    "map_upper_limb_torques_to_qfrc": "myosuite.integrations.emg.upper_limb_sources",
    "read_wfdb_record": "myosuite.integrations.emg.upper_limb_sources",
    "require_myosim_scapulothoracic_rhythm": "myosuite.integrations.emg.upper_limb_sources",
    "resolve_stem_channel_map": "myosuite.integrations.emg.trial_io",
    "save_emg_trial": "myosuite.integrations.emg.trial_io",
    "thoracohumeral_to_upper_limb_coords": "myosuite.integrations.emg.upper_limb_sources",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        mod = importlib.import_module(_EXPORTS[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
