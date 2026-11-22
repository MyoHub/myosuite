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
    "EMG_TRIAL_META_KEY": "myosuite.integrations.emg.trial_io",
    "NpzDirectoryEMGSource": "myosuite.integrations.emg.trial_io",
    "load_emg_trial": "myosuite.integrations.emg.trial_io",
    "resolve_stem_channel_map": "myosuite.integrations.emg.trial_io",
    "save_emg_trial": "myosuite.integrations.emg.trial_io",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        mod = importlib.import_module(_EXPORTS[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
