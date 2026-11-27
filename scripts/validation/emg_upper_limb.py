# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Compatibility facade re-exporting upper-limb and hand EMG source adapters."""

from __future__ import annotations

from typing import Any

from myosuite.integrations.emg import upper_limb_sources as _impl
from myosuite.integrations.emg.upper_limb_sources import *  # noqa: F403

__all__ = list(_impl.__all__)


def __getattr__(name: str) -> Any:
    return getattr(_impl, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_impl)))
