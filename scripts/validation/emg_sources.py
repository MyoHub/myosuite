# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Compatibility re-exports for gait and custom-manifest EMG dataset adapters."""

from __future__ import annotations

from myosuite.integrations.emg.gait_sources import (
    CAMARGO_CHANNELS,
    DIMITROV_SHANK_ACTUATORS,
    LENCIONI_CHANNELS,
    Camargo2021Source,
    Dimitrov2023Source,
    Lencioni2019Source,
    ManifestEMGSource,
    RajagopalWalkingSource,
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
