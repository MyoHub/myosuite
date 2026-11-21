# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Compatibility facade re-exporting public capture I/O and force-plate wrench mapping.

File decoding and gap interpolation live in :mod:`myosuite.integrations.capture`;
generalized-force projection lives in :mod:`myosuite.biomechanics.external_loads`.
"""

from __future__ import annotations

from myosuite.biomechanics.external_loads import force_plates_to_qfrc
from myosuite.integrations.capture import (
    C3DTrial,
    ForcePlateData,
    MarkerClip,
    fill_marker_gaps,
    open_c3d,
    read_c3d_trial,
    read_markers,
    write_trc,
)

__all__ = [
    "C3DTrial",
    "ForcePlateData",
    "MarkerClip",
    "fill_marker_gaps",
    "force_plates_to_qfrc",
    "open_c3d",
    "read_c3d_trial",
    "read_markers",
    "write_trc",
]
