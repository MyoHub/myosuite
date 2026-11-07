# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""OpenSim scale files and the Rajagopal segments of MyoLeg and MyoFullBody.

:func:`myosuite.core.body_scaling.scale_model` reads an OpenSim ScaleSet or a
scaled ``.osim`` model with these helpers and maps its segments onto the
model's bodies. They are private: pass the file's path to ``scale_model``
instead of calling them.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

_LEG_SEGMENTS = (
    "femur",
    "tibia",
    "patella",
    "talus",
    "calcn",
    "toes",
)

# A segment's first bodies and, for each of their axes, the OpenSim axis it lies
# along (OpenSim frames: x anterior, y up, z to the right). Every other body
# takes its nearest mapped ancestor's factors, turned into its own frame.
_Segments = dict[str, tuple[tuple[str, ...], tuple[int, int, int]]]
_OPENSIM_AXES = (0, 1, 2)
_Z_UP = (0, 2, 1)  # x anterior, y lateral, z up

# Rajagopal model segments in MyoLeg (``myolegs_with_torso``). Its root and
# torso are z-up; the pelvis below the root and the legs use OpenSim's frames.
_MYOLEG_SEGMENTS: _Segments = {
    "pelvis": (("root",), _Z_UP),
    "torso": (("torso",), _Z_UP),
    **{
        f"{seg}_{side}": ((f"{seg}_{side}",), _OPENSIM_AXES)
        for seg in _LEG_SEGMENTS
        for side in "rl"
    },
}

# Rajagopal model segments in the MuscleMimic MyoFullBody. The lumbar spine,
# thorax, shoulder girdle and head follow ``torso``; the 27 hand bodies follow
# ``hand``. Its root body ("Full Body") is z-up with a lateral x.
_MYOFULLBODY_SEGMENTS: _Segments = {
    "pelvis": (("Full Body",), (2, 0, 1)),
    "torso": (("lumbar5",), _OPENSIM_AXES),
    **{
        f"{seg}_{side}": ((f"{seg}_{side}",), _OPENSIM_AXES)
        for seg in (*_LEG_SEGMENTS, "humerus", "ulna", "radius")
        for side in "rl"
    },
    "hand_r": (("lunate_r",), _OPENSIM_AXES),
    "hand_l": (("lunate_l",), _OPENSIM_AXES),
}


def _segments_for(spec: mujoco.MjSpec) -> _Segments:
    """The Rajagopal segment mapping whose first bodies are all in *spec*.

    MyoFullBody also has bodies named like MyoLeg's (``root``, ``torso``), so
    the mapping with the most segments wins.
    """
    names = {b.name for b in spec.bodies[1:]}
    tables = [
        table
        for table in (_MYOFULLBODY_SEGMENTS, _MYOLEG_SEGMENTS)
        if all(body in names for bodies, _ in table.values() for body in bodies)
    ]
    if not tables:
        raise ValueError(
            "No OpenSim segment mapping fits this model (MyoLeg and MyoFullBody "
            "have one); pass per-body factors instead."
        )
    return max(tables, key=len)


def _read_opensim_scales(path: str | os.PathLike[str]) -> dict[str, np.ndarray]:
    """Segment name -> ``(sx, sy, sz)`` from a ScaleSet or a scaled ``.osim`` model.

    A scaled model carries the applied factors as its geometry's
    ``scale_factors`` (one for the unscaled Rajagopal model).
    """
    root = ET.parse(path).getroot()
    return _scale_set(root) or _model_scales(root)


def _scale_set(root: ET.Element) -> dict[str, np.ndarray]:
    scales = {}
    for scale in root.iter("Scale"):
        if (scale.findtext("apply") or "true").strip().lower() == "false":
            continue
        segment, values = scale.findtext("segment"), scale.findtext("scales")
        if segment and values:
            scales[segment.strip()] = np.array(values.split(), dtype=float)
    return scales


def _model_scales(root: ET.Element) -> dict[str, np.ndarray]:
    scales = {}
    for body in root.iter("Body"):
        name, factors = body.get("name"), body.find(".//scale_factors")
        if name and factors is not None and factors.text:
            scales[name] = np.array(factors.text.split(), dtype=float)
    return scales
