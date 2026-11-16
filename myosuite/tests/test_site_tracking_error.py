# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The playback site-tracking error matches clip columns to model sites by name."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from myosuite.core.trajectory_io import MotionClip
from myosuite.envs.myo.tasks.mimic.clip_env import _CLIP_SITE_ORDER
from myosuite.integrations.musclemimic.fullbody_native_playback import (
    _compute_site_tracking_error,
)

pytestmark = pytest.mark.tier1

# Model sites in neither the model-dict order nor the clip order, after an unrelated site.
_MODEL_SITES = ("left_hand_mimic", "pelvis_mimic", "head_mimic", "right_knee_mimic")


def _model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    sites = "".join(
        f'<site name="{name}" pos="{0.1 * i} {0.2 * i} {0.3 + 0.1 * i}"/>'
        for i, name in enumerate(_MODEL_SITES, start=1)
    )
    model = mujoco.MjModel.from_xml_string(
        f'<mujoco><worldbody><site name="other"/><body>{sites}</body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _clip(names: list[str], data: mujoco.MjData, model: mujoco.MjModel) -> np.ndarray:
    xpos = np.full((2, len(names), 3), 5.0)
    for col, name in enumerate(names):
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site >= 0:
            xpos[:, col] = data.site_xpos[site]
    return xpos


@pytest.mark.parametrize("named", [True, False], ids=["site_names", "17-site order"])
def test_a_body_on_the_reference_has_zero_error(named: bool) -> None:
    model, data = _model()
    names = list(_CLIP_SITE_ORDER)
    xpos = _clip(names, data, model)
    clip = MotionClip(
        qpos=np.zeros((2, 1)),
        qvel=None,
        site_xpos=xpos,
        site_names=names if named else None,
    )
    assert _compute_site_tracking_error(model, data, clip, 0) == pytest.approx(
        0.0, abs=1e-12
    )
    xpos[1, names.index("head_mimic"), 2] += 0.4
    expected = 0.4 / len(_MODEL_SITES)
    assert _compute_site_tracking_error(model, data, clip, 1) == pytest.approx(expected)


def test_a_clip_without_the_sites_has_no_error() -> None:
    model, data = _model()
    clip = MotionClip(
        qpos=np.zeros((2, 1)),
        qvel=None,
        site_xpos=np.zeros((2, 3, 3)),
        site_names=["a", "b", "c"],
    )
    assert _compute_site_tracking_error(model, data, clip, 0) is None
