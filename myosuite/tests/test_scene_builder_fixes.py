# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Regression tests for SceneSpec MJCF XML and mjlab EntityCfg builders."""

from __future__ import annotations

import mujoco
import pytest

from myosuite.scenes.library import SCENES, ObjectSpec, SceneSpec
from myosuite.scenes.scene_builder import (
    get_scene,
    scene_to_mjcf_xml,
    scene_to_mjlab_cfg,
)

pytestmark = pytest.mark.tier1


def test_all_registered_scenes_compile_and_get_scene_returns_deepcopy() -> None:
    s1 = get_scene("cube_reach")
    s1.objects.clear()
    assert len(get_scene("cube_reach").objects) == 1

    for name in SCENES:
        xml = scene_to_mjcf_xml(get_scene(name))
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

    multi = SceneSpec(
        objects=[ObjectSpec(name="cap", shape="capsule", size=0.04, count=3)]
    )
    model_multi = mujoco.MjModel.from_xml_string(scene_to_mjcf_xml(multi))
    assert model_multi.nbody == 4
    assert isinstance(scene_to_mjlab_cfg(multi), dict)
