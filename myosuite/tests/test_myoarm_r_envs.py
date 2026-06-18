# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Numeric sanity tests for the myoarm_r-based task models.

These tests verify that the models built from PR #96 assets load correctly,
have the expected DOF/actuator counts, and produce stable physics.
"""

from __future__ import annotations

import numpy as np
import pytest

import mujoco

from myosuite.utils.simhive_path import resolve_model_xml_path
from pathlib import Path

_ASSETS = Path(__file__).resolve().parents[1] / "envs" / "myo" / "assets"
_RELOCATE_XML = _ASSETS / "arm" / "myoarm_relocate.xml"
_TABLETENNIS_XML = _ASSETS / "arm" / "myoarm_tabletennis.xml"


@pytest.fixture(scope="module")
def relocate_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(resolve_model_xml_path(_RELOCATE_XML)))


@pytest.fixture(scope="module")
def tabletennis_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(resolve_model_xml_path(_TABLETENNIS_XML)))


class TestRelocateModel:
    def test_loads(self, relocate_model: mujoco.MjModel) -> None:
        assert relocate_model is not None

    def test_nq(self, relocate_model: mujoco.MjModel) -> None:
        # 38 arm DOF + 6 object joints (3 slide + 3 hinge)
        assert relocate_model.nq == 44

    def test_nu(self, relocate_model: mujoco.MjModel) -> None:
        # 63 arm muscles
        assert relocate_model.nu == 63

    def test_physics_stable(self, relocate_model: mujoco.MjModel) -> None:
        d = mujoco.MjData(relocate_model)
        mujoco.mj_forward(relocate_model, d)
        for _ in range(100):
            mujoco.mj_step(relocate_model, d)
        assert not np.any(np.isnan(d.qpos)), "NaN in qpos after 100 steps"
        assert not np.any(np.isnan(d.qvel)), "NaN in qvel after 100 steps"

    def test_clavicle_at_shoulder_height(self, relocate_model: mujoco.MjModel) -> None:
        d = mujoco.MjData(relocate_model)
        mujoco.mj_forward(relocate_model, d)
        bid = mujoco.mj_name2id(relocate_model, mujoco.mjtObj.mjOBJ_BODY, "clavicle_r")
        assert bid >= 0, "clavicle_r body not found"
        z = float(d.xpos[bid][2])
        assert (
            1.2 < z < 1.8
        ), f"clavicle_r z={z:.3f} not in expected shoulder-height range [1.2, 1.8]"

    def test_object_site_exists(self, relocate_model: mujoco.MjModel) -> None:
        sid = mujoco.mj_name2id(relocate_model, mujoco.mjtObj.mjOBJ_SITE, "object_o")
        assert sid >= 0, "object_o site not found"

    def test_target_site_exists(self, relocate_model: mujoco.MjModel) -> None:
        sid = mujoco.mj_name2id(relocate_model, mujoco.mjtObj.mjOBJ_SITE, "target_o")
        assert sid >= 0, "target_o site not found"


class TestTableTennisModel:
    def test_loads(self, tabletennis_model: mujoco.MjModel) -> None:
        assert tabletennis_model is not None

    def test_nq_reasonable(self, tabletennis_model: mujoco.MjModel) -> None:
        # Full body (torso + arm + legs) + paddle freejoint + pingpong freejoint
        # Exact count depends on model; verify it's in a reasonable range
        assert (
            80 < tabletennis_model.nq < 150
        ), f"nq={tabletennis_model.nq} outside expected range"

    def test_nu_has_arm_and_pelvis(self, tabletennis_model: mujoco.MjModel) -> None:
        # At least 63 arm + 2 pelvis actuators
        assert tabletennis_model.nu >= 65, f"nu={tabletennis_model.nu} too low"

    def test_physics_stable(self, tabletennis_model: mujoco.MjModel) -> None:
        d = mujoco.MjData(tabletennis_model)
        mujoco.mj_forward(tabletennis_model, d)
        for _ in range(100):
            mujoco.mj_step(tabletennis_model, d)
        assert not np.any(np.isnan(d.qpos)), "NaN in qpos after 100 steps"
        assert not np.any(np.isnan(d.qvel)), "NaN in qvel after 100 steps"

    def test_pingpong_site_exists(self, tabletennis_model: mujoco.MjModel) -> None:
        sid = mujoco.mj_name2id(tabletennis_model, mujoco.mjtObj.mjOBJ_SITE, "pingpong")
        assert sid >= 0, "pingpong site not found"

    def test_paddle_site_exists(self, tabletennis_model: mujoco.MjModel) -> None:
        sid = mujoco.mj_name2id(tabletennis_model, mujoco.mjtObj.mjOBJ_SITE, "paddle")
        assert sid >= 0, "paddle site not found"
