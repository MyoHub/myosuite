# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Regression tests for packaging metadata, wrapper pickling, and core helpers."""

from __future__ import annotations

import pickle
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import mujoco
import numpy as np
import pytest

from myosuite.core.model_builder import build_from_recipe
from myosuite.core.muscle_conditions import (
    apply_sarcopenia_to_model,
    apply_sarcopenia_to_spec,
)
from myosuite.core.trajectory_io import load_motion_clip, motion_clip_from_states
from myosuite.envs.myo.tasks.mimic.tracking.env import MimicEnv
from myosuite.envs.waypoint import WaypointEnv, WaypointTaskCfg
from myosuite.envs.wrappers import FatigueWrapper, MotorNoiseWrapper, SarcopeniaWrapper
from myosuite.terms.base_action import MotorNoiseCfg

pytestmark = pytest.mark.tier1

_HYBRID_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body name="b" pos="0 0 0.5">
      <joint name="j1" type="hinge" axis="0 1 0" range="-1 1"/>
      <joint name="j2" type="hinge" axis="1 0 0" range="-1 1"/>
      <geom type="sphere" size="0.05" mass="1"/>
      <site name="tip" pos="0 0 0"/>
    </body>
  </worldbody>
  <tendon>
    <fixed name="t1"><joint joint="j1" coef="0.05"/></fixed>
  </tendon>
  <actuator>
    <muscle name="m1" tendon="t1" force="120"/>
    <general name="aff" joint="j2" gaintype="affine" gainprm="10 0 -3.5" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""


def test_pyproject_package_data_and_docs_deps() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    pkg_data = cfg["tool"]["setuptools"]["package-data"]["myosuite"]
    excl = cfg["tool"]["setuptools"]["exclude-package-data"]["myosuite"]
    docs_deps = cfg["project"]["optional-dependencies"]["docs"]

    assert "biomechanics/**" in pkg_data
    assert "robot/**" not in pkg_data
    assert "**/.myosuite_resolved_*.xml" in excl
    assert any(dep.startswith("sphinx-autoapi") for dep in docs_deps)


def test_sarcopenia_masks_non_muscle_actuators() -> None:
    spec = mujoco.MjSpec.from_string(_HYBRID_XML)
    apply_sarcopenia_to_spec(spec, force_scale=0.5)
    m_spec = spec.compile()
    assert m_spec.actuator_gainprm[0, 2] == pytest.approx(60.0)
    assert m_spec.actuator_gainprm[1, 2] == pytest.approx(-3.5)

    m_direct = mujoco.MjModel.from_xml_string(_HYBRID_XML)
    apply_sarcopenia_to_model(m_direct, force_scale=0.5)
    assert m_direct.actuator_gainprm[0, 2] == pytest.approx(60.0)
    assert m_direct.actuator_gainprm[1, 2] == pytest.approx(-3.5)


def test_wrapped_waypoint_and_mimic_pickling_preserves_single_scaling() -> None:
    base_model = mujoco.MjModel.from_xml_string(_HYBRID_XML)
    wp = WaypointEnv(
        base_model,
        task=WaypointTaskCfg(site_name="tip", waypoints=((0.2, 0.0),)),
        frame_skip=1,
    )
    wrapped = MotorNoiseWrapper(
        FatigueWrapper(SarcopeniaWrapper(wp, force_scale=0.5)),
        MotorNoiseCfg(signal_dependent_std=0.05),
    )
    restored = pickle.loads(pickle.dumps(wrapped))
    assert restored.unwrapped.model.actuator_gainprm[0, 2] == pytest.approx(60.0)
    assert base_model.actuator_gainprm[0, 2] == pytest.approx(120.0)
    restored.reset(seed=7)
    restored.step(np.zeros(restored.action_space.shape, dtype=np.float32))
    restored.close()
    wrapped.close()

    mimic = MotorNoiseWrapper(
        FatigueWrapper(SarcopeniaWrapper(MimicEnv(frame_skip=10), force_scale=0.5)),
        MotorNoiseCfg(constant_std=0.02),
    )
    f0 = float(mimic.unwrapped.model.actuator_gainprm[0, 2])
    mimic_restored = pickle.loads(pickle.dumps(mimic))
    assert float(
        mimic_restored.unwrapped.model.actuator_gainprm[0, 2]
    ) == pytest.approx(f0)
    mimic_restored.close()
    mimic.close()


def test_hand_pen_rgba_and_musclemimic_bimanual_spec_recipe() -> None:
    pen_model, _ = build_from_recipe("hand_pen")
    sid = mujoco.mj_name2id(pen_model, mujoco.mjtObj.mjOBJ_SITE, "target_ball")
    assert sid >= 0
    assert np.all((pen_model.site_rgba[sid] >= 0.0) & (pen_model.site_rgba[sid] <= 1.0))

    try:
        bim_model, _ = build_from_recipe("musclemimic_bimanual")
    except (FileNotFoundError, ImportError) as exc:
        pytest.skip(f"musclemimic_models unavailable: {exc}")
    assert bim_model.nq > 0


def test_trajectory_io_closes_npz_and_differentiates_in_float64(tmp_path: Path) -> None:
    xml = """
    <mujoco><worldbody><body name="b"><joint name="x" type="slide" axis="1 0 0"/>
    <geom size="0.1"/><site name="s"/></body></worldbody></mujoco>
    """
    m = mujoco.MjModel.from_xml_string(xml)
    dt = 0.01
    qpos = 1000.0 + np.arange(5, dtype=np.float64)[:, None] * 1e-4
    clip = motion_clip_from_states(m, qpos, dt, ("s",))
    assert clip.qvel is not None
    np.testing.assert_allclose(clip.qvel[:, 0], 0.01, rtol=1e-4)

    npz_path = tmp_path / "clip.npz"
    np.savez(npz_path, qpos=qpos)
    loaded = load_motion_clip(npz_path, expected_nq=1, expected_nv=1)
    assert loaded.qpos.shape == (5, 1)
    npz_path.unlink()
