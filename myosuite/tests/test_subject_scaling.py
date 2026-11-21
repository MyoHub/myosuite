# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for bilevel multi-frame subject scaling and coupled LandmarkIK."""

from __future__ import annotations

import importlib.util
import sys
import types
from collections.abc import Mapping

import mujoco
import numpy as np
import pytest

from myosuite.biomechanics.retarget import Landmark, LandmarkMap
from myosuite.biomechanics.subject_scaling import (
    BilevelScaleConfig,
    BilevelScaleError,
    fit_subject_scale_and_ik,
    select_diverse_frames,
)
from myosuite.core.body_scaling import scale_spec
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings

pytestmark = pytest.mark.tier1


def _synthetic_scale_spec(
    spec: mujoco.MjSpec,
    scales: Mapping[str, float],
    *,
    total_mass: float | None = None,
) -> mujoco.MjSpec:
    """Minimal kinematic test double for ``myo_sim.scaling.scale_spec`` when unavailable."""
    out = spec.copy()

    def _walk(body: mujoco.MjsBody, s_parent: float) -> None:
        s_here = float(scales.get(body.name, s_parent))
        for st in body.sites:
            st.pos = np.asarray(st.pos, dtype=np.float64) * s_here
        for child in body.bodies:
            child.pos = np.asarray(child.pos, dtype=np.float64) * s_here
            _walk(child, s_here)

    for root in out.worldbody.bodies:
        _walk(root, float(scales.get(root.name, 1.0)))
    if total_mass is not None:
        cur = sum(float(b.mass) for b in out.bodies if b.name != "world")
        if cur > 0.0:
            ratio = float(total_mass) / cur
            for b in out.bodies:
                if b.name != "world":
                    b.mass = float(b.mass) * ratio
    return out


@pytest.fixture
def scaling_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use installed ``myo_sim.scaling`` when present, otherwise stub it for unit tests."""
    if importlib.util.find_spec("myo_sim.scaling") is None:
        fake = types.ModuleType("myo_sim.scaling")
        fake.scale_spec = _synthetic_scale_spec  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "myo_sim.scaling", fake)


def _build_coupled_biped_spec() -> tuple[mujoco.MjSpec, LandmarkMap]:
    xml = """
    <mujoco model="coupled_biped">
      <compiler angle="radian"/>
      <worldbody>
        <body name="pelvis" pos="0 0 0.90">
          <freejoint name="root"/>
          <inertial pos="0 0 0" mass="12.0" diaginertia="0.12 0.12 0.12"/>
          <site name="s_pelvis_front" pos="0.10 0 0.05" size="0.01"/>
          <site name="s_l_asis" pos="0.06 0.12 0.0" size="0.01"/>
          <site name="s_r_asis" pos="0.06 -0.12 0.0" size="0.01"/>
          <body name="femur_l" pos="0 0.10 0">
            <joint name="hip_l" type="hinge" axis="0 1 0" range="-1.4 1.2"/>
            <inertial pos="0 0 -0.20" mass="7.0" diaginertia="0.08 0.08 0.02"/>
            <site name="s_l_hip" pos="0 0.04 0" size="0.01"/>
            <site name="s_l_thigh_pad" pos="0.04 0.05 -0.22" size="0.01"/>
            <site name="s_l_knee" pos="0 0.04 -0.40" size="0.01"/>
            <body name="tibia_l" pos="0 0 -0.40">
              <joint name="knee_l" type="hinge" axis="0 1 0" range="-2.1 0.05"/>
              <inertial pos="0 0 -0.20" mass="4.0" diaginertia="0.05 0.05 0.01"/>
              <site name="s_l_shank_pad" pos="0.03 0.04 -0.18" size="0.01"/>
              <site name="s_l_ankle" pos="0 0.03 -0.40" size="0.01"/>
              <body name="patella_l" pos="0.04 0 0">
                <joint name="knee_l_cpl" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
                <inertial pos="0 0 0" mass="0.1" diaginertia="0.001 0.001 0.001"/>
              </body>
            </body>
          </body>
          <body name="femur_r" pos="0 -0.10 0">
            <joint name="hip_r" type="hinge" axis="0 1 0" range="-1.4 1.2"/>
            <inertial pos="0 0 -0.20" mass="7.0" diaginertia="0.08 0.08 0.02"/>
            <site name="s_r_hip" pos="0 -0.04 0" size="0.01"/>
            <site name="s_r_thigh_pad" pos="0.04 -0.05 -0.22" size="0.01"/>
            <site name="s_r_knee" pos="0 -0.04 -0.40" size="0.01"/>
            <body name="tibia_r" pos="0 0 -0.40">
              <joint name="knee_r" type="hinge" axis="0 1 0" range="-2.1 0.05"/>
              <inertial pos="0 0 -0.20" mass="4.0" diaginertia="0.05 0.05 0.01"/>
              <site name="s_r_shank_pad" pos="0.03 -0.04 -0.18" size="0.01"/>
              <site name="s_r_ankle" pos="0 -0.03 -0.40" size="0.01"/>
              <body name="patella_r" pos="0.04 0 0">
                <joint name="knee_r_cpl" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
                <inertial pos="0 0 0" mass="0.1" diaginertia="0.001 0.001 0.001"/>
              </body>
            </body>
          </body>
        </body>
      </worldbody>
      <equality>
        <joint joint1="knee_l_cpl" joint2="knee_l" polycoef="0 0.35 0.05 0 0"/>
        <joint joint1="knee_r_cpl" joint2="knee_r" polycoef="0 0.35 0.05 0 0"/>
      </equality>
      <tendon>
        <spatial name="quad_l">
          <site site="s_l_hip"/>
          <site site="s_l_shank_pad"/>
        </spatial>
      </tendon>
      <actuator>
        <muscle name="rect_fem_l" tendon="quad_l" lengthrange="0.18 0.28" force="600"/>
      </actuator>
    </mujoco>
    """
    spec = mujoco.MjSpec.from_string(xml)
    site_pairs = (
        ("pelvis_front", "pelvis", "s_pelvis_front"),
        ("l_asis", "pelvis", "s_l_asis"),
        ("r_asis", "pelvis", "s_r_asis"),
        ("l_hip", "femur_l", "s_l_hip"),
        ("l_thigh_pad", "femur_l", "s_l_thigh_pad"),
        ("l_knee", "femur_l", "s_l_knee"),
        ("l_shank_pad", "tibia_l", "s_l_shank_pad"),
        ("l_ankle", "tibia_l", "s_l_ankle"),
        ("r_hip", "femur_r", "s_r_hip"),
        ("r_thigh_pad", "femur_r", "s_r_thigh_pad"),
        ("r_knee", "femur_r", "s_r_knee"),
        ("r_shank_pad", "tibia_r", "s_r_shank_pad"),
        ("r_ankle", "tibia_r", "s_r_ankle"),
    )
    lmap = LandmarkMap(
        name="coupled_biped_markers",
        version="1.0",
        landmarks=tuple(Landmark(src, body, st) for src, body, st in site_pairs),
    )
    return spec, lmap


def _simulate_marker_trial(
    spec: mujoco.MjSpec,
    lmap: LandmarkMap,
    true_scales: dict[str, float],
    pad_shifts: dict[str, np.ndarray],
    n_frames: int = 24,
) -> np.ndarray:
    gt_spec = scale_spec(spec, true_scales)
    spec_sites = {st.name: st for b in gt_spec.bodies for st in b.sites}
    for lm in lmap.landmarks:
        if lm.source in pad_shifts and lm.site is not None:
            st = spec_sites[lm.site]
            st.pos = np.asarray(st.pos, dtype=np.float64) + pad_shifts[lm.source]
    gt_model = gt_spec.compile()
    gt_data = mujoco.MjData(gt_model)
    couplings = JointCouplings.from_model(gt_model)

    t = np.linspace(0.0, 2.0 * np.pi, n_frames)
    markers = np.empty((n_frames, len(lmap.landmarks), 3), dtype=np.float64)
    for k in range(n_frames):
        q = gt_model.qpos0.copy()
        q[0] = 0.04 * k
        q[2] = 0.90 + 0.03 * np.sin(t[k])
        q[gt_model.jnt_qposadr[gt_model.joint("hip_l").id]] = 0.45 * np.sin(t[k])
        q[gt_model.jnt_qposadr[gt_model.joint("knee_l").id]] = -0.85 * (
            0.5 + 0.5 * np.sin(t[k] + 0.4)
        )
        q[gt_model.jnt_qposadr[gt_model.joint("hip_r").id]] = -0.45 * np.sin(t[k])
        q[gt_model.jnt_qposadr[gt_model.joint("knee_r").id]] = -0.85 * (
            0.5 + 0.5 * np.cos(t[k])
        )
        project_joint_couplings(couplings, q)
        gt_data.qpos[:] = q
        mujoco.mj_kinematics(gt_model, gt_data)
        for i, lm in enumerate(lmap.landmarks):
            assert lm.site is not None
            markers[k, i] = gt_data.site_xpos[gt_model.site(lm.site).id]
    return markers


def test_missing_myo_sim_scaling_raises_actionable_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, lmap = _build_coupled_biped_spec()
    markers = np.zeros((4, len(lmap.landmarks), 3), dtype=np.float64)
    monkeypatch.setitem(sys.modules, "myo_sim.scaling", None)
    with pytest.raises(ImportError, match="myo_sim PR #149"):
        fit_subject_scale_and_ik(spec, markers, lmap)


def test_bilevel_scaling_recovers_segment_scales_marker_offsets_and_couplings(
    scaling_backend: None,
) -> None:
    del scaling_backend
    spec, lmap = _build_coupled_biped_spec()
    true_scales = {
        "pelvis": 1.08,
        "femur_l": 1.16,
        "femur_r": 1.16,
        "tibia_l": 0.88,
        "tibia_r": 0.88,
    }
    pad_shifts = {
        "l_thigh_pad": np.array([0.016, 0.010, -0.012]),
        "r_thigh_pad": np.array([0.016, -0.010, -0.012]),
    }
    rng = np.random.default_rng(12)
    markers = _simulate_marker_trial(spec, lmap, true_scales, pad_shifts, n_frames=24)
    markers_noisy = markers + rng.normal(0.0, 0.0008, size=markers.shape)
    markers_noisy[5, 4] = np.nan

    bony = (
        "pelvis_front",
        "l_asis",
        "r_asis",
        "l_hip",
        "l_knee",
        "l_ankle",
        "r_hip",
        "r_knee",
        "r_ankle",
    )
    cfg = BilevelScaleConfig(
        max_calibration_frames=10,
        optimize_marker_offsets=True,
        fixed_landmarks=bony,
        offset_reg_weight=0.5,
        scale_reg_weight=1e-4,
    )
    res = fit_subject_scale_and_ik(spec, markers_noisy, lmap, config=cfg)

    assert res.converged is True
    assert res.status > 0
    assert np.isfinite(res.optimality) and res.optimality >= 0.0
    assert res.scales["pelvis"] == pytest.approx(1.08, abs=0.03)
    assert res.scales["femur_l"] == pytest.approx(1.16, abs=0.025)
    assert res.scales["tibia_l"] == pytest.approx(0.88, abs=0.025)
    assert res.initial_rmse_m > 0.025
    assert res.final_rmse_m < 0.003

    np.testing.assert_allclose(res.marker_offsets["l_hip"], 0.0, atol=1e-12)
    np.testing.assert_allclose(
        res.marker_offsets["l_thigh_pad"], pad_shifts["l_thigh_pad"], atol=0.004
    )

    cpl = JointCouplings.from_model(res.scaled_model)
    q_proj = res.qpos.copy()
    project_joint_couplings(cpl, q_proj)
    np.testing.assert_allclose(res.qpos, q_proj, atol=1e-10)


def test_select_diverse_frames_asymmetric_scaling_and_convergence_policy(
    scaling_backend: None,
) -> None:
    del scaling_backend
    spec, lmap = _build_coupled_biped_spec()
    asym_scales = {
        "pelvis": 1.0,
        "femur_l": 1.14,
        "femur_r": 0.92,
        "tibia_l": 1.05,
        "tibia_r": 0.95,
    }
    markers = _simulate_marker_trial(spec, lmap, asym_scales, {}, n_frames=20)
    idx = select_diverse_frames(markers, max_frames=6)
    assert len(idx) == 6
    assert np.all(np.diff(idx) > 0)

    cfg = BilevelScaleConfig(
        bilateral_symmetry=False,
        max_calibration_frames=8,
        optimize_marker_offsets=False,
        scale_reg_weight=1e-4,
    )
    res = fit_subject_scale_and_ik(spec, markers, lmap, config=cfg)
    assert res.converged is True
    assert res.scales["femur_l"] == pytest.approx(1.14, abs=0.02)
    assert res.scales["femur_r"] == pytest.approx(0.92, abs=0.02)
    assert res.final_rmse_m < 0.002

    with pytest.raises(BilevelScaleError, match="did not converge"):
        fit_subject_scale_and_ik(
            spec,
            markers,
            lmap,
            config=BilevelScaleConfig(outer_max_nfev=1, on_failure="raise"),
        )

    with pytest.warns(RuntimeWarning, match="did not converge"):
        unconv = fit_subject_scale_and_ik(
            spec,
            markers,
            lmap,
            config=BilevelScaleConfig(outer_max_nfev=1, on_failure="warn"),
        )
    assert unconv.converged is False
    assert unconv.status == 0
