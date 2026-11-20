# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for standalone InstantHMR subject tracking and clip extraction."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

_PKG_ROOT = str(Path(__file__).resolve().parents[1])
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from myosuite_instanthmr import (  # noqa: E402
    MHR70_NUM_JOINTS,
    TrackerConfig,
    box_iou,
    extract_predictions,
)

pytestmark = pytest.mark.tier1


@dataclass
class _DummyPerson:
    bbox: np.ndarray
    joints_3d_cam: np.ndarray
    shape_params: np.ndarray | None = None
    score: float = 0.95


@dataclass
class _DummyResult:
    persons: list[_DummyPerson]


def _make_person(
    bbox: tuple[float, float, float, float],
    pelvis_xyz: tuple[float, float, float],
    *,
    shape_val: float | None = 0.25,
    score: float = 0.95,
) -> _DummyPerson:
    joints = np.zeros((MHR70_NUM_JOINTS, 3), dtype=np.float32)
    joints[:] = np.asarray(pelvis_xyz, dtype=np.float32)
    shape = np.full(10, shape_val, dtype=np.float32) if shape_val is not None else None
    return _DummyPerson(
        bbox=np.asarray(bbox, dtype=np.float32),
        joints_3d_cam=joints,
        shape_params=shape,
        score=score,
    )


def test_box_iou_basic() -> None:
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50.0 / 150.0)


def test_bystander_rejected_when_primary_disappears() -> None:
    """Regression: A disappears and distant B remains as sole detection (zero overlap)."""
    fps = 30.0
    results: list[_DummyResult] = []
    for t in range(12):
        bystander_b = _make_person(
            (500.0, 120.0, 580.0, 320.0), (2.8, 0.0, 5.5), shape_val=-0.9
        )
        if t < 6:
            subject_a = _make_person(
                (100.0 + 3.0 * t, 80.0, 240.0 + 3.0 * t, 420.0),
                (-0.4 + 0.02 * t, 0.0, 2.5),
                shape_val=0.4,
            )
            results.append(_DummyResult([subject_a, bystander_b]))
        else:
            # Subject A is gone; only distant B remains (len(persons) == 1).
            results.append(_DummyResult([bystander_b]))

    clip = extract_predictions(results, fps=fps)
    assert clip["valid"][:6].all()
    assert not clip["valid"][6:].any()
    assert np.isnan(clip["joints_3d_cam"][6:]).all()
    assert "track_lost" in clip["quality_flags"].tolist()


def test_short_occlusion_reacquires_subject_ignoring_bystander() -> None:
    fps = 30.0
    results: list[_DummyResult] = []
    for t in range(12):
        bystander_b = _make_person((520.0, 100.0, 600.0, 310.0), (2.5, 0.0, 5.0))
        if t in (4, 5):
            results.append(_DummyResult([bystander_b]))
        else:
            subject_a = _make_person(
                (100.0 + 2.0 * t, 80.0, 240.0 + 2.0 * t, 420.0),
                (0.03 * t, 0.0, 2.5),
            )
            results.append(_DummyResult([bystander_b, subject_a]))

    clip = extract_predictions(results, fps=fps)
    expected_valid = np.array([t not in (4, 5) for t in range(12)], dtype=bool)
    np.testing.assert_array_equal(clip["valid"], expected_valid)
    assert clip["quality_flags"].tolist() == []
    assert np.allclose(clip["joints_3d_cam"][6, 0], [0.18, 0.0, 2.5], atol=1e-5)


def test_long_occlusion_terminates_track_without_switching() -> None:
    fps = 30.0
    cfg = TrackerConfig(max_gap_s=0.1)  # at most 3 missed frames at 30 Hz
    results: list[_DummyResult] = []
    for t in range(15):
        if t < 4:
            results.append(
                _DummyResult([_make_person((100, 80, 240, 420), (0.0, 0.0, 2.5))])
            )
        elif t < 10:
            results.append(_DummyResult([]))
        else:
            # New person enters near the old box long after track expiration.
            results.append(
                _DummyResult([_make_person((105, 82, 245, 422), (0.0, 0.0, 2.5))])
            )

    clip = extract_predictions(results, fps=fps, config=cfg)
    assert clip["valid"][:4].all()
    assert not clip["valid"][4:].any()
    assert "track_lost" in clip["quality_flags"].tolist()


def test_depth_jump_rejected_even_with_2d_box_overlap(tmp_path: Path) -> None:
    fps = 30.0
    results = [
        _DummyResult([_make_person((100, 80, 240, 420), (0.0, 0.0, 2.2))]),
        _DummyResult([_make_person((102, 80, 242, 420), (0.02, 0.0, 2.2))]),
        # Same 2D region, but 3.5 m deeper in camera space -> background person!
        _DummyResult([_make_person((104, 80, 244, 420), (0.04, 0.0, 5.7))]),
    ]
    clip = extract_predictions(results, fps=fps)
    np.testing.assert_array_equal(clip["valid"], [True, True, False])
    out = tmp_path / "out.npz"
    np.savez_compressed(out, **clip)
    with np.load(out, allow_pickle=False) as loaded:
        assert "track_lost" in loaded["quality_flags"].tolist()


def _make_walking_person(
    bbox: tuple[float, float, float, float],
    pelvis_xyz: tuple[float, float, float],
    swing_phase: float,
    swing_amp_m: float,
) -> _DummyPerson:
    p = _make_person(bbox, pelvis_xyz)
    hip = np.asarray(pelvis_xyz, dtype=np.float32)
    p.joints_3d_cam[9] = hip + np.array([-0.10, 0.0, 0.0], dtype=np.float32)
    p.joints_3d_cam[10] = hip + np.array([0.10, 0.0, 0.0], dtype=np.float32)
    s = float(swing_amp_m * np.sin(swing_phase))
    p.joints_3d_cam[13] = hip + np.array([-0.10, 0.80, s], dtype=np.float32)
    p.joints_3d_cam[14] = hip + np.array([0.10, 0.80, -s], dtype=np.float32)
    return p


def test_leg_swing_selection_locks_walking_subject_over_large_bystander() -> None:
    fps = 30.0
    results: list[_DummyResult] = []
    for t in range(18):
        # Huge foreground static bystander (clinician standing near camera).
        bystander = _make_walking_person(
            (20.0, 20.0, 420.0, 620.0), (-0.8, 0.0, 1.8), 0.0, 0.0
        )
        # Smaller walking patient further away with active 3D leg swing.
        walker = _make_walking_person(
            (480.0 + 2.0 * t, 120.0, 580.0 + 2.0 * t, 380.0),
            (0.5 + 0.03 * t, 0.0, 3.4),
            2.0 * np.pi * 1.5 * (t / fps),
            0.25,
        )
        results.append(_DummyResult([bystander, walker]))

    clip_area = extract_predictions(
        results, fps=fps, config=TrackerConfig(selection_mode="bbox_area")
    )
    clip_swing = extract_predictions(
        results, fps=fps, config=TrackerConfig(selection_mode="leg_swing")
    )
    assert clip_area["joints_3d_cam"][0, 0, 0] == pytest.approx(-0.8)
    assert clip_swing["valid"].all()
    assert clip_swing["joints_3d_cam"][0, 0, 0] == pytest.approx(0.5)
    assert clip_swing["joints_3d_cam"][-1, 0, 0] == pytest.approx(0.5 + 0.03 * 17)


def test_leg_swing_selection_waits_for_late_walker_entry() -> None:
    fps = 30.0
    results: list[_DummyResult] = []
    for t in range(12):
        bystander = _make_walking_person(
            (30.0, 30.0, 380.0, 580.0), (-0.7, 0.0, 1.9), 0.0, 0.0
        )
        if t < 3:
            results.append(_DummyResult([bystander]))
        else:
            walker = _make_walking_person(
                (460.0 + 2.0 * t, 110.0, 560.0 + 2.0 * t, 370.0),
                (0.6 + 0.03 * t, 0.0, 3.2),
                2.0 * np.pi * 1.5 * (t / fps),
                0.25,
            )
            results.append(_DummyResult([bystander, walker]))

    clip = extract_predictions(
        results, fps=fps, config=TrackerConfig(selection_mode="leg_swing")
    )
    assert not clip["valid"][:3].any()
    assert clip["valid"][3:].all()
    assert clip["joints_3d_cam"][3, 0, 0] == pytest.approx(0.6 + 0.03 * 3)
