# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for trajectory_io helpers."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from myosuite.integrations.musclemimic.trajectory_io import (
    MotionClip,
    expand_motion_clip_to_model,
    load_motion_clip,
    resolve_motion_path,
)


def _write_motion(path: Path, nq: int, nv: int) -> None:
    frames = 4
    np.savez(
        path,
        qpos=np.zeros((frames, nq), dtype=np.float32),
        qvel=np.zeros((frames, nv), dtype=np.float32),
        frequency=np.array(100.0, dtype=np.float32),
    )


def test_resolve_motion_path_absolute(tmp_path: Path) -> None:
    """Absolute NPZ path should resolve as-is."""
    p = tmp_path / "m.npz"
    _write_motion(p, nq=3, nv=2)
    got = resolve_motion_path(str(p))
    assert got == p.resolve()


def test_load_motion_clip_validates_dimensions(tmp_path: Path) -> None:
    """Loader validates qpos/qvel widths against model nq/nv."""
    p = tmp_path / "m.npz"
    _write_motion(p, nq=5, nv=4)
    clip = load_motion_clip(p, expected_nq=5, expected_nv=4)
    assert clip.qpos.shape == (4, 5)
    assert clip.qvel is not None
    assert clip.frequency_hz == pytest.approx(100.0)


def test_load_motion_clip_rejects_bad_qpos(tmp_path: Path) -> None:
    """Wrong qpos width should raise ValueError."""
    p = tmp_path / "bad.npz"
    np.savez(p, qpos=np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="qpos width mismatch"):
        load_motion_clip(p, expected_nq=4, expected_nv=3)


def test_load_motion_clip_accepts_partial_joint_subset(tmp_path: Path) -> None:
    """Named joint subsets should load without full-width qpos/qvel arrays."""
    p = tmp_path / "partial.npz"
    np.savez(
        p,
        qpos=np.array([[0.1], [0.2]], dtype=np.float32),
        qvel=np.array([[0.0], [0.1]], dtype=np.float32),
        joint_names=np.array(["joint_a"]),
    )

    clip = load_motion_clip(p, expected_nq=2, expected_nv=2)

    assert clip.qpos.shape == (2, 1)
    assert clip.qvel is not None and clip.qvel.shape == (2, 1)
    assert clip.qpos_joint_names == ["joint_a"]
    assert clip.qvel_joint_names == ["joint_a"]
    assert clip.qpos_model_indices is None
    assert clip.qvel_model_indices is None


def test_expand_motion_clip_to_model_expands_named_subset() -> None:
    """Partial clips should expand to model width using joint-name metadata."""
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body>
              <joint name="joint_a" type="hinge"/>
              <joint name="joint_b" type="hinge"/>
              <geom type="sphere" size="0.01"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    clip = MotionClip(
        qpos=np.array([[0.1], [0.2]], dtype=np.float64),
        qvel=np.array([[0.0], [0.3]], dtype=np.float64),
        site_xpos=None,
        site_names=None,
        qpos_joint_names=["joint_a"],
        qvel_joint_names=["joint_a"],
    )
    expanded = expand_motion_clip_to_model(clip, model)

    assert expanded.qpos is not None and expanded.qpos.shape == (2, 2)
    assert expanded.qvel is not None and expanded.qvel.shape == (2, 2)
    np.testing.assert_allclose(expanded.qpos[:, 0], [0.1, 0.2])
    np.testing.assert_allclose(expanded.qvel[:, 0], [0.0, 0.3])
    np.testing.assert_allclose(expanded.qpos[:, 1], 0.0)
    np.testing.assert_allclose(expanded.qvel[:, 1], 0.0)
    np.testing.assert_array_equal(
        expanded.qpos_model_indices, np.array([0], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        expanded.qvel_model_indices, np.array([0], dtype=np.int32)
    )
