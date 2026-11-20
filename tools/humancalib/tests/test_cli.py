"""CLI smoke test for myosuite-humancalib."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from myosuite.biomechanics.camera import (
    PinholeCamera,
    load_pose2sim_cameras,
    save_pose2sim_cameras,
)
from myosuite_humancalib.cli import main

pytestmark = pytest.mark.tier2


def test_cli_writes_calibrated_pose2sim_toml(tmp_path: Path) -> None:
    k = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
    gt = [
        PinholeCamera(
            k,
            Rotation.from_euler("y", deg, degrees=True).as_matrix(),
            np.array([tx, 0.0, 3.6]),
            f"cam0{i + 1}",
        )
        for i, (deg, tx) in enumerate(((-20.0, -0.9), (20.0, 0.9)))
    ]
    intr_only = [PinholeCamera(k, np.eye(3), np.zeros(3), c.calibration_id) for c in gt]
    intr_toml = save_pose2sim_cameras(tmp_path / "intrinsics.toml", intr_only)

    t = np.linspace(0.0, 1.5, 16)
    pts = np.zeros((16, 4, 3))
    pts[:, 0] = np.column_stack(
        (0.6 * t - 0.4, np.zeros_like(t), np.full_like(t, 1.65))
    )
    pts[:, 1] = np.column_stack((0.6 * t - 0.4, np.full_like(t, 0.1), np.zeros_like(t)))
    pts[:, 2] = np.column_stack(
        (0.6 * t - 0.4, np.full_like(t, -0.1), np.zeros_like(t))
    )
    pts[:, 3] = np.column_stack((0.6 * t - 0.4, np.zeros_like(t), np.full_like(t, 0.9)))

    pixels_2d = np.stack([c.project(pts) for c in gt], axis=0)
    poses_3d = np.stack([pts @ c.rotation.T + c.translation for c in gt], axis=0)
    npz_path = tmp_path / "poses.npz"
    np.savez(
        npz_path,
        pixels_2d=pixels_2d,
        poses_3d=poses_3d,
        bones=np.array([[0, 3], [3, 1], [3, 2]]),
        head_indices=np.array([0]),
        sole_indices=np.array([1, 2]),
        left_heel_index=1,
        right_heel_index=2,
    )
    out_toml = tmp_path / "Calib_scene_calibrated.toml"
    rc = main([str(intr_toml), str(npz_path), "-o", str(out_toml), "--height", "1.78"])
    assert rc == 0 and out_toml.is_file()
    calibrated = load_pose2sim_cameras(out_toml)
    assert len(calibrated) == 2
