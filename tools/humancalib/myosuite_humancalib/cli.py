# Copyright (c) MyoSuite Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Calibrate multi-camera extrinsics from extracted 2D/3D pose ``.npz`` files."""

from __future__ import annotations

import argparse
from pathlib import Path
from collections.abc import Sequence

import numpy as np

from myosuite.biomechanics.camera import load_pose2sim_cameras, save_pose2sim_cameras
from myosuite.biomechanics.camera_calibration import calibrate_cameras_from_motion


def main(argv: Sequence[str] | None = None) -> int:
    """Run multi-camera human calibration from an intrinsics TOML and pose NPZ."""
    parser = argparse.ArgumentParser(prog="myosuite-humancalib")
    parser.add_argument("intrinsics_toml", type=Path, help="Pose2Sim intrinsics TOML")
    parser.add_argument(
        "poses_npz",
        type=Path,
        help="NPZ with pixels_2d (C,T,J,2) and poses_3d (C,T,J,3)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output calibrated Pose2Sim TOML",
    )
    parser.add_argument(
        "--height", type=float, default=None, help="Subject barefoot height [m]"
    )
    parser.add_argument(
        "--root-relative", action="store_true", help="poses_3d are root-relative"
    )
    parser.add_argument(
        "--max-mrad", type=float, default=15.0, help="Acceptance limit [mrad]"
    )
    args = parser.parse_args(argv)

    cams = load_pose2sim_cameras(args.intrinsics_toml)
    with np.load(args.poses_npz) as raw:
        pixels_2d = np.asarray(raw["pixels_2d"], dtype=float)
        poses_3d = np.asarray(raw["poses_3d"], dtype=float)
        conf = (
            np.asarray(raw["confidences"], dtype=float)
            if "confidences" in raw
            else None
        )
        bones = np.asarray(raw["bones"], dtype=int) if "bones" in raw else None
        head_idx = (
            tuple(int(i) for i in raw["head_indices"]) if "head_indices" in raw else ()
        )
        sole_idx = (
            tuple(int(i) for i in raw["sole_indices"]) if "sole_indices" in raw else ()
        )
        l_heel = int(raw["left_heel_index"]) if "left_heel_index" in raw else None
        r_heel = int(raw["right_heel_index"]) if "right_heel_index" in raw else None

    result = calibrate_cameras_from_motion(
        [c.intrinsic for c in cams],
        pixels_2d,
        poses_3d,
        confidences=conf,
        poses_are_root_relative=args.root_relative,
        bones=bones,
        camera_names=[c.calibration_id for c in cams],
        distortions=[c.distortion for c in cams],
        image_sizes=[c.image_size for c in cams],
        fisheye=[c.fisheye for c in cams],
        height_m=args.height,
        head_indices=head_idx,
        sole_indices=sole_idx,
        left_heel_index=l_heel,
        right_heel_index=r_heel,
        max_mre_mrad=args.max_mrad,
    )
    if not result.accepted:
        raise SystemExit(
            f"Calibration rejected: worst camera MRE {result.worst_camera_mre_mrad:.2f} mrad > {args.max_mrad:.2f} mrad"
        )
    save_pose2sim_cameras(args.output, result.cameras)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
