# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Convert GVHMR Y-up pose outputs into Z-up AMASS ``*_poses.npz`` clips.

    python scripts/mimic/convert_gvhmr.py input_results.pt --out clip_poses.npz \\
        --fps 30 --body-model /path/to/SMPLX_NEUTRAL.npz

The written ``.npz`` can be passed straight to ``scripts/mimic/retarget_amass.py``
to produce ``myoMimic-v0`` reference trajectories.

Note: SMPL-X / SMPL+H body model files passed to ``--body-model`` are not distributed
with MyoSuite and must be obtained directly from MPI-IS under their upstream
non-commercial license terms (https://smpl-x.is.tue.mpg.de/).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from myosuite.integrations.gvhmr import (
    load_gvhmr_result,
    save_amass_npz,
    to_amass_clip,
)


def _load_any_gvhmr(path: Path) -> dict[str, Any]:
    """Load GVHMR ``.pt``, ``.pth``, ``.npz``, or ``.pkl`` results."""
    if path.suffix.lower() in (".pt", ".pth"):
        import torch

        raw = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(raw, dict):
            raise TypeError(f"Expected dict in {path}, got {type(raw).__name__}.")
        return raw
    return load_gvhmr_result(path)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point converting one GVHMR result file or directory of files."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog=(
            "Body models passed via --body-model are governed by MPI-IS's upstream "
            "SMPL-X / SMPL+H non-commercial license terms."
        ),
    )
    parser.add_argument("input", type=Path, help="GVHMR .pt/.npz file or directory")
    parser.add_argument(
        "--out", type=Path, required=True, help="Output .npz file or directory"
    )
    parser.add_argument(
        "--fps", type=float, default=30.0, help="Motion frame rate (Hz)"
    )
    parser.add_argument(
        "--body-model",
        type=Path,
        default=None,
        help="Neutral SMPL-X / SMPL+H model (.npz or .pkl) when pelvis_j0 is not stored",
    )
    parser.add_argument(
        "--layout",
        choices=("smplx", "smplh"),
        default="smplx",
        help="Pose array layout in output .npz (default: smplx)",
    )
    args = parser.parse_args(argv)

    if args.input.is_dir():
        sources = sorted(
            p
            for p in args.input.rglob("*")
            if p.suffix.lower() in (".pt", ".pth", ".npz", ".pkl")
        )
        for src in sources:
            rel = src.relative_to(args.input).with_suffix("")
            dst = args.out / f"{rel}_poses.npz"
            result = _load_any_gvhmr(src)
            clip = to_amass_clip(
                result,
                motion_fps=args.fps,
                body_model=args.body_model,
                pose_layout=args.layout,
            )
            save_amass_npz(clip, dst)
        print(f"Converted {len(sources)} GVHMR file(s) -> {args.out}")
        return 0

    result = _load_any_gvhmr(args.input)
    clip = to_amass_clip(
        result,
        motion_fps=args.fps,
        body_model=args.body_model,
        pose_layout=args.layout,
    )
    written = save_amass_npz(clip, args.out)
    print(f"Wrote {written} ({len(clip['poses'])} frames @ {args.fps:g} Hz)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
