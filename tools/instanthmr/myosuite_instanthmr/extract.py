# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Extract single-subject MHR70 ``.npz`` clips from monocular video via InstantHMR."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MHR70_NUM_JOINTS = 70
_LEFT_HIP_IDX = 9
_RIGHT_HIP_IDX = 10


def box_iou(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> float:
    """Intersection-over-union of two ``(x1, y1, x2, y2)`` bounding boxes."""
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    denom = area_a + area_b - inter
    return inter / denom if denom > 0.0 else 0.0


@dataclass(frozen=True)
class TrackerConfig:
    """Identity-continuity gates for single-subject video extraction.

    Attributes:
        min_iou: Minimum 2D bounding-box IoU against the last seen or velocity-predicted
            subject box required to accept a detection.
        max_gap_s: Maximum occlusion duration (s) across which the tracker will re-acquire
            the same subject; longer gaps terminate the track and mark subsequent frames
            invalid rather than switching to another person.
        max_speed_mps: Maximum plausible 3D camera-space pelvis speed (m/s).
        pos_margin_m: Base 3D camera-space pelvis displacement tolerance (m).
        min_confidence: Optional minimum detector confidence when exposed on detections.
        init_bbox: Optional initial ``(x1, y1, x2, y2)`` box selecting the target person.
    """

    min_iou: float = 0.2
    max_gap_s: float = 0.25
    max_speed_mps: float = 8.0
    pos_margin_m: float = 0.6
    min_confidence: float = 0.0
    init_bbox: tuple[float, float, float, float] | None = None


def _hip_root(joints_3d_cam: np.ndarray) -> np.ndarray:
    return 0.5 * (joints_3d_cam[_LEFT_HIP_IDX] + joints_3d_cam[_RIGHT_HIP_IDX])


class SubjectTracker:
    """Stateful single-subject association preserving identity continuity across frames."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self.locked: bool = False
        self.lost: bool = False
        self.last_box: np.ndarray | None = None
        self.box_vel: np.ndarray = np.zeros(4, dtype=np.float64)
        self.last_root: np.ndarray | None = None
        self.missed_frames: int = 0

    def step(self, persons: Sequence[Any], fps: float) -> Any | None:
        """Select the tracked person for the current frame, or ``None`` if missing."""
        cfg = self.config
        cands: list[tuple[Any, np.ndarray, np.ndarray]] = []
        for p in persons:
            score = float(getattr(p, "score", getattr(p, "confidence", 1.0)))
            if score < cfg.min_confidence:
                continue
            box = np.asarray(p.bbox, dtype=np.float64)
            if box.shape != (4,) or not np.isfinite(box).all():
                continue
            if (box[2] - box[0]) <= 0.0 or (box[3] - box[1]) <= 0.0:
                continue
            joints = np.asarray(p.joints_3d_cam, dtype=np.float64)
            if joints.shape != (MHR70_NUM_JOINTS, 3) or not np.isfinite(joints).all():
                continue
            cands.append((p, box, _hip_root(joints)))

        if not self.locked:
            if not cands:
                return None
            if cfg.init_bbox is not None:
                init_box = np.asarray(cfg.init_bbox, dtype=np.float64)
                best_p, best_box, best_root = max(
                    cands, key=lambda item: box_iou(init_box, item[1])
                )
                if box_iou(init_box, best_box) < cfg.min_iou:
                    return None
            else:
                best_p, best_box, best_root = max(
                    cands,
                    key=lambda item: float(
                        (item[1][2] - item[1][0]) * (item[1][3] - item[1][1])
                    ),
                )
            self.locked = True
            self.last_box = best_box
            self.last_root = best_root
            self.missed_frames = 0
            return best_p

        if self.lost:
            return None

        assert self.last_box is not None and self.last_root is not None
        max_gap_frames = max(0, int(np.floor(cfg.max_gap_s * fps + 1e-9)))
        step_frames = self.missed_frames + 1
        pred_box = self.last_box + self.box_vel * step_frames
        max_disp = cfg.max_speed_mps * (step_frames / fps) + cfg.pos_margin_m

        admissible: list[tuple[float, Any, np.ndarray, np.ndarray]] = []
        for p, box, root in cands:
            iou = max(box_iou(self.last_box, box), box_iou(pred_box, box))
            disp = float(np.linalg.norm(root - self.last_root))
            if iou >= cfg.min_iou and disp <= max_disp:
                admissible.append((iou, p, box, root))

        if admissible:
            _, chosen_p, chosen_box, chosen_root = max(
                admissible, key=lambda item: item[0]
            )
            inst_vel = (chosen_box - self.last_box) / step_frames
            self.box_vel = 0.5 * self.box_vel + 0.5 * inst_vel
            self.last_box = chosen_box
            self.last_root = chosen_root
            self.missed_frames = 0
            return chosen_p

        self.missed_frames += 1
        if self.missed_frames > max_gap_frames:
            self.lost = True
        return None


def extract_predictions(
    frame_results: Iterable[Any],
    *,
    fps: float,
    config: TrackerConfig | None = None,
) -> dict[str, Any]:
    """Convert per-frame ``PosePipeline`` outputs into an MHR70 clip dictionary."""
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be finite and positive, got {fps}")
    tracker = SubjectTracker(config)
    joints_list: list[np.ndarray] = []
    bboxes_list: list[np.ndarray] = []
    shapes_list: list[np.ndarray | None] = []
    valid_list: list[bool] = []

    for res in frame_results:
        persons = getattr(res, "persons", res)
        chosen = tracker.step(persons, fps)
        if chosen is not None:
            joints_list.append(np.asarray(chosen.joints_3d_cam, dtype=np.float32))
            bboxes_list.append(np.asarray(chosen.bbox, dtype=np.float32))
            sp = getattr(chosen, "shape_params", None)
            shapes_list.append(
                np.asarray(sp, dtype=np.float32) if sp is not None else None
            )
            valid_list.append(True)
        else:
            joints_list.append(np.full((MHR70_NUM_JOINTS, 3), np.nan, dtype=np.float32))
            bboxes_list.append(np.full(4, np.nan, dtype=np.float32))
            shapes_list.append(None)
            valid_list.append(False)

    if sum(valid_list) < 2:
        raise ValueError(
            f"fewer than 2 frames with a tracked subject ({sum(valid_list)}/{len(valid_list)})"
        )

    flags: list[str] = []
    if tracker.lost or tracker.missed_frames > 0:
        flags.append("track_lost")

    out: dict[str, Any] = {
        "joints_3d_cam": np.stack(joints_list, axis=0),
        "bboxes": np.stack(bboxes_list, axis=0),
        "fps": np.float64(fps),
        "valid": np.asarray(valid_list, dtype=bool),
        "quality_flags": np.asarray(flags, dtype=str),
    }
    first_shape = next((s for s in shapes_list if s is not None), None)
    if first_shape is not None:
        nan_shape = np.full_like(first_shape, np.nan, dtype=np.float32)
        out["shape_params"] = np.stack(
            [s if s is not None else nan_shape for s in shapes_list], axis=0
        )
    return out


def extract_video_clip(
    video_path: str | Path,
    out_npz: str | Path,
    *,
    hmr_model: str = "medium-commercial",
    fit: int = 2,
    device: str = "cuda",
    focal: float | None = None,
    pose_filter: str | None = "one-euro",
    tracker_config: TrackerConfig | None = None,
    pipeline: Any | None = None,
) -> Path:
    """Run ``instanthmr.PosePipeline`` on *video_path* and write an MHR70 ``.npz`` clip."""
    import cv2  # noqa: PLC0415

    if pipeline is None:
        from instanthmr import PosePipeline  # noqa: PLC0415

        pipeline = PosePipeline(
            hmr_model, fit=fit, device=device, focal=focal, pose_filter=pose_filter
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0

        def _iter_results() -> Iterable[Any]:
            frame_idx = 0
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                yield pipeline(rgb, timestamp=frame_idx / fps)
                frame_idx += 1

        clip_data = extract_predictions(_iter_results(), fps=fps, config=tracker_config)
    finally:
        cap.release()

    out = Path(out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **clip_data)
    tmp.replace(out)
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``myosuite-instanthmr-extract``."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video", type=Path, help="Input monocular video file")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .npz clip path (default: <video_stem>_mhr70.npz)",
    )
    p.add_argument("--hmr-model", default="medium-commercial")
    p.add_argument("--hmr-fit", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--focal", type=float, default=None)
    p.add_argument("--pose-filter", default="one-euro")
    p.add_argument("--min-iou", type=float, default=0.2)
    p.add_argument("--max-gap-s", type=float, default=0.25)
    p.add_argument("--max-speed-mps", type=float, default=8.0)
    p.add_argument(
        "--init-bbox",
        type=float,
        nargs=4,
        default=None,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Optional initial bounding box selecting the target person",
    )
    args = p.parse_args(argv)

    out_npz = args.output or args.video.with_name(f"{args.video.stem}_mhr70.npz")
    pose_filter = None if args.pose_filter == "none" else args.pose_filter
    cfg = TrackerConfig(
        min_iou=args.min_iou,
        max_gap_s=args.max_gap_s,
        max_speed_mps=args.max_speed_mps,
        init_bbox=tuple(args.init_bbox) if args.init_bbox is not None else None,
    )
    written = extract_video_clip(
        args.video,
        out_npz,
        hmr_model=args.hmr_model,
        fit=args.hmr_fit,
        device=args.device,
        focal=args.focal,
        pose_filter=pose_filter,
        tracker_config=cfg,
    )
    print(f"Wrote {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
