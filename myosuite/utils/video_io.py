# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Shared helpers for writing and showing tutorial/demo videos."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

if TYPE_CHECKING:
    from IPython.display import HTML

logger = logging.getLogger(__name__)


def _resolve_fps(
    fps: int | float,
    inputdict: dict[str, str] | None,
    outputdict: dict[str, str] | None,
) -> int | float:
    """Resolve output FPS from explicit arg and skvideo-style dicts."""
    for cfg in (inputdict, outputdict):
        if not cfg:
            continue
        if "-r" in cfg:
            try:
                return float(cfg["-r"])
            except (TypeError, ValueError):
                logger.warning(
                    "Ignoring invalid -r value in video options: %r", cfg["-r"]
                )
    return fps


def write_video(
    video_path: str | Path,
    video_frames: list[np.ndarray] | np.ndarray,
    fps: int | float = 25,
    *,
    inputdict: dict[str, str] | None = None,
    outputdict: dict[str, str] | None = None,
) -> None:
    """Write MP4 frames to disk without scikit-video.

    Args:
        video_path: Destination file path.
        video_frames: Frame sequence with shape ``(T, H, W, 3)`` or list of frames.
        fps: Output frame rate unless overridden by ``inputdict``/``outputdict``.
            Defaults to 25 to match historical ``skvideo.io.vwrite`` behavior.
        inputdict: Backward-compatible skvideo-style ffmpeg options.
        outputdict: Backward-compatible skvideo-style ffmpeg options.
    """
    import imageio

    path = Path(video_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_fps = _resolve_fps(fps=fps, inputdict=inputdict, outputdict=outputdict)
    codec = (outputdict or {}).get("-vcodec", "libx264")
    pix_fmt = (outputdict or {}).get("-pix_fmt", "yuv420p")
    ffmpeg_params = ["-pix_fmt", pix_fmt] if pix_fmt else None

    frames = np.asarray(video_frames, dtype=np.uint8)
    with imageio.get_writer(
        str(path),
        fps=resolved_fps,
        codec=codec,
        format="FFMPEG",
        ffmpeg_params=ffmpeg_params,
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


def add_text_to_frame(frame, text, pos=(20, 20), color=(255, 0, 0), fontsize=12):
    if isinstance(frame, np.ndarray):
        frame = PIL.Image.fromarray(frame)

    draw = PIL.ImageDraw.Draw(frame)
    font = PIL.ImageFont.truetype("arial.ttf", fontsize)
    draw.text(pos, text, fill=color, font=font)
    return frame


def show_video(video_path: str | Path, video_width: int = 400) -> HTML:
    """Embed a local MP4 in Jupyter output.

    Args:
        video_path: Path to an MP4 file on disk.
        video_width: HTML video element width in pixels.

    Returns:
        An ``IPython.display.HTML`` object with a base64 data-URL, or empty
        HTML if the file does not exist.
    """
    from base64 import b64encode

    from IPython.display import HTML

    path = Path(video_path)
    if not path.is_file():
        logger.warning("Video file not found: %s", path)
        return HTML("")

    data = path.read_bytes()
    video_url = f"data:video/mp4;base64,{b64encode(data).decode()}"
    return HTML(
        f'<video autoplay width={video_width} controls><source src="{video_url}"></video>'
    )


__all__ = ["show_video", "write_video", "add_text_to_frame"]
