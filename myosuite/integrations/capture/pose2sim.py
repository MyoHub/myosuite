# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Pose2Sim ``.toml`` camera calibration adapter without external runtime dependencies."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from myosuite.biomechanics.camera import PinholeCamera


def _strip_toml_comment(line: str) -> str:
    in_str: str | None = None
    escaped = False
    for idx, ch in enumerate(line):
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_str == '"':
            escaped = True
            continue
        if ch in ('"', "'"):
            if in_str is None:
                in_str = ch
            elif in_str == ch:
                in_str = None
            continue
        if ch == "#" and in_str is None:
            return line[:idx]
    return line


def _split_toml_header(header_inner: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    in_str: str | None = None
    escaped = False
    for ch in header_inner:
        if escaped:
            buf.append(ch)
            escaped = False
            continue
        if ch == "\\" and in_str == '"':
            buf.append(ch)
            escaped = True
            continue
        if ch in ('"', "'"):
            buf.append(ch)
            if in_str is None:
                in_str = ch
            elif in_str == ch:
                in_str = None
            continue
        if ch == "." and in_str is None:
            parts.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    out: list[str] = []
    for raw_part in parts:
        if not raw_part:
            continue
        if (raw_part.startswith('"') and raw_part.endswith('"')) or (
            raw_part.startswith("'") and raw_part.endswith("'")
        ):
            out.append(str(ast.literal_eval(raw_part)))
        else:
            out.append(raw_part)
    return out


def _parse_pose2sim_toml_fallback(text: str) -> dict[str, Any]:
    """Minimal stdlib parser for Pose2Sim ``Calib*.toml`` files when ``tomllib`` is absent."""
    root: dict[str, Any] = {}
    target: dict[str, Any] = root
    lines = text.splitlines()
    idx = 0
    bool_re = re.compile(r'(?<!["\'\w])(true|false)(?!["\'\w])')

    while idx < len(lines):
        raw = _strip_toml_comment(lines[idx]).strip()
        idx += 1
        if not raw:
            continue
        if raw.startswith("[") and raw.endswith("]") and "=" not in raw:
            sec_path = _split_toml_header(raw[1:-1])
            cur = root
            for part in sec_path:
                nxt = cur.setdefault(part, {})
                if not isinstance(nxt, dict):
                    raise ValueError(
                        f"Invalid TOML section header [{'.'.join(sec_path)}]"
                    )
                cur = nxt
            target = cur
            continue
        if "=" not in raw:
            raise ValueError(f"Expected key = value in Pose2Sim TOML, got: {raw!r}")
        key_part, val_part = raw.split("=", 1)
        key = key_part.strip().strip("\"'")
        val_buf = val_part.strip()
        while val_buf.count("[") > val_buf.count("]") and idx < len(lines):
            val_buf += " " + _strip_toml_comment(lines[idx]).strip()
            idx += 1
        py_expr = bool_re.sub(
            lambda m: "True" if m.group(1) == "true" else "False", val_buf
        )
        target[key] = ast.literal_eval(py_expr)
    return root


def _loads_toml(text: str) -> dict[str, Any]:
    try:
        import tomllib

        return dict(tomllib.loads(text))
    except ModuleNotFoundError:
        try:
            import tomli  # type: ignore[import-not-found]

            return dict(tomli.loads(text))
        except ModuleNotFoundError:
            return _parse_pose2sim_toml_fallback(text)


def load_pose2sim_cameras(
    path: str | Path, *, camera_names: Sequence[str] | None = None
) -> list[PinholeCamera]:
    """Load calibrated or intrinsics-only cameras from a Pose2Sim ``.toml`` file.

    Pose2Sim stores camera extrinsics in world-to-camera OpenCV convention
    (``X_cam = R @ X_world + t``, ``rotation`` as a 3-vector Rodrigues axis-angle in rad
    or a 3x3 matrix, ``translation`` in metres, ``matrix`` in pixels, ``size`` as
    ``[width, height]``). Returned :class:`PinholeCamera` instances preserve that exact
    metric convention.
    """
    data = _loads_toml(Path(path).read_text(encoding="utf-8"))
    keys = (
        list(camera_names)
        if camera_names is not None
        else [k for k, v in data.items() if isinstance(v, dict) and "matrix" in v]
    )
    if not keys:
        raise ValueError(f"No camera sections with 'matrix' found in {path}")
    cameras: list[PinholeCamera] = []
    for key in keys:
        sec = data[key]
        raw_rot = sec.get("rotation")
        arr = None if raw_rot is None else np.asarray(raw_rot, dtype=float)
        rot = (
            np.eye(3)
            if arr is None
            else (Rotation.from_rotvec(arr).as_matrix() if arr.shape == (3,) else arr)
        )
        sz = sec.get("size")
        cameras.append(
            PinholeCamera(
                intrinsic=np.asarray(sec["matrix"], dtype=float),
                rotation=rot,
                translation=np.asarray(
                    sec.get("translation", [0.0, 0.0, 0.0]), dtype=float
                ),
                calibration_id=str(sec.get("name") or key),
                distortion=tuple(float(c) for c in sec.get("distortions", ())),
                image_size=(int(round(sz[0])), int(round(sz[1])))
                if sz is not None
                else None,
                fisheye=bool(sec.get("fisheye", False)),
            )
        )
    return cameras


def save_pose2sim_cameras(path: str | Path, cameras: Sequence[PinholeCamera]) -> Path:
    """Write ``cameras`` in Pose2Sim ``Calib_scene.toml`` format."""
    if not cameras:
        raise ValueError("cameras must be non-empty")
    blocks: list[str] = []
    for cam in cameras:
        rvec = Rotation.from_matrix(cam.rotation).as_rotvec().tolist()
        dist = list(cam.distortion) if cam.distortion else [0.0, 0.0, 0.0, 0.0]
        sz = (
            list(cam.image_size)
            if cam.image_size is not None
            else [
                int(round(2 * cam.intrinsic[0, 2])),
                int(round(2 * cam.intrinsic[1, 2])),
            ]
        )
        rows = ", ".join(
            "[" + ", ".join(f"{v:.10g}" for v in row) + "]" for row in cam.intrinsic
        )
        sec_key = (
            cam.calibration_id
            if re.fullmatch(r"[A-Za-z0-9_-]+", cam.calibration_id)
            else json.dumps(cam.calibration_id)
        )
        blocks.append(
            f"[{sec_key}]\nname = {json.dumps(cam.calibration_id)}\n"
            f"size = [{float(sz[0]):.1f}, {float(sz[1]):.1f}]\nmatrix = [{rows}]\n"
            f"distortions = [{', '.join(f'{v:.10g}' for v in dist)}]\n"
            f"rotation = [{', '.join(f'{v:.10g}' for v in rvec)}]\n"
            f"translation = [{', '.join(f'{v:.10g}' for v in cam.translation)}]\n"
            f"fisheye = {'true' if cam.fisheye else 'false'}\n"
        )
    out = Path(path)
    out.write_text("\n".join(blocks), encoding="utf-8")
    return out


__all__ = [
    "load_pose2sim_cameras",
    "save_pose2sim_cameras",
]
