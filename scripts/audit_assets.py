#!/usr/bin/env python3
"""Audit 3D meshes and textures used by MyoSuite.

This script scans the `myosuite` tree for mesh and texture assets, reports
largest files, and estimates complexity (triangle count for meshes, dimensions
for textures). By default it excludes `myosuite/simhive`.
"""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable


MESH_EXTENSIONS = {
    ".obj",
    ".stl",
    ".ply",
    ".dae",
    ".fbx",
    ".gltf",
    ".glb",
    ".msh",
}
TEXTURE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tga", ".gif", ".webp"}
ASSET_EXTENSIONS = MESH_EXTENSIONS | TEXTURE_EXTENSIONS | {".mtl"}


@dataclass(frozen=True)
class AssetRecord:
    """Single asset file entry.

    Attributes:
        path: Absolute file path.
        size_bytes: File size in bytes.
    """

    path: Path
    size_bytes: int


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed CLI namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("myosuite"),
        help="Directory to scan (default: myosuite).",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=["myosuite/simhive"],
        help=(
            "Path prefix to exclude. Repeat flag for multiple values. "
            "Defaults to myosuite/simhive."
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top files to report per section.",
    )
    return parser.parse_args()


def should_exclude(path: Path, excluded_roots: Iterable[Path]) -> bool:
    """Check whether a path is under excluded roots.

    Args:
        path: File path to test.
        excluded_roots: Absolute excluded root paths.

    Returns:
        True if the path should be skipped.
    """

    for excluded in excluded_roots:
        try:
            path.relative_to(excluded)
            return True
        except ValueError:
            continue
    return False


def iter_assets(scan_root: Path, excluded_roots: list[Path]) -> list[AssetRecord]:
    """Collect assets under a root directory.

    Args:
        scan_root: Directory to scan recursively.
        excluded_roots: Absolute excluded root paths.

    Returns:
        Sorted list of asset records by descending size.
    """

    records: list[AssetRecord] = []
    for file_path in scan_root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in ASSET_EXTENSIONS:
            continue
        if should_exclude(file_path, excluded_roots):
            continue
        records.append(AssetRecord(path=file_path, size_bytes=file_path.stat().st_size))
    records.sort(key=lambda record: record.size_bytes, reverse=True)
    return records


def count_obj_faces(path: Path) -> int:
    """Count face lines in an OBJ file.

    Args:
        path: OBJ file.

    Returns:
        Number of `f` lines.
    """

    faces = 0
    with path.open("r", errors="ignore") as handle:
        for line in handle:
            if line.startswith("f "):
                faces += 1
    return faces


def count_stl_faces(path: Path) -> int | None:
    """Count triangles in STL file (binary preferred, ASCII fallback).

    Args:
        path: STL file.

    Returns:
        Triangle count, or None if parsing fails.
    """

    file_size = path.stat().st_size
    if file_size >= 84:
        with path.open("rb") as handle:
            handle.read(80)
            raw_count = handle.read(4)
        if len(raw_count) == 4:
            count = struct.unpack("<I", raw_count)[0]
            if 84 + count * 50 == file_size:
                return int(count)

    # ASCII fallback
    try:
        facets = 0
        with path.open("r", errors="ignore") as handle:
            for line in handle:
                if line.lstrip().startswith("facet normal"):
                    facets += 1
        return facets or None
    except UnicodeDecodeError:
        return None


def image_dimensions(path: Path) -> tuple[int, int] | None:
    """Read image dimensions for PNG/JPEG using headers only.

    Args:
        path: Image path.

    Returns:
        Width/height tuple if recognized, else None.
    """

    ext = path.suffix.lower()
    if ext == ".png":
        with path.open("rb") as handle:
            header = handle.read(24)
        if len(header) >= 24 and header[:8] == b"\x89PNG\r\n\x1a\n":
            width, height = struct.unpack(">II", header[16:24])
            return width, height
        return None

    if ext in {".jpg", ".jpeg"}:
        with path.open("rb") as handle:
            data = handle.read()
        offset = 0
        while offset + 9 < len(data):
            if data[offset] == 0xFF and data[offset + 1] in {
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC9,
                0xCA,
                0xCB,
            }:
                height = int.from_bytes(data[offset + 5 : offset + 7], "big")
                width = int.from_bytes(data[offset + 7 : offset + 9], "big")
                return width, height
            if data[offset] == 0xFF and offset + 4 < len(data):
                marker_length = int.from_bytes(
                    data[offset + 2 : offset + 4],
                    "big",
                )
                offset += max(marker_length + 2, 1)
            else:
                offset += 1
    return None


def fmt_bytes(value: int) -> str:
    """Format bytes in a compact human-readable string.

    Args:
        value: Byte count.

    Returns:
        Formatted size string.
    """

    units = ["B", "KB", "MB", "GB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f}{unit}"
        size /= 1024
    return f"{value}B"


def print_section(title: str) -> None:
    """Print section heading.

    Args:
        title: Section title.
    """

    print()
    print(f"== {title} ==")


def main() -> None:
    """Run the asset audit."""

    args = parse_args()
    repo_root = Path.cwd().resolve()
    scan_root = (repo_root / args.root).resolve()
    excluded_roots = [(repo_root / Path(prefix)).resolve() for prefix in args.exclude]

    if not scan_root.exists():
        raise SystemExit(f"Scan root does not exist: {scan_root}")

    records = iter_assets(scan_root=scan_root, excluded_roots=excluded_roots)
    if not records:
        print("No assets found.")
        return

    total_bytes = sum(record.size_bytes for record in records)
    mesh_records = [
        record for record in records if record.path.suffix.lower() in MESH_EXTENSIONS
    ]
    texture_records = [
        record for record in records if record.path.suffix.lower() in TEXTURE_EXTENSIONS
    ]

    print("Asset audit scope")
    print(f"- root: {scan_root.relative_to(repo_root)}")
    print(
        "- exclude: "
        + ", ".join(str(excluded.relative_to(repo_root)) for excluded in excluded_roots)
    )
    print(f"- files: {len(records)}")
    print(f"- total size: {fmt_bytes(total_bytes)}")
    print(f"- meshes: {len(mesh_records)} | textures: {len(texture_records)}")

    print_section(f"Top {args.top} Largest Assets")
    for record in records[: args.top]:
        print(
            f"{fmt_bytes(record.size_bytes):>10}  "
            f"{record.path.relative_to(repo_root)}"
        )

    print_section(f"Top {args.top} Meshes By Triangles")
    tri_rows: list[tuple[int, AssetRecord]] = []
    for record in mesh_records:
        suffix = record.path.suffix.lower()
        triangle_count = None
        if suffix == ".stl":
            triangle_count = count_stl_faces(record.path)
        elif suffix == ".obj":
            triangle_count = count_obj_faces(record.path)
        if triangle_count is not None:
            tri_rows.append((triangle_count, record))
    tri_rows.sort(key=lambda row: row[0], reverse=True)
    for triangles, record in tri_rows[: args.top]:
        print(
            f"{triangles:>10} tris  "
            f"{fmt_bytes(record.size_bytes):>10}  "
            f"{record.path.relative_to(repo_root)}"
        )

    print_section(f"Top {args.top} Textures By Dimensions")
    tex_rows: list[tuple[int, int, int, AssetRecord]] = []
    for record in texture_records:
        dims = image_dimensions(record.path)
        if dims is None:
            continue
        width, height = dims
        tex_rows.append((width * height, width, height, record))
    tex_rows.sort(key=lambda row: row[0], reverse=True)
    for pixel_count, width, height, record in tex_rows[: args.top]:
        est_rgba = pixel_count * 4
        print(
            f"{width}x{height: <5}  "
            f"rgba~{fmt_bytes(est_rgba):>8}  "
            f"{fmt_bytes(record.size_bytes):>10}  "
            f"{record.path.relative_to(repo_root)}"
        )


if __name__ == "__main__":
    main()
