# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Local Denver Visible Human tissue geometry and explicit volume constraints."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal
from zipfile import ZipFile

import numpy as np

from myosuite.core.subject_model import ScaleEstimate
from myosuite.integrations.capture.markers import MarkerScaleResult

_GEOMETRY = re.compile(
    r"(?P<subject>VH[FM])_(?P<side>Left|Right)_"
    r"(?P<tissue>Bone|Muscle|Cartilage|Ligament|Fat)_(?P<label>.+)_smooth\.stl"
)
_SOURCE = "https://digitalcommons.du.edu/visiblehuman/"
_CITATION = "Andreassen et al. (2023), https://doi.org/10.1038/s41597-022-01905-2"


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class DenverGeometry:
    """Identity of one original local STL archive member."""

    name: str
    member: str
    side: str
    tissue: str
    label: str


@dataclass(frozen=True)
class TissueSummary:
    """Derived geometry measures and an independent published-volume check.

    ``volume_m3`` is unavailable for open, inconsistently wound or inward-facing
    surfaces. Published scalar data are available at runtime for comparison;
    distribute aggregate checks rather than reproducing the metadata table.
    """

    geometry: DenverGeometry
    volume_m3: float | None
    signed_volume_m3: float
    surface_area_m2: float
    published_volume_m3: float
    relative_volume_error: float | None
    watertight: bool
    winding_consistent: bool
    _source_json: str = field(repr=False)

    def metadata(self) -> dict[str, Any]:
        """Return source hashes, attribution and geometry quality evidence."""
        return json.loads(self._source_json)


@dataclass(frozen=True)
class DenverReference:
    """Read a subject-matched local Denver final-STL and metadata archive pair.

    Args:
        mesh_archive: Original ``Final 3D STL Models-stl.zip`` download.
        metadata_archive: Original ``MetaData.zip`` containing final volumes.
        subject: ``VHF`` or ``VHM``, verified against internal member names.

    Original mesh coordinates and published volume tables use millimetres and
    mm³. Geometry is loaded only in memory; the reader extracts no source files.
    Optional ``myosuite[anatomy]`` provides trimesh and read-only XLSX support.
    """

    mesh_archive: Path
    metadata_archive: Path
    subject: Literal["VHF", "VHM"]
    _geometries: tuple[DenverGeometry, ...] = field(init=False, repr=False)
    _volumes: Mapping[str, float] = field(init=False, repr=False)
    _hashes: tuple[str, str] = field(init=False, repr=False)
    _signatures: tuple[tuple[int, int], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.subject not in ("VHF", "VHM"):
            raise ValueError("subject must be VHF or VHM.")
        paths = (
            Path(self.mesh_archive).resolve(),
            Path(self.metadata_archive).resolve(),
        )
        if any(not path.is_file() for path in paths):
            raise FileNotFoundError("Both local Denver source archives are required.")
        object.__setattr__(self, "mesh_archive", paths[0])
        object.__setattr__(self, "metadata_archive", paths[1])
        geometries = []
        with ZipFile(paths[0]) as archive:
            members = [
                name for name in archive.namelist() if name.lower().endswith(".stl")
            ]
            if not members or len(set(members)) != len(members):
                raise ValueError("Mesh archive needs unique Denver STL members.")
            for member in members:
                parts = member.split("/")
                match = _GEOMETRY.fullmatch(parts[-1])
                if (
                    len(parts) != 3
                    or parts[0] != "Final 3D STL Models-stl"
                    or match is None
                    or match["subject"] != self.subject
                    or parts[1] != match["side"]
                ):
                    raise ValueError(
                        "Mesh archive subject, side or member layout is inconsistent."
                    )
                geometries.append(
                    DenverGeometry(
                        parts[-1],
                        member,
                        match["side"],
                        match["tissue"],
                        match["label"],
                    )
                )
        volumes = self._read_volumes()
        if set(volumes) != {geometry.name for geometry in geometries}:
            raise ValueError(
                "Final volume table and mesh archive geometries do not match."
            )
        object.__setattr__(
            self, "_geometries", tuple(sorted(geometries, key=lambda g: g.name))
        )
        object.__setattr__(self, "_volumes", MappingProxyType(volumes))
        object.__setattr__(self, "_hashes", tuple(_file_hash(path) for path in paths))
        object.__setattr__(
            self,
            "_signatures",
            tuple((path.stat().st_size, path.stat().st_mtime_ns) for path in paths),
        )

    def _read_volumes(self) -> dict[str, float]:
        try:
            import openpyxl
        except ImportError as exc:
            raise ImportError(
                "Denver metadata requires `pip install 'myosuite[anatomy]'`."
            ) from exc
        expected = {
            f"MetaData/Volume Data/{self.subject}{side}_Final_volumes.xlsx": side
            for side in ("L", "R")
        }
        volumes = {}
        with ZipFile(self.metadata_archive) as archive:
            members = [
                name
                for name in archive.namelist()
                if name.endswith("_Final_volumes.xlsx")
            ]
            if set(members) != set(expected) or len(members) != len(expected):
                raise ValueError(
                    "Metadata archive does not match the selected Denver subject."
                )
            for member in members:
                workbook = openpyxl.load_workbook(
                    BytesIO(archive.read(member)), read_only=True, data_only=True
                )
                try:
                    rows = workbook.active.iter_rows(values_only=True)
                    if tuple(next(rows, ())) != ("Geometry", "Volume (mm^3)"):
                        raise ValueError("Unsupported Denver volume-table columns.")
                    side = "Left" if expected[member] == "L" else "Right"
                    for row in rows:
                        if all(value is None for value in row):
                            continue
                        name, volume = row
                        match = (
                            _GEOMETRY.fullmatch(name) if isinstance(name, str) else None
                        )
                        if (
                            match is None
                            or match["subject"] != self.subject
                            or match["side"] != side
                        ):
                            raise ValueError(
                                "Volume-table geometry subject/side is inconsistent."
                            )
                        if name in volumes:
                            raise ValueError("Duplicate volume-table geometry.")
                        try:
                            value = float(volume)
                        except (TypeError, ValueError) as exc:
                            raise ValueError(
                                "Published volume must be numeric."
                            ) from exc
                        if not np.isfinite(value) or value <= 0:
                            raise ValueError(
                                "Published volume must be positive and finite."
                            )
                        volumes[name] = value * 1e-9
                finally:
                    workbook.close()
        return volumes

    def _check_unchanged(self) -> None:
        paths = (self.mesh_archive, self.metadata_archive)
        current = tuple(
            (path.stat().st_size, path.stat().st_mtime_ns) for path in paths
        )
        if current != self._signatures:
            raise ValueError(
                "Denver source archive changed after reference initialization."
            )

    def metadata(self) -> dict[str, Any]:
        """Return source identity and hashes without exposing source surfaces/tables."""
        self._check_unchanged()
        return {
            "dataset": "University of Denver Visible Human final tissue geometry",
            "subject": self.subject,
            "url": _SOURCE,
            "attribution": _CITATION,
            "geometry_license": "CC-BY-4.0",
            "mesh_archive_sha256": self._hashes[0],
            "metadata_archive_sha256": self._hashes[1],
            "mesh_units": "mm",
            "published_volume_units": "mm^3",
            "limitations": [
                "Same two Visible Human donors as BoneHub; not an independent cohort.",
                "Tissue surfaces do not establish muscle fibre, tendon or force parameters.",
                "Volume and one length identify transverse area only, not separate widths.",
            ],
        }

    def geometries(self) -> tuple[DenverGeometry, ...]:
        """List verified local bone, muscle, cartilage, ligament or fat members."""
        self._check_unchanged()
        return self._geometries

    def summarize_geometry(
        self, name: str, *, max_relative_volume_error: float = 0.001
    ) -> TissueSummary:
        """Measure a local mesh, checking volume against its independent table.

        The default 0.1% discrepancy limit detects unit/source mismatches;
        it is not an anatomical reconstruction accuracy threshold.
        """
        try:
            import trimesh
        except ImportError as exc:
            raise ImportError(
                "Denver geometry requires `pip install 'myosuite[anatomy]'`."
            ) from exc
        self._check_unchanged()
        if not np.isfinite(max_relative_volume_error) or max_relative_volume_error <= 0:
            raise ValueError("max_relative_volume_error must be finite and positive.")
        matches = [geometry for geometry in self._geometries if geometry.name == name]
        if not matches:
            raise KeyError(f"Unknown Denver geometry: {name!r}.")
        geometry = matches[0]
        with ZipFile(self.mesh_archive) as archive:
            raw = archive.read(geometry.member)
        mesh = trimesh.load(BytesIO(raw), file_type="stl", force="mesh", process=True)
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            raise ValueError("Denver member contains no triangle mesh.")
        signed_volume = float(mesh.volume) * 1e-9
        area = float(mesh.area) * 1e-6
        if not np.all(np.isfinite([signed_volume, area])) or area <= 0:
            raise ValueError("Denver mesh geometry measures are invalid.")
        watertight = bool(mesh.is_watertight)
        winding = bool(mesh.is_winding_consistent)
        valid_volume = watertight and winding and signed_volume > 0
        published = self._volumes[name]
        discrepancy = (
            abs(signed_volume - published) / published if valid_volume else None
        )
        if discrepancy is not None and discrepancy > max_relative_volume_error:
            raise ValueError(
                "Mesh and published volume disagree; check source pairing and millimetre units."
            )
        evidence = {
            **self.metadata(),
            "geometry": geometry.name,
            "member_sha256": hashlib.sha256(raw).hexdigest(),
            "watertight": watertight,
            "winding_consistent": winding,
            "volume_usable": valid_volume,
        }
        return TissueSummary(
            geometry,
            signed_volume if valid_volume else None,
            signed_volume,
            area,
            published,
            discrepancy,
            watertight,
            winding,
            json.dumps(evidence, sort_keys=True, allow_nan=False),
        )


@dataclass(frozen=True)
class VolumeScaleConstraint:
    """A measured volume ratio and independently supplied longitudinal scale."""

    subject_id: str
    segment: str
    reference_volume_m3: float
    observed_volume_m3: float
    longitudinal_factor: float
    axis: Literal["x", "y", "z"]
    reference_source: str
    observed_source: str

    def __post_init__(self) -> None:
        for name in ("subject_id", "segment", "reference_source", "observed_source"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string.")
        for name in (
            "reference_volume_m3",
            "observed_volume_m3",
            "longitudinal_factor",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        if self.axis not in ("x", "y", "z"):
            raise ValueError("axis must be a target-local x, y or z axis.")


def volume_constrained_scale_estimate(
    constraint: VolumeScaleConstraint,
    *,
    assume_equal_transverse: bool,
    scale_bounds: tuple[float, float] = (0.5, 2.0),
) -> MarkerScaleResult:
    """Constrain transverse area from volume, with an explicit shape assumption.

    Args:
        constraint: Subject volume, reference volume and an independent length
            factor in a known target-local axis. Volumes alone supply no length.
        assume_equal_transverse: Must explicitly be True. Width and depth are not
            separately identifiable from a volume and one length measurement.
        scale_bounds: Positive factor bounds detecting obvious unit mistakes.

    Returns:
        Existing subject-builder input with two equal transverse factors
        ``sqrt((observed_volume / reference_volume) / longitudinal_factor)``.
        This consistency constraint does not establish anatomical accuracy.
    """
    if assume_equal_transverse is not True:
        raise ValueError("Explicit assume_equal_transverse=True is required.")
    bounds = np.asarray(scale_bounds, dtype=float)
    if (
        bounds.shape != (2,)
        or not np.all(np.isfinite(bounds))
        or not 0 < bounds[0] < bounds[1]
    ):
        raise ValueError("scale_bounds must be positive, finite and increasing.")
    ratio = float(constraint.observed_volume_m3 / constraint.reference_volume_m3)
    transverse = float(np.sqrt(ratio / constraint.longitudinal_factor))
    factors = np.full(3, transverse)
    factors["xyz".index(constraint.axis)] = constraint.longitudinal_factor
    if not np.all(np.isfinite(factors)) or np.any(
        (factors < bounds[0]) | (factors > bounds[1])
    ):
        raise ValueError(
            "Volume-constrained factors exceed bounds; check units and anatomy."
        )
    evidence = {
        "schema": 1,
        "adapter": "volume-and-longitudinal-constraint",
        "subject_id": constraint.subject_id,
        "segment": constraint.segment,
        "reference_source": constraint.reference_source,
        "observed_source": constraint.observed_source,
        "volume_ratio": ratio,
        "longitudinal_factor": float(constraint.longitudinal_factor),
        "local_axis": constraint.axis,
        "equal_transverse_assumption": True,
        "factors": factors.tolist(),
        "implied_volume_ratio": float(np.prod(factors)),
        "isotropic_length_only_volume_ratio": float(constraint.longitudinal_factor**3),
    }
    identity = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
    estimate = ScaleEstimate(
        {constraint.segment: factors},
        source=f"volume-constraint:{constraint.subject_id}@{identity}",
        warnings=(
            "Equal transverse factors are assumed; width/depth were not independently measured.",
        ),
    )
    return MarkerScaleResult(estimate, evidence)
