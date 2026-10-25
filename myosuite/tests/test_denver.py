# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Synthetic local archive and volume-constraint checks without source assets."""

from __future__ import annotations

import json
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

from myosuite.integrations.denver import (
    DenverReference,
    VolumeScaleConstraint,
    volume_constrained_scale_estimate,
)

pytestmark = pytest.mark.tier1


def _archives(
    directory: Path,
    subject: str = "VHF",
    *,
    open_surface: bool = False,
    volume_mm3: float = 48000.0,
    scale: float = 1.0,
) -> tuple[Path, Path]:
    trimesh = pytest.importorskip("trimesh")
    openpyxl = pytest.importorskip("openpyxl")
    mesh = trimesh.creation.box(extents=np.array([20, 40, 60]) * scale)
    if open_surface:
        mesh.update_faces(np.arange(len(mesh.faces) - 1))
    name = f"{subject}_Left_Bone_Tibia_smooth.stl"
    mesh_path, metadata_path = (
        directory / f"{subject}_meshes.zip",
        directory / f"{subject}_metadata.zip",
    )
    with ZipFile(mesh_path, "w", ZIP_DEFLATED) as archive:
        archive.writestr(
            f"Final 3D STL Models-stl/Left/{name}", mesh.export(file_type="stl")
        )
    with ZipFile(metadata_path, "w", ZIP_DEFLATED) as archive:
        for side in ("L", "R"):
            workbook = openpyxl.Workbook()
            sheet = workbook.active
            sheet.append(["Geometry", "Volume (mm^3)"])
            if side == "L":
                sheet.append([name, volume_mm3 * scale**3])
            data = BytesIO()
            workbook.save(data)
            workbook.close()
            archive.writestr(
                f"MetaData/Volume Data/{subject}{side}_Final_volumes.xlsx",
                data.getvalue(),
            )
    return mesh_path, metadata_path


def _constraint() -> VolumeScaleConstraint:
    return VolumeScaleConstraint(
        "synthetic-person",
        "tibia_r",
        0.0001,
        0.00011,
        1.2,
        "y",
        "independently authored reference volume",
        "independently measured subject volume",
    )


def test_local_geometry_is_measured_in_si_against_independent_volume_metadata(
    tmp_path: Path,
) -> None:
    meshes, metadata = _archives(tmp_path)
    reference = DenverReference(meshes, metadata, "VHF")
    assert len(reference.geometries()) == 1
    geometry = reference.geometries()[0]
    summary = reference.summarize_geometry(geometry.name)
    assert geometry.tissue == "Bone" and geometry.side == "Left"
    assert summary.watertight and summary.winding_consistent
    assert summary.volume_m3 == pytest.approx(4.8e-5)
    assert summary.surface_area_m2 == pytest.approx(0.0088)
    assert summary.published_volume_m3 == pytest.approx(4.8e-5)
    assert summary.relative_volume_error < 1e-12
    source = summary.metadata()
    assert source["subject"] == "VHF"
    assert source["geometry_license"] == "CC-BY-4.0"
    assert (
        len(source["mesh_archive_sha256"])
        == len(source["metadata_archive_sha256"])
        == 64
    )
    assert len(source["member_sha256"]) == 64
    assert not list(tmp_path.rglob("*.stl"))
    assert not list(tmp_path.rglob("*.xlsx"))
    json.dumps(source, allow_nan=False)


def test_open_mesh_has_no_usable_volume_constraint(tmp_path: Path) -> None:
    reference = DenverReference(*_archives(tmp_path, open_surface=True), "VHF")
    summary = reference.summarize_geometry(reference.geometries()[0].name)
    assert not summary.watertight
    assert summary.volume_m3 is None
    assert summary.relative_volume_error is None
    assert not summary.metadata()["volume_usable"]


def test_mismatched_subject_archive_pair_and_unknown_geometry_are_rejected(
    tmp_path: Path,
) -> None:
    female_mesh, female_metadata = _archives(tmp_path, "VHF")
    _, male_metadata = _archives(tmp_path, "VHM")
    with pytest.raises(ValueError, match="selected Denver subject"):
        DenverReference(female_mesh, male_metadata, "VHF")
    with pytest.raises(ValueError, match="subject, side"):
        DenverReference(female_mesh, female_metadata, "VHM")
    with pytest.raises(KeyError, match="Unknown Denver geometry"):
        DenverReference(female_mesh, female_metadata, "VHF").summarize_geometry(
            "invented"
        )


def test_nonmatching_volume_table_and_unsafe_members_are_rejected(
    tmp_path: Path,
) -> None:
    mesh_path, metadata = _archives(tmp_path)
    with ZipFile(mesh_path, "a") as archive:
        archive.writestr("../injected.stl", b"not a source mesh")
    with pytest.raises(ValueError, match="member layout"):
        DenverReference(mesh_path, metadata, "VHF")
    mesh_path, metadata = _archives(tmp_path, volume_mm3=0)
    with pytest.raises(ValueError, match="Published volume"):
        DenverReference(mesh_path, metadata, "VHF")


def test_reference_hashes_cannot_silently_survive_archive_edits(tmp_path: Path) -> None:
    mesh, metadata = _archives(tmp_path)
    reference = DenverReference(mesh, metadata, "VHF")
    with mesh.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed after"):
        reference.metadata()


def test_metadata_volume_check_rejects_units_or_source_mismatch(tmp_path: Path) -> None:
    reference = DenverReference(*_archives(tmp_path, volume_mm3=48000 * 1000), "VHF")
    with pytest.raises(ValueError, match="millimetre units"):
        reference.summarize_geometry(reference.geometries()[0].name)


def test_volume_ratio_and_length_constrain_transverse_area_only_with_explicit_assumption() -> (
    None
):
    constraint = _constraint()
    with pytest.raises(TypeError, match="assume_equal_transverse"):
        volume_constrained_scale_estimate(constraint)
    with pytest.raises(ValueError, match="Explicit assume_equal_transverse"):
        volume_constrained_scale_estimate(constraint, assume_equal_transverse=False)
    result = volume_constrained_scale_estimate(constraint, assume_equal_transverse=True)
    factors = np.asarray(result.estimate.factors["tibia_r"])
    assert factors[1] == 1.2
    assert factors[0] == factors[2] == pytest.approx(np.sqrt(1.1 / 1.2))
    assert np.prod(factors) == pytest.approx(1.1)
    assert result.evidence["isotropic_length_only_volume_ratio"] == pytest.approx(1.728)
    assert result.evidence["equal_transverse_assumption"]
    assert "not independently measured" in result.estimate.warnings[0]
    assert result.estimate.confidence is None
    assert result.estimate.mass_kg is None
    json.dumps(result.evidence, allow_nan=False)


@pytest.mark.parametrize(
    "change",
    [
        {"reference_volume_m3": 0},
        {"observed_volume_m3": float("nan")},
        {"longitudinal_factor": -1},
        {"axis": "world-y"},
        {"reference_source": ""},
        {"subject_id": ""},
    ],
)
def test_invalid_volume_constraints_fail(change: dict) -> None:
    with pytest.raises(ValueError):
        replace(_constraint(), **change)


def test_unreasonable_volume_factors_and_invalid_bounds_fail() -> None:
    with pytest.raises(ValueError, match="exceed bounds"):
        volume_constrained_scale_estimate(
            replace(_constraint(), observed_volume_m3=100),
            assume_equal_transverse=True,
        )
    with pytest.raises(ValueError, match="scale_bounds"):
        volume_constrained_scale_estimate(
            _constraint(), assume_equal_transverse=True, scale_bounds=(2, 1)
        )
