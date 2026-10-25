# Copyright (c) MyoSuite Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Derived mesh descriptors preserve geometric units and reject open volume."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

pytestmark = pytest.mark.tier1


def _module() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts/validate_vsd_geometry.py"
    spec = importlib.util.spec_from_file_location("vsd_geometry", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_principal_spans_and_dimensionless_volume_are_scale_and_pose_consistent(
    tmp_path: Path,
) -> None:
    trimesh = pytest.importorskip("trimesh")
    module = _module()
    mesh = trimesh.creation.box(extents=[20, 100, 30])
    path = tmp_path / "synthetic.stl"
    mesh.export(path)
    result = module.mesh_descriptors(path, "mm")
    assert result["volume_m3"] == pytest.approx(0.02 * 0.1 * 0.03)
    np.testing.assert_allclose(result["principal_axis_spans_m"], [0.1, 0.03, 0.02])
    mesh.apply_scale(2)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(0.4, [1, 2, 3]))
    mesh.apply_translation([100, 200, -50])
    mesh.export(path)
    other = module.mesh_descriptors(path, "mm")
    np.testing.assert_allclose(
        other["principal_axis_spans_m"],
        np.array(result["principal_axis_spans_m"]) * 2,
        atol=1e-7,
    )
    assert other["volume_over_long_span_cubed"] == pytest.approx(
        result["volume_over_long_span_cubed"], rel=1e-5
    )


def test_open_mesh_is_excluded_and_missing_cache_reports_coverage(
    tmp_path: Path,
) -> None:
    trimesh = pytest.importorskip("trimesh")
    module = _module()
    mesh = trimesh.creation.box()
    mesh.update_faces(np.arange(len(mesh.faces)) != 0)
    path = tmp_path / "open.stl"
    mesh.export(path)
    assert not module.mesh_descriptors(path, "m")["volume_usable"]
    report = module.validate(tmp_path / "empty", units="mm", cases=("001",))
    assert report["coverage"] == {
        "requested": 1,
        "source_complete_and_volume_usable": 0,
        "excluded_counts": {"not_cached": 1},
    }
    assert report["volume_m3"] is None
    assert not report["sources"]
