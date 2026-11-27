# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Pure array geometry tests for myosuite.physics.spatial_enclosure."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import myosuite.physics.spatial_enclosure as se
from myosuite.physics.spatial_enclosure import (
    enclosure_margins,
    excess_contact_force,
    min_enclosure_clearance,
)

pytestmark = pytest.mark.tier1


def test_spatial_enclosure_module_has_no_accessor_or_terms_imports() -> None:
    tree = ast.parse(Path(se.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    for mod in imported:
        assert not mod.startswith(("myosuite.terms", "myosuite.core")), mod


def test_enclosure_margins_aabb_and_per_env_bounds() -> None:
    pos_single = np.array([[0.0, 0.0, 0.0], [0.6, -0.2, 0.1]], dtype=np.float64)
    m = enclosure_margins(np, pos_single, (-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))
    assert m.shape == (2, 6)
    np.testing.assert_allclose(m[0], np.full(6, 0.5))
    assert float(m[1, 3]) == pytest.approx(-0.1)

    pos_batch = np.zeros((2, 3, 3), dtype=np.float32)
    lo_batch = np.array([[-0.5, -0.5, -0.5], [-1.0, -1.0, -1.0]], dtype=np.float32)
    hi_batch = np.array([[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]], dtype=np.float32)
    min_c = min_enclosure_clearance(np, pos_batch, lo_batch, hi_batch)
    np.testing.assert_allclose(min_c, [0.5, 1.0], atol=1e-6)


def test_non_unit_plane_normal_scales_offset_in_numpy_and_torch() -> None:
    pos_np = np.array([[1.5, 0.0, 0.0]], dtype=np.float32)
    bounds_lo = (-10.0, -10.0, -10.0)
    bounds_hi = (10.0, 10.0, 10.0)

    margins_non_unit = enclosure_margins(
        np,
        pos_np,
        bounds_lo,
        bounds_hi,
        plane_normals=[[2.0, 0.0, 0.0]],
        plane_offsets=[2.0],
    )
    margins_unit = enclosure_margins(
        np,
        pos_np,
        bounds_lo,
        bounds_hi,
        plane_normals=[[1.0, 0.0, 0.0]],
        plane_offsets=[1.0],
    )
    assert float(margins_non_unit[0, 6]) == pytest.approx(0.5)
    np.testing.assert_allclose(margins_non_unit, margins_unit, atol=1e-6)

    with pytest.raises(ValueError, match="plane_normals and plane_offsets"):
        enclosure_margins(
            np, pos_np, bounds_lo, bounds_hi, plane_normals=[[1.0, 0.0, 0.0]]
        )

    torch = pytest.importorskip("torch")
    pos_t = torch.tensor(pos_np[None, ...])
    margins_t = enclosure_margins(
        torch,
        pos_t,
        bounds_lo,
        bounds_hi,
        plane_normals=[[2.0, 0.0, 0.0]],
        plane_offsets=[2.0],
    )
    assert float(margins_t[0, 0, 6].item()) == pytest.approx(0.5)


def test_excess_contact_force_scalar_and_vector() -> None:
    ref = np.zeros(3, dtype=np.float32)
    vec_f = np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [3.0, 4.0, 0.0]], dtype=np.float32
    )
    ex = excess_contact_force(np, vec_f, threshold=1.0, reference=ref, batched=True)
    np.testing.assert_allclose(ex, [0.0, 0.0, 4.0], atol=1e-6)
    ex_none = excess_contact_force(np, None, threshold=1.0, reference=ref, batched=True)
    np.testing.assert_allclose(ex_none, [0.0, 0.0, 0.0], atol=1e-6)
