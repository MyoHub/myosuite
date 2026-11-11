# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Selected source-model edits preserve anatomy and fail transactionally."""

import copy
import sys

import mujoco
import numpy as np
import pytest

from myosuite.integrations.muskemo import _inertia, model_fingerprint
from myosuite.integrations.muskemo_edits import apply_model_patch, mesh_inertia
from myosuite.tests.support.muskemo import (
    _assert_source_unchanged,
    _edit,
    _fusion_spec,
    _patch,
    _source_state,
    _elbow_model,
)

pytestmark = pytest.mark.tier1


@pytest.fixture(scope="module")
def elbow():
    return _elbow_model()


def test_elbow_site_and_inertia_preserve_other_anatomy(elbow):
    spec, source = elbow
    patch = _edit(source)
    before = _source_state(spec)
    edited = apply_model_patch(spec, patch)
    model = edited.compile()
    assert edited is not spec
    _assert_source_unchanged(spec, before)
    assert model_fingerprint(spec.compile()) == patch["fingerprint"]
    np.testing.assert_array_equal(model.site("BRA-P2").pos, patch["sites"][0]["pos"])
    body = model.body("r_ulna_radius_hand").id
    np.testing.assert_allclose(model.body_mass[body], 1.7)
    np.testing.assert_allclose(model.body_ipos[body], patch["inertias"][0]["com"])
    np.testing.assert_allclose(
        _inertia(model.body_inertia[body], model.body_iquat[body]),
        patch["inertias"][0]["inertia"],
        atol=1e-12,
    )
    untouched_sites = np.arange(source.nsite) != source.site("BRA-P2").id
    np.testing.assert_array_equal(
        model.site_pos[untouched_sites], source.site_pos[untouched_sites]
    )


@pytest.mark.parametrize(
    "edit",
    [
        {"schema": True},
        {"schema": 2},
        {"fingerprint": "wrong"},
        {"mujoco_version": "different"},
        {"unknown": 1},
        {"sites": [{"name": "unknown", "pos": [0, 0, 0]}]},
        {"sites": [{"name": "BRA-P2", "pos": [0, 0, 0], "extra": 1}]},
        {"sites": [{"name": "BRA-P2", "pos": [0, 0, np.inf]}]},
        {"sites": [{"name": "BRA-P2", "pos": [0, 0]}]},
        {"sites": [{"name": "BRA-P2", "pos": ["0", "0", "0"]}]},
        {"sites": [{"name": "BRA-P2", "pos": [0, 0, 0]}] * 2},
        {
            "inertias": [
                {
                    "name": "world",
                    "mass": 1,
                    "com": [0, 0, 0],
                    "inertia": np.eye(3).tolist(),
                }
            ]
        },
    ],
)
def test_invalid_patch_does_not_mutate_source(elbow, edit):
    spec, model = elbow
    before = _source_state(spec)
    patch = _patch(model)
    patch.update(copy.deepcopy(edit))
    with pytest.raises(ValueError):
        apply_model_patch(spec, patch)
    _assert_source_unchanged(spec, before)


@pytest.mark.parametrize(
    "edit",
    [
        {"mass": 0},
        {"mass": np.nan},
        {"com": [0, 1, np.nan]},
        {"inertia": np.diag([1, 1, -1]).tolist()},
        {"inertia": np.diag([1, 1, 3]).tolist()},
        {"inertia": [[1, 0.1, 0], [0.2, 1, 0], [0, 0, 1]]},
        {"inertia": [[True, False, False], [False, True, False], [False, False, True]]},
    ],
)
def test_invalid_inertia_is_rejected(elbow, edit):
    spec, model = elbow
    patch = _edit(model)
    patch["inertias"][0].update(edit)
    with pytest.raises(ValueError):
        apply_model_patch(spec, patch)


def test_duplicate_inertias_and_changed_source_are_rejected(elbow):
    spec, model = elbow
    patch = _edit(model)
    patch["inertias"] *= 2
    with pytest.raises(ValueError, match="Duplicate"):
        apply_model_patch(spec, patch)
    changed = spec.copy()
    changed.site("BRA-P2").pos[0] += 0.001
    with pytest.raises(ValueError, match="fingerprint"):
        apply_model_patch(changed, _edit(model))


def test_compile_failure_keeps_original_spec(elbow, monkeypatch):
    spec, model = elbow
    patch = _edit(model)
    before = _source_state(spec)
    compile = mujoco.MjSpec.compile
    calls = []

    def fail_edited(candidate):
        calls.append(candidate)
        if len(calls) == 2:
            raise ValueError("compiler rejected the edit")
        return compile(candidate)

    with monkeypatch.context() as context:
        context.setattr(mujoco.MjSpec, "compile", fail_edited)
        with pytest.raises(ValueError, match="compiler rejected"):
            apply_model_patch(spec, patch)
    assert all(candidate is not spec for candidate in calls)
    _assert_source_unchanged(spec, before)


def test_compiler_cannot_silently_ignore_explicit_inertia():
    spec = mujoco.MjSpec.from_string(
        '<mujoco><compiler inertiafromgeom="true"/><worldbody><body name="b">'
        '<joint/><geom size=".1"/></body></worldbody></mujoco>'
    )
    patch = _patch(spec.compile())
    patch["inertias"] = [
        dict(name="b", mass=1, com=[0, 0, 0], inertia=np.eye(3).tolist())
    ]
    with pytest.raises(ValueError, match="ignored"):
        apply_model_patch(spec, patch)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("fusestatic", [False, True])
def test_framed_site_uses_body_local_position_and_preserves_orientation(
    nested, fusestatic
):
    inner = (
        '<frame name="inner" pos=".03 -.02 .1" euler="20 30 0">'
        '<site name="s" pos=".1 0 0" euler="10 0 25"/></frame>'
        if nested
        else '<site name="s" pos=".1 0 0" euler="10 0 25"/>'
    )
    spec = mujoco.MjSpec.from_string(
        f'<mujoco><compiler fusestatic="{str(fusestatic).lower()}"/><worldbody><body name="moving"><joint/><geom size=".1"/>'
        '<frame name="outer" pos=".1 .2 .3" euler="0 0 90">'
        '<site name="untouched" pos="0 .2 0"/>'
        + inner
        + "</frame></body></worldbody></mujoco>"
    )
    source = spec.copy().compile()
    before = _source_state(spec)
    patch = _patch(source)
    patch["sites"] = [dict(name="s", pos=[0.2, 0.4, 0.6])]
    result = apply_model_patch(spec, patch).compile()
    np.testing.assert_allclose(result.site("s").pos, [0.2, 0.4, 0.6], atol=1e-12)
    np.testing.assert_allclose(result.site("s").quat, source.site("s").quat, atol=1e-12)
    assert result.site("s").bodyid == source.site("s").bodyid
    np.testing.assert_array_equal(
        result.site("untouched").pos, source.site("untouched").pos
    )
    _assert_source_unchanged(spec, before)
    assert model_fingerprint(spec.compile()) == patch["fingerprint"]


def test_unsupported_fromto_frame_edit_is_transactional():
    spec = mujoco.MjSpec.from_string(
        '<mujoco><worldbody><body name="moving"><joint/><geom size=".1"/>'
        '<frame pos=".1 .2 .3" euler="0 0 90">'
        '<site name="s" type="capsule" size=".01" fromto="0 0 0 0 0 .1"/>'
        "</frame></body></worldbody></mujoco>"
    )
    patch = _patch(spec.copy().compile())
    patch["sites"] = [dict(name="s", pos=[0.2, 0.4, 0.6])]
    before = _source_state(spec)
    with pytest.raises(ValueError):
        apply_model_patch(spec, patch)
    _assert_source_unchanged(spec, before)


def test_compiled_site_postcondition_rejects_a_transformed_result(monkeypatch):
    spec = mujoco.MjSpec.from_string(
        '<mujoco><worldbody><site name="s"/></worldbody></mujoco>'
    )
    patch = _patch(spec.compile())
    patch["sites"] = [dict(name="s", pos=[0.2, 0.4, 0.6])]
    before = _source_state(spec)
    compile = mujoco.MjSpec.compile
    calls = []

    def transformed(candidate):
        result = compile(candidate)
        calls.append(candidate)
        if len(calls) == 2:
            result.site("s").pos[0] += 0.1
        return result

    with monkeypatch.context() as context:
        context.setattr(mujoco.MjSpec, "compile", transformed)
        with pytest.raises(ValueError, match="body-local site"):
            apply_model_patch(spec, patch)
    _assert_source_unchanged(spec, before)


def test_mesh_inertia_uses_homogeneous_watertight_mass_properties(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.creation.box(extents=[0.1, 0.2, 0.3])
    mesh.apply_translation([0.2, -0.3, 0.4])
    path = tmp_path / "box.ply"
    mesh.export(path)
    result = mesh_inertia(path, 1000)
    np.testing.assert_allclose(result["mass"], 6, rtol=1e-6)
    np.testing.assert_allclose(result["com"], [0.2, -0.3, 0.4], atol=1e-7)
    np.testing.assert_allclose(
        result["inertia"], np.diag([0.065, 0.05, 0.025]), rtol=1e-6, atol=1e-8
    )
    open_mesh = trimesh.Trimesh(mesh.vertices, mesh.faces[:-1], process=False)
    open_mesh.export(path)
    with pytest.raises(ValueError, match="watertight"):
        mesh_inertia(path, 1000)
    for density in (0, -1, np.nan):
        with pytest.raises(ValueError):
            mesh_inertia(path, density)


def test_mesh_inertia_is_optional(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "trimesh", None)
    with pytest.raises(ImportError, match="myosuite\\[muskemo\\]"):
        mesh_inertia(tmp_path / "unused.stl", 1000)


def test_actual_static_fusion_is_rejected_without_changing_other_anatomy():
    spec = _fusion_spec()
    source = spec.copy().compile()
    before = _source_state(spec)
    patch = _patch(source)
    patch["sites"] = [dict(name="s", pos=[0.2, 0.4, 0.6])]
    with pytest.raises(ValueError, match="fusion"):
        apply_model_patch(spec, patch)
    _assert_source_unchanged(spec, before)
    unchanged = spec.copy().compile()
    for field in ("geom_pos", "geom_quat", "site_pos", "site_quat"):
        np.testing.assert_array_equal(getattr(unchanged, field), getattr(source, field))
