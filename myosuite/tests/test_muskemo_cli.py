# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Anatomy CLI staging and native model archives preserve the source model."""

import importlib.util
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np
import pytest

from myosuite.integrations.muskemo import joint_sweep
from myosuite.integrations.muskemo_edits import apply_model_patch
from myosuite.tests.support.muskemo import (
    _assert_source_unchanged,
    _edit,
    _fusion_spec,
    _source_state,
    _elbow_model,
)

pytestmark = pytest.mark.tier1


def _curves(model):
    data = mujoco.MjData(model)
    lengths, moments = [], []
    for pose in joint_sweep(model, "r_elbow_flex", 21):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        dense = np.zeros((model.nu, model.nv))
        mujoco.mju_sparse2dense(
            dense,
            data.actuator_moment,
            data.moment_rownnz,
            data.moment_rowadr,
            data.moment_colind,
        )
        lengths.append(data.actuator_length.copy())
        moments.append(dense.copy())
    return np.array(lengths), np.array(moments)


@pytest.fixture(scope="module")
def elbow():
    return _elbow_model()


@pytest.fixture
def cli():
    location = Path(__file__).parents[2] / "scripts" / "muskemo.py"
    spec = importlib.util.spec_from_file_location("muskemo_cli", location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_export_apply_and_protected_outputs(cli, tmp_path):
    xml = tmp_path / "source.xml"
    xml.write_text(
        '<mujoco><worldbody><body name="b"><joint name="q" range="-30 30"/>'
        '<geom size=".1"/><site name="s" pos=".1 0 0"/></body></worldbody></mujoco>'
    )
    output = tmp_path / "export.json"
    assert (
        cli.main(
            [
                "export",
                "--xml",
                str(xml),
                "--joint-sweep",
                "q",
                "--samples",
                "3",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    document = json.loads(output.read_text())
    assert len(document["frames"]) == 3
    patch = {k: document[k] for k in ("schema", "fingerprint", "mujoco_version")}
    patch.update(sites=[dict(name="s", pos=[0.2, 0, 0])], inertias=[])
    patch_file = tmp_path / "edits.json"
    patch_file.write_text(json.dumps(patch))
    archive = tmp_path / "model.zip"
    args = [
        "apply",
        "--xml",
        str(xml),
        "--patch",
        str(patch_file),
        "--output",
        str(archive),
    ]
    assert cli.main(args) == 0
    np.testing.assert_allclose(
        mujoco.MjSpec.from_zip(archive).compile().site("s").pos, [0.2, 0, 0]
    )
    before = archive.read_bytes()
    with pytest.raises(SystemExit):
        cli.main(args)
    assert archive.read_bytes() == before
    patch["fingerprint"] = "wrong"
    patch_file.write_text(json.dumps(patch))
    with pytest.raises(SystemExit):
        cli.main([*args, "--overwrite"])
    assert archive.read_bytes() == before


def test_blender_cli_stages_and_protects_outputs(cli, tmp_path, monkeypatch):
    script = tmp_path / "muskemo_blender.py"
    script.write_text("# external importer test fixture\n")
    monkeypatch.setattr(cli.muskemo, "__file__", str(tmp_path / "muskemo.py"))
    blend, report = tmp_path / "scene.blend", tmp_path / "report.json"
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs == {"check": True} and command[1:3] == [
            "--background",
            "--factory-startup",
        ]
        Path(command[command.index("--output") + 1]).write_bytes(b"blend")
        Path(command[command.index("--report") + 1]).write_text("{}")

    monkeypatch.setattr(cli.subprocess, "run", run)
    args = [
        "blender",
        str(tmp_path / "input.json"),
        "--addon",
        str(tmp_path / "MuSkeMo"),
        "--blender",
        "blender.exe",
        "--output",
        str(blend),
        "--report",
        str(report),
    ]
    assert (
        cli.main(args) == 0
        and blend.read_bytes() == b"blend"
        and report.read_text() == "{}"
    )
    with pytest.raises(SystemExit):
        cli.main(args)
    assert len(calls) == 1

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "blender")

    monkeypatch.setattr(cli.subprocess, "run", fail)
    with pytest.raises(SystemExit):
        cli.main([*args, "--overwrite"])
    assert blend.read_bytes() == b"blend" and report.read_text() == "{}"


def test_archive_rejects_ambiguous_external_asset_names(cli, tmp_path):
    obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 2 3 4\nf 3 1 4\n"
    for folder, contents in (("a", obj), ("b", obj.replace("v 1 0 0", "v 2 0 0"))):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "same.obj").write_text(contents)
    xml = tmp_path / "source.xml"
    xml.write_text(
        '<mujoco><asset><mesh name="a" file="a/same.obj"/>'
        '<mesh name="b" file="b/same.obj"/></asset><worldbody>'
        '<geom type="mesh" mesh="a"/><geom type="mesh" mesh="b"/></worldbody></mujoco>'
    )
    spec = mujoco.MjSpec.from_file(str(xml))
    with pytest.raises(ValueError, match="Ambiguous"):
        cli._write_zip(spec, tmp_path / "model.zip")
    assert not (tmp_path / "model.zip").exists()


@pytest.mark.parametrize(
    "vfs_key",
    [
        None,
        "assets/tetra.obj",
        "tetra.obj",
        "old/path/tetra.obj",
        "assets/old/path/tetra.obj",
    ],
)
@pytest.mark.parametrize("filename", ["old/path/tetra.obj", r"old\path\tetra.obj"])
def test_native_archive_honors_strippath(cli, tmp_path, vfs_key, filename):
    data = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 2 3 4\nf 3 1 4\n"
    xml = (
        '<mujoco><compiler strippath="true" meshdir="assets"/><asset>'
        f'<mesh name="tetra" file="{filename}"/></asset><worldbody>'
        '<geom type="mesh" mesh="tetra"/></worldbody></mujoco>'
    )
    if vfs_key is None:
        (tmp_path / "assets").mkdir()
        (tmp_path / "assets/tetra.obj").write_bytes(data)
        source = tmp_path / "source.xml"
        source.write_text(xml)
        spec = mujoco.MjSpec.from_file(str(source))
    else:
        spec = mujoco.MjSpec.from_string(xml, assets={vfs_key: data})
    original = spec.compile()
    archive = tmp_path / "model.zip"
    cli._write_zip(spec, archive)
    loaded = mujoco.MjSpec.from_zip(archive).compile()
    for field in (
        "mesh_vert",
        "mesh_face",
        "geom_type",
        "geom_pos",
        "geom_quat",
        "geom_size",
    ):
        np.testing.assert_allclose(
            getattr(loaded, field), getattr(original, field), atol=1e-6, err_msg=field
        )


def test_elbow_native_archive_preserves_edited_anatomy(elbow, tmp_path, cli):
    spec, source = elbow
    before = _source_state(spec)
    edited = apply_model_patch(spec, _edit(source))
    model = edited.compile()
    for field in (
        "geom_type",
        "geom_size",
        "geom_pos",
        "geom_quat",
        "geom_dataid",
        "mesh_vert",
        "mesh_face",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_dynprm",
        "actuator_lengthrange",
        "eq_data",
    ):
        np.testing.assert_array_equal(
            getattr(model, field), getattr(source, field), err_msg=field
        )
    old_length, old_moment = _curves(source)
    new_length, new_moment = _curves(model)
    selected = model.actuator("BRA").id
    assert np.max(np.abs(new_length[:, selected] - old_length[:, selected])) > 1e-4
    assert np.max(np.abs(new_moment[:, selected] - old_moment[:, selected])) > 1e-5
    others = np.arange(model.nu) != selected
    np.testing.assert_array_equal(new_length[:, others], old_length[:, others])
    np.testing.assert_allclose(
        new_moment[:, others], old_moment[:, others], atol=1e-12, rtol=1e-12
    )
    archive = tmp_path / "edited.zip"
    cli._write_zip(edited, archive)
    loaded = mujoco.MjSpec.from_zip(archive).compile()
    for field in (
        "geom_type",
        "geom_size",
        "geom_pos",
        "geom_quat",
        "mesh_vert",
        "mesh_face",
        "site_pos",
    ):
        np.testing.assert_allclose(
            getattr(loaded, field), getattr(model, field), atol=1e-6, err_msg=field
        )
    np.testing.assert_allclose(_curves(loaded)[0], new_length, atol=1e-12)
    np.testing.assert_allclose(_curves(loaded)[1], new_moment, atol=1e-12)
    _assert_source_unchanged(spec, before)


def test_archive_refuses_static_fusion_without_changing_source(cli, tmp_path):
    spec = _fusion_spec()
    source = spec.copy().compile()
    before = _source_state(spec)
    with pytest.raises(ValueError, match="fusion"):
        cli._write_zip(spec, tmp_path / "fused.zip")
    assert not (tmp_path / "fused.zip").exists()
    _assert_source_unchanged(spec, before)
    unchanged = spec.copy().compile()
    for field in ("geom_pos", "geom_quat", "site_pos", "site_quat"):
        np.testing.assert_array_equal(getattr(unchanged, field), getattr(source, field))
