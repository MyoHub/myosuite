# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""The asset resolver writes each patched model XML once, under a stable name."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import mujoco
import numpy as np
import pytest
from PIL import Image

import myosuite
from myosuite.utils import asset_path_resolver as apr
from myosuite import make_env

pytestmark = pytest.mark.tier1

_ASSETS = Path(apr.__file__).resolve().parents[1] / "envs" / "myo" / "assets"
_TETRA = "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
_PYRAMID = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv .5 .5 1\nf 1 3 2\nf 1 4 3\nf 1 2 5\nf 2 3 5\nf 3 4 5\nf 4 1 5\n"


def _write_model(model_dir: Path) -> Path:
    """A model the resolver rewrites, with paths MuJoCo resolves from its directory.

    The include and the mesh and texture files it names are relative to the model's
    directory, not the include's, so they only resolve from the model's directory.
    """
    (model_dir / "parts").mkdir(parents=True)
    (model_dir / "main.obj").write_text(_TETRA)
    (model_dir / "inc.obj").write_text(_PYRAMID)
    Image.new("RGB", (4, 2), (200, 30, 30)).save(model_dir / "inc.png")
    (model_dir / "parts" / "assets.xml").write_text(
        '<mujoco><asset><mesh name="inc" file="inc.obj"/>'
        '<texture name="tex" type="2d" file="inc.png"/>'
        '<material name="mat" texture="tex"/></asset></mujoco>'
    )
    model = model_dir / "model.xml"
    model.write_text(
        # convexhull is stripped by the resolver, so the model gets a patched copy.
        '<mujoco><compiler convexhull="false"/><include file="parts/assets.xml"/>'
        '<asset><mesh name="main" file="main.obj"/></asset><worldbody>'
        '<geom type="mesh" mesh="main"/><geom type="mesh" mesh="inc" material="mat"/>'
        "</worldbody></mujoco>"
    )
    return model


def test_simhive_folder_is_not_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``myosuite/simhive/myo_sim`` folder (e.g. a stale pre-3.0 copy) is ignored.

    Assets come from the bundled subsets and pip; a local myo_sim checkout is used
    through an editable install instead.
    """
    package = tmp_path / "myosuite"
    (package / "simhive" / "myo_sim" / "meshes").mkdir(parents=True)
    (package / "utils").mkdir()
    monkeypatch.setattr(
        apr, "__file__", str(package / "utils" / "asset_path_resolver.py")
    )

    pip_root = apr._pip_myo_sim_models_root()
    assert pip_root is not None
    assert apr.get_sim_asset_root("myo_sim") == pip_root
    assert apr._resolve_myo_sim_rel("meshes/hat_cervical.stl") == (
        pip_root / "meshes" / "hat_cervical.stl"
    )


def _resolved_copies(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.glob(".myosuite_resolved_*"))


def _model_arrays(path: Path) -> dict[str, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(path))
    return {name: getattr(model, name).copy() for name in ("mesh_vert", "tex_data")}


def test_resolve_model_xml_path_reuses_one_copy(tmp_path: Path) -> None:
    """Repeated calls return the same file; only that one copy is ever written."""
    model = _write_model(tmp_path / "model")

    first = apr.resolve_model_xml_path(model)
    second = apr.resolve_model_xml_path(model)

    assert first == second != model
    assert first.parent == model.parent
    assert _resolved_copies(model.parent) == [first.name]
    assert _model_arrays(first)["mesh_vert"].shape == (4 + 5, 3)


def test_gym_make_twice_writes_no_new_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second ``gym.make`` of an env whose models are rewritten adds no XML file."""

    monkeypatch.setattr(apr, "patched_xml_dir", lambda: tmp_path)

    def written() -> set[Path]:
        return set(_ASSETS.rglob(".myosuite_resolved_*")) | set(tmp_path.iterdir())

    make_env("myoTorsoPoseFixed-v0").close()
    before = written()
    make_env("myoTorsoPoseFixed-v0").close()
    assert written() == before


def test_names_are_stable_across_processes(tmp_path: Path) -> None:
    """Names are content digests, not Python's per-process salted ``hash``."""
    model = _write_model(tmp_path / "model")
    code = (
        "import sys, pathlib\n"
        "import xml.etree.ElementTree as ET\n"
        "from myosuite.utils import asset_path_resolver as apr\n"
        "model = pathlib.Path(sys.argv[1])\n"
        "print(apr.resolve_model_xml_path(model).name)\n"
        "print(apr.write_patched_xml(ET.parse(model), model, 'probe').name)\n"
    )
    repo = str(Path(myosuite.__file__).resolve().parents[1])
    names = set()
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [repo, env.get("PYTHONPATH")]))
        out = subprocess.run(
            [sys.executable, "-c", code, str(model)],
            capture_output=True,
            text=True,
            env=env,
        )
        assert out.returncode == 0, out.stderr
        names.add(tuple(out.stdout.split()))
    assert len(names) == 1
    assert _resolved_copies(model.parent) == [names.pop()[0]]


def test_read_only_model_dir_falls_back_to_the_patched_xml_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model in a directory that refuses writes still resolves to the same model."""
    writable = _write_model(tmp_path / "writable")
    expected = _model_arrays(apr.resolve_model_xml_path(writable))

    model = _write_model(tmp_path / "read_only")
    patched = tmp_path / "patched"
    monkeypatch.setattr(apr, "patched_xml_dir", lambda: patched)
    mkstemp = apr.tempfile.mkstemp

    def refuse_model_dir(*args: object, **kwargs: object) -> tuple[int, str]:
        if Path(str(kwargs["dir"])).resolve() == model.parent.resolve():
            raise PermissionError(13, "Permission denied", str(model.parent))
        return mkstemp(*args, **kwargs)

    monkeypatch.setattr(apr.tempfile, "mkstemp", refuse_model_dir)
    resolved = apr.resolve_model_xml_path(model)

    assert resolved.parent == patched
    assert sorted(p.name for p in model.parent.iterdir()) == [
        "inc.obj",
        "inc.png",
        "main.obj",
        "model.xml",
        "parts",
    ]
    actual = _model_arrays(resolved)
    for name, value in expected.items():
        np.testing.assert_array_equal(actual[name], value)
    assert apr.resolve_model_xml_path(model) == resolved


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="needs POSIX permissions and a non-root user",
)
def test_read_only_model_dir_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same as above with a directory the OS really refuses to write."""
    model = _write_model(tmp_path / "read_only")
    monkeypatch.setattr(apr, "patched_xml_dir", lambda: tmp_path / "patched")
    model.parent.chmod(0o555)
    try:
        resolved = apr.resolve_model_xml_path(model)
    finally:
        model.parent.chmod(0o755)
    assert resolved.parent == tmp_path / "patched"
    assert _model_arrays(resolved)["mesh_vert"].shape == (4 + 5, 3)


_DATA = b"<mujoco/>"


def _published_name(tmp_path: Path) -> str:
    """Name ``_write_once`` gives ``_DATA`` written from ``source.xml``."""
    return apr._write_once(
        tmp_path / "names", "probe", tmp_path / "source.xml", _DATA
    ).name


def _lose_race(error: type[OSError]) -> Callable[[Path, Path], Path]:
    """A rename/replace that loses to a writer of the same bytes, raising *error*."""

    def lose(self: Path, target: Path) -> Path:
        Path(target).write_bytes(_DATA)  # the other process got there first ...
        raise error(13, "in use", str(target))  # ... and is still renaming it

    return lose


@pytest.mark.parametrize("error", [FileExistsError, PermissionError])
def test_write_once_reuses_the_file_of_a_concurrent_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: type[OSError]
) -> None:
    """Losing the rename to an identical writer is not an error (Windows refuses it)."""
    monkeypatch.setattr(Path, "rename", _lose_race(error))
    monkeypatch.setattr(Path, "replace", _lose_race(error))
    path = apr._write_once(tmp_path, "probe", tmp_path / "source.xml", _DATA)
    monkeypatch.undo()

    assert path.read_bytes() == _DATA
    assert [p.name for p in tmp_path.iterdir()] == [path.name]


def test_write_once_retries_until_the_concurrent_file_is_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows also refuses a first read of the file being renamed; the write retries."""
    read_bytes = Path.read_bytes
    refused: list[Path] = []

    def refuse_first_read(self: Path) -> bytes:
        if self.exists() and not refused:
            refused.append(self)
            raise PermissionError(13, "in use", str(self))
        return read_bytes(self)

    sleeps: list[float] = []
    monkeypatch.setattr(Path, "rename", _lose_race(PermissionError))
    monkeypatch.setattr(Path, "replace", _lose_race(PermissionError))
    monkeypatch.setattr(Path, "read_bytes", refuse_first_read)
    monkeypatch.setattr(apr.time, "sleep", sleeps.append)
    path = apr._write_once(tmp_path, "probe", tmp_path / "source.xml", _DATA)
    monkeypatch.undo()

    assert refused == [path]
    assert sleeps == [apr._RETRY_DELAYS[0]]
    assert path.read_bytes() == _DATA
    assert [p.name for p in tmp_path.iterdir()] == [path.name]


def test_write_once_raises_when_refusals_outlast_the_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rename Windows keeps refusing fails after the last attempt, leaving no temp file."""
    attempts: list[Path] = []

    def refuse(self: Path, target: Path) -> Path:
        attempts.append(Path(target))
        raise PermissionError(13, "in use", str(target))

    sleeps: list[float] = []
    monkeypatch.setattr(Path, "rename", refuse)
    monkeypatch.setattr(Path, "replace", refuse)
    monkeypatch.setattr(apr.time, "sleep", sleeps.append)
    with pytest.raises(PermissionError):
        apr._write_once(tmp_path, "probe", tmp_path / "source.xml", _DATA)
    monkeypatch.undo()

    assert len(attempts) == len(apr._RETRY_DELAYS) + 1
    assert sleeps == list(apr._RETRY_DELAYS)
    assert list(tmp_path.iterdir()) == []


def test_write_once_leaves_a_published_file_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file another writer published meanwhile is kept, not replaced under its readers."""
    path = tmp_path / _published_name(tmp_path)
    mkstemp = apr.tempfile.mkstemp

    def publish_meanwhile(*args: object, **kwargs: object) -> tuple[int, str]:
        path.write_bytes(_DATA)  # the other process publishes while this one writes
        return mkstemp(*args, **kwargs)

    def must_not_publish(self: Path, target: Path) -> Path:
        raise AssertionError(f"replaced the published {target}")

    monkeypatch.setattr(apr.tempfile, "mkstemp", publish_meanwhile)
    monkeypatch.setattr(Path, "rename", must_not_publish)
    monkeypatch.setattr(Path, "replace", must_not_publish)
    assert apr._write_once(tmp_path, "probe", tmp_path / "source.xml", _DATA) == path
    monkeypatch.undo()

    assert sorted(p.name for p in tmp_path.iterdir()) == ["names", path.name]


def test_write_once_replaces_a_stale_file(tmp_path: Path) -> None:
    """A file under the digest name with other bytes (e.g. truncated) is rewritten."""
    path = tmp_path / _published_name(tmp_path)
    path.write_bytes(b"<muj")
    assert apr._write_once(tmp_path, "probe", tmp_path / "source.xml", _DATA) == path
    assert path.read_bytes() == _DATA
    assert sorted(p.name for p in tmp_path.iterdir()) == ["names", path.name]
