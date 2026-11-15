# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Compass cache reuse must identify source content and generated artifacts."""

import json
from pathlib import Path

import numpy as np
import pytest

from myosuite.core.trajectory_yaw import clip_natural_yaw, write_rotated_clip_npz
from myosuite.integrations.musclemimic import steered_teacher as teacher
from myosuite.tests.test_fullbody_obs_adapter import _toy_model, _write_clip

pytestmark = pytest.mark.tier2


@pytest.fixture
def compass_case(tmp_path: Path) -> tuple:
    model = _toy_model(3)
    source = tmp_path / "source" / "walk.npz"
    source.parent.mkdir()
    _write_clip(model, source, n_frames=6)
    return model, source, tmp_path / "cache"


def _ensure(model, source: Path, cache: Path) -> dict:
    return teacher.ensure_compass_clips(
        source, cache, expected_nq=model.nq, expected_nv=model.nv
    )


def test_same_named_native_sources_do_not_share_generated_motion(
    compass_case, tmp_path: Path
) -> None:
    model, source, cache = compass_case
    other = tmp_path / "another" / source.name
    other.parent.mkdir()
    _write_clip(model, other, n_frames=9)
    first = _ensure(model, source, cache)
    second = _ensure(model, other, cache)
    assert all(clip.num_frames == 6 for clip in first.values())
    assert all(clip.num_frames == 9 for clip in second.values())
    assert first["E"].source_path != second["E"].source_path


def test_overwritten_source_creates_new_cache_without_changing_old_clip_paths(
    compass_case,
) -> None:
    model, source, cache = compass_case
    first = _ensure(model, source, cache)
    old_path = first["E"].source_path
    old_bytes = old_path.read_bytes()
    _write_clip(model, source, n_frames=12)
    second = _ensure(model, source, cache)
    assert all(clip.num_frames == 12 for clip in second.values())
    assert second["E"].source_path != old_path
    assert old_path.read_bytes() == old_bytes


def test_verified_hit_skips_generation_and_preserves_native_outputs(
    compass_case, tmp_path: Path, monkeypatch
) -> None:
    model, source, cache = compass_case
    original_source = source.read_bytes()
    first = _ensure(model, source, cache)
    manifest = first["E"].source_path.parent / "manifest.json"
    metadata = json.loads(manifest.read_text())
    assert metadata["identity"]["schema"] == 1
    assert len(metadata["identity"]["yaw_sha256"]) == 64
    before = {
        p: p.stat().st_mtime_ns
        for p in [manifest, *[c.source_path for c in first.values()]]
    }

    def unexpected_generation(*args, **kwargs) -> None:
        raise AssertionError("cache hit regenerated clips")

    monkeypatch.setattr(teacher, "write_rotated_clip_npz", unexpected_generation)
    second = _ensure(model, source, cache)
    natural = clip_natural_yaw(first["E"].qpos, first["E"].qvel)
    with np.load(source, allow_pickle=False) as original:
        source_natural = clip_natural_yaw(original["qpos"], original["qvel"])
    for i, label in enumerate(teacher.COMPASS_LABELS):
        reference = write_rotated_clip_npz(
            source,
            tmp_path / f"reference_{label}.npz",
            i * teacher.COMPASS_STEP - source_natural,
        )
        with (
            np.load(reference, allow_pickle=False) as expected,
            np.load(second[label].source_path, allow_pickle=False) as actual,
        ):
            assert set(actual.files) == set(expected.files)
            for name in expected.files:
                np.testing.assert_array_equal(actual[name], expected[name])
        assert second[label].source_path.name == f"walk_{label}.npz"
    assert natural == pytest.approx(0.0, abs=1e-7)
    assert source.read_bytes() == original_source
    assert {p: p.stat().st_mtime_ns for p in before} == before


@pytest.mark.parametrize("corruption", ["truncated", "foreign", "missing"])
def test_only_corrupt_generated_heading_is_rebuilt(
    compass_case, monkeypatch, corruption: str
) -> None:
    model, source, cache = compass_case
    first = _ensure(model, source, cache)
    target = first["E"].source_path
    untouched = {
        c.source_path: c.source_path.read_bytes()
        for label, c in first.items()
        if label != "E"
    }
    if corruption == "truncated":
        target.write_bytes(target.read_bytes()[:30])
    elif corruption == "foreign":
        _write_clip(model, target, n_frames=9)
    else:
        target.unlink()
    generated = []
    native_write = teacher.write_rotated_clip_npz

    def record(src, dst, yaw):
        generated.append(dst.name)
        return native_write(src, dst, yaw)

    monkeypatch.setattr(teacher, "write_rotated_clip_npz", record)
    result = _ensure(model, source, cache)
    assert generated == ["walk_E.npz"]
    np.testing.assert_array_equal(result["E"].qpos, first["E"].qpos)
    assert all(path.read_bytes() == value for path, value in untouched.items())


@pytest.mark.parametrize("invalid", ["missing", "malformed", "wrong_identity"])
def test_unverified_manifest_is_rebuilt(compass_case, invalid: str) -> None:
    model, source, cache = compass_case
    first = _ensure(model, source, cache)
    manifest = first["E"].source_path.parent / "manifest.json"
    if invalid == "missing":
        manifest.unlink()
    elif invalid == "malformed":
        manifest.write_bytes(b"{incomplete")
    else:
        data = json.loads(manifest.read_text())
        data["identity"]["source_sha256"] = "wrong"
        manifest.write_text(json.dumps(data))
    result = _ensure(model, source, cache)
    assert json.loads(manifest.read_text())["identity"]["source_sha256"] != "wrong"
    np.testing.assert_array_equal(result["E"].qpos, first["E"].qpos)


@pytest.mark.parametrize("change", ["schema", "yaw_implementation"])
def test_generation_contract_changes_invalidate_cache(
    compass_case, tmp_path: Path, monkeypatch, change: str
) -> None:
    model, source, cache = compass_case
    first = _ensure(model, source, cache)
    if change == "schema":
        monkeypatch.setattr(teacher, "_COMPASS_CACHE_SCHEMA", 2)
    else:
        alternate = tmp_path / "yaw_revision.py"
        alternate.write_bytes(
            Path(teacher.trajectory_yaw.__file__).read_bytes() + b"\n# revision\n"
        )
        monkeypatch.setattr(teacher.trajectory_yaw, "__file__", str(alternate))
    second = _ensure(model, source, cache)
    assert second["E"].source_path != first["E"].source_path
    np.testing.assert_array_equal(second["E"].qpos, first["E"].qpos)


def test_legacy_cache_is_ignored_without_overwriting_it(compass_case) -> None:
    model, source, cache = compass_case
    cache.mkdir()
    legacy = cache / "walk_E.npz"
    _write_clip(model, legacy, n_frames=9)
    original = legacy.read_bytes()
    result = _ensure(model, source, cache)
    assert result["E"].num_frames == 6
    assert result["E"].source_path != legacy and legacy.read_bytes() == original


def test_failed_generation_preserves_existing_artifacts_and_cleans_staging(
    compass_case, monkeypatch
) -> None:
    model, source, cache = compass_case
    first = _ensure(model, source, cache)
    target = first["E"].source_path
    target.write_bytes(b"corrupt")
    before = {p: p.read_bytes() for p in target.parent.iterdir()}

    def fail_write(src, dst, yaw) -> None:
        dst.write_bytes(b"partial output")
        raise OSError("disk full")

    with monkeypatch.context() as scoped:
        scoped.setattr(teacher, "write_rotated_clip_npz", fail_write)
        with pytest.raises(OSError, match="disk full"):
            _ensure(model, source, cache)
    assert {p: p.read_bytes() for p in target.parent.iterdir()} == before
    assert _ensure(model, source, cache)["E"].num_frames == 6


def test_source_change_during_generation_publishes_nothing(
    compass_case, monkeypatch
) -> None:
    model, source, cache = compass_case
    native_write = teacher.write_rotated_clip_npz
    changed = False

    def change_source(src, dst, yaw):
        nonlocal changed
        result = native_write(src, dst, yaw)
        if not changed:
            _write_clip(model, src, n_frames=12)
            changed = True
        return result

    with monkeypatch.context() as scoped:
        scoped.setattr(teacher, "write_rotated_clip_npz", change_source)
        with pytest.raises(ValueError, match="Source motion changed"):
            _ensure(model, source, cache)
    assert list(cache.rglob("*.npz")) == []
    assert list(cache.rglob("manifest.json")) == []
    assert all(p.is_dir() for p in cache.iterdir())
    assert all(clip.num_frames == 12 for clip in _ensure(model, source, cache).values())


def test_relative_inputs_keep_resolved_persistent_clip_paths(
    compass_case, monkeypatch
) -> None:
    model, source, cache = compass_case
    monkeypatch.chdir(source.parent.parent)
    result = _ensure(model, Path("source/walk.npz"), Path("cache"))
    assert all(
        clip.source_path.is_absolute() and clip.source_path.is_file()
        for clip in result.values()
    )
