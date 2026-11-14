"""Content validation, concurrency and execution controls of motion preprocessing."""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest

from myosuite.envs.myo.tasks.mimic.tracking import motion
from myosuite.envs.myo.tasks.mimic.tracking.config import MotionCfg, resolve_task
from myosuite.envs.myo.tasks.mimic.tracking.models import compiled_scene, model_cfg

pytestmark = pytest.mark.tier2
Inputs = tuple[list[str], mujoco.MjModel, tuple[str, ...]]


@pytest.fixture
def inputs(tmp_path: Path) -> Inputs:
    model = copy.copy(compiled_scene("myofullbody"))
    files = [str(tmp_path / f"clip{i}.npz") for i in range(2)]
    for i, path in enumerate(files):
        motion.write_synthetic_clip(path, model, 0.2, 1.0 + i, 0.1, i)
    tracked = model_cfg("myofullbody").tracked_bodies
    return files, model, tracked


def _same(a: motion.MotionLib, b: motion.MotionLib) -> None:
    assert a.names == b.names and a.dt == b.dt and a.quality_flags == b.quality_flags
    for field in (
        "clip_start",
        "clip_len",
        "clip_end",
        "clip_id",
        "bin_clip",
        "bin_start",
        "frame_bin",
    ):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    for name in a.frames:
        np.testing.assert_array_equal(a.frames[name], b.frames[name], err_msg=name)


def test_workers_are_equivalent_and_cache_hit_skips_preprocessing(
    inputs: Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files, model, tracked = inputs
    expected = motion.build_motion_lib(files, model, tracked, 0.02, workers=1)
    parallel = motion.build_motion_lib(files, model, tracked, 0.02, workers=2)
    _same(expected, parallel)
    cache = tmp_path / "cache"
    _same(
        expected, motion.build_motion_lib(files, model, tracked, 0.02, cache_dir=cache)
    )

    def unexpected(*args: Any, **kwargs: Any) -> None:
        pytest.fail("valid cache must skip clip preprocessing")

    monkeypatch.setattr(motion, "_process_clip", unexpected)
    _same(
        expected, motion.build_motion_lib(files, model, tracked, 0.02, cache_dir=cache)
    )
    with np.load(next(cache.glob("*.npz")), allow_pickle=False) as data:
        assert all(data[k].dtype.kind != "O" for k in data.files)


@pytest.mark.parametrize(
    "change", ["source", "order", "model", "tracked", "dt", "bins", "schema"]
)
def test_cache_invalidates_every_input(
    inputs: Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    files, model, tracked = inputs
    cache = tmp_path / "cache"
    kwargs = {"dt": 0.02, "bin_s": 1.0, "cache_dir": cache}
    motion.build_motion_lib(files, model, tracked, **kwargs)
    if change == "source":
        stat = Path(files[0]).stat()
        motion.write_synthetic_clip(files[0], model, 0.2, 3.0, 0.1, 0)
        os.utime(files[0], ns=(stat.st_atime_ns, stat.st_mtime_ns))
    elif change == "order":
        files = files[::-1]
    elif change == "model":
        model.body_pos[model.body("head").id, 0] += 0.1
    elif change == "tracked":
        tracked = (tracked[0], tracked[2], tracked[1], *tracked[3:])
    elif change == "dt":
        kwargs["dt"] = 0.01
    elif change == "bins":
        kwargs["bin_s"] = 0.04
    else:
        monkeypatch.setattr(motion, "_CACHE_SCHEMA", motion._CACHE_SCHEMA + 1)
    expected = motion.build_motion_lib(
        files, model, tracked, kwargs["dt"], kwargs["bin_s"]
    )
    _same(expected, motion.build_motion_lib(files, model, tracked, **kwargs))
    assert len(list(cache.glob("*.npz"))) == 2


@pytest.mark.parametrize(
    "damage", ["truncated", "dtype", "shape", "values", "object", "metadata"]
)
def test_invalid_cache_is_rebuilt(inputs: Inputs, tmp_path: Path, damage: str) -> None:
    files, model, tracked = inputs
    cache = tmp_path / "cache"
    expected = motion.build_motion_lib(files, model, tracked, 0.02, cache_dir=cache)
    path = next(cache.glob("*.npz"))
    if damage == "truncated":
        path.write_bytes(b"broken archive")
    else:
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        if damage == "dtype":
            arrays["body_pos"] = arrays["body_pos"].astype(np.float64)
        elif damage == "shape":
            arrays["body_pos"] = arrays["body_pos"][:-1]
        elif damage == "values":
            arrays["body_pos"][0, 0, 0] += 1
        elif damage == "object":
            arrays["body_pos"] = np.array(["never unpickle"], dtype=object)
        else:
            meta = json.loads(str(arrays["metadata"]))
            meta["names"] = ["wrong", "names"]
            arrays["metadata"] = np.array(json.dumps(meta))
        np.savez(path, **arrays)
    with pytest.warns(RuntimeWarning, match="Ignoring invalid motion cache"):
        actual = motion.build_motion_lib(files, model, tracked, 0.02, cache_dir=cache)
    _same(expected, actual)
    _same(
        expected, motion.build_motion_lib(files, model, tracked, 0.02, cache_dir=cache)
    )


def test_concurrent_writers_publish_complete_archives(
    inputs: Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files, model, tracked = inputs
    cache = tmp_path / "cache"
    original = motion._process_clip
    barrier = threading.Barrier(2)

    def synchronized(*args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
        barrier.wait(timeout=30)
        return original(*args, **kwargs)

    monkeypatch.setattr(motion, "_process_clip", synchronized)
    with ThreadPoolExecutor(2) as pool:
        results = list(
            pool.map(
                lambda _: motion.build_motion_lib(
                    files, model, tracked, 0.02, cache_dir=cache
                ),
                range(2),
            )
        )
    _same(*results)
    assert len(list(cache.glob("*.npz"))) == 1
    monkeypatch.setattr(
        motion, "_process_clip", lambda *a, **k: pytest.fail("cache miss")
    )
    _same(
        results[0],
        motion.build_motion_lib(files, model, tracked, 0.02, cache_dir=cache),
    )


def test_sources_changed_during_processing_are_not_published(
    inputs: Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files, model, tracked = inputs
    original = motion._process_clip

    def changed(path: str, **kwargs: Any) -> dict[str, np.ndarray]:
        out = original(path, **kwargs)
        if path == files[0]:
            motion.write_synthetic_clip(path, model, 0.2, 3.0, 0.1, 0)
        return out

    monkeypatch.setattr(motion, "_process_clip", changed)
    cache = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="inputs changed"):
        motion.build_motion_lib(files, model, tracked, 0.02, cache_dir=cache)
    assert not list(cache.glob("*.npz"))


def test_memory_cache_not_stale_after_source_edit(inputs: Inputs) -> None:
    files, model, _ = inputs
    before = motion.load_motion_lib(files[0], "", "myofullbody", True, 0.02, 1.0)
    motion.write_synthetic_clip(files[0], model, 0.2, 3.0, 0.1, 0)
    after = motion.load_motion_lib(files[0], "", "myofullbody", True, 0.02, 1.0)
    assert before is not after
    assert after.frames["root_pos"][-1, 0] > before.frames["root_pos"][-1, 0]


def test_memory_cache_hit_reads_no_clip(
    inputs: Inputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repeated load with unchanged clips neither hashes nor checks a clip file;
    a clip with new contents (same size, newer modification time) is read again."""
    from myosuite.biomechanics import retarget

    files, model, _ = inputs
    args = (files[0], "", "myofullbody", True, 0.02, 1.0)
    first = motion.load_motion_lib(*args)
    reads: list[str] = []
    sha256, check_meta = retarget.file_sha256, retarget.check_refit_meta

    def counted_sha256(path: Any) -> str:
        reads.append(Path(path).name)
        return sha256(path)

    def counted_check(path: Any, *rest: Any) -> Any:
        reads.append(Path(path).name)
        return check_meta(path, *rest)

    monkeypatch.setattr(retarget, "file_sha256", counted_sha256)
    monkeypatch.setattr(retarget, "check_refit_meta", counted_check)
    assert motion.load_motion_lib(*args) is first
    assert reads == []

    size = os.path.getsize(files[0])
    motion.write_synthetic_clip(files[0], model, 0.2, 3.0, 0.1, 0)
    assert os.path.getsize(files[0]) == size
    stat = os.stat(files[0])
    os.utime(files[0], ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    after = motion.load_motion_lib(*args)
    assert after is not first and Path(files[0]).name in reads
    assert after.frames["root_pos"][-1, 0] > first.frames["root_pos"][-1, 0]


def test_execution_options_do_not_change_task_contract() -> None:
    original = resolve_task()
    changed = dataclasses.replace(
        original,
        motion=dataclasses.replace(original.motion, workers=3, cache_dir="elsewhere"),
    )
    assert original.fingerprint() == changed.fingerprint()
    for workers in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            MotionCfg(workers=workers)


def test_unwritable_cache_falls_back_to_preprocessing(
    inputs: Inputs, tmp_path: Path
) -> None:
    files, model, tracked = inputs
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    expected = motion.build_motion_lib(files, model, tracked, 0.02)
    with pytest.warns(RuntimeWarning):
        actual = motion.build_motion_lib(files, model, tracked, 0.02, cache_dir=blocked)
    _same(expected, actual)


def test_readers_never_observe_partial_writes(inputs: Inputs, tmp_path: Path) -> None:
    files, model, tracked = inputs
    key = motion._motion_key(files, model, tracked, 0.02, 1.0)
    lib = motion.build_motion_lib(files, model, tracked, 0.02)
    path = tmp_path / "motion.npz"

    def writer() -> None:
        for _ in range(10):
            motion._write_cache(path, key, lib)

    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(writer)
        for _ in range(10):
            got = motion._read_cache(path, key, model, tracked, files, 0.02, 1.0)
            if got is not None:
                _same(lib, got)
        future.result(timeout=30)
    _same(lib, motion._read_cache(path, key, model, tracked, files, 0.02, 1.0))


def test_cache_is_reused_in_a_fresh_process(inputs: Inputs, tmp_path: Path) -> None:
    files, model, tracked = inputs
    cache = tmp_path / "cache"
    expected = motion.build_motion_lib(files, model, tracked, 0.02, cache_dir=cache)
    script = """
import sys
from myosuite.envs.myo.tasks.mimic.tracking import motion
from myosuite.envs.myo.tasks.mimic.tracking.models import compiled_scene, model_cfg
def fail(*args, **kwargs):
    raise AssertionError('preprocessing instead of reading the disk cache')
motion._process_clip = fail
lib = motion.build_motion_lib(sys.argv[2:], compiled_scene('myofullbody'),
    model_cfg('myofullbody').tracked_bodies, 0.02, cache_dir=sys.argv[1])
print(lib.num_frames)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(cache), *files],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip().splitlines()[-1] == str(expected.num_frames)
