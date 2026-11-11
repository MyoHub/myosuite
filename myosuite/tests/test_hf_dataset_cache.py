# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Dataset cache publication never exposes incomplete downloads."""

from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from myosuite.core.hf_dataset_cache import download_hf_dataset_file

pytestmark = pytest.mark.tier1


@pytest.fixture
def download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    source = tmp_path / "upstream.npz"
    source.write_bytes(b"complete motion data")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kwargs: str(source))
    return source


@pytest.mark.parametrize("existing", [False, True])
def test_interrupted_copy_preserves_published_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    download: Path,
    existing: bool,
) -> None:
    destination = tmp_path / "cache" / "motion.npz"
    destination.parent.mkdir()
    if existing:
        destination.write_bytes(b"previous complete motion")

    def interrupted_copy(source: str, target: Path) -> None:
        target.write_bytes(b"partial")
        raise OSError("interrupted copy")

    monkeypatch.setattr(shutil, "copy2", interrupted_copy)
    assert (
        download_hf_dataset_file(
            repo_id="offline/test", remote_path="motion.npz", local_path=destination
        )
        is None
    )
    if existing:
        assert destination.read_bytes() == b"previous complete motion"
    else:
        assert not destination.exists()
    assert sorted(destination.parent.iterdir()) == ([destination] if existing else [])


def test_readers_see_complete_files_during_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, download: Path
) -> None:
    destination = tmp_path / "cache" / "motion.npz"
    destination.parent.mkdir()
    destination.write_bytes(b"previous complete motion")
    copying, resume = Event(), Event()

    def paused_copy(source: str, target: Path) -> None:
        target.write_bytes(b"partial")
        copying.set()
        assert resume.wait(timeout=10)
        target.write_bytes(Path(source).read_bytes())

    monkeypatch.setattr(shutil, "copy2", paused_copy)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            download_hf_dataset_file,
            repo_id="offline/test",
            remote_path="motion.npz",
            local_path=destination,
        )
        try:
            assert copying.wait(timeout=10)
            assert destination.read_bytes() == b"previous complete motion"
        finally:
            resume.set()
        assert future.result(timeout=10) == destination
    assert destination.read_bytes() == download.read_bytes()
    assert list(destination.parent.iterdir()) == [destination]


def test_unwritable_parent_returns_download_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, download: Path
) -> None:
    destination = tmp_path / "cache" / "motion.npz"

    def deny_mkdir(self: Path, *args: object, **kwargs: object) -> None:
        raise PermissionError("read-only cache")

    monkeypatch.setattr(Path, "mkdir", deny_mkdir)
    assert (
        download_hf_dataset_file(
            repo_id="offline/test", remote_path="motion.npz", local_path=destination
        )
        is None
    )
    assert not destination.exists()
