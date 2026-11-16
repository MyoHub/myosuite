# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The steering library cache follows the clips' content and holds data only (synthetic clips)."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pytest
import torch

pytestmark = pytest.mark.tier2

_UNPICKLED: list[str] = []


def _record(tag: str) -> str:
    _UNPICKLED.append(tag)
    return tag


class _Tripwire:
    """Records a call when unpickled, where a malicious file would run code."""

    def __reduce__(self) -> tuple[Any, ...]:
        return (_record, ("unpickled",))


@pytest.fixture(name="kit")
def kit_fixture(
    tmp_path: Path, body: Any, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, list[str]]:
    """A clip root with two locomotion clips, and a training split listing them."""
    from chasetag_teacher import library

    from myosuite.envs.myo.tasks.mimic.tracking.motion import write_synthetic_clip

    root = tmp_path / "kit"
    names = ["KIT/1/walking_medium01_poses.npz", "KIT/2/LeftTurn01_poses.npz"]
    files = [str(root / n) for n in names]
    for i, f in enumerate(files):
        Path(f).parent.mkdir(parents=True, exist_ok=True)
        write_synthetic_clip(f, body, 3.0, 1.0, 0.05, i)
    split = tmp_path / "train.txt"
    split.write_text("".join(f"{n}\n" for n in names))
    monkeypatch.setattr(library, "kit_split", lambda name: split)
    return root, files


def test_a_clip_rewritten_in_place_rebuilds_the_library(
    kit: tuple[Path, list[str]], body: Any
) -> None:
    from chasetag_teacher.library import load_library

    from myosuite.envs.myo.tasks.mimic.tracking.motion import write_synthetic_clip

    root, files = kit
    before = load_library(str(root), limit=2)
    write_synthetic_clip(files[0], body, 3.0, 0.4, 0.05, 0)
    after = load_library(str(root), limit=2)
    assert after.names == before.names
    n = int(before.clip_len[0])
    assert torch.allclose(before.v[:n, 0], torch.tensor(1.0), atol=1e-3)
    assert torch.allclose(after.v[:n, 0], torch.tensor(0.4), atol=1e-3)


def test_cache_files_are_never_unpickled(kit: tuple[Path, list[str]]) -> None:
    from chasetag_teacher.library import cache_dir, load_library

    root, _ = kit
    load_library(str(root), limit=2)
    cached = [p for p in cache_dir().iterdir() if p.is_file()]
    assert cached
    for path in cached:
        torch.save(_Tripwire(), path)
    _UNPICKLED.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # the invalid cache is ignored
        lib = load_library(str(root), limit=2)
    assert _UNPICKLED == []
    assert len(lib.names) == 2
