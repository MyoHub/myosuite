# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for fullbody checkpoint path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from myosuite.integrations.musclemimic.fullbody_checkpoint_io import (
    describe_checkpoint_dir,
    resolve_checkpoint_ref,
)


def test_resolve_checkpoint_ref_local_dir(tmp_path: Path) -> None:
    """Local directory checkpoint paths should resolve directly."""
    ref = resolve_checkpoint_ref(str(tmp_path))
    assert ref.local_path == tmp_path.resolve()


def test_resolve_checkpoint_ref_missing_local_raises(tmp_path: Path) -> None:
    """Missing local checkpoint path should raise FileNotFoundError."""
    missing = tmp_path / "nope"
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint_ref(str(missing))


def test_describe_checkpoint_dir_flags(tmp_path: Path) -> None:
    """Directory diagnostics should reflect expected marker files."""
    (tmp_path / "_METADATA").write_text("x")
    (tmp_path / "params").write_text("x")
    flags = describe_checkpoint_dir(tmp_path)
    assert flags["exists"]
    assert flags["is_dir"]
    assert flags["has_orbax_metadata"]
    assert flags["has_params_file"]
