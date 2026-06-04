# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for shared Hugging Face IO utilities."""

from __future__ import annotations

import pathlib
from pathlib import Path

import pytest

from myosuite.core.hf_io import (
    default_musclemimic_cache_root,
    parse_hf_ref,
)


def test_parse_hf_ref_with_subpath() -> None:
    """hf path parser splits repo id and subpath."""
    ref = parse_hf_ref("hf://owner/repo/sub/dir")
    assert ref.repo_id == "owner/repo"
    assert ref.subpath == "sub/dir"


def test_parse_hf_ref_invalid_raises() -> None:
    """Invalid hf path should raise ValueError."""
    with pytest.raises(ValueError):
        parse_hf_ref("hf://only-owner")


def test_default_cache_root_path() -> None:
    """Default cache root should point to MuscleMimic AMASS cache."""
    root = default_musclemimic_cache_root()
    assert isinstance(root, pathlib.Path)
    assert str(root).endswith("/.musclemimic/caches/AMASS")


def test_default_cache_root_prefers_musclemimic_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Upstream MuscleMimic env var should override the legacy home path."""
    monkeypatch.setenv(
        "MUSCLEMIMIC_CONVERTED_AMASS_PATH",
        "~/scratch/.musclemimic/caches/AMASS",
    )
    monkeypatch.delenv("CONVERTED_AMASS_PATH", raising=False)

    root = default_musclemimic_cache_root()

    assert root == Path("~/scratch/.musclemimic/caches/AMASS").expanduser()


def test_default_cache_root_falls_back_to_converted_amass_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The generic upstream cache env var should be used when set."""
    monkeypatch.delenv("MUSCLEMIMIC_CONVERTED_AMASS_PATH", raising=False)
    monkeypatch.setenv(
        "CONVERTED_AMASS_PATH",
        "~/scratch/.musclemimic/caches/AMASS",
    )

    root = default_musclemimic_cache_root()

    assert root == Path("~/scratch/.musclemimic/caches/AMASS").expanduser()
