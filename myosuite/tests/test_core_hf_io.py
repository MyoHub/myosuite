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
    download_baseline_checkpoint,
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


def test_download_baseline_checkpoint_returns_the_env_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A snapshot containing the env's checkpoint resolves to its subdirectory."""
    (tmp_path / "checkpoints" / "myoElbowPose1D6MRandom-v0").mkdir(parents=True)
    (
        tmp_path / "checkpoints" / "myoElbowPose1D6MRandom-v0" / "model_200.pt"
    ).write_bytes(b"")

    def fake_snapshot_download(*, repo_id: str, repo_type: str, allow_patterns) -> str:
        assert repo_id == "myohub/myosuite-3-baselines"
        assert repo_type == "model"
        assert allow_patterns == ["checkpoints/myoElbowPose1D6MRandom-v0/*"]
        return str(tmp_path)

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", fake_snapshot_download, raising=False
    )

    local = download_baseline_checkpoint("myoElbowPose1D6MRandom-v0")

    assert local == tmp_path / "checkpoints" / "myoElbowPose1D6MRandom-v0"


def test_download_baseline_checkpoint_missing_env_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An env not hosted on the baselines repo (e.g. below the success threshold) is None."""
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **kwargs: str(tmp_path),
        raising=False,
    )

    assert download_baseline_checkpoint("motorFingerPoseFixed-v0") is None


def test_download_baseline_checkpoint_network_error_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A network/auth/repo error is treated as "no baseline", not raised."""

    def raises(**kwargs):
        raise OSError("no network")

    monkeypatch.setattr("huggingface_hub.snapshot_download", raises, raising=False)

    assert download_baseline_checkpoint("myoElbowPose1D6MRandom-v0") is None
