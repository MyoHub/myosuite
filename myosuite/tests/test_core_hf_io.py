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
    baseline_revisions,
    default_baseline_revision,
    default_musclemimic_cache_root,
    download_baseline_checkpoint,
    download_baseline_file,
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

    def fake_snapshot_download(
        *, repo_id: str, repo_type: str, revision: str, allow_patterns
    ) -> str:
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


def test_default_baseline_revision_follows_the_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tag is v<major>.<minor> of the installed version, unless overridden."""
    monkeypatch.delenv("MYOSUITE_BASELINES_REVISION", raising=False)
    monkeypatch.setattr("myosuite.__version__", "3.1.4")
    assert default_baseline_revision() == "v3.1"
    monkeypatch.setenv("MYOSUITE_BASELINES_REVISION", "my-branch")
    assert default_baseline_revision() == "my-branch"


def test_baseline_revisions_try_the_patch_tag_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A patch release looks for its own tag before the minor-release tag."""
    monkeypatch.delenv("MYOSUITE_BASELINES_REVISION", raising=False)
    monkeypatch.setattr("myosuite.__version__", "3.0.1")
    assert baseline_revisions() == ["v3.0.1", "v3.0"]
    monkeypatch.setattr("myosuite.__version__", "3.1")
    assert baseline_revisions() == ["v3.1"]
    monkeypatch.setenv("MYOSUITE_BASELINES_REVISION", "my-branch")
    assert baseline_revisions() == ["my-branch"]


def test_download_baseline_checkpoint_falls_back_to_the_minor_tag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without a patch tag on the Hub, the minor-release tag is used (not main)."""
    from huggingface_hub.errors import RevisionNotFoundError

    class MissingTag(RevisionNotFoundError):
        """RevisionNotFoundError without the HTTP response the Hub client attaches."""

        def __init__(self) -> None:
            Exception.__init__(self, "no such tag")

    env = tmp_path / "checkpoints" / "myoElbowPose1D6MRandom-v0"
    env.mkdir(parents=True)
    (env / "model_200.pt").write_bytes(b"")
    asked: list[str] = []

    def fake_snapshot_download(*, revision: str, **kwargs) -> str:
        asked.append(revision)
        if revision == "v3.0.1":
            raise MissingTag()
        return str(tmp_path)

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", fake_snapshot_download, raising=False
    )
    monkeypatch.setattr(
        "myosuite.core.hf_io.baseline_revisions", lambda: ["v3.0.1", "v3.0"]
    )

    assert download_baseline_checkpoint("myoElbowPose1D6MRandom-v0") == env
    assert asked == ["v3.0.1", "v3.0"]


def test_download_baseline_checkpoint_falls_back_to_main(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A release tag that is not on the Hub yet falls back to the main branch."""
    from huggingface_hub.errors import RevisionNotFoundError

    class MissingTag(RevisionNotFoundError):
        """RevisionNotFoundError without the HTTP response the Hub client attaches."""

        def __init__(self) -> None:
            Exception.__init__(self, "no such tag")

    env = tmp_path / "checkpoints" / "myoElbowPose1D6MRandom-v0"
    env.mkdir(parents=True)
    (env / "model_200.pt").write_bytes(b"")
    asked: list[str] = []

    def fake_snapshot_download(*, revision: str, **kwargs) -> str:
        asked.append(revision)
        if revision == "v3.0":
            raise MissingTag()
        return str(tmp_path)

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", fake_snapshot_download, raising=False
    )
    monkeypatch.setattr("myosuite.core.hf_io.baseline_revisions", lambda: ["v3.0"])

    assert download_baseline_checkpoint("myoElbowPose1D6MRandom-v0") == env
    assert asked == ["v3.0", "main"]


def test_download_baseline_file_uses_the_release_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A single file (and its folder's manifest) is fetched at the release tag; the revision is printed."""
    target = tmp_path / "model_81380.pt"
    asked: list[tuple[str, str, str]] = []

    def fake_hf_hub_download(
        repo_id: str, filename: str, *, repo_type: str, revision: str
    ) -> str:
        asked.append((repo_id, filename, revision))
        return str(target)

    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", fake_hf_hub_download, raising=False
    )
    monkeypatch.setattr("myosuite.core.hf_io.baseline_revisions", lambda: ["v3.0"])

    path = download_baseline_file("checkpoints/x/model_81380.pt")

    assert path == target
    assert asked == [
        ("myohub/myosuite-3-baselines", "checkpoints/x/model_81380.pt", "v3.0"),
        ("myohub/myosuite-3-baselines", "checkpoints/x/manifest.json", "v3.0"),
    ]
    assert "revision 'v3.0'" in capsys.readouterr().out


def test_download_baseline_checkpoint_prints_the_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """When the release tag is missing the message says that the latest revision was used."""
    from huggingface_hub.errors import RevisionNotFoundError

    class MissingTag(RevisionNotFoundError):
        """RevisionNotFoundError without the HTTP response the Hub client attaches."""

        def __init__(self) -> None:
            Exception.__init__(self, "no such tag")

    env = tmp_path / "checkpoints" / "myoElbowPose1D6MRandom-v0"
    env.mkdir(parents=True)
    (env / "model_200.pt").write_bytes(b"")

    def fake_snapshot_download(*, revision: str, **kwargs) -> str:
        if revision != "main":
            raise MissingTag()
        return str(tmp_path)

    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", fake_snapshot_download, raising=False
    )
    monkeypatch.setattr("myosuite.core.hf_io.baseline_revisions", lambda: ["v3.0"])

    assert download_baseline_checkpoint("myoElbowPose1D6MRandom-v0") == env
    out = capsys.readouterr().out
    assert "revision 'main'" in out and "tag 'v3.0' not found" in out
