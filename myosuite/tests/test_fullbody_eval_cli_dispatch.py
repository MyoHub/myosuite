# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Dispatch tests for fullbody_eval_cli."""

from __future__ import annotations

from typing import Any
import contextlib
import io

import pytest

from myosuite.integrations.musclemimic import fullbody_eval_cli as cli


def test_path_dispatch_calls_native_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When --path is present, CLI should route to native path runner."""
    seen: dict[str, Any] = {}

    def _fake_native(argv: list[str]) -> int:
        seen["argv"] = list(argv)
        return 7

    monkeypatch.setattr(cli, "_native_path_main", _fake_native)
    code = cli.main(["--path", "hf://foo/bar", "--motion_path", "KIT/a"])
    assert code == 7
    assert "--path" in seen["argv"]


def test_path_dispatch_strips_backend_before_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--backend`` is for smoke-test dispatch only; strip it for native playback."""
    seen: dict[str, Any] = {}

    def _fake_native(argv: list[str]) -> int:
        seen["argv"] = list(argv)
        return 0

    monkeypatch.setattr(cli, "_native_path_main", _fake_native)
    code = cli.main(
        [
            "--path",
            "hf://foo/bar",
            "--motion_path",
            "KIT/a",
            "--backend",
            "mjlab",
        ]
    )
    assert code == 0
    assert "--backend" not in seen["argv"]
    assert "mjlab" not in seen["argv"]


def test_path_mujoco_viewer_dispatch_calls_local_runner_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Viewer path mode should route to local path runner by default."""
    seen: dict[str, Any] = {}

    def _fake_local(argv: list[str]) -> int:
        seen["argv"] = list(argv)
        return 11

    monkeypatch.setattr(cli, "_maybe_reexec_preview_under_mjpython", lambda _argv: None)
    monkeypatch.setattr(cli, "_policy_path_main", lambda _argv: 99)
    monkeypatch.setattr(cli, "_path_mode_main", _fake_local)
    code = cli.main(
        [
            "--path",
            "hf://foo/bar",
            "--motion_path",
            "KIT/a",
            "--use_mujoco",
            "--mujoco_viewer",
        ]
    )
    assert code == 11
    assert "--mujoco_viewer" in seen["argv"]


def test_path_mujoco_viewer_importerror_falls_back_to_native_when_upstream_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If upstream backend import fails, route to native path mode."""
    monkeypatch.setenv("MYOSUITE_MUSCLEMIMIC_USE_UPSTREAM", "1")
    monkeypatch.setattr(
        cli,
        "_policy_path_main",
        lambda _argv: (_ for _ in ()).throw(ImportError("no fullbody")),
    )
    monkeypatch.setattr(cli, "_maybe_reexec_preview_under_mjpython", lambda _argv: None)
    monkeypatch.setattr(cli, "_path_mode_main", lambda _argv: 13)
    code = cli.main(
        [
            "--path",
            "hf://foo/bar",
            "--motion_path",
            "KIT/a",
            "--use_mujoco",
            "--mujoco_viewer",
        ]
    )
    assert code == 13


def test_path_dispatch_does_not_use_preview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--path mode must not fall through to preview mode."""
    called = {"preview": False}

    def _fake_preview(_argv: list[str]) -> int:
        called["preview"] = True
        return 0

    monkeypatch.setattr(cli, "_path_mode_main", lambda _argv: 2)
    monkeypatch.setattr(
        "myosuite.integrations.musclemimic.fullbody_mujoco_preview.main",
        _fake_preview,
        raising=False,
    )
    code = cli.main(["--path", "hf://foo/bar", "--motion_path", "KIT/a"])
    assert code == 2
    assert not called["preview"]


def test_native_path_argparse_errors_return_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native argparse failures should map to nonzero exit."""
    from myosuite.integrations.musclemimic import (
        fullbody_native_playback as native,
    )

    def _fake_main(_argv: list[str]) -> int:
        raise SystemExit(2)

    monkeypatch.setattr(native, "main", _fake_main)
    code = cli.main(["--path", "hf://foo/bar", "--motion_path", "KIT/a"])
    assert code == 2


def test_deprecated_playback_flag_returns_usage_error() -> None:
    """Legacy --playback flag should fail with a clear error."""
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        code = cli.main(["--playback", "legacy_mode"])
    assert code == 2
    assert "obsolete" in stderr.getvalue().lower()
