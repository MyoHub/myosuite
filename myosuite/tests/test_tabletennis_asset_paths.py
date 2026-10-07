# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The TableTennis model builds and compiles from any working directory."""

from __future__ import annotations

from pathlib import Path

import pytest

import myosuite  # noqa: F401  (registers the envs)
from myosuite import make_env

pytestmark = pytest.mark.tier1

TT_P0 = "myoChallengeTableTennisP0-v0"


def test_tabletennis_furniture_asset_paths_are_absolute() -> None:
    """The meshes and textures the recipe adds do not depend on the working directory."""
    env = make_env(TT_P0)
    spec = env.unwrapped._mj_spec
    files = [
        spec.mesh(name).file
        for name in ("tabletennis_table", "tabletennis_net_mesh", "paddle_mesh")
    ]
    files += [spec.texture(name).file for name in ("tabletennis_tex", "paddle_tex")]
    assert all(Path(file).is_absolute() for file in files), files
    env.close()


def test_tabletennis_spec_compiles_after_a_directory_change(monkeypatch, tmp_path):
    """An env's spec still compiles after the working directory changes."""
    env = make_env(TT_P0)
    monkeypatch.chdir(tmp_path)
    env.unwrapped._mj_spec.compile()
    env.close()


def test_tabletennis_makes_from_another_working_directory(monkeypatch, tmp_path):
    """Making the env from another directory works (another drive on Windows CI)."""
    monkeypatch.chdir(tmp_path)
    env = make_env(TT_P0)
    env.reset(seed=0)
    env.close()
