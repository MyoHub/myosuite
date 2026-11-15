# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""mjviser stays compatible with the core mujoco<4 pin."""

from __future__ import annotations

from pathlib import Path

from packaging.requirements import Requirement
import pytest

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

pytestmark = pytest.mark.tier1


def test_mjviser_capped_before_mujoco_core_4_while_mujoco_under_4() -> None:
    """While core pins mujoco<4, mjviser must exclude >=0.0.17 (requires mujoco-core>=4.0.1)."""
    config = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text())
    deps = {
        req.name: req
        for raw in config["project"]["dependencies"]
        if (req := Requirement(raw)).marker is None
    }
    mujoco_spec = deps["mujoco"].specifier
    mjviser_spec = deps["mjviser"].specifier
    if not mujoco_spec.contains("4.0.1"):
        assert mjviser_spec.contains("0.0.14")
        assert mjviser_spec.contains("0.0.16")
        assert not mjviser_spec.contains("0.0.17")
