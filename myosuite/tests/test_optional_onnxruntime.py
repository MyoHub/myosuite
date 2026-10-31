# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""ONNX dependency markers and the independent native CPU path."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from packaging.requirements import Requirement
import pytest

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from myosuite import make_env
from myosuite.utils.onnxruntime_utils import require_onnxruntime

pytestmark = pytest.mark.tier1


@pytest.mark.parametrize("version", ["3.10", "3.11", "3.12", "3.13", "3.14", "3.15"])
def test_default_runtime_markers_preserve_supported_versions(version: str) -> None:
    config = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text())
    dependencies = [Requirement(value) for value in config["project"]["dependencies"]]
    active = [
        dependency
        for dependency in dependencies
        if dependency.name == "onnxruntime"
        and dependency.marker.evaluate({"python_version": version})
    ]
    assert len(active) == (0 if version == "3.15" else 1)
    if active:
        assert active[0].specifier.contains("1.20.1" if version < "3.14" else "1.31.0")
    extra = [
        Requirement(value)
        for value in config["project"]["optional-dependencies"]["onnx"]
    ]
    assert any(
        dependency.marker.evaluate({"python_version": version}) for dependency in extra
    )


def test_native_cpu_environment_does_not_import_onnxruntime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    env = make_env("myoElbowPose1D6MRandom-v0")
    try:
        observation, _ = env.reset(seed=0)
        result = env.step(np.zeros(env.action_space.shape))
        assert len(result) == 5
        assert np.isfinite(observation).all()
        assert np.isfinite(result[0]).all()
    finally:
        env.close()


def test_missing_runtime_explains_optional_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(ImportError, match=r"myosuite\[onnx\]"):
        require_onnxruntime()


def test_onnx_actor_reports_missing_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    from myosuite.integrations.musclemimic.actor_onnx import load_onnx_session

    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(ImportError, match=r"myosuite\[onnx\]"):
        load_onnx_session("actor.onnx")
