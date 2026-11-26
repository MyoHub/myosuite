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
from myosuite.policies.onnxruntime_utils import load_onnx_session, require_onnxruntime

pytestmark = pytest.mark.tier1


@pytest.mark.parametrize("version", ["3.10", "3.11", "3.12", "3.13", "3.14", "3.15"])
def test_default_runtime_markers_preserve_supported_versions(version: str) -> None:
    config = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text())
    dependencies = [Requirement(value) for value in config["project"]["dependencies"]]
    assert not any(
        dependency.name == "onnxruntime" for dependency in dependencies
    ), "onnxruntime must live in myosuite[onnx], not base dependencies"
    extra = [
        Requirement(value)
        for value in config["project"]["optional-dependencies"]["onnx"]
    ]
    active = [
        dependency
        for dependency in extra
        if dependency.name == "onnxruntime"
        and (
            dependency.marker is None
            or dependency.marker.evaluate({"python_version": version})
        )
    ]
    assert len(active) == 1
    assert active[0].specifier.contains("1.20.1" if version < "3.14" else "1.31.0")

    # `examples` (and therefore `all = ["myosuite[examples,musclemimic]"]`) pulls `[onnx]` on <3.15 only.
    examples_extra = [
        Requirement(value)
        for value in config["project"]["optional-dependencies"]["examples"]
    ]
    onnx_refs = [
        req
        for req in examples_extra
        if req.name == "myosuite"
        and "onnx" in req.extras
        and (req.marker is None or req.marker.evaluate({"python_version": version}))
    ]
    assert len(onnx_refs) == (0 if version == "3.15" else 1)
    all_extra = [
        Requirement(value)
        for value in config["project"]["optional-dependencies"]["all"]
    ]
    assert any(req.name == "myosuite" and "examples" in req.extras for req in all_extra)


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


def test_require_onnxruntime_reports_install_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(ImportError, match=r"myosuite\[onnx\]"):
        require_onnxruntime()


def test_onnx_actor_reports_missing_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    dummy = tmp_path / "actor.onnx"
    dummy.write_bytes(b"onnx")
    with pytest.raises(ImportError, match=r"myosuite\[onnx\]"):
        load_onnx_session(dummy)


def test_require_onnxruntime_reraises_broken_install_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    def _missing_transitive(name: str):
        raise ModuleNotFoundError("No module named 'flatbuffers'", name="flatbuffers")

    monkeypatch.setattr(importlib, "import_module", _missing_transitive)
    with pytest.raises(ModuleNotFoundError, match="flatbuffers") as exc_info:
        require_onnxruntime()
    assert "onnxruntime-gpu" not in str(exc_info.value)

    def _missing_internal_c_ext(name: str):
        raise ModuleNotFoundError(
            "No module named 'onnxruntime.capi.onnxruntime_pybind11_state'",
            name="onnxruntime.capi.onnxruntime_pybind11_state",
        )

    monkeypatch.setattr(importlib, "import_module", _missing_internal_c_ext)
    with pytest.raises(
        ModuleNotFoundError, match="onnxruntime_pybind11_state"
    ) as c_ext_exc:
        require_onnxruntime()
    assert "myosuite[onnx]" not in str(c_ext_exc.value)

    def _dll_load_failure(name: str):
        raise ImportError("DLL load failed while importing onnxruntime_pybind11_state")

    monkeypatch.setattr(importlib, "import_module", _dll_load_failure)
    with pytest.raises(
        ImportError, match="DLL load failed while importing onnxruntime_pybind11_state"
    ) as dll_exc:
        require_onnxruntime()
    assert "onnxruntime-gpu" not in str(dll_exc.value)
