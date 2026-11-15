# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""``compare_onnx_across_backends`` fails when the second backend never ran."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from myosuite.utils import export_onnx

pytestmark = pytest.mark.tier1


class _Env:
    def __init__(self, reward: float) -> None:
        self.reward = reward

    def reset(self, seed: int = 0):  # noqa: ANN201
        return np.zeros(3, np.float32), {}

    def step(self, action):  # noqa: ANN001, ANN201
        return np.zeros(3, np.float32), self.reward, False, False, {}

    def close(self) -> None:
        pass


class _Session:
    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        pass

    def get_inputs(self):  # noqa: ANN201
        return [SimpleNamespace(name="obs")]

    def get_outputs(self):  # noqa: ANN201
        return [SimpleNamespace(name="action")]

    def run(self, names, feed):  # noqa: ANN001, ANN201
        return [np.zeros((1, 2), np.float32)]


@pytest.fixture(autouse=True)
def _fake_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "myosuite.utils.onnxruntime_utils.require_onnxruntime",
        lambda: SimpleNamespace(InferenceSession=_Session),
    )


def _make_env(mjx_reward: float | None):  # noqa: ANN202
    def make(env_id: str, **kwargs):  # noqa: ANN003, ANN202
        if env_id.startswith("Mjx"):
            if mjx_reward is None:
                raise RuntimeError("Environment MjxLegWalk doesn't exist")
            return _Env(mjx_reward)
        return _Env(1.0)

    return make


def test_a_missing_backend_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(export_onnx, "make_env", _make_env(None))
    results = export_onnx.compare_onnx_across_backends("policy.onnx", n_steps=3)
    assert results["passed"] is False
    assert "MjxLegWalk" in results["mjx_error"]


def test_a_reward_difference_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(export_onnx, "make_env", _make_env(2.0))
    assert (
        export_onnx.compare_onnx_across_backends("p.onnx", n_steps=3)["passed"] is False
    )
    monkeypatch.setattr(export_onnx, "make_env", _make_env(1.0))
    assert (
        export_onnx.compare_onnx_across_backends("p.onnx", n_steps=3)["passed"] is True
    )
