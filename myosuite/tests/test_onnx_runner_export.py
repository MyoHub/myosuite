# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""ONNX export of OnnxCheckpointingMjlabRunner for actors with one or several observation groups."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("mjlab")
ort = pytest.importorskip("onnxruntime")
TensorDict = pytest.importorskip("tensordict").TensorDict

from myosuite.integrations.musclemimic.mjlab_policy_runner import (  # noqa: E402
    OnnxCheckpointingMjlabRunner,
)


class _MlpActor(torch.nn.Module):
    """One 1D group, like the RSL-RL MLP model."""

    obs_groups = ["proprioception"]

    def __init__(self) -> None:
        super().__init__()
        self.mlp = torch.nn.Linear(5, 2)

    def forward(self, obs: TensorDict) -> torch.Tensor:
        return self.mlp(obs["proprioception"])


class _CnnActor(torch.nn.Module):
    """A 1D group plus an image group, like the RSL-RL CNN model (``obs_groups_2d``)."""

    obs_groups = ["proprioception"]
    obs_groups_2d = ["vision_mono"]

    def __init__(self) -> None:
        super().__init__()
        self.cnn = torch.nn.Sequential(torch.nn.Conv2d(2, 3, 3), torch.nn.Flatten())
        self.mlp = torch.nn.Linear(5 + 3 * 6 * 6, 2)

    def forward(self, obs: TensorDict) -> torch.Tensor:
        return self.mlp(
            torch.cat([obs["proprioception"], self.cnn(obs["vision_mono"])], dim=-1)
        )


def _export(
    actor: torch.nn.Module, observations: dict[str, torch.Tensor], path: Path
) -> None:
    runner = object.__new__(OnnxCheckpointingMjlabRunner)
    runner.alg = SimpleNamespace(actor=actor)
    runner.env = SimpleNamespace(
        get_observations=lambda: TensorDict(observations, batch_size=[4])
    )
    runner._obs_dim = observations["proprioception"].shape[-1]
    runner._export_actor_onnx(path)


def test_single_group_actor_keeps_the_obs_input(tmp_path: Path) -> None:
    actor = _MlpActor()
    obs = {"proprioception": torch.randn(4, 5)}
    _export(actor, obs, tmp_path / "a.onnx")
    session = ort.InferenceSession(str(tmp_path / "a.onnx"))
    assert [i.name for i in session.get_inputs()] == ["obs"]
    (out,) = session.run(None, {"obs": obs["proprioception"].numpy()})
    np.testing.assert_allclose(
        out, actor(TensorDict(obs, batch_size=[4])).detach().numpy(), atol=1e-5
    )


def test_actor_with_an_image_group_exports_one_input_per_group(tmp_path: Path) -> None:
    actor = _CnnActor()
    obs = {"proprioception": torch.randn(4, 5), "vision_mono": torch.randn(4, 2, 8, 8)}
    _export(actor, obs, tmp_path / "a.onnx")
    session = ort.InferenceSession(str(tmp_path / "a.onnx"))
    assert [i.name for i in session.get_inputs()] == ["proprioception", "vision_mono"]
    (out,) = session.run(None, {k: v.numpy() for k, v in obs.items()})
    np.testing.assert_allclose(
        out, actor(TensorDict(obs, batch_size=[4])).detach().numpy(), atol=1e-5
    )
