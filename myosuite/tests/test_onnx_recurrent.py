# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for stateful GRU/LSTM ONNX policy export and playback."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from myosuite.utils.export_onnx import (
    export_recurrent_to_onnx,
    verify_onnx_on_cpu,
)

pytestmark = pytest.mark.tier1


def test_recurrent_gru_and_lstm_onnx_export_and_cpu_playback(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    from myosuite import make_env

    with make_env("myoElbowPose1D6MRandom-v0") as env:
        obs_dim = int(env.observation_space.shape[0])
        act_dim = int(env.action_space.shape[0])

    class _GRUActor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gru = torch.nn.GRU(input_size=obs_dim, hidden_size=16, num_layers=1)
            self.head = torch.nn.Linear(16, act_dim)

        def forward(
            self, obs: torch.Tensor, h_in: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            y, h_out = self.gru(obs.unsqueeze(0), h_in)
            return torch.tanh(self.head(y.squeeze(0))), h_out

    gru_out = tmp_path / "gru_actor.onnx"
    export_recurrent_to_onnx(
        _GRUActor().eval(),
        gru_out,
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=16,
        num_layers=1,
        rnn_type="gru",
        verify=True,
        write_spec=True,
    )
    gru_spec = json.loads(
        gru_out.with_suffix(".policy_spec.json").read_text(encoding="utf-8")
    )
    assert [i["name"] for i in gru_spec["model"]["onnx"]["inputs"]] == ["obs", "h_in"]
    assert [o["name"] for o in gru_spec["model"]["onnx"]["outputs"]] == [
        "action",
        "h_out",
    ]
    metrics = verify_onnx_on_cpu(
        gru_out, n_steps=5, seed=0, env_id="myoElbowPose1D6MRandom-v0"
    )
    assert metrics["n_steps"] == 5

    class _LSTMActor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = torch.nn.LSTM(input_size=10, hidden_size=12, num_layers=2)
            self.head = torch.nn.Linear(12, 4)

        def forward(
            self, obs: torch.Tensor, h_in: torch.Tensor, c_in: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            y, (h_out, c_out) = self.lstm(obs.unsqueeze(0), (h_in, c_in))
            return torch.tanh(self.head(y.squeeze(0))), h_out, c_out

    lstm_out = tmp_path / "lstm_actor.onnx"
    export_recurrent_to_onnx(
        _LSTMActor().eval(),
        lstm_out,
        obs_dim=10,
        act_dim=4,
        hidden_dim=12,
        num_layers=2,
        rnn_type="lstm",
        verify=True,
        write_spec=True,
    )
    lstm_spec = json.loads(
        lstm_out.with_suffix(".policy_spec.json").read_text(encoding="utf-8")
    )
    assert [i["name"] for i in lstm_spec["model"]["onnx"]["inputs"]] == [
        "obs",
        "h_in",
        "c_in",
    ]
    assert [o["name"] for o in lstm_spec["model"]["onnx"]["outputs"]] == [
        "action",
        "h_out",
        "c_out",
    ]


def test_recurrent_parity_checks_state_outputs_and_missing_onnxruntime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    from myosuite.utils import onnxruntime_utils
    from myosuite.utils.export_onnx import verify_onnx_parity

    class _LSTMActor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = torch.nn.LSTM(input_size=6, hidden_size=8, num_layers=1)
            self.head = torch.nn.Linear(8, 3)

        def forward(
            self, obs: torch.Tensor, h_in: torch.Tensor, c_in: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            y, (h_out, c_out) = self.lstm(obs.unsqueeze(0), (h_in, c_in))
            return torch.tanh(self.head(y.squeeze(0))), h_out, c_out

    base = _LSTMActor().eval()
    onnx_path = tmp_path / "lstm.onnx"
    export_recurrent_to_onnx(
        base,
        onnx_path,
        obs_dim=6,
        act_dim=3,
        hidden_dim=8,
        num_layers=1,
        rnn_type="lstm",
        verify=True,
        write_spec=True,
    )

    class _PerturbedCellState(torch.nn.Module):
        """Matches action on step 0 while perturbing c_out so state parity must catch it."""

        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(
            self, obs: torch.Tensor, h_in: torch.Tensor, c_in: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            action, h_out, c_out = self.inner(obs, h_in, c_in)
            return action, h_out, c_out + 0.5

    with pytest.raises(RuntimeError, match="c_out"):
        verify_onnx_parity(
            _PerturbedCellState(base).eval(),
            onnx_path,
            obs_dim=6,
            recurrent_shapes=((1, 8), (1, 8)),
            steps=1,
        )

    prev_onnx = onnx_path.read_bytes()
    prev_spec = onnx_path.with_suffix(".policy_spec.json").read_bytes()
    monkeypatch.setattr(
        onnxruntime_utils,
        "require_onnxruntime",
        lambda: (_ for _ in ()).throw(
            ImportError("ONNX inference requires onnxruntime.")
        ),
    )
    with pytest.raises(ImportError, match="--no-verify"):
        export_recurrent_to_onnx(
            base,
            onnx_path,
            obs_dim=6,
            act_dim=3,
            hidden_dim=8,
            num_layers=1,
            rnn_type="lstm",
            verify=True,
            write_spec=True,
        )
    assert onnx_path.read_bytes() == prev_onnx
    assert onnx_path.with_suffix(".policy_spec.json").read_bytes() == prev_spec
