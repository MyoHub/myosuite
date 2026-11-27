# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for RSL-RL v3/v4/v5 checkpoint extraction and custom trunk builder hooks."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from myosuite.utils.export_onnx import export_rslrl_to_onnx  # noqa: E402
from myosuite.utils.rslrl_policy import (  # noqa: E402
    _build_mlp,
    _build_normalizer,
    extract_actor_state_dict,
    load_rslrl_policy,
    register_actor_trunk_builder,
    unregister_actor_trunk_builder,
)

pytestmark = pytest.mark.tier1


class _ToyLayerNormTrunk(nn.Module):
    """Minimal non-MLP trunk whose first parameter is 1-D (LayerNorm weight)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.tanh(self.proj(self.norm(x))))


def test_extract_actor_state_dict_supports_v3_and_v5() -> None:
    v5_raw = {
        "actor_state_dict": {
            "mlp.0.weight": torch.ones(8, 4),
            "mlp.0.bias": torch.zeros(8),
            "distribution.std_param": torch.full((3,), 0.25),
        }
    }
    assert set(extract_actor_state_dict(v5_raw)) == set(v5_raw["actor_state_dict"])

    v3_raw = {
        "model_state_dict": {
            "actor.0.weight": torch.ones(8, 4),
            "actor.0.bias": torch.zeros(8),
            "actor_obs_normalizer._mean": torch.zeros(1, 4),
            "actor_obs_normalizer._std": torch.ones(1, 4),
            "std": torch.full((3,), 0.4),
            "critic.0.weight": torch.ones(8, 4),
        }
    }
    v3_out = extract_actor_state_dict(v3_raw)
    assert "mlp.0.weight" in v3_out
    assert "obs_normalizer._mean" in v3_out
    assert "distribution.std_param" in v3_out
    assert all(not k.startswith("critic.") for k in v3_out)

    v3_ac = {
        "actor_critic_state_dict": {
            "actor.0.weight": torch.ones(3, 4),
            "log_std": torch.zeros(3),
        }
    }
    ac_out = extract_actor_state_dict(v3_ac)
    assert "mlp.0.weight" in ac_out
    assert "distribution.log_std_param" in ac_out

    with pytest.raises(ValueError, match="neither 'actor_state_dict' nor"):
        extract_actor_state_dict({"optimizer_state_dict": {}})

    with pytest.raises(ValueError, match="Could not locate actor weights"):
        extract_actor_state_dict(
            {"model_state_dict": {"critic.0.weight": torch.ones(2, 2)}}
        )


def test_v3_mlp_checkpoint_loads_and_matches_forward(tmp_path: Path) -> None:
    torch.manual_seed(7)
    obs_dim, act_dim = 12, 5
    mlp = _build_mlp(obs_dim, act_dim, [16, 16], "swish").eval()
    norm = _build_normalizer(obs_dim).eval()
    norm._mean.copy_(torch.linspace(-0.2, 0.2, obs_dim).unsqueeze(0))
    norm._var.copy_(torch.linspace(0.64, 1.44, obs_dim).unsqueeze(0))
    norm._std.copy_(norm._var.sqrt())

    v3_state: dict[str, torch.Tensor] = {
        f"actor.{k}": v for k, v in mlp.state_dict().items()
    }
    v3_state["actor_obs_normalizer._mean"] = norm._mean.clone()
    v3_state["actor_obs_normalizer._std"] = norm._std.clone()
    v3_state["log_std"] = torch.full((act_dim,), -0.5)

    run_dir = tmp_path / "run_v3"
    (run_dir / "params").mkdir(parents=True)
    (run_dir / "params" / "agent.yaml").write_text(
        yaml.safe_dump({"policy": {"activation": "swish"}}), encoding="utf-8"
    )
    ckpt = run_dir / "model_50.pt"
    torch.save({"model_state_dict": v3_state}, ckpt)

    loaded = load_rslrl_policy(ckpt, act_dim)
    sample = torch.randn(4, obs_dim)
    with torch.no_grad():
        torch.testing.assert_close(
            loaded(sample), mlp(norm(sample)), atol=1e-6, rtol=1e-6
        )
    torch.testing.assert_close(loaded.std, torch.full((act_dim,), -0.5).exp())


def test_registered_actor_trunk_builder_loads_and_exports_onnx(tmp_path: Path) -> None:
    torch.manual_seed(11)
    obs_dim, hidden_dim, act_dim = 10, 20, 4
    trunk = _ToyLayerNormTrunk(obs_dim, hidden_dim, act_dim).eval()
    norm = _build_normalizer(obs_dim).eval()
    norm._mean.copy_(torch.randn(1, obs_dim))
    norm._var.copy_(torch.rand(1, obs_dim) + 0.25)
    norm._std.copy_(norm._var.sqrt())

    state: dict[str, torch.Tensor] = {
        f"mlp.{k}": v for k, v in trunk.state_dict().items()
    }
    for k, v in norm.state_dict().items():
        state[f"obs_normalizer.{k}"] = v
    state["distribution.std_param"] = torch.full((act_dim,), 0.2)

    run_dir = tmp_path / "custom_run"
    (run_dir / "params").mkdir(parents=True)
    (run_dir / "params" / "agent.yaml").write_text(
        yaml.safe_dump({"actor": {"class_name": "pkg.custom:ToyModel"}}),
        encoding="utf-8",
    )
    ckpt = run_dir / "model_100.pt"
    torch.save({"actor_state_dict": state}, ckpt)

    # Without the builder registered, loading reports a clear ValueError naming class_name.
    with pytest.raises(ValueError, match="pkg.custom:ToyModel"):
        load_rslrl_policy(ckpt, act_dim)

    def _toy_builder(
        mlp_state: dict[str, torch.Tensor], actor_cfg: dict[str, Any]
    ) -> tuple[nn.Module, int, int] | None:
        if "proj.weight" not in mlp_state or "head.weight" not in mlp_state:
            return None
        in_d = int(mlp_state["proj.weight"].shape[1])
        hid_d = int(mlp_state["proj.weight"].shape[0])
        out_d = int(mlp_state["head.weight"].shape[0])
        mod = _ToyLayerNormTrunk(in_d, hid_d, out_d)
        mod.load_state_dict(mlp_state)
        return mod, in_d, out_d

    with pytest.raises(ValueError, match="non-empty"):
        register_actor_trunk_builder("", _toy_builder)

    register_actor_trunk_builder("test.toy", _toy_builder)
    try:
        loaded = load_rslrl_policy(ckpt, act_dim)
        assert loaded.obs_dim == obs_dim
        sample = torch.randn(5, obs_dim)
        with torch.no_grad():
            expected = trunk(norm(sample))
            torch.testing.assert_close(loaded(sample), expected, atol=1e-6, rtol=1e-6)

        if (
            importlib.util.find_spec("onnx") is not None
            and importlib.util.find_spec("onnxruntime") is not None
        ):
            import onnxruntime as ort

            onnx_path = tmp_path / "toy_policy.onnx"
            export_rslrl_to_onnx(ckpt, onnx_path, obs_dim, act_dim)
            sess = ort.InferenceSession(
                str(onnx_path), providers=["CPUExecutionProvider"]
            )
            actual = sess.run(["action"], {"obs": sample.numpy()})[0]
            np.testing.assert_allclose(actual, expected.numpy(), atol=1e-5, rtol=1e-5)
    finally:
        unregister_actor_trunk_builder("test.toy")

    with pytest.raises(ValueError, match="No mlp.<index>.weight layers"):
        load_rslrl_policy(ckpt, act_dim)
