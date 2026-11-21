# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for atomic ONNX staging and policy_spec.json sidecar generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from myosuite.utils.export_onnx import _build_parser, _write_policy_spec

pytestmark = pytest.mark.tier1


def test_policy_spec_sidecar_and_cli_flags(tmp_path: Path) -> None:
    dummy_ckpt = tmp_path / "model_50.pt"
    dummy_ckpt.write_bytes(b"synthetic-checkpoint-bytes")
    onnx_path = tmp_path / "policy.onnx"

    spec_path = _write_policy_spec(
        onnx_path,
        obs_dim=36,
        act_dim=80,
        checkpoint=dummy_ckpt,
        framework="rslrl",
    )
    assert spec_path == tmp_path / "policy.policy_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert spec["schema_version"] == 1
    assert spec["observations"]["dim"] == 36
    assert spec["actions"]["dim"] == 80
    input_names = [i["name"] for i in spec["model"]["onnx"]["inputs"]]
    output_names = [o["name"] for o in spec["model"]["onnx"]["outputs"]]
    assert input_names == ["obs"]
    assert output_names == ["action"]
    assert len(spec["provenance"]["checkpoint"]["sha256"]) == 64

    args = _build_parser().parse_args(
        [
            "export",
            "--framework",
            "rslrl",
            "--checkpoint",
            str(dummy_ckpt),
            "--output",
            str(onnx_path),
            "--obs-dim",
            "36",
            "--act-dim",
            "80",
            "--no-verify",
            "--policy-spec",
        ]
    )
    assert args.no_verify is True
    assert args.policy_spec is True


def test_atomic_staging_parity_and_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    from myosuite.utils import export_onnx
    from myosuite.utils.export_onnx import _export_policy, verify_onnx_parity

    mlp = torch.nn.Sequential(torch.nn.Linear(6, 4), torch.nn.Tanh()).eval()
    good_out = tmp_path / "good.onnx"
    _export_policy(
        mlp, obs_dim=6, output=good_out, opset=17, verify=True, write_spec=True
    )
    assert good_out.is_file()
    assert good_out.with_suffix(".policy_spec.json").is_file()
    assert verify_onnx_parity(mlp, good_out, obs_dim=6) < 1e-4

    target = tmp_path / "existing.onnx"
    sentinel = b"KEEP-EXISTING-ONNX"
    target.write_bytes(sentinel)

    def _boom(*_a: object, **_kw: object) -> float:
        raise RuntimeError("simulated parity mismatch")

    monkeypatch.setattr(export_onnx, "verify_onnx_parity", _boom)
    with pytest.raises(RuntimeError, match="simulated parity mismatch"):
        _export_policy(mlp, obs_dim=6, output=target, opset=17, verify=True)
    assert target.read_bytes() == sentinel
