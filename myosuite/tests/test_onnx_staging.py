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


def test_missing_onnxruntime_fails_unless_no_verify_and_preserves_previous_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")

    import hashlib

    from myosuite.utils import onnxruntime_utils
    from myosuite.utils.export_onnx import _export_policy

    mlp = torch.nn.Sequential(torch.nn.Linear(6, 4), torch.nn.Tanh()).eval()
    target = tmp_path / "policy.onnx"
    spec_path = target.with_suffix(".policy_spec.json")
    prev_onnx = b"PREVIOUS-VERIFIED-ONNX"
    prev_spec = b'{"previous": true}\n'
    target.write_bytes(prev_onnx)
    spec_path.write_bytes(prev_spec)

    def _missing_ort() -> object:
        raise ImportError("ONNX inference requires onnxruntime.")

    monkeypatch.setattr(onnxruntime_utils, "require_onnxruntime", _missing_ort)
    with pytest.raises(ImportError, match="--no-verify"):
        _export_policy(
            mlp, obs_dim=6, output=target, opset=17, verify=True, write_spec=True
        )
    assert target.read_bytes() == prev_onnx
    assert spec_path.read_bytes() == prev_spec

    _export_policy(
        mlp, obs_dim=6, output=target, opset=17, verify=False, write_spec=True
    )
    assert target.read_bytes() != prev_onnx
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert (
        spec["model"]["onnx"]["sha256"]
        == hashlib.sha256(target.read_bytes()).hexdigest()
    )


def test_sidecar_write_failure_keeps_graph_and_provenance_consistent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")

    from myosuite.utils import export_onnx
    from myosuite.utils.export_onnx import _export_policy

    torch.manual_seed(0)
    mlp_v1 = torch.nn.Sequential(torch.nn.Linear(6, 4), torch.nn.Tanh()).eval()
    ckpt_v1 = tmp_path / "v1.pt"
    ckpt_v1.write_bytes(b"ckpt-v1")
    target = tmp_path / "policy.onnx"
    spec_path = target.with_suffix(".policy_spec.json")

    _export_policy(
        mlp_v1,
        obs_dim=6,
        output=target,
        opset=17,
        verify=True,
        write_spec=True,
        checkpoint=ckpt_v1,
    )
    v1_onnx = target.read_bytes()
    v1_spec = spec_path.read_bytes()

    torch.manual_seed(99)
    mlp_v2 = torch.nn.Sequential(torch.nn.Linear(6, 4), torch.nn.Tanh()).eval()
    ckpt_v2 = tmp_path / "v2.pt"
    ckpt_v2.write_bytes(b"ckpt-v2")
    real_replace = export_onnx._replace_file

    def _fail_on_spec_commit(src: Path, dst: Path) -> None:
        if dst == spec_path and src.name == spec_path.name:
            raise OSError("simulated sidecar commit failure")
        real_replace(src, dst)

    monkeypatch.setattr(export_onnx, "_replace_file", _fail_on_spec_commit)
    with pytest.raises(OSError, match="simulated sidecar commit failure"):
        _export_policy(
            mlp_v2,
            obs_dim=6,
            output=target,
            opset=17,
            verify=True,
            write_spec=True,
            checkpoint=ckpt_v2,
        )
    assert target.read_bytes() == v1_onnx
    assert spec_path.read_bytes() == v1_spec

    monkeypatch.undo()
    _export_policy(
        mlp_v2, obs_dim=6, output=target, opset=17, verify=True, write_spec=False
    )
    assert target.read_bytes() != v1_onnx
    assert not spec_path.exists()
