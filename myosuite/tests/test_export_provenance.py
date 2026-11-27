"""Real CLI export/verification carries and enforces deployment provenance."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from myosuite import make_env
from myosuite.utils.deployment_manifest import artifact_record

pytestmark = pytest.mark.tier1


def test_cli_export_and_verification_authenticate_the_real_onnx(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sb3 = pytest.importorskip("stable_baselines3")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from myosuite.utils import export_onnx

    env_id = "myoElbowPose1D6MFixed-v0"
    checkpoint = tmp_path / "checkpoint.zip"
    output = tmp_path / "policy.onnx"
    with make_env(env_id) as env:
        sb3.PPO(
            "MlpPolicy",
            env,
            n_steps=8,
            batch_size=8,
            policy_kwargs={"net_arch": [8]},
            device="cpu",
        ).save(checkpoint)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export",
            "export",
            "--framework",
            "sb3",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--obs-dim",
            "9",
            "--act-dim",
            "6",
            "--env",
            env_id,
        ],
    )
    export_onnx.main()
    manifest = json.loads(Path(str(output) + ".manifest.json").read_text())
    assert manifest["checkpoint"] == artifact_record(checkpoint)
    assert manifest["artifacts"][output.name] == artifact_record(output)
    assert manifest["normalization"]["input"] == "raw"
    assert export_onnx.verify_onnx_on_cpu(output, 2, env_id=env_id)["n_steps"] == 2
    # A bad graph must be rejected before ONNX Runtime attempts deserialization.
    output.write_bytes(b"not a valid graph")
    called = []
    monkeypatch.setattr(
        "myosuite.utils.onnxruntime_utils.require_onnxruntime",
        lambda: SimpleNamespace(InferenceSession=lambda *a, **k: called.append(a)),
    )
    with pytest.raises(ValueError, match="artifact policy.onnx"):
        export_onnx.verify_onnx_on_cpu(output, 2, env_id=env_id)
    assert not called


def test_provenance_export_refuses_unpinned_hf_before_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from myosuite.utils import export_onnx

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export",
            "export",
            "--framework",
            "orbax",
            "--checkpoint",
            "hf://owner/repo",
            "--output",
            "x.onnx",
            "--env",
            "myoMimic-v0",
        ],
    )
    called = []
    monkeypatch.setattr(
        export_onnx, "export_orbax_to_onnx", lambda *a: called.append(a)
    )
    with pytest.raises(SystemExit):
        export_onnx.main()
    assert not called
