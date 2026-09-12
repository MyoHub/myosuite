from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

import myosuite.utils.onnx_checkpoint as onnx_checkpoint
from myosuite.utils.onnx_checkpoint import (
    bundle_onnx_with_checkpoint,
    extract_checkpoint_from_onnx,
    get_wandb_onnx_checkpoint_path,
    normalize_onnx_checkpoint_name,
    read_onnx_checkpoint_metadata,
)


def test_onnx_checkpoint_bundle_round_trip() -> None:
    pytest.importorskip("onnx")

    class _Tiny(torch.nn.Module):
        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            return obs + 1.0

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        onnx_path = tmp / "policy.onnx"
        ckpt_path = tmp / "policy_state.bin"
        ckpt_path.write_bytes(b"resume-me")
        torch.onnx.export(
            _Tiny(),
            torch.zeros(1, 3, dtype=torch.float32),
            str(onnx_path),
            input_names=["obs"],
            output_names=["action"],
            dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )

        bundle_onnx_with_checkpoint(
            onnx_path=onnx_path,
            checkpoint_path=ckpt_path,
            framework="unit-test",
            metadata={"step": 123},
        )

        meta = read_onnx_checkpoint_metadata(onnx_path)
        assert meta["framework"] == "unit-test"
        assert meta["metadata"]["step"] == 123

        extracted, extracted_meta, temp_dir = extract_checkpoint_from_onnx(onnx_path)
        try:
            assert extracted.read_bytes() == b"resume-me"
            assert extracted_meta["checkpoint_name"] == "policy_state.bin"
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()


def test_normalize_onnx_checkpoint_name_accepts_pt_aliases() -> None:
    assert normalize_onnx_checkpoint_name("model_42.pt") == "model_42.onnx"
    assert normalize_onnx_checkpoint_name("model_final.pt") == "model_final.onnx"
    assert normalize_onnx_checkpoint_name("custom.onnx") == "custom.onnx"


def test_get_wandb_onnx_checkpoint_path_prefers_final_and_normalizes_alias(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _FakeFile:
        def __init__(self, name: str) -> None:
            self.name = name

        def download(self, root: str, replace: bool = True) -> None:
            del replace
            target = Path(root) / self.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.name.encode("utf-8"))

    class _FakeRun:
        def __init__(self, names: list[str]) -> None:
            self._files = [_FakeFile(name) for name in names]

        def files(self) -> list[_FakeFile]:
            return list(self._files)

        def file(self, name: str) -> _FakeFile:
            for file in self._files:
                if file.name == name:
                    return file
            raise KeyError(name)

    class _FakeApi:
        def __init__(self, run: _FakeRun) -> None:
            self._run = run

        def run(self, path: str) -> _FakeRun:
            assert path == "org/project/run-id"
            return self._run

    fake_run = _FakeRun(["model_10.onnx", "model_20.onnx", "model_final.onnx"])
    monkeypatch.setattr(onnx_checkpoint.wandb, "Api", lambda: _FakeApi(fake_run))

    latest_path, was_cached = get_wandb_onnx_checkpoint_path(
        tmp_path, Path("org/project/run-id")
    )
    assert latest_path.name == "model_final.onnx"
    assert was_cached is False
    assert latest_path.read_bytes() == b"model_final.onnx"

    aliased_path, was_cached = get_wandb_onnx_checkpoint_path(
        tmp_path,
        Path("org/project/run-id"),
        checkpoint_name="model_20.pt",
    )
    assert aliased_path.name == "model_20.onnx"
    assert was_cached is False
    assert aliased_path.read_bytes() == b"model_20.onnx"
