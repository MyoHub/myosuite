# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Helpers for bundling resumable training checkpoints into ONNX files.

The exported ONNX graph remains directly usable for inference, while a compressed
copy of the framework-native training checkpoint is embedded in the model
metadata so scripts can resume training from the same ``.onnx`` file.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any

import wandb

from myosuite.integrations.musclemimic.actor_onnx import load_onnx_session

_BUNDLE_META_KEY = "myosuite.checkpoint_bundle.v1.meta"
_BUNDLE_PAYLOAD_KEY = "myosuite.checkpoint_bundle.v1.payload_gzip_base64"
_MODEL_STEP_ONNX_RE = re.compile(r"^model_(\d+)\.onnx$")
_MODEL_STEP_PT_RE = re.compile(r"^model_(\d+)\.pt$")


def _load_model_props(onnx_path: Path) -> tuple[Any, dict[str, str]]:
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    props = {entry.key: entry.value for entry in model.metadata_props}
    return model, props


def bundle_onnx_with_checkpoint(
    onnx_path: str | Path,
    checkpoint_path: str | Path,
    *,
    framework: str,
    metadata: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Embed a framework-native checkpoint inside an ONNX file.

    Args:
        onnx_path: Existing ONNX model path.
        checkpoint_path: Native checkpoint file to embed.
        framework: Training stack identifier, e.g. ``"sb3-ppo"``.
        metadata: Extra JSON-serializable metadata to persist.
        output_path: Optional destination path. Defaults to in-place update.

    Returns:
        Resolved path to the bundled ONNX file.
    """
    import onnx

    source_onnx = Path(onnx_path)
    checkpoint = Path(checkpoint_path)
    target = Path(output_path) if output_path is not None else source_onnx
    target.parent.mkdir(parents=True, exist_ok=True)

    model, props = _load_model_props(source_onnx)
    payload = checkpoint.read_bytes()
    meta = {
        "framework": framework,
        "checkpoint_name": checkpoint.name,
        "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
        "metadata": metadata or {},
    }
    props[_BUNDLE_META_KEY] = json.dumps(meta, sort_keys=True)
    props[_BUNDLE_PAYLOAD_KEY] = base64.b64encode(gzip.compress(payload)).decode(
        "ascii"
    )
    onnx.helper.set_model_props(model, props)
    onnx.save_model(model, str(target), save_as_external_data=False)
    return target.resolve()


def read_onnx_checkpoint_metadata(onnx_path: str | Path) -> dict[str, Any]:
    """Return embedded checkpoint metadata from an ONNX file."""
    _, props = _load_model_props(Path(onnx_path))
    if _BUNDLE_META_KEY not in props:
        raise ValueError(f"ONNX checkpoint bundle metadata missing in {onnx_path}.")
    return json.loads(props[_BUNDLE_META_KEY])


def extract_checkpoint_from_onnx(
    onnx_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> tuple[Path, dict[str, Any], tempfile.TemporaryDirectory[str] | None]:
    """Extract the embedded native checkpoint from an ONNX bundle.

    Args:
        onnx_path: ONNX bundle created by :func:`bundle_onnx_with_checkpoint`.
        output_path: Optional extraction destination. When omitted, a temporary
            file is created and its owning temp directory is returned.

    Returns:
        Tuple of ``(checkpoint_path, metadata, temp_dir)``. ``temp_dir`` is only
        non-``None`` when *output_path* was omitted.
    """
    props_path = Path(onnx_path)
    meta = read_onnx_checkpoint_metadata(props_path)
    _, props = _load_model_props(props_path)
    if _BUNDLE_PAYLOAD_KEY not in props:
        raise ValueError(f"ONNX checkpoint payload missing in {onnx_path}.")

    payload = gzip.decompress(
        base64.b64decode(props[_BUNDLE_PAYLOAD_KEY].encode("ascii"))
    )
    payload_sha = hashlib.sha256(payload).hexdigest()
    if payload_sha != meta["checkpoint_sha256"]:
        raise ValueError(
            f"ONNX checkpoint payload hash mismatch for {onnx_path}: "
            f"expected {meta['checkpoint_sha256']}, got {payload_sha}."
        )

    if output_path is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="myosuite-onnx-ckpt-")
        destination = Path(temp_dir.name) / str(meta["checkpoint_name"])
    else:
        temp_dir = None
        destination = Path(output_path)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return destination.resolve(), meta, temp_dir


def normalize_onnx_checkpoint_name(checkpoint_name: str) -> str:
    """Normalize legacy checkpoint names to their ONNX bundle equivalents."""
    if checkpoint_name == "model_final.pt":
        return "model_final.onnx"
    match = _MODEL_STEP_PT_RE.fullmatch(checkpoint_name)
    if match is not None:
        return f"model_{match.group(1)}.onnx"
    return checkpoint_name


def is_onnx_checkpoint_name(checkpoint_name: str) -> bool:
    """Return whether *checkpoint_name* matches the ONNX checkpoint convention."""
    return checkpoint_name == "model_final.onnx" or (
        _MODEL_STEP_ONNX_RE.fullmatch(checkpoint_name) is not None
    )


def onnx_checkpoint_sort_key(checkpoint_name: str) -> tuple[int, int]:
    """Sort numeric checkpoints by step and keep ``model_final.onnx`` last."""
    if checkpoint_name == "model_final.onnx":
        return (1, 0)
    match = _MODEL_STEP_ONNX_RE.fullmatch(checkpoint_name)
    if match is None:
        return (-1, -1)
    return (0, int(match.group(1)))


def get_wandb_onnx_checkpoint_path(
    log_path: Path,
    run_path: Path,
    checkpoint_name: str | None = None,
) -> tuple[Path, bool]:
    run_id = str(run_path).split("/")[-1]
    download_dir = log_path / "wandb_checkpoints" / run_id
    api = wandb.Api()
    wandb_run = api.run(str(run_path))
    files = [
        file.name for file in wandb_run.files() if is_onnx_checkpoint_name(file.name)
    ]
    if not files:
        raise FileNotFoundError(f"No ONNX checkpoints found in W&B run {run_path}.")
    if checkpoint_name is None:
        checkpoint_file = max(files, key=onnx_checkpoint_sort_key)
    else:
        checkpoint_name = normalize_onnx_checkpoint_name(checkpoint_name)
        if checkpoint_name not in files:
            raise ValueError(
                f"Checkpoint '{checkpoint_name}' not found in run {run_path}. "
                f"Available: {files}"
            )
        checkpoint_file = checkpoint_name
    checkpoint_path = download_dir / checkpoint_file
    was_cached = checkpoint_path.exists()
    if not was_cached:
        download_dir.mkdir(parents=True, exist_ok=True)
        wandb_run.file(checkpoint_file).download(str(download_dir), replace=True)
    return checkpoint_path, was_cached


class OnnxPolicy:
    def __init__(self, onnx_path: Path, device: str) -> None:
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device.startswith("cuda")
            else ["CPUExecutionProvider"]
        )
        self._session = load_onnx_session(onnx_path, providers=providers)
        self._device = device

    def __call__(self, obs):
        actor_obs = obs["actor"] if "actor" in obs.keys() else obs
        return self._session.act_torch(actor_obs).to(device=self._device)

    def reset(self) -> None:
        return None


__all__ = [
    "bundle_onnx_with_checkpoint",
    "extract_checkpoint_from_onnx",
    "read_onnx_checkpoint_metadata",
    "get_wandb_onnx_checkpoint_path",
    "is_onnx_checkpoint_name",
    "normalize_onnx_checkpoint_name",
    "onnx_checkpoint_sort_key",
    "OnnxPolicy",
]
