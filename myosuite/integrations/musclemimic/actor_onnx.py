# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""ONNX export and inference session for the MuscleMimic actor.

Exports :class:`~myosuite.integrations.musclemimic.actor_torch.MimicActorModule`
to an ONNX file with a dynamic batch axis so it can be loaded by
``onnxruntime`` for efficient batched CPU/GPU inference.

Typical usage::

    from myosuite.integrations.musclemimic.fullbody_local_policy import (
        load_local_policy_artifacts,
    )
    from myosuite.integrations.musclemimic.actor_torch import MimicActorModule
    from myosuite.integrations.musclemimic.actor_onnx import (
        export_to_onnx,
        load_onnx_session,
    )

    artifacts = load_local_policy_artifacts(checkpoint_root)
    module = MimicActorModule.from_artifacts(artifacts)

    onnx_path = export_to_onnx(module, checkpoint_root / "actor.onnx")
    session = load_onnx_session(onnx_path)

    # Batched inference: (N, obs_dim) → (N, act_dim)
    obs_np = np.zeros((N, artifacts.obs_dim), dtype=np.float32)
    actions = session.act(obs_np)
"""

from __future__ import annotations

import logging
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from myosuite.integrations.musclemimic.actor_torch import (
        MimicActorModule,
        DenseActorModule,
    )

    _TorchActor = MimicActorModule | DenseActorModule
else:
    _TorchActor = Any

logger = logging.getLogger(__name__)

_OPSET = 17


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_to_onnx(
    module: _TorchActor,
    path: Path | str,
    *,
    opset: int = _OPSET,
) -> Path:
    """Export *module* to an ONNX file at *path*.

    The exported model accepts raw (un-normalised) observations; obs
    normalisation using the checkpoint's stored stats is fused into the graph.

    Args:
        module: :class:`~...MimicActorModule` in eval mode.
        path: Destination ``.onnx`` file path.
        opset: ONNX opset version (default 17).

    Returns:
        Resolved path to the written ``.onnx`` file.

    Raises:
        ImportError: If ``torch.onnx`` cannot serialise the model (unexpected).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if importlib.util.find_spec("onnx") is None:
        raise ImportError(
            "ONNX export requires the `onnx` package. Install it with: pip install onnx"
        )

    module = module.eval().cpu()
    dummy_obs = torch.zeros(1, module.obs_dim, dtype=torch.float32)

    torch.onnx.export(
        module,
        dummy_obs,
        str(path),
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "batch_size"}, "actions": {0: "batch_size"}},
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    logger.info(
        "Exported MimicActorModule to %s (obs_dim=%d, act_dim=%d, opset=%d)",
        path,
        module.obs_dim,
        module.act_dim,
        opset,
    )
    return path.resolve()


def actor_to_onnx_bytes(module: _TorchActor, *, opset: int = _OPSET) -> bytes:
    """Serialize *module* to ONNX bytes without writing a file.

    Useful for in-memory testing or embedding the model in a checkpoint.

    Args:
        module: Eval-mode :class:`~...MimicActorModule`.
        opset: ONNX opset version.

    Returns:
        Raw ONNX protobuf bytes.
    """
    import tempfile

    module = module.eval().cpu()
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmp_path = Path(f.name)
    try:
        export_to_onnx(module, tmp_path, opset=opset)
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Inference session
# ---------------------------------------------------------------------------


@dataclass
class OnnxActorSession:
    """Thin wrapper around an ``onnxruntime`` session for actor inference.

    Keeps the session and input/output names together so callers do not need
    to know ORT internals.  Inference runs on whichever provider was selected
    at load time (CPU or CUDA).

    Args:
        _session: Underlying ``onnxruntime.InferenceSession``.
        obs_dim: Expected observation dimension.
        act_dim: Output action dimension.
    """

    _session: object  # onnxruntime.InferenceSession
    obs_dim: int
    act_dim: int
    _input_name: str = field(repr=False)
    _output_name: str = field(repr=False)

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Run batched inference.

        Args:
            obs: Raw observations, shape ``(N, obs_dim)``, dtype ``float32``.

        Returns:
            Deterministic mean actions, shape ``(N, act_dim)``, dtype ``float32``.

        Raises:
            ValueError: If ``obs`` dtype is not ``float32`` or shape is wrong.
        """
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[np.newaxis, :]
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(
                f"Expected obs last dim {self.obs_dim}, got {obs.shape[-1]}"
            )
        outputs = self._session.run([self._output_name], {self._input_name: obs})
        return np.asarray(outputs[0], dtype=np.float32)

    def act_torch(self, obs: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper accepting and returning torch tensors.

        Moves to CPU for ORT inference then returns a CPU tensor.

        Args:
            obs: Shape ``(N, obs_dim)``.

        Returns:
            Shape ``(N, act_dim)``.
        """
        result = self.act(obs.detach().cpu().numpy())
        return torch.as_tensor(result, dtype=torch.float32)


def load_onnx_session(
    path: Path | str,
    *,
    providers: list[str] | None = None,
) -> OnnxActorSession:
    """Load an ONNX actor from *path* into an ``onnxruntime`` session.

    Args:
        path: Path to ``.onnx`` file written by :func:`export_to_onnx`.
        providers: ORT execution providers.  Defaults to
                   ``["CUDAExecutionProvider", "CPUExecutionProvider"]``.

    Returns:
        :class:`OnnxActorSession` ready for :meth:`~OnnxActorSession.act`.

    Raises:
        ImportError: If ``onnxruntime`` is not installed.
        FileNotFoundError: If *path* does not exist.
    """
    try:
        import onnxruntime as ort
    except ImportError as err:
        raise ImportError(
            "onnxruntime is required to load ONNX actors: "
            "pip install onnxruntime  (CPU)  or  onnxruntime-gpu  (CUDA)"
        ) from err

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX actor file not found: {path}")

    if providers is None:
        available = ort.get_available_providers()
        providers = [
            p
            for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if p in available
        ]
        if not providers:
            providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(str(path), providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    obs_dim = int(session.get_inputs()[0].shape[1])
    act_dim = int(session.get_outputs()[0].shape[1])

    logger.info(
        "Loaded ONNX actor from %s (obs_dim=%d, act_dim=%d, providers=%s)",
        path,
        obs_dim,
        act_dim,
        providers,
    )
    return OnnxActorSession(
        _session=session,
        obs_dim=obs_dim,
        act_dim=act_dim,
        _input_name=input_name,
        _output_name=output_name,
    )


def load_onnx_session_from_bytes(
    data: bytes,
    *,
    providers: list[str] | None = None,
) -> OnnxActorSession:
    """Load an ONNX session directly from bytes (no file needed).

    Args:
        data: Raw ONNX protobuf bytes from :func:`actor_to_onnx_bytes`.
        providers: ORT execution providers (same default as :func:`load_onnx_session`).

    Returns:
        :class:`OnnxActorSession` ready for inference.
    """
    try:
        import onnxruntime as ort
    except ImportError as err:
        raise ImportError("onnxruntime is required: pip install onnxruntime") from err

    if providers is None:
        available = ort.get_available_providers()
        providers = [
            p
            for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if p in available
        ]
        if not providers:
            providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(data, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    obs_dim = int(session.get_inputs()[0].shape[1])
    act_dim = int(session.get_outputs()[0].shape[1])
    return OnnxActorSession(
        _session=session,
        obs_dim=obs_dim,
        act_dim=act_dim,
        _input_name=input_name,
        _output_name=output_name,
    )


__all__ = [
    "OnnxActorSession",
    "actor_to_onnx_bytes",
    "export_to_onnx",
    "load_onnx_session",
    "load_onnx_session_from_bytes",
]
