# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Optional ONNX Runtime loader and lightweight actor inference session."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


def require_onnxruntime() -> ModuleType:
    """Return ONNX Runtime via :func:`~myosuite.core.import_utils.require_extra` with CPU/GPU guidance.

    Raises:
        ImportError: If ONNX Runtime is not installed (re-raises internal import failures untouched).
    """
    from myosuite.core.import_utils import require_extra

    return require_extra(
        "onnxruntime",
        "onnx",
        "ONNX inference",
        hint=(
            "(CPU), or install onnxruntime-gpu. "
            "If no wheel is available for your Python version, use an "
            "interpreter supported by ONNX Runtime or build it from source. "
            "Native MyoSuite CPU environments do not require ONNX Runtime."
        ),
    )


@dataclass
class OnnxActorSession:
    """Thin wrapper around an ``onnxruntime`` session for batched actor inference.

    Keeps the session and input/output names together so callers do not need
    to know ORT internals. Inference runs on whichever provider was selected
    at load time (CPU or CUDA).

    Args:
        _session: Underlying ``onnxruntime.InferenceSession``.
        obs_dim: Expected observation dimension.
        act_dim: Output action dimension.
    """

    _session: Any
    obs_dim: int
    act_dim: int
    _input_name: str = field(repr=False)
    _output_name: str = field(repr=False)

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Run batched inference.

        Args:
            obs: Raw observations, shape ``(N, obs_dim)`` or ``(obs_dim,)``, dtype ``float32``.

        Returns:
            Deterministic mean actions, shape ``(N, act_dim)``, dtype ``float32``.

        Raises:
            ValueError: If ``obs`` last dimension does not match ``obs_dim``.
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
        """Convenience wrapper accepting and returning CPU torch tensors.

        Args:
            obs: Shape ``(N, obs_dim)``.

        Returns:
            Shape ``(N, act_dim)``.
        """
        import torch

        result = self.act(obs.detach().cpu().numpy())
        return torch.as_tensor(result, dtype=torch.float32)


def load_onnx_session(
    path: Path | str,
    *,
    providers: list[str] | None = None,
) -> OnnxActorSession:
    """Load an ONNX actor from *path* into an ``onnxruntime`` session.

    Args:
        path: Path to ``.onnx`` file.
        providers: ORT execution providers (defaults to CUDA then CPU).

    Returns:
        :class:`OnnxActorSession` ready for :meth:`~OnnxActorSession.act`.

    Raises:
        ImportError: If ``onnxruntime`` is not installed.
        FileNotFoundError: If *path* does not exist.
    """
    ort = require_onnxruntime()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX actor file not found: {path}")
    if providers is None:
        available = ort.get_available_providers()
        providers = [
            p
            for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if p in available
        ] or ["CPUExecutionProvider"]
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
        data: Raw ONNX protobuf bytes from :func:`~myosuite.integrations.musclemimic.actor_onnx.actor_to_onnx_bytes`.
        providers: ORT execution providers (same default as :func:`load_onnx_session`).

    Returns:
        :class:`OnnxActorSession` ready for inference.

    Raises:
        ImportError: If ``onnxruntime`` is not installed.
    """
    ort = require_onnxruntime()
    if providers is None:
        available = ort.get_available_providers()
        providers = [
            p
            for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if p in available
        ] or ["CPUExecutionProvider"]
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
    "load_onnx_session",
    "load_onnx_session_from_bytes",
    "require_onnxruntime",
]
