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

import copy
import importlib.util
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from myosuite.policies.onnxruntime_utils import (
    OnnxActorSession,
    load_onnx_session,
    load_onnx_session_from_bytes,
)


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

    import torch

    module = copy.deepcopy(module).eval().cpu()
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

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmp_path = Path(f.name)
    try:
        export_to_onnx(module, tmp_path, opset=opset)
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)


__all__ = [
    "OnnxActorSession",
    "actor_to_onnx_bytes",
    "export_to_onnx",
    "load_onnx_session",
    "load_onnx_session_from_bytes",
]
