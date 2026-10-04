# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Batched mjlab policy runner for MuscleMimic actor inference.

Wraps either a :class:`~...MimicActorModule` (PyTorch, GPU-native) or
an :class:`~...OnnxActorSession` (onnxruntime, CPU or CUDA) behind a unified
interface that accepts ``(N, obs_dim)`` torch tensors and returns
``(N, act_dim)`` torch tensors — matching the mjlab training loop's data
flow.

Usage with torch backend (GPU-native, recommended for mjlab)::

    from myosuite.integrations.musclemimic.actor_torch import MimicActorModule
    from myosuite.integrations.musclemimic.mjlab_policy_runner import (
        MjlabPolicyRunner,
    )

    module = MimicActorModule.from_artifacts(artifacts).to("cuda")
    runner = MjlabPolicyRunner.from_torch(module)

    obs = torch.zeros(1024, artifacts.obs_dim, device="cuda")
    actions = runner.act(obs)   # (1024, act_dim), clamped to [-1, 1]

Usage with ONNX backend::

    from myosuite.integrations.musclemimic.actor_onnx import load_onnx_session
    from myosuite.integrations.musclemimic.mjlab_policy_runner import (
        MjlabPolicyRunner,
    )

    session = load_onnx_session("actor.onnx")
    runner = MjlabPolicyRunner.from_onnx(session)

    obs = torch.zeros(1024, session.obs_dim, device="cpu")
    actions = runner.act(obs)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

# Re-exported: the batched full-body obs builder used to live in this module.
from myosuite.integrations.musclemimic.fullbody_obs_torch import (
    TorchFullbodyObsAdapter,
)

if TYPE_CHECKING:
    from myosuite.integrations.musclemimic.actor_onnx import OnnxActorSession
    from myosuite.integrations.musclemimic.actor_torch import (
        MimicActorModule,
        DenseActorModule,
    )

    _TorchActor = MimicActorModule | DenseActorModule
else:
    _TorchActor = Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol-like ABC (no ABC dep to keep things simple)
# ---------------------------------------------------------------------------


class _ActorBackend:
    """Internal interface for actor backends."""

    def infer(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class _TorchBackend(_ActorBackend):
    def __init__(self, module: _TorchActor) -> None:
        self._module = module

    def infer(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._module(obs)


class _OnnxBackend(_ActorBackend):
    def __init__(self, session: OnnxActorSession) -> None:
        self._session = session

    def infer(self, obs: torch.Tensor) -> torch.Tensor:
        # ORT runs on CPU; move back to obs device afterwards
        device = obs.device
        actions_np = self._session.act(obs.detach().cpu().numpy())
        return torch.as_tensor(actions_np, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------


@dataclass
class MjlabPolicyRunner:
    """Batched inference runner compatible with the mjlab training loop.

    Accepts ``(N, obs_dim)`` observations on any device and returns
    ``(N, act_dim)`` actions on the same device, clamped to ``[-1, 1]``.

    Do not construct directly; use :meth:`from_torch` or :meth:`from_onnx`.

    Args:
        obs_dim: Expected observation dimension.
        act_dim: Output action dimension.
        _backend: Internal actor backend (torch or ONNX).
    """

    obs_dim: int
    act_dim: int
    _backend: _ActorBackend

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_torch(cls, module: _TorchActor) -> MjlabPolicyRunner:
        """Create a runner backed by a :class:`~...MimicActorModule`.

        The module should already be on the target device and in eval mode.

        Args:
            module: Eval-mode torch actor module.

        Returns:
            :class:`MjlabPolicyRunner` using the torch backend.
        """
        return cls(
            obs_dim=module.obs_dim,
            act_dim=module.act_dim,
            _backend=_TorchBackend(module),
        )

    @classmethod
    def from_onnx(cls, session: OnnxActorSession) -> MjlabPolicyRunner:
        """Create a runner backed by an :class:`~...OnnxActorSession`.

        Args:
            session: Loaded ONNX session from :func:`~...load_onnx_session`.

        Returns:
            :class:`MjlabPolicyRunner` using the ONNX backend.
        """
        return cls(
            obs_dim=session.obs_dim,
            act_dim=session.act_dim,
            _backend=_OnnxBackend(session),
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Return deterministic actions for a batch of observations.

        Args:
            obs: Raw observations, shape ``(N, obs_dim)``, any device.
                 Will be cast to ``float32`` if needed.

        Returns:
            Actions clamped to ``[-1, 1]``, shape ``(N, act_dim)``,
            on the same device as *obs*.

        Raises:
            ValueError: If ``obs`` last dimension does not match ``obs_dim``.
        """
        if obs.dtype != torch.float32:
            obs = obs.float()
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(
                f"Expected obs last dim {self.obs_dim}, got {obs.shape[-1]}"
            )
        actions = self._backend.infer(obs)
        return torch.clamp(actions, -1.0, 1.0)

    def act_stochastic(
        self,
        obs: torch.Tensor,
        log_std: np.ndarray | torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample actions from the policy's Gaussian distribution.

        Args:
            obs: Raw observations, shape ``(N, obs_dim)``.
            log_std: Log standard deviations, shape ``(act_dim,)``.
            generator: Optional torch RNG for reproducibility.

        Returns:
            Sampled actions clamped to ``[-1, 1]``, shape ``(N, act_dim)``.
        """
        mean = self._backend.infer(obs if obs.dtype == torch.float32 else obs.float())
        log_std_t = torch.as_tensor(
            np.asarray(log_std, dtype=np.float32), device=obs.device
        )
        std = torch.exp(torch.clamp(log_std_t, -20.0, 2.0))
        noise = torch.randn(
            mean.shape,
            dtype=mean.dtype,
            device=mean.device,
            generator=generator,
        )
        return torch.clamp(mean + noise * std, -1.0, 1.0)


__all__ = ["MjlabPolicyRunner", "TorchFullbodyObsAdapter"]


# ---------------------------------------------------------------------------
# ONNX-checkpointing runner and training utilities
# (moved from tutorials/mc26 mimic-init training scripts)
# ---------------------------------------------------------------------------

try:
    import tempfile
    from pathlib import Path

    import wandb
    from mjlab.rl import MjlabOnPolicyRunner
    from tensordict import TensorDict

    import torch.nn as nn

    from myosuite.integrations.musclemimic.actor_torch import (
        _load_dense_checkpoint_into_model,
    )
    from myosuite.integrations.musclemimic.fullbody_checkpoint_io import (
        resolve_checkpoint_ref,
    )
    from myosuite.integrations.musclemimic.fullbody_local_policy import (
        load_local_policy_artifacts,
    )
    from myosuite.utils.onnx_checkpoint import (
        _FATIGUE_STATE_KEY,
        bundle_onnx_with_checkpoint,
        extract_checkpoint_from_onnx,
        get_env_fatigue_state,
        set_env_fatigue_state,
    )

    _TRAINING_DEPS_AVAILABLE = True
except ImportError:
    _TRAINING_DEPS_AVAILABLE = False

    class MjlabOnPolicyRunner:  # type: ignore[no-redef]
        """Placeholder base so this module stays importable without mjlab.

        ``MjlabPolicyRunner`` inference needs only torch/onnxruntime, so the
        module must import on Pythons where the mjlab training stack is
        unavailable (e.g. 3.14). Only the training runner below is unusable.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "OnnxCheckpointingMjlabRunner requires the mjlab training stack; "
                "install it with `pip install -e '.[mjlab]'` (needs Python <3.14)."
            )


class _ActorExportWrapper(torch.nn.Module):
    """Wrap mjlab's Gaussian actor to export a deterministic mean-action ONNX."""

    def __init__(self, actor: torch.nn.Module) -> None:
        super().__init__()
        self.actor = actor

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Use the model's own obs_groups so the dummy TensorDict has the right key.
        # For models with a single group (the common case) this resolves to that
        # group name (e.g. "proprioception"); fall back to "actor" for legacy models.
        obs_groups = getattr(self.actor, "obs_groups", None)
        key = obs_groups[0] if obs_groups and len(obs_groups) == 1 else "actor"
        obs_td = TensorDict({key: obs}, batch_size=[obs.shape[0]])
        return self.actor(obs_td)


class OnnxCheckpointingMjlabRunner(MjlabOnPolicyRunner):
    """RSL-RL runner that saves ONNX bundles instead of raw ``.pt`` files.

    Extends :class:`~mjlab.rl.MjlabOnPolicyRunner` so that every
    ``runner.save(path)`` call produces an ONNX-bundled checkpoint that can be
    loaded back with :meth:`load_onnx` or evaluated with the standard myosuite
    ONNX playback tools.
    """

    def __init__(
        self,
        env: Any,
        train_cfg: dict[str, Any],
        log_dir: str | None,
        device: str,
        *,
        task_id: str = "",
    ) -> None:
        super().__init__(env=env, train_cfg=train_cfg, log_dir=log_dir, device=device)
        self._task_id = task_id
        first_layer = self.alg.actor.state_dict()["mlp.0.weight"]
        self._obs_dim = int(first_layer.shape[1])
        self._act_dim = int(getattr(self.env, "num_actions"))

    def _export_actor_onnx(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        actor_device = next(self.alg.actor.parameters()).device
        wrapper = _ActorExportWrapper(self.alg.actor)
        was_training = self.alg.actor.training
        self.alg.actor.eval()
        try:
            dummy_obs = torch.zeros(
                1, self._obs_dim, dtype=torch.float32, device=actor_device
            )
            torch.onnx.export(
                wrapper,
                dummy_obs,
                str(output_path),
                input_names=["obs"],
                output_names=["action"],
                dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
                opset_version=17,
                do_constant_folding=True,
                dynamo=False,
            )
        finally:
            if was_training:
                self.alg.actor.train()

    def save(self, path: str, infos: Any = None) -> None:
        native_path = Path(path)
        onnx_path = native_path.with_suffix(".onnx")
        with tempfile.TemporaryDirectory(prefix="onnx-ckpt-") as tmp_dir:
            tmp_root = Path(tmp_dir)
            tmp_native = tmp_root / native_path.name
            tmp_onnx = tmp_root / onnx_path.name
            env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
            saved_dict = self.alg.save()
            saved_dict["iter"] = self.current_learning_iteration
            saved_dict["infos"] = {**(infos or {}), "env_state": env_state}
            torch.save(saved_dict, tmp_native)
            self._export_actor_onnx(tmp_onnx)
            metadata: dict[str, Any] = {
                "task_id": self._task_id,
                "obs_dim": self._obs_dim,
                "act_dim": self._act_dim,
                "iteration": int(self.current_learning_iteration),
            }
            fatigue_state = get_env_fatigue_state(self.env)
            if fatigue_state is not None:
                metadata[_FATIGUE_STATE_KEY] = fatigue_state
            bundle_onnx_with_checkpoint(
                onnx_path=tmp_onnx,
                checkpoint_path=tmp_native,
                framework="mjlab-rslrl",
                metadata=metadata,
                output_path=onnx_path,
            )
            if self.cfg["upload_model"] and wandb.run is not None:
                self.logger.save_model(str(onnx_path), self.current_learning_iteration)

    def load(self, path: str, **kwargs: Any) -> dict[str, Any]:
        """Load a checkpoint; delegates to :meth:`load_onnx` for ``.onnx`` bundles."""
        if Path(path).suffix == ".onnx":
            return self.load_onnx(path)
        return super().load(path, **kwargs)  # type: ignore[return-value]

    def load_onnx(self, path: str | Path) -> dict[str, Any]:
        """Load a previously saved ONNX bundle back into this runner."""
        checkpoint_path, meta, temp_dir = extract_checkpoint_from_onnx(path)
        try:
            result = self.load(str(checkpoint_path), map_location=self.device)
            fatigue_state = meta.get("metadata", {}).get(_FATIGUE_STATE_KEY)
            if fatigue_state is not None:
                set_env_fatigue_state(self.env, fatigue_state)
            return result
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()


# ---------------------------------------------------------------------------
# Convenience utilities for the mimic-init training workflow
# ---------------------------------------------------------------------------


def _initialize_runner_from_mimic_checkpoint(
    runner: OnnxCheckpointingMjlabRunner,
    checkpoint_root: Path,
) -> None:
    """Load actor and critic weights from a mimic Orbax checkpoint into *runner*."""
    checkpoint = resolve_checkpoint_ref(str(checkpoint_root))
    artifacts = load_local_policy_artifacts(checkpoint.local_path)
    _load_dense_checkpoint_into_model(
        runner.alg.actor,
        artifacts.params["actor"],
        obs_mean=artifacts.obs_mean,
        obs_var=artifacts.obs_var,
        obs_count=artifacts.obs_count,
    )
    _load_dense_checkpoint_into_model(
        runner.alg.critic,
        artifacts.params["critic"],
        obs_mean=artifacts.obs_mean,
        obs_var=artifacts.obs_var,
        obs_count=artifacts.obs_count,
    )


def _maybe_freeze_actor_std(
    runner: OnnxCheckpointingMjlabRunner, *, learn_actor_std: bool
) -> None:
    """Freeze the Gaussian exploration std unless ``learn_actor_std`` is ``True``."""
    if learn_actor_std:
        return
    distribution = getattr(runner.alg.actor, "distribution", None)
    if distribution is None:
        return
    for attr in ("std_param", "log_std_param"):
        param = getattr(distribution, attr, None)
        if isinstance(param, nn.Parameter):
            param.requires_grad_(False)
