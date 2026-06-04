# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""PyTorch re-implementation of the MuscleMimic actor network.

Converts Orbax/JAX parameter trees (loaded via
:func:`~myosuite.integrations.musclemimic.fullbody_local_policy.load_local_policy_artifacts`)
into a :class:`torch.nn.Module` suitable for GPU-batched inference in mjlab.

Architecture (mirrors ``_actor_forward`` in ``fullbody_local_policy.py``):

    obs  ->  [obs_norm]  ->  ResBlock_*  ->  optional Tail  ->  output

Each ``ResBlock`` contains:
  - layer0: Linear → LayerNorm → SiLU
  - layer1: Linear → LayerNorm
  - optional projection on the residual shortcut
  - scalar gated residual: ``SiLU(shortcut + sigmoid(gate) * y)``

The Tail applies one more Linear → LayerNorm → SiLU before the output linear.
Obs normalisation uses the checkpoint's stored ``obs_mean`` / ``obs_var`` (fixed,
no running update) and is fused into the module so the ONNX export is
self-contained.

Example::

    from myosuite.integrations.musclemimic.fullbody_local_policy import (
        load_local_policy_artifacts,
    )
    from myosuite.integrations.musclemimic.actor_torch import MimicActorModule

    artifacts = load_local_policy_artifacts(checkpoint_root)
    module = MimicActorModule.from_artifacts(artifacts)
    module.eval()

    obs = torch.zeros(N, artifacts.obs_dim)
    actions = module(obs)   # (N, act_dim), values in (-1, 1) approx
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from myosuite.integrations.musclemimic.fullbody_local_policy import (
        LocalPolicyArtifacts,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _t(arr: Any) -> torch.Tensor:
    """Convert numpy array to float32 torch tensor."""
    return torch.as_tensor(np.array(arr, dtype=np.float32, copy=True))


@dataclass(frozen=True)
class _ResBlockWeights:
    """Raw numpy weights for one residual block."""

    l0_kernel: np.ndarray  # (in_dim, hidden_dim) JAX convention
    l0_bias: np.ndarray
    l0_ln_scale: np.ndarray
    l0_ln_bias: np.ndarray
    l1_kernel: np.ndarray  # (hidden_dim, hidden_dim)
    l1_bias: np.ndarray
    l1_ln_scale: np.ndarray
    l1_ln_bias: np.ndarray
    proj_kernel: np.ndarray | None  # (in_dim, hidden_dim) or None
    proj_bias: np.ndarray | None
    gate_raw: float  # scalar logit; gate = sigmoid(gate_raw)


def _read_res_block(actor: dict[str, Any], idx: int) -> _ResBlockWeights:
    d0 = actor[f"block{idx}_layer0_dense"]
    ln0 = actor[f"block{idx}_layer0_ln"]
    d1 = actor[f"block{idx}_layer1_dense"]
    ln1 = actor[f"block{idx}_layer1_ln"]
    proj = actor.get(f"block{idx}_proj")
    gate = actor[f"res_gate_{idx}"]
    return _ResBlockWeights(
        l0_kernel=np.asarray(d0["kernel"], dtype=np.float32),
        l0_bias=np.asarray(d0["bias"], dtype=np.float32),
        l0_ln_scale=np.asarray(ln0["scale"], dtype=np.float32),
        l0_ln_bias=np.asarray(ln0["bias"], dtype=np.float32),
        l1_kernel=np.asarray(d1["kernel"], dtype=np.float32),
        l1_bias=np.asarray(d1["bias"], dtype=np.float32),
        l1_ln_scale=np.asarray(ln1["scale"], dtype=np.float32),
        l1_ln_bias=np.asarray(ln1["bias"], dtype=np.float32),
        proj_kernel=(
            np.asarray(proj["kernel"], dtype=np.float32) if proj is not None else None
        ),
        proj_bias=(
            np.asarray(proj["bias"], dtype=np.float32) if proj is not None else None
        ),
        gate_raw=float(np.asarray(gate, dtype=np.float32).reshape(())),
    )


def _residual_block_indices(actor: dict[str, Any]) -> list[int]:
    """Return sorted residual block indices present in an actor tree."""
    indices: list[int] = []
    suffix = "_layer0_dense"
    for key in actor:
        if not key.startswith("block") or not key.endswith(suffix):
            continue
        idx = key[len("block") : -len(suffix)]
        if idx.isdigit():
            indices.append(int(idx))
    return sorted(indices)


def _sorted_prefix_indices(tree: dict[str, Any], prefix: str) -> list[int]:
    """Return sorted integer suffixes for keys like ``f"{prefix}{i}"``."""
    indices: list[int] = []
    for key in tree:
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix) :]
        if suffix.isdigit():
            indices.append(int(suffix))
    return sorted(indices)


def _linear_from_jax(kernel: np.ndarray, bias: np.ndarray) -> nn.Linear:
    """Build nn.Linear from JAX-convention (in, out) kernel."""
    out_dim, in_dim = kernel.T.shape
    layer = nn.Linear(in_dim, out_dim, bias=True)
    with torch.no_grad():
        layer.weight.copy_(_t(kernel.T))  # PyTorch: (out, in)
        layer.bias.copy_(_t(bias))
    return layer


def _layer_norm_from_jax(scale: np.ndarray, bias: np.ndarray) -> nn.LayerNorm:
    ln = nn.LayerNorm(scale.shape[0], elementwise_affine=True, eps=1e-5)
    with torch.no_grad():
        ln.weight.copy_(_t(scale))
        ln.bias.copy_(_t(bias))
    return ln


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------


class _ResidualBlock(nn.Module):
    """One gated residual block of the MuscleMimic actor."""

    def __init__(self, weights: _ResBlockWeights) -> None:
        super().__init__()
        self.layer0 = _linear_from_jax(weights.l0_kernel, weights.l0_bias)
        self.ln0 = _layer_norm_from_jax(weights.l0_ln_scale, weights.l0_ln_bias)
        self.layer1 = _linear_from_jax(weights.l1_kernel, weights.l1_bias)
        self.ln1 = _layer_norm_from_jax(weights.l1_ln_scale, weights.l1_ln_bias)

        self.proj: nn.Linear | None = None
        if weights.proj_kernel is not None:
            assert weights.proj_bias is not None
            self.proj = _linear_from_jax(weights.proj_kernel, weights.proj_bias)

        # Scalar gate stored as a non-trainable parameter so it moves with .to(device)
        self.register_buffer("gate_raw", torch.tensor(weights.gate_raw))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        y = F.silu(self.ln0(self.layer0(h)))
        y = self.ln1(self.layer1(y))
        shortcut = h if self.proj is None else self.proj(h)
        gate = torch.sigmoid(self.gate_raw)
        return F.silu(shortcut + gate * y)


# ---------------------------------------------------------------------------
# Full actor module
# ---------------------------------------------------------------------------


class MimicActorModule(nn.Module):
    """Actor as a PyTorch module with fused obs normalisation.

    Obs normalisation uses the checkpoint's stored ``obs_mean`` / ``obs_var``
    as fixed buffers — no running update at inference time.

    Args:
        params: Param tree from :class:`~...LocalPolicyArtifacts` (already
                converted to numpy via ``_to_numpy_tree``).
        obs_mean: Stored running mean, shape ``(obs_dim,)``.
        obs_var: Stored running variance, shape ``(obs_dim,)``.

    Returns:
        ``(N, act_dim)`` tensor of deterministic mean actions.
    """

    def __init__(
        self,
        params: dict[str, Any],
        obs_mean: np.ndarray,
        obs_var: np.ndarray,
    ) -> None:
        super().__init__()
        actor = params["actor"]
        block_indices = _residual_block_indices(actor)
        if not block_indices:
            raise ValueError("Unsupported actor params: no residual blocks found.")
        self.blocks = nn.ModuleList(
            [_ResidualBlock(_read_res_block(actor, i)) for i in block_indices]
        )

        self.tail: nn.Linear | None = None
        self.tail_ln: nn.LayerNorm | None = None
        tail_d = actor.get("tail_dense")
        tail_ln = actor.get("tail_ln")
        if tail_d is not None:
            if tail_ln is None:
                raise ValueError(
                    "Unsupported actor params: tail_dense without tail_ln."
                )
            self.tail = _linear_from_jax(
                np.asarray(tail_d["kernel"], dtype=np.float32),
                np.asarray(tail_d["bias"], dtype=np.float32),
            )
            self.tail_ln = _layer_norm_from_jax(
                np.asarray(tail_ln["scale"], dtype=np.float32),
                np.asarray(tail_ln["bias"], dtype=np.float32),
            )
        out = actor["output"]
        self.output = _linear_from_jax(
            np.asarray(out["kernel"], dtype=np.float32),
            np.asarray(out["bias"], dtype=np.float32),
        )

        self.register_buffer("obs_mean", _t(obs_mean))
        self.register_buffer("obs_std", torch.sqrt(_t(obs_var) + 1e-8))

        self.obs_dim: int = int(obs_mean.shape[0])
        self.act_dim: int = int(np.asarray(out["bias"]).shape[0])

    @classmethod
    def from_artifacts(cls, artifacts: LocalPolicyArtifacts) -> MimicActorModule:
        """Construct from a loaded :class:`~...LocalPolicyArtifacts` object.

        Args:
            artifacts: Loaded policy artifacts (params + normalisation stats).

        Returns:
            ``MimicActorModule`` in eval mode with weights copied from
            the checkpoint.
        """
        module = cls(artifacts.params, artifacts.obs_mean, artifacts.obs_var)
        module.eval()
        return module

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        """Apply checkpoint obs normalisation: ``(obs - mean) / std``.

        Args:
            obs: Raw observations, shape ``(..., obs_dim)``.

        Returns:
            Normalised observations, same shape.
        """
        return (obs - self.obs_mean) / self.obs_std

    def forward_normalized(self, norm_obs: torch.Tensor) -> torch.Tensor:
        """Run the actor on already-normalised observations.

        Useful when normalisation is applied externally (e.g. in a batched
        mjlab pipeline where normalisation is done once per batch).

        Args:
            norm_obs: Pre-normalised observations, shape ``(N, obs_dim)``.

        Returns:
            Deterministic mean actions, shape ``(N, act_dim)``.
        """
        h = norm_obs
        for block in self.blocks:
            h = block(h)
        if self.tail is not None:
            assert self.tail_ln is not None
            h = F.silu(self.tail_ln(self.tail(h)))
        return self.output(h)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalise obs then run the actor.

        Args:
            obs: Raw observations, shape ``(N, obs_dim)``.

        Returns:
            Deterministic mean actions, shape ``(N, act_dim)``.
        """
        return self.forward_normalized(self.normalize(obs))


class DenseActorModule(nn.Module):
    """Actor for simple ``Dense_*`` / ``LayerNorm_*`` layouts."""

    def __init__(
        self,
        params: dict[str, Any],
        obs_mean: np.ndarray,
        obs_var: np.ndarray,
    ) -> None:
        super().__init__()
        actor = params["actor"]
        dense_indices = _sorted_prefix_indices(actor, "Dense_")
        if not dense_indices:
            raise ValueError("Unsupported actor params: no Dense_* layers found.")

        self.hidden = nn.ModuleList()
        self.hidden_ln = nn.ModuleList()
        for idx in dense_indices[:-1]:
            dense = actor[f"Dense_{idx}"]
            self.hidden.append(
                _linear_from_jax(
                    np.asarray(dense["kernel"], dtype=np.float32),
                    np.asarray(dense["bias"], dtype=np.float32),
                )
            )
            ln = actor.get(f"LayerNorm_{idx}")
            if ln is None:
                self.hidden_ln.append(nn.Identity())
            else:
                self.hidden_ln.append(
                    _layer_norm_from_jax(
                        np.asarray(ln["scale"], dtype=np.float32),
                        np.asarray(ln["bias"], dtype=np.float32),
                    )
                )

        out = actor[f"Dense_{dense_indices[-1]}"]
        self.output = _linear_from_jax(
            np.asarray(out["kernel"], dtype=np.float32),
            np.asarray(out["bias"], dtype=np.float32),
        )

        self.register_buffer("obs_mean", _t(obs_mean))
        self.register_buffer("obs_std", torch.sqrt(_t(obs_var) + 1e-8))

        self.obs_dim: int = int(obs_mean.shape[0])
        self.act_dim: int = int(np.asarray(out["bias"]).shape[0])

    @classmethod
    def from_artifacts(cls, artifacts: LocalPolicyArtifacts) -> DenseActorModule:
        """Construct from a loaded :class:`~...LocalPolicyArtifacts` object."""
        module = cls(artifacts.params, artifacts.obs_mean, artifacts.obs_var)
        module.eval()
        return module

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        """Apply checkpoint obs normalisation: ``(obs - mean) / std``."""
        return (obs - self.obs_mean) / self.obs_std

    def forward_normalized(self, norm_obs: torch.Tensor) -> torch.Tensor:
        """Run the actor on already-normalised observations."""
        h = norm_obs
        for layer, norm in zip(self.hidden, self.hidden_ln, strict=True):
            h = F.silu(norm(layer(h)))
        return self.output(h)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalise obs then run the actor."""
        return self.forward_normalized(self.normalize(obs))


def make_actor_module(
    artifacts: LocalPolicyArtifacts,
) -> MimicActorModule | DenseActorModule:
    """Build the appropriate Torch actor module for a checkpoint artifact tree."""
    actor = artifacts.params["actor"]
    if "output" in actor:
        return MimicActorModule.from_artifacts(artifacts)
    if _sorted_prefix_indices(actor, "Dense_"):
        return DenseActorModule.from_artifacts(artifacts)
    raise ValueError("Unsupported actor params: no residual or Dense actor found.")


# Backward-compatible aliases for the initial public names from this integration.
MuscleMimicActorModule = MimicActorModule
MuscleMimicDenseActorModule = DenseActorModule
make_musclemimic_actor_module = make_actor_module


class BCActorNet(nn.Module):
    """LayerNorm-MLP actor with baked-in obs normalisation.

    Mirrors the simple ``Dense_* / LayerNorm_*`` Flax checkpoint architecture:
    N hidden layers each followed by LayerNorm + SiLU, then a linear output.
    Obs normalisation is stored as fixed (non-learnable) buffers so the module
    is fully self-contained.

    Args:
        obs_dim: Observation space dimension.
        action_dim: Action space dimension.
        hidden_dims: Width of each hidden layer.
        obs_mean: Checkpoint obs normalisation mean, shape ``(obs_dim,)``.
        obs_std: Checkpoint obs normalisation std, shape ``(obs_dim,)``.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
    ) -> None:
        super().__init__()
        self.register_buffer("obs_mean", _t(obs_mean))
        self.register_buffer("obs_std", _t(obs_std))

        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.SiLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalise obs and run the MLP.

        Args:
            obs: Raw observations, shape ``(..., obs_dim)``.

        Returns:
            Raw logits, shape ``(..., action_dim)``.
        """
        x = (obs - self.obs_mean) / (self.obs_std + 1e-8)
        return self.net(x)

    @torch.no_grad()
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Predict clipped action from a raw numpy observation.

        Args:
            obs: Shape ``(obs_dim,)``, dtype float32.

        Returns:
            Action array, shape ``(action_dim,)``, dtype float32, in ``[-1, 1]``.
        """
        t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        return self.forward(t).squeeze(0).clamp(-1.0, 1.0).cpu().numpy()


# ---------------------------------------------------------------------------
# RSL-RL checkpoint-compatible actor architecture
# (used by tutorials/mc26 mimic-init training scripts)
# ---------------------------------------------------------------------------

try:
    from collections.abc import Sequence as _Sequence
    from rsl_rl.models.mlp_model import MLPModel as _MLPModel
    from rsl_rl.modules.distribution import Distribution as _Distribution
    from rsl_rl.utils import resolve_callable as _resolve_callable
    from tensordict import TensorDict as _TensorDict
except ImportError:
    _Sequence = list  # type: ignore[misc,assignment]
    _MLPModel = object  # type: ignore[misc,assignment]
    _Distribution = object  # type: ignore[misc,assignment]
    _resolve_callable = None  # type: ignore[assignment]
    _TensorDict = None  # type: ignore[assignment]


def _writable_float32_array(value: Any) -> np.ndarray:
    """Return a writable float32 NumPy array (safe for Torch tensor copies)."""
    return np.array(value, dtype=np.float32, copy=True)


def _module_device(module: nn.Module) -> torch.device:
    """Return the device hosting a module's first parameter or buffer."""
    param = next(module.parameters(), None)
    if param is not None:
        return param.device
    buffer = next(module.buffers(), None)
    if buffer is not None:
        return buffer.device
    return torch.device("cpu")


def _has_checkpoint_normalizer(model: nn.Module) -> bool:
    """Return whether *model* exposes the frozen checkpoint-normalizer buffers.

    Uses attribute-level duck-typing rather than ``isinstance`` so the check
    survives dual-module scenarios that arise when running a script directly
    (the class referenced by dotted string path may differ from ``__main__``).
    """
    normalizer = getattr(model, "obs_normalizer", None)
    return all(hasattr(normalizer, attr) for attr in ("_mean", "_var", "_std", "count"))


def _sorted_dense_indices(tree: dict[str, Any]) -> list[int]:
    """Return sorted integer suffixes of ``Dense_*`` keys in a checkpoint tree."""
    return sorted(
        int(k[6:]) for k in tree if k.startswith("Dense_") and k[6:].isdigit()
    )


class CheckpointNormalization(nn.Module):
    """Frozen observation normalizer that matches the mimic checkpoint path.

    Statistics are loaded from the checkpoint and remain fixed during training
    (``update`` is a no-op so RSL-RL's running-stats machinery leaves them alone).
    """

    def __init__(self, shape: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = float(eps)
        self.register_buffer(
            "_mean", torch.zeros(shape, dtype=torch.float32).unsqueeze(0)
        )
        self.register_buffer(
            "_var", torch.ones(shape, dtype=torch.float32).unsqueeze(0)
        )
        self.register_buffer(
            "_std", torch.ones(shape, dtype=torch.float32).unsqueeze(0)
        )
        self.register_buffer("count", torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x: torch.Tensor) -> None:
        del x  # statistics are frozen


class LayerNormSiLUMLP(nn.Sequential):
    """Dense/LayerNorm/SiLU stack matching the mimic-checkpoint architecture."""

    def __init__(
        self, input_dim: int, output_dim: int, hidden_dims: _Sequence[int]
    ) -> None:
        super().__init__()
        in_dim = int(input_dim)
        idx = 0
        for hidden_dim in hidden_dims:
            self.add_module(str(idx), nn.Linear(in_dim, int(hidden_dim)))
            idx += 1
            self.add_module(str(idx), nn.LayerNorm(int(hidden_dim), eps=1e-5))
            idx += 1
            self.add_module(str(idx), nn.SiLU())
            idx += 1
            in_dim = int(hidden_dim)
        self.add_module(str(idx), nn.Linear(in_dim, int(output_dim)))


class LayerNormSiLUMLPModel(_MLPModel):
    """RSL-RL ``MLPModel`` subclass whose architecture matches the mimic checkpoint.

    Replaces the default ELU activations with a Dense/LayerNorm/SiLU stack so
    that checkpoint weights can be loaded directly via
    :func:`_load_dense_checkpoint_into_model` after construction.
    """

    def __init__(
        self,
        obs: _TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (1024, 1024, 1024, 1024, 1024),
        activation: str = "silu",
        obs_normalization: bool = True,
        distribution_cfg: dict[str, Any] | None = None,
    ) -> None:
        del activation  # activation is fixed to SiLU by the architecture
        nn.Module.__init__(self)
        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        self.obs_normalization = bool(obs_normalization)
        self.obs_normalizer = (
            CheckpointNormalization(self.obs_dim)
            if self.obs_normalization
            else nn.Identity()
        )
        if distribution_cfg is not None:
            distribution_cfg = dict(distribution_cfg)
            dist_class: type = _resolve_callable(distribution_cfg.pop("class_name"))
            self.distribution = dist_class(output_dim, **distribution_cfg)
            mlp_output_dim = int(self.distribution.input_dim)
        else:
            self.distribution = None
            mlp_output_dim = int(output_dim)
        self.mlp = LayerNormSiLUMLP(
            input_dim=self.obs_dim,
            output_dim=mlp_output_dim,
            hidden_dims=hidden_dims,
        )
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)


def _load_dense_checkpoint_into_model(
    model: nn.Module,
    layer_tree: dict[str, Any],
    *,
    obs_mean: np.ndarray,
    obs_var: np.ndarray,
    obs_count: np.ndarray,
) -> None:
    """Load Dense/LayerNorm checkpoint weights into a :class:`LayerNormSiLUMLPModel`.

    Copies observation-normalizer statistics and all linear/layer-norm weights
    from the JAX parameter tree *layer_tree* (as returned by
    :func:`~myosuite.integrations.musclemimic.fullbody_local_policy.load_local_policy_artifacts`)
    into *model* in place.
    """
    if not _has_checkpoint_normalizer(model):
        raise TypeError("Checkpoint import requires CheckpointNormalization.")
    dev = _module_device(model)
    dense_indices = _sorted_dense_indices(layer_tree)
    if not dense_indices:
        raise ValueError("Checkpoint layer tree does not expose any Dense_* layers.")
    hidden_indices = dense_indices[:-1]
    out_index = dense_indices[-1]
    linear_layers = [m for m in model.mlp if isinstance(m, nn.Linear)]
    norm_layers = [m for m in model.mlp if isinstance(m, nn.LayerNorm)]
    if len(linear_layers) != len(hidden_indices) + 1:
        raise ValueError(
            f"Linear count mismatch: model has {len(linear_layers)}, "
            f"checkpoint has {len(hidden_indices) + 1}."
        )
    if len(norm_layers) != len(hidden_indices):
        raise ValueError(
            f"LayerNorm count mismatch: model has {len(norm_layers)}, "
            f"checkpoint has {len(hidden_indices)}."
        )
    with torch.no_grad():

        def _t(a: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(_writable_float32_array(a), device=dev)

        model.obs_normalizer._mean.copy_(_t(obs_mean).unsqueeze(0))
        model.obs_normalizer._var.copy_(_t(obs_var).unsqueeze(0))
        model.obs_normalizer._std.copy_(torch.sqrt(_t(obs_var).unsqueeze(0) + 1e-8))
        model.obs_normalizer.count.copy_(_t(obs_count))
        for dense_idx, linear, norm in zip(
            hidden_indices, linear_layers[:-1], norm_layers, strict=True
        ):
            d = layer_tree[f"Dense_{dense_idx}"]
            linear.weight.copy_(_t(np.asarray(d["kernel"]).T))
            linear.bias.copy_(_t(np.asarray(d["bias"])))
            ln = layer_tree.get(f"LayerNorm_{dense_idx}")
            if ln is None:
                raise ValueError(f"Missing LayerNorm_{dense_idx}.")
            norm.weight.copy_(_t(np.asarray(ln["scale"])))
            norm.bias.copy_(_t(np.asarray(ln["bias"])))
        out = layer_tree[f"Dense_{out_index}"]
        linear_layers[-1].weight.copy_(_t(np.asarray(out["kernel"]).T))
        linear_layers[-1].bias.copy_(_t(np.asarray(out["bias"])))


def load_mimic_checkpoint_policy(
    checkpoint_root: Any,
    *,
    device: str | torch.device = "cpu",
) -> LayerNormSiLUMLPModel:
    """Load the mimic checkpoint actor as a standalone :class:`LayerNormSiLUMLPModel`.

    Infers the architecture from the checkpoint's Dense_* layers, constructs the
    model, loads the weights, and returns it in eval mode on *device*.

    Intended for evaluation / sanity-checking; wrap the result with
    :func:`~myosuite.integrations.musclemimic.model_bridge.make_fullbody_checkpoint_bridged_policy`
    to run it on a native mjlab saber environment.
    """
    from myosuite.integrations.musclemimic.fullbody_checkpoint_io import (
        resolve_checkpoint_ref,
    )
    from myosuite.integrations.musclemimic.fullbody_local_policy import (
        load_local_policy_artifacts,
    )

    checkpoint = resolve_checkpoint_ref(str(checkpoint_root))
    artifacts = load_local_policy_artifacts(checkpoint.local_path)
    actor_tree = artifacts.params["actor"]
    dense_indices = _sorted_dense_indices(actor_tree)
    if not dense_indices:
        raise ValueError("No Dense_* layers found in mimic checkpoint actor params.")
    hidden_dims = tuple(
        int(np.asarray(actor_tree[f"Dense_{i}"]["bias"]).shape[0])
        for i in dense_indices[:-1]
    )
    policy_device = torch.device(device)
    obs = torch.zeros(
        1, int(artifacts.obs_dim), dtype=torch.float32, device=policy_device
    )
    policy = LayerNormSiLUMLPModel(
        obs=_TensorDict(
            {"actor": obs, "critic": obs.clone()},
            batch_size=[obs.shape[0]],
            device=policy_device,
        ),
        obs_groups={"actor": ["actor"], "critic": ["critic"]},
        obs_set="actor",
        output_dim=int(artifacts.action_dim),
        hidden_dims=hidden_dims,
        distribution_cfg=None,
    ).to(policy_device)
    _load_dense_checkpoint_into_model(
        policy,
        actor_tree,
        obs_mean=artifacts.obs_mean,
        obs_var=artifacts.obs_var,
        obs_count=artifacts.obs_count,
    )
    policy.eval()
    return policy


__all__ = [
    "BCActorNet",
    "MimicActorModule",
    "DenseActorModule",
    "make_actor_module",
    "MuscleMimicActorModule",
    "MuscleMimicDenseActorModule",
    "make_musclemimic_actor_module",
    "CheckpointNormalization",
    "LayerNormSiLUMLP",
    "LayerNormSiLUMLPModel",
    "load_mimic_checkpoint_policy",
    "_load_dense_checkpoint_into_model",
]
