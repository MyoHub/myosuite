# Copyright 2025-2026 DAVIAN-Robotics (SimBaV2 Authors) and Embodied AI Nexus (Eden Contributors).
# Modifications Copyright (c) MyoSuite Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
# See NOTICE.txt in this package for full upstream attribution and license notices.
#
# Upstream sources:
# - SimBaV2: Lee et al. (2025), "Hyperspherical Normalization for Scalable Deep
#   Reinforcement Learning", arXiv:2502.15280, https://github.com/DAVIAN-Robotics/SimbaV2
#   (Apache-2.0) and https://github.com/xiaohu-art/Weave.
# - Eden: https://github.com/embodied-ai-nexus/Eden (Apache-2.0),
#   packages/eden/src/eden/learning/wrappers/rsl_rl/simba/{networks,model}.py.
"""SimBaV2 hyperspherical backbone, per-group encoders, checkpoint loader, and ONNX export.

Adapted upstream components:
    :func:`l2normalize`, :class:`Scaler`, :class:`HyperDense`, :class:`HyperEmbedder`,
    :class:`HyperMLP`, :class:`HyperLERPBlock`, :class:`SimBa`, :class:`MLPEncoder`,
    :class:`GroupEncoder`, and :class:`SimBaModel` (adapted from SimBaV2 / Eden).

MyoSuite-specific integration:
    :func:`build_simba_trunk`, :func:`rebuild_simba_trunk`,
    :func:`register_simba_rslrl_loader` (plugging into
    :func:`myosuite.utils.rslrl_policy.register_actor_trunk_builder`),
    :func:`load_simba_policy`, and :func:`export_simba_to_onnx`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from myosuite.utils.rslrl_policy import (
    _resolve_activation as resolve_activation,
)
from myosuite.utils.rslrl_policy import (
    extract_actor_state_dict,
    load_rslrl_policy,
    register_actor_trunk_builder,
)
from torch import nn

SIMBA_MODEL_CLASS_NAME = "eden_rl.simba:SimBaModel"


def l2normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Project ``x`` onto the unit hypersphere along ``dim``."""
    return x / torch.linalg.vector_norm(x, dim=dim, keepdim=True).clamp_min(eps)


class Scaler(nn.Module):
    """Per-dimension learnable scale ``scaler * (init / scale) * x``."""

    def __init__(self, dim: int, init: float = 1.0, scale: float = 1.0) -> None:
        super().__init__()
        self.scaler = nn.Parameter(torch.full((dim,), float(scale)))
        self.forward_scale = float(init) / float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scaler * self.forward_scale * x


class HyperDense(nn.Module):
    """Bias-free linear layer with row-wise L2-normalized orthogonal weights."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.orthogonal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, F.normalize(self.weight, dim=1))


class HyperEmbedder(nn.Module):
    """Hyperspherical input projection appending a constant shift coordinate."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        scaler_init: float,
        scaler_scale: float,
        c_shift: float,
    ) -> None:
        super().__init__()
        self.c_shift = float(c_shift)
        self.dense = HyperDense(in_dim + 1, hidden_dim)
        self.scaler = Scaler(hidden_dim, scaler_init, scaler_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shift = torch.full_like(x[..., :1], self.c_shift)
        x = l2normalize(torch.cat([x, shift], dim=-1))
        return l2normalize(self.scaler(self.dense(x)))


class HyperMLP(nn.Module):
    """Expanded hyperspherical MLP block ``HyperDense -> Scaler -> ReLU -> HyperDense -> L2``."""

    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        expansion: int,
        scaler_init: float,
        scaler_scale: float,
    ) -> None:
        super().__init__()
        mid = hidden_dim * expansion
        self.up = HyperDense(hidden_dim, mid)
        self.scaler = Scaler(mid, scaler_init, scaler_scale)
        self.down = HyperDense(mid, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return l2normalize(self.down(F.relu(self.scaler(self.up(x)))))


class HyperLERPBlock(nn.Module):
    """Residual spherical interpolation block ``l2norm(x + alpha * (MLP(x) - x))``."""

    def __init__(
        self,
        hidden_dim: int,
        expansion: int,
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
    ) -> None:
        super().__init__()
        s_corr = math.sqrt(expansion)
        self.mlp = HyperMLP(
            hidden_dim,
            hidden_dim,
            expansion,
            scaler_init / s_corr,
            scaler_scale / s_corr,
        )
        self.alpha = Scaler(hidden_dim, alpha_init, alpha_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return l2normalize(x + self.alpha(self.mlp(x) - x))


class SimBa(nn.Module):
    """SimBaV2 hyperspherical residual trunk with orthogonal low-gain output head."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        num_blocks: int = 2,
        expansion: int = 4,
        c_shift: float = 3.0,
    ) -> None:
        super().__init__()
        self.in_features = input_dim
        self.out_features = output_dim

        s_init = math.sqrt(2.0 / hidden_dim)
        a_init = 1.0 / (num_blocks + 1)
        a_scale = 1.0 / math.sqrt(hidden_dim)

        self.embedder = HyperEmbedder(input_dim, hidden_dim, s_init, s_init, c_shift)
        self.blocks = nn.Sequential(
            *[
                HyperLERPBlock(hidden_dim, expansion, s_init, s_init, a_init, a_scale)
                for _ in range(num_blocks)
            ]
        )
        self.head_dense = HyperDense(hidden_dim, hidden_dim)
        self.head_scaler = Scaler(hidden_dim, s_init, s_init)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        nn.init.orthogonal_(self.output_proj.weight, gain=0.01)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(self.embedder(x))
        return self.output_proj(self.head_scaler(self.head_dense(x)))


class MLPEncoder(nn.Sequential):
    """Per-group observation encoder ``Linear -> act -> ... -> Linear(latent_dim)``."""

    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        hidden_dims: list[int],
        activation: nn.Module,
    ) -> None:
        layers: list[nn.Module] = []
        for a, b in zip([in_dim, *hidden_dims[:-1]], hidden_dims, strict=True):
            layers.extend([nn.Linear(a, b), type(activation)()])
        layers.append(nn.Linear(hidden_dims[-1] if hidden_dims else in_dim, latent_dim))
        super().__init__(*layers)


class GroupEncoder(nn.Module):
    """Splits concatenated observations into groups and encodes each to ``latent_dim``."""

    def __init__(
        self,
        group_dims: list[int],
        hidden_dims: list[list[int]],
        latent_dim: int,
        activation: nn.Module | str = "swish",
    ) -> None:
        super().__init__()
        if len(group_dims) != len(hidden_dims):
            raise ValueError(
                f"GroupEncoder got {len(group_dims)} group_dims vs {len(hidden_dims)} hidden_dims."
            )
        act_mod = (
            resolve_activation(activation)
            if isinstance(activation, str)
            else activation
        )
        self.group_dims = list(group_dims)
        self.in_features = sum(self.group_dims)
        self.out_features = latent_dim * len(self.group_dims)
        self.encoders = nn.ModuleList(
            [
                MLPEncoder(dim, latent_dim, list(dims), act_mod)
                for dim, dims in zip(self.group_dims, hidden_dims, strict=True)
            ]
        )
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.kaiming_uniform_(mod.weight, nonlinearity="relu")
                nn.init.zeros_(mod.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = torch.split(x, self.group_dims, dim=-1)
        return torch.cat(
            [enc(chunks[i]) for i, enc in enumerate(self.encoders)], dim=-1
        )


def build_simba_trunk(
    group_dims: list[int] | int,
    output_dim: int,
    *,
    hidden_dim: int = 512,
    num_blocks: int = 2,
    expansion: int = 4,
    latent_dim: int = 128,
    encoder_hidden_dims: list[list[int]] | None = None,
    default_encoder_hidden_dims: tuple[int, ...] = (256,),
    c_shift: float = 3.0,
    activation: str = "swish",
) -> nn.Module:
    """Build standalone :class:`SimBa` (``int``) or ``nn.Sequential(GroupEncoder, SimBa)``."""
    if isinstance(group_dims, int):
        return SimBa(group_dims, output_dim, hidden_dim, num_blocks, expansion, c_shift)

    specs = (
        encoder_hidden_dims
        if encoder_hidden_dims is not None
        else [list(default_encoder_hidden_dims) for _ in group_dims]
    )
    encoder = GroupEncoder(group_dims, specs, latent_dim, activation=activation)
    simba = SimBa(
        encoder.out_features, output_dim, hidden_dim, num_blocks, expansion, c_shift
    )
    trunk = nn.Sequential(encoder, simba)
    trunk.in_features = encoder.in_features  # type: ignore[assignment]
    trunk.out_features = output_dim  # type: ignore[assignment]
    return trunk


def rebuild_simba_trunk(
    mlp_state: dict[str, torch.Tensor], actor_cfg: dict[str, Any] | None = None
) -> tuple[nn.Module, int, int] | None:
    """Reconstruct ``SimBa`` or ``Sequential(GroupEncoder, SimBa)`` from ``mlp.*`` weights.

    Conforms to :data:`myosuite.utils.rslrl_policy.ActorTrunkBuilder`: returns
    ``(trunk, obs_dim, out_dim)`` when ``mlp_state`` holds SimBa weights, or ``None``
    for standard MLP/other checkpoints.

    Args:
        mlp_state: Checkpoint actor trunk weights with ``"mlp."`` stripped.
        actor_cfg: Optional ``actor`` / ``policy`` config dictionary from ``agent.yaml``.

    Returns:
        ``(trunk_module, obs_dim, out_dim)`` if matched, else ``None``.
    """
    if not any(k.endswith("embedder.dense.weight") for k in mlp_state):
        return None
    cfg = actor_cfg or {}
    c_shift = float(cfg.get("c_shift", 3.0))
    act_name = str(cfg.get("activation", "swish"))

    has_group_enc = "1.embedder.dense.weight" in mlp_state
    prefix = "1." if has_group_enc else ""
    emb_w = mlp_state[f"{prefix}embedder.dense.weight"]
    hidden_dim = int(emb_w.shape[0])
    simba_in = int(emb_w.shape[1]) - 1
    out_dim = int(mlp_state[f"{prefix}output_proj.weight"].shape[0])

    block_ids = sorted(
        {
            int(k[len(prefix) :].split(".")[1])
            for k in mlp_state
            if k.startswith(f"{prefix}blocks.")
        }
    )
    num_blocks = len(block_ids)
    expansion = (
        int(mlp_state[f"{prefix}blocks.0.mlp.up.weight"].shape[0]) // hidden_dim
        if num_blocks > 0
        else int(cfg.get("expansion", 4))
    )
    simba = SimBa(simba_in, out_dim, hidden_dim, num_blocks, expansion, c_shift)

    if not has_group_enc:
        simba.load_state_dict(mlp_state)
        return simba, simba_in, out_dim

    enc_ids = sorted(
        {
            int(k.split(".")[2])
            for k in mlp_state
            if k.startswith("0.encoders.") and k.endswith(".weight")
        }
    )
    group_dims: list[int] = []
    hidden_specs: list[list[int]] = []
    latent_dim = 0
    for idx in enc_ids:
        layer_keys = sorted(
            (
                k
                for k in mlp_state
                if k.startswith(f"0.encoders.{idx}.") and k.endswith(".weight")
            ),
            key=lambda k: int(k.split(".")[3]),
        )
        shapes = [tuple(mlp_state[k].shape) for k in layer_keys]
        group_dims.append(int(shapes[0][1]))
        hidden_specs.append([int(s[0]) for s in shapes[:-1]])
        latent_dim = int(shapes[-1][0])

    group_enc = GroupEncoder(group_dims, hidden_specs, latent_dim, act_name)
    trunk = nn.Sequential(group_enc, simba)
    trunk.in_features = group_enc.in_features  # type: ignore[assignment]
    trunk.out_features = out_dim  # type: ignore[assignment]
    trunk.load_state_dict(mlp_state)
    return trunk, group_enc.in_features, out_dim


def register_simba_rslrl_loader() -> bool:
    """Register :func:`rebuild_simba_trunk` with :mod:`myosuite.utils.rslrl_policy`.

    Returns:
        ``True`` once ``rebuild_simba_trunk`` is registered.
    """
    from myosuite.utils import rslrl_policy

    rslrl_policy.register_actor_trunk_builder("eden_rl.simba", rebuild_simba_trunk)
    return True


def load_simba_policy(
    checkpoint: str | Path, action_dim: int, action_map_applied: bool = False
) -> nn.Module:
    """Load a SimBa RSL-RL (v3/v4/v5) checkpoint via :func:`load_rslrl_policy`.

    Args:
        checkpoint: Path to ``model_*.pt``.
        action_dim: Expected action dimension.
        action_map_applied: Forwarded to ``load_rslrl_policy``.

    Returns:
        Deterministic policy ``nn.Module`` with ``obs_dim`` and ``action_dim`` attributes.
    """
    ckpt = Path(checkpoint)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    actor_state = extract_actor_state_dict(state, ckpt)
    if not any(k.endswith("embedder.dense.weight") for k in actor_state):
        raise ValueError(f"Checkpoint {ckpt} does not contain a SimBa actor trunk.")
    register_actor_trunk_builder("eden_rl.simba", rebuild_simba_trunk)
    return load_rslrl_policy(ckpt, action_dim, action_map_applied=action_map_applied)


def export_simba_to_onnx(
    checkpoint: str | Path,
    output: str | Path,
    obs_dim: int,
    act_dim: int,
    opset: int = 17,
) -> Path:
    """Export a SimBa RSL-RL checkpoint to a self-contained MyoSuite ``.onnx`` file.

    Args:
        checkpoint: Path to ``model_*.pt``.
        output: Destination ``.onnx`` file path.
        obs_dim: Expected input observation width.
        act_dim: Expected action output width.
        opset: ONNX opset version.

    Returns:
        Resolved ``output`` path.
    """
    out_path = Path(output)
    policy = load_simba_policy(checkpoint, act_dim)
    actual_obs = int(getattr(policy, "obs_dim", obs_dim))
    if actual_obs != obs_dim:
        raise ValueError(f"obs_dim={obs_dim} != checkpoint input size {actual_obs}")

    try:
        from myosuite.utils.export_onnx import _export_policy
    except ImportError:
        _export_policy = None  # type: ignore[assignment]

    if _export_policy is not None:
        _export_policy(policy, obs_dim, out_path, opset)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            policy,
            torch.zeros(1, obs_dim, dtype=torch.float32),
            str(out_path),
            opset_version=opset,
            input_names=["obs"],
            output_names=["action"],
            dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
            do_constant_folding=True,
            dynamo=False,
        )
    return out_path


register_simba_rslrl_loader()


try:
    from rsl_rl.models.mlp_model import MLPModel as _RslMLPModel
    from rsl_rl.utils import resolve_nn_activation as _rsl_resolve_act
    from rsl_rl.utils import unpad_trajectories as _rsl_unpad
except ImportError:  # pragma: no cover - optional [rsl] extra
    _RslMLPModel = None  # type: ignore[assignment,misc]


if _RslMLPModel is not None:

    class SimBaModel(_RslMLPModel):  # type: ignore[misc]
        """RSL-RL >= 5 ``MLPModel`` subclass swapping ``self.mlp`` for ``GroupEncoder -> SimBa``."""

        def __init__(
            self,
            obs: Any,
            obs_groups: dict[str, list[str]],
            obs_set: str,
            output_dim: int,
            hidden_dim: int = 512,
            num_blocks: int = 2,
            expansion: int = 4,
            latent_dim: int = 128,
            encoder_hidden_dims: dict[str, list[int]] | None = None,
            default_encoder_hidden_dims: list[int] | tuple[int, ...] = (256,),
            c_shift: float = 3.0,
            compile: bool = False,
            hidden_dims: list[int] | tuple[int, ...] = (),
            activation: str = "swish",
            obs_normalization: bool = True,
            distribution_cfg: dict[str, Any] | None = None,
        ) -> None:
            if len(hidden_dims) != 0:
                raise ValueError(
                    f"SimBaModel requires empty hidden_dims=(), got {list(hidden_dims)}; "
                    "configure hidden_dim / num_blocks / expansion instead."
                )
            encoder_hidden_dims = dict(encoder_hidden_dims or {})
            super().__init__(
                obs,
                obs_groups,
                obs_set,
                output_dim,
                hidden_dims=[1],
                activation=activation,
                obs_normalization=obs_normalization,
                distribution_cfg=distribution_cfg,
            )
            trunk_out = (
                self.distribution.input_dim
                if self.distribution is not None
                else output_dim
            )
            g_dims = [int(obs[g].shape[-1]) for g in self.obs_groups]
            specs = [
                list(encoder_hidden_dims.get(g, default_encoder_hidden_dims))
                for g in self.obs_groups
            ]
            encoder = GroupEncoder(
                g_dims, specs, latent_dim, _rsl_resolve_act(activation)
            )
            self.mlp = nn.Sequential(
                encoder,
                SimBa(
                    encoder.out_features,
                    trunk_out,
                    hidden_dim,
                    num_blocks,
                    expansion,
                    c_shift,
                ),
            )
            if self.distribution is not None:
                self.distribution.init_mlp_weights(self.mlp)
            self._compile_trunk = compile
            self._compiled_trunk: Any = None

        def forward(
            self,
            obs: Any,
            masks: torch.Tensor | None = None,
            hidden_state: Any = None,
            stochastic_output: bool = False,
        ) -> torch.Tensor:
            if not self._compile_trunk:
                return super().forward(obs, masks, hidden_state, stochastic_output)
            if self._compiled_trunk is None:
                self._compiled_trunk = torch.compile(
                    self.mlp.forward,
                    backend="inductor",
                    mode="default",
                    fullgraph=True,
                    dynamic=False,
                )
            obs = (
                _rsl_unpad(obs, masks)
                if masks is not None and not self.is_recurrent
                else obs
            )
            latent = self.get_latent(obs, masks, hidden_state)
            trunk_out = self._compiled_trunk(latent)
            if self.distribution is not None:
                if stochastic_output:
                    self.distribution.update(trunk_out)
                    return self.distribution.sample()
                return self.distribution.deterministic_output(trunk_out)
            return trunk_out

        def __getstate__(self) -> dict[str, Any]:
            state = dict(super().__getstate__())
            state["_compiled_trunk"] = None
            return state
