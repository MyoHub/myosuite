# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Load an mjlab / RSL-RL checkpoint (v3, v4, or v5) as a deterministic policy.

mjlab tasks of the basic suite are CPU twins (same obs vector, action space and
control timing as the CPU env of the same ``env_id``), so the actor of an
mjlab training run can drive the CPU env directly. This module rebuilds that
actor from ``model_*.pt``, translating both ``rsl-rl-lib`` 3.x
(``model_state_dict``) and 4.x/5.x (``actor_state_dict``) checkpoint layouts.
Method-specific training packages under ``training/`` may register custom actor
trunk builders via :func:`register_actor_trunk_builder`.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from myosuite.utils.checkpoint_manifest import CfgLoader, action_map_kind

try:
    from rsl_rl.modules import MLP as _RslMLP
    from rsl_rl.modules import EmpiricalNormalization as _RslEmpiricalNormalization
except ImportError:  # pragma: no cover - fallback when rsl-rl-lib is not installed
    _RslMLP = None
    _RslEmpiricalNormalization = None

ActorTrunkBuilder = Callable[
    [dict[str, torch.Tensor], dict[str, Any]],
    tuple[nn.Module, int, int] | None,
]

_ACTOR_TRUNK_BUILDERS: dict[str, ActorTrunkBuilder] = {}


def register_actor_trunk_builder(name: str, builder: ActorTrunkBuilder) -> None:
    """Register a custom actor trunk builder for :func:`load_rslrl_policy`.

    Training packages under ``training/<method>/`` can register a builder that
    inspects the checkpoint's ``mlp.*`` state dict (with the ``"mlp."`` prefix
    removed) and optional ``actor`` config dict, returning
    ``(trunk, obs_dim, out_dim)`` when matched or ``None`` to fall through to
    default MLP reconstruction.

    Args:
        name: Unique identifier for the builder (e.g. ``"eden_rl.simba"``).
        builder: Callable returning ``(trunk, obs_dim, out_dim)`` or ``None``.

    Raises:
        ValueError: If ``name`` is empty.
    """
    if not name:
        raise ValueError("Actor trunk builder name must be non-empty.")
    _ACTOR_TRUNK_BUILDERS[name] = builder


def unregister_actor_trunk_builder(name: str) -> None:
    """Remove a previously registered actor trunk builder if present.

    Args:
        name: Identifier passed to :func:`register_actor_trunk_builder`.
    """
    _ACTOR_TRUNK_BUILDERS.pop(name, None)


def _resolve_activation(name: str) -> nn.Module:
    key = name.lower()
    table: dict[str, type[nn.Module]] = {
        "elu": nn.ELU,
        "selu": nn.SELU,
        "relu": nn.ReLU,
        "leakyrelu": nn.LeakyReLU,
        "lrelu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "gelu": nn.GELU,
        "identity": nn.Identity,
    }
    if key not in table:
        raise ValueError(
            f"Unsupported activation {name!r}; choose from {sorted(table)}."
        )
    return table[key]()


class _FallbackMLP(nn.Sequential):
    """State-dict-compatible mirror of ``rsl_rl.modules.MLP``."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        activation: str = "elu",
    ) -> None:
        layers: list[nn.Module] = []
        dims = [input_dim, *hidden_dims]
        for a, b in itertools.pairwise(dims):
            layers.append(nn.Linear(a, b))
            layers.append(_resolve_activation(activation))
        layers.append(nn.Linear(dims[-1], output_dim))
        super().__init__(*layers)
        self.in_features = input_dim
        self.out_features = output_dim


class _FallbackEmpiricalNormalization(nn.Module):
    """State-dict-compatible mirror of ``rsl_rl.modules.EmpiricalNormalization``."""

    def __init__(self, shape: int, eps: float = 1e-2) -> None:
        super().__init__()
        self.eps = eps
        self.until = None
        self._mean: torch.Tensor
        self._var: torch.Tensor
        self._std: torch.Tensor
        self.count: torch.Tensor
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / (self._std + self.eps)


def _build_mlp(
    obs_dim: int, out_dim: int, hidden: list[int], activation: str
) -> nn.Module:
    if _RslMLP is not None:
        return _RslMLP(obs_dim, out_dim, hidden, activation)
    return _FallbackMLP(obs_dim, out_dim, hidden, activation)


def _build_normalizer(obs_dim: int) -> nn.Module:
    if _RslEmpiricalNormalization is not None:
        return _RslEmpiricalNormalization(obs_dim)
    return _FallbackEmpiricalNormalization(obs_dim)


def extract_actor_state_dict(
    state: dict[str, Any], checkpoint: Path | str = "<memory>"
) -> dict[str, torch.Tensor]:
    """Normalize ``rsl-rl-lib`` 3.x, 4.x, or 5.x checkpoints to ``actor_state_dict``.

    Args:
        state: Deserialized checkpoint mapping.
        checkpoint: Source path or label for error messages.

    Returns:
        Normalized ``actor_state_dict`` mapping with ``mlp.*``, ``obs_normalizer.*``,
        and ``distribution.*`` keys.

    Raises:
        ValueError: If no recognized actor state dictionary or actor weights exist.
    """
    if "actor_state_dict" in state:
        return dict(state["actor_state_dict"])

    raw = state.get("model_state_dict") or state.get("actor_critic_state_dict")
    if raw is None or not isinstance(raw, dict):
        raise ValueError(
            f"Checkpoint {checkpoint} has neither 'actor_state_dict' nor 'model_state_dict' "
            f"(keys: {sorted(state)})."
        )

    out: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if k.startswith("actor."):
            out[f"mlp.{k.removeprefix('actor.')}"] = v
        elif k.startswith("actor_obs_normalizer."):
            out[f"obs_normalizer.{k.removeprefix('actor_obs_normalizer.')}"] = v
        elif k == "std":
            out["distribution.std_param"] = v
        elif k == "log_std":
            out["distribution.log_std_param"] = v
    if not out:
        raise ValueError(
            f"Could not locate actor weights in {checkpoint} (keys: {sorted(raw)})."
        )
    return out


class RslRlPolicy(torch.nn.Module):
    """RSL-RL actor: ``mlp(normalizer(obs))`` -> action mean (deterministic).

    :meth:`sample` adds the training Gaussian noise, i.e. the policy the training
    success rate was measured with. The two can behave very differently for muscle
    tasks (the action goes through a sigmoid), so a policy can succeed when sampled
    and fail with its mean action.

    Args:
        mlp: Actor trunk module.
        normalizer: Observation normalizer (``Identity`` when training used none).
        action_dim: Action dimension; a ``2 * action_dim`` trunk output (state-
            dependent std) is reduced to its mean slice.
        std: Learned action standard deviation (scalar or per action), if known.
        obs_dim: Explicit input observation width (inferred from ``mlp`` when ``None``).
    """

    def __init__(
        self,
        mlp: torch.nn.Module,
        normalizer: torch.nn.Module,
        action_dim: int,
        std: torch.Tensor | None = None,
        obs_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.mlp = mlp
        self.normalizer = normalizer
        self.action_dim = action_dim
        if obs_dim is not None:
            self.obs_dim = int(obs_dim)
        elif hasattr(mlp, "in_features") and isinstance(mlp.in_features, int):
            self.obs_dim = int(mlp.in_features)
        else:
            first_linear = next(
                (m for m in mlp.modules() if isinstance(m, torch.nn.Linear)), None
            )
            if first_linear is None:
                raise ValueError("Actor MLP must contain at least one Linear layer.")
            self.obs_dim = first_linear.in_features
        self.std: torch.Tensor | None
        self.register_buffer("std", None if std is None else std.flatten().clone())
        self.generator: torch.Generator | None = None

    def seed(self, seed: int) -> None:
        """Seed the noise of :meth:`sample` (CPU) without touching torch's global RNG."""
        self.generator = torch.Generator().manual_seed(int(seed))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(
                f"The policy expects {self.obs_dim}-d observations but the env gives "
                f"{obs.shape[-1]}-d. It was trained on an env whose observation differs "
                "from this one (different env, or a backend with a different observation)."
            )
        out = self.mlp(self.normalizer(obs))
        if out.shape[-1] != self.action_dim:
            out = out.reshape(*out.shape[:-1], 2, self.action_dim)[..., 0, :]
        return out

    def sample(self, obs: torch.Tensor) -> torch.Tensor:
        """Action mean plus Gaussian noise with the learned std (as in training)."""
        if self.std is None:
            raise ValueError("The checkpoint has no action std to sample from.")
        mean = self(obs)
        if self.generator is None:
            return mean + self.std * torch.randn_like(mean)
        noise = torch.randn(mean.shape, generator=self.generator, dtype=mean.dtype)
        return mean + self.std * noise.to(mean.device)

    @torch.no_grad()
    def act(self, obs: np.ndarray, stochastic: bool = False) -> np.ndarray:
        """Numpy convenience wrapper (single or batched observations)."""
        param = next(self.parameters(), None)
        buf = next(self.buffers(), None)
        device = (
            param.device
            if param is not None
            else (buf.device if buf is not None else torch.device("cpu"))
        )
        x = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=device)
        return (self.sample(x) if stochastic else self(x)).cpu().numpy()


def _agent_params(checkpoint: Path) -> dict[str, Any]:
    """``params/agent.yaml`` written by ``scripts/train_mjlab.py`` (if present)."""
    path = checkpoint.parent / "params" / "agent.yaml"
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.load(f, Loader=CfgLoader) or {}


def load_rslrl_policy(
    checkpoint: str | Path, action_dim: int, action_map_applied: bool = False
) -> RslRlPolicy:
    """Rebuild the deterministic actor of an RSL-RL (v3, v4, or v5) checkpoint.

    Layer sizes are read from the weights (or delegated to a registered
    :data:`ActorTrunkBuilder`); the activation comes from the run's
    ``params/agent.yaml`` (default ``elu``). If that file gives the actor's
    ``distribution_cfg.std_range``, the std used by :meth:`RslRlPolicy.sample` is
    clamped to it, as in training (the saved parameter can lie just outside the
    range); without one the checkpoint's std is used as is.

    Args:
        checkpoint: ``model_*.pt`` from an mjlab or RSL-RL training run.
        action_dim: Action dimension of the env the policy will drive.
        action_map_applied: The env applies the action map the run was trained
            through (see :mod:`myosuite.utils.action_map`).

    Returns:
        The policy in eval mode on CPU.

    Raises:
        ValueError: The checkpoint records an action map and ``action_map_applied``
            is not set; no matching actor trunk builder or ``mlp.<index>.weight``
            layers are found; or the actor's outputs do not fit ``action_dim``.
    """
    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    kind = action_map_kind(state.get("infos") or {}, checkpoint)
    if kind is not None and not action_map_applied:
        raise ValueError(
            f"{checkpoint} was trained through the {kind!r} action map: its actions are "
            "not one action per muscle. Load it with myosuite.utils.action_map.load_policy, "
            "or evaluate it with scripts/eval_mjlab_policy.py."
        )
    actor_state = extract_actor_state_dict(state, checkpoint)
    agent_params = _agent_params(checkpoint)
    actor_cfg = agent_params.get("actor") or agent_params.get("policy") or {}

    mlp_state = {
        k.removeprefix("mlp."): v
        for k, v in actor_state.items()
        if k.startswith("mlp.")
    }
    built: tuple[nn.Module, int, int] | None = None
    for builder in _ACTOR_TRUNK_BUILDERS.values():
        built = builder(mlp_state, actor_cfg)
        if built is not None:
            break

    if built is not None:
        mlp, obs_dim, out_dim = built
    else:
        linear = sorted(
            (
                k
                for k in mlp_state
                if k.endswith(".weight") and k.split(".")[0].isdigit()
            ),
            key=lambda k: int(k.split(".")[0]),
        )
        if not linear:
            cls_hint = actor_cfg.get("class_name")
            extra = (
                f" (actor class_name={cls_hint!r}; import its training package "
                "to register its trunk builder, or export to ONNX)"
                if cls_hint
                else ""
            )
            raise ValueError(
                f"No mlp.<index>.weight layers found in {checkpoint}{extra}."
            )
        shapes = [tuple(mlp_state[k].shape) for k in linear]
        obs_dim, out_dim = shapes[0][1], shapes[-1][0]
        hidden = [s[0] for s in shapes[:-1]]
        mlp = _build_mlp(obs_dim, out_dim, hidden, actor_cfg.get("activation", "elu"))
        mlp.load_state_dict(mlp_state)

    if any(k.startswith("obs_normalizer.") for k in actor_state):
        normalizer: torch.nn.Module = _build_normalizer(obs_dim)
        norm_state = {
            k.removeprefix("obs_normalizer."): v
            for k, v in actor_state.items()
            if k.startswith("obs_normalizer.")
        }
        incompat = normalizer.load_state_dict(norm_state, strict=False)
        if incompat.unexpected_keys:
            raise RuntimeError(
                f"Unexpected obs_normalizer keys in {checkpoint}: {incompat.unexpected_keys}"
            )
    else:
        normalizer = torch.nn.Identity()

    learned_std = any(
        k in actor_state
        for k in ("distribution.std_param", "distribution.log_std_param")
    )
    if out_dim != action_dim and (learned_std or out_dim != 2 * action_dim):
        raise ValueError(
            f"{checkpoint} outputs {out_dim} values"
            f"{' (one per action, with a learned std)' if learned_std else ''}; "
            f"the env takes {action_dim} actions. It was trained on another env."
        )
    std = None
    if "distribution.std_param" in actor_state:
        std = actor_state["distribution.std_param"]
    elif "distribution.log_std_param" in actor_state:
        std = actor_state["distribution.log_std_param"].exp()
    std_range = (actor_cfg.get("distribution_cfg") or {}).get("std_range")
    if std is not None and std_range is not None:
        # Same bounds as rsl_rl's GaussianDistribution (lower limit 1e-6).
        std = std.clamp(max(float(std_range[0]), 1e-6), float(std_range[1]))
    return RslRlPolicy(mlp, normalizer, action_dim, std, obs_dim=obs_dim).eval()


__all__ = [
    "ActorTrunkBuilder",
    "RslRlPolicy",
    "extract_actor_state_dict",
    "load_rslrl_policy",
    "register_actor_trunk_builder",
    "unregister_actor_trunk_builder",
]
