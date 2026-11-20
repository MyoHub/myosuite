# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Multi-teacher trajectory-swap DAgger and offline BC distillation in PyTorch.

Implements:
- Dataset Aggregation (Ross, Gordon & Bagnell, AISTATS 2011).
- Multi-teacher `(teacher, reset_options)` chunk-swap scheduling, teacher-registry
  deduplication, sticky `beta` rollouts, Gaussian NLL policy distillation, and
  optional value-head MSE distillation adapted from LocoMuJoCo
  (`robfiras/loco-mujoco`, `vanilla_dagger_jax.py` and
  `examples/training_examples/jax_distill/experiment_traj_swap.py`, MIT License).
"""

from __future__ import annotations

import copy
import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from myosuite_traj_dagger.buffer import DaggerRolloutState, PersistentDaggerBuffer

TeacherKind = Literal["callable", "torch", "torchscript", "rslrl", "sb3", "onnx"]

_TWO_PI_LOG = float(math.log(2.0 * math.pi))


def _activation_module(name: str) -> nn.Module:
    key = name.lower()
    mapping: dict[str, type[nn.Module]] = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
        "gelu": nn.GELU,
    }
    if key not in mapping:
        raise ValueError(
            f"Unsupported activation {name!r}; expected one of {sorted(mapping)}."
        )
    return mapping[key]()


def _build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dims: Sequence[int],
    activation: str = "elu",
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = int(in_dim)
    for h in hidden_dims:
        layers.append(nn.Linear(prev, int(h)))
        layers.append(_activation_module(activation))
        prev = int(h)
    layers.append(nn.Linear(prev, int(out_dim)))
    return nn.Sequential(*layers)


class RunningMeanStd(nn.Module):
    """Running observation normalizer embedded in student checkpoints and exports.

    Buffers (`mean`, `var`, `count`) update explicitly via `update` during data
    collection; `forward` performs pure normalization `(x - mean) / sqrt(var + eps)`
    so TorchScript and ONNX graphs remain stateless and deterministic.
    """

    mean: torch.Tensor
    var: torch.Tensor
    count: torch.Tensor

    def __init__(
        self,
        shape: int | tuple[int, ...],
        *,
        eps: float = 1e-5,
        clip: float = 10.0,
    ) -> None:
        super().__init__()
        dims = (shape,) if isinstance(shape, int) else tuple(shape)
        self.eps = float(eps)
        self.clip = float(clip)
        self.register_buffer("mean", torch.zeros(dims, dtype=torch.float32))
        self.register_buffer("var", torch.ones(dims, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(1e-4, dtype=torch.float64))

    @torch.no_grad()
    def update(self, x: np.ndarray | torch.Tensor) -> None:
        """Update running mean and variance from a batch of observations.

        Args:
            x: Batch of observations with trailing shape matching `self.mean.shape`.
        """
        t = torch.as_tensor(x, dtype=torch.float64, device=self.mean.device)
        if t.shape == self.mean.shape:
            t = t.unsqueeze(0)
        flat = t.reshape(-1, *self.mean.shape)
        batch_count = int(flat.shape[0])
        if batch_count == 0:
            return

        batch_mean = flat.mean(dim=0)
        batch_var = flat.var(dim=0, unbiased=False)

        delta = batch_mean - self.mean.double()
        total_count = self.count + float(batch_count)
        new_mean = self.mean.double() + delta * (float(batch_count) / total_count)

        m_a = self.var.double() * self.count
        m_b = batch_var * float(batch_count)
        m2 = m_a + m_b + delta.pow(2) * self.count * float(batch_count) / total_count

        self.mean.copy_(new_mean.float())
        self.var.copy_((m2 / total_count).clamp_min(1e-8).float())
        self.count.copy_(total_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize observations without mutating running statistics."""
        normed = (x - self.mean) / torch.sqrt(self.var + self.eps)
        return torch.clamp(normed, -self.clip, self.clip)


class StudentActorCritic(nn.Module):
    """Gaussian student policy with embedded `RunningMeanStd` and optional value head.

    Supports architectures independent of (e.g. smaller than) the teacher policies,
    Gaussian policy NLL distillation, optional critic MSE distillation, PPO updates,
    and standalone TorchScript / ONNX export (`obs -> action`).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        *,
        actor_hidden_dims: Sequence[int] = (128, 128),
        critic_hidden_dims: Sequence[int] = (128, 128),
        activation: str = "elu",
        init_log_std: float = -0.5,
        log_std_bounds: tuple[float, float] = (-4.0, 1.0),
        normalize_obs: bool = True,
        use_value_head: bool = True,
        action_limit: float | None = 1.0,
    ) -> None:
        super().__init__()
        if obs_dim < 1 or act_dim < 1:
            raise ValueError(
                f"obs_dim ({obs_dim}) and act_dim ({act_dim}) must be >= 1."
            )
        if log_std_bounds[0] >= log_std_bounds[1]:
            raise ValueError(f"Invalid log_std_bounds: {log_std_bounds}.")

        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.actor_hidden_dims = tuple(int(d) for d in actor_hidden_dims)
        self.critic_hidden_dims = tuple(int(d) for d in critic_hidden_dims)
        self.activation = activation
        self.log_std_min = float(log_std_bounds[0])
        self.log_std_max = float(log_std_bounds[1])
        self.normalize_obs = bool(normalize_obs)
        self.use_value_head = bool(use_value_head)
        self.action_limit = float(action_limit) if action_limit is not None else None

        self.obs_rms: nn.Module = (
            RunningMeanStd(self.obs_dim) if self.normalize_obs else nn.Identity()
        )
        self.actor = _build_mlp(
            self.obs_dim, self.act_dim, self.actor_hidden_dims, activation
        )
        self.log_std = nn.Parameter(
            torch.full((self.act_dim,), float(init_log_std), dtype=torch.float32)
        )
        self.critic: nn.Sequential | None = (
            _build_mlp(self.obs_dim, 1, self.critic_hidden_dims, activation)
            if self.use_value_head
            else None
        )

    @property
    def device(self) -> torch.device:
        """Device holding the student parameters."""
        return self.log_std.device

    def distribution_params(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute unclamped Gaussian action mean and clamped `log_std`.

        Args:
            obs: Raw observation tensor `(..., obs_dim)`.

        Returns:
            Tuple `(mean, log_std)` with `mean` of shape `(..., act_dim)` and
            `log_std` of shape `(act_dim,)`.
        """
        norm_obs = self.obs_rms(obs)
        mean = self.actor(norm_obs)
        clamped_log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        return mean, clamped_log_std

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic policy mapping `obs -> action` for inference and export."""
        norm_obs = self.obs_rms(obs)
        mean = self.actor(norm_obs)
        if self.action_limit is not None:
            return torch.clamp(mean, -self.action_limit, self.action_limit)
        return mean

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict scalar state value `V_student(s)`.

        Args:
            obs: Raw observation tensor `(..., obs_dim)`.

        Returns:
            Value predictions of shape `(...)`.
        """
        if self.critic is None:
            raise RuntimeError(
                "StudentActorCritic was built with use_value_head=False."
            )
        norm_obs = self.obs_rms(obs)
        return self.critic(norm_obs).squeeze(-1)

    def policy_nll(
        self, obs: torch.Tensor, target_action: torch.Tensor
    ) -> torch.Tensor:
        """Mean Gaussian negative log-likelihood `-log pi_student(a_teacher | s)`.

        Args:
            obs: Batch of observations `(B, obs_dim)`.
            target_action: Batch of target teacher actions `(B, act_dim)`.

        Returns:
            Scalar NLL loss averaged across the batch (summed over action dims).
        """
        mean, log_std = self.distribution_params(obs)
        inv_var = torch.exp(-2.0 * log_std)
        per_dim = 0.5 * (
            (target_action - mean).pow(2) * inv_var + 2.0 * log_std + _TWO_PI_LOG
        )
        return per_dim.sum(dim=-1).mean()

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log-probabilities, entropy, and values for PPO updates.

        Args:
            obs: Observations `(B, obs_dim)`.
            actions: Unclamped sampled actions `(B, act_dim)`.

        Returns:
            Tuple `(log_prob, entropy, values)` each of shape `(B,)`.
        """
        mean, log_std = self.distribution_params(obs)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = (
            self.value(obs)
            if self.critic is not None
            else torch.zeros(obs.shape[0], dtype=obs.dtype, device=obs.device)
        )
        return log_prob, entropy, values

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray | torch.Tensor,
        *,
        deterministic: bool = True,
        generator: torch.Generator | None = None,
    ) -> np.ndarray | torch.Tensor:
        """Compute environment-ready actions for NumPy or Torch observations.

        Args:
            obs: Single `(obs_dim,)` or batched `(B, obs_dim)` observations.
            deterministic: Return mean action if `True`, else sample from Gaussian.
            generator: Optional RNG for stochastic sampling.

        Returns:
            Actions in the same array/tensor container as `obs`.
        """
        is_numpy = isinstance(obs, np.ndarray)
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        mean, log_std = self.distribution_params(x)
        if deterministic:
            out = mean
        else:
            std = torch.exp(log_std)
            if generator is None:
                noise = torch.randn_like(mean)
            else:
                noise = torch.randn(
                    mean.shape,
                    dtype=mean.dtype,
                    device=generator.device,
                    generator=generator,
                ).to(mean.device)
            out = mean + std * noise
        if self.action_limit is not None:
            out = torch.clamp(out, -self.action_limit, self.action_limit)
        if is_numpy:
            return out.detach().cpu().numpy().astype(np.float32, copy=False)
        return out

    def export_torchscript(self, output: str | Path) -> Path:
        """Export deterministic student actor (`obs -> action`) as TorchScript."""
        return export_torchscript(self, output)

    def export_onnx(self, output: str | Path, *, opset: int = 17) -> Path:
        """Export deterministic student actor (`obs -> action`) to ONNX."""
        return export_onnx(self, output, opset=opset)


class _DeterministicActorExport(nn.Module):
    """Self-contained deterministic actor module for TorchScript and ONNX export."""

    def __init__(self, student: StudentActorCritic) -> None:
        super().__init__()
        self.obs_rms = copy.deepcopy(student.obs_rms).cpu().eval()
        self.actor = copy.deepcopy(student.actor).cpu().eval()
        self.action_limit = student.action_limit

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.actor(self.obs_rms(obs))
        if self.action_limit is not None:
            return torch.clamp(out, -self.action_limit, self.action_limit)
        return out


def export_torchscript(student: StudentActorCritic, output: str | Path) -> Path:
    """Export a `StudentActorCritic` deterministic policy to a TorchScript file.

    Args:
        student: Trained student policy.
        output: Destination file path.

    Returns:
        Resolved `Path` to the written TorchScript archive.
    """
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = _DeterministicActorExport(student).eval()
    example = torch.zeros((1, student.obs_dim), dtype=torch.float32)
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore", DeprecationWarning)
        traced = torch.jit.trace(wrapper, example)
        torch.jit.save(traced, str(path))
    return path


def export_onnx(
    student: StudentActorCritic, output: str | Path, *, opset: int = 17
) -> Path:
    """Export a `StudentActorCritic` deterministic policy to a single `.onnx` file.

    Matches `myosuite.utils.export_onnx` interface (`obs [batch, obs_dim] ->
    action [batch, act_dim]`, `dynamo=False` for single-file portability).

    Args:
        student: Trained student policy.
        output: Destination `.onnx` path.
        opset: ONNX opset version.

    Returns:
        `Path` to the written `.onnx` file.
    """
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = _DeterministicActorExport(student).eval()
    dummy = torch.zeros((1, student.obs_dim), dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        dummy,
        str(path),
        opset_version=int(opset),
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        do_constant_folding=True,
        dynamo=False,
    )
    return path


@dataclass(frozen=True)
class TeacherSpec:
    """Specification for a specialist teacher policy in `TeacherRegistry`.

    Repeating a `TeacherSpec` across multiple `SwapPair` entries in a schedule
    increases its chunk-sampling weight while `TeacherRegistry` loads and caches
    the underlying model once by `key`.

    Attributes:
        key: Canonical cache identifier.
        kind: Loader backend (`"callable"`, `"torch"`, `"torchscript"`, `"rslrl"`,
            `"sb3"`, or `"onnx"`).
        source: Callable, `nn.Module`, or checkpoint `Path` / string.
        value_fn: Optional value callable `V_teacher(obs)` for callable/ONNX teachers.
        act_dim: Action dimension required by `load_rslrl_policy` when `kind="rslrl"`.
        sb3_algo: Optional SB3 algorithm name (`"PPO"`, `"SAC"`, `"TD3"`).
        device: Inference device string.
    """

    key: str
    kind: TeacherKind
    source: Any
    value_fn: (
        Callable[[np.ndarray | torch.Tensor], np.ndarray | torch.Tensor] | None
    ) = None
    act_dim: int | None = None
    sb3_algo: str | None = None
    device: str = "cpu"

    @classmethod
    def from_callable(
        cls,
        fn: Callable[[np.ndarray | torch.Tensor], np.ndarray | torch.Tensor],
        *,
        key: str,
        value_fn: Callable[[np.ndarray | torch.Tensor], np.ndarray | torch.Tensor]
        | None = None,
    ) -> TeacherSpec:
        """Create a `TeacherSpec` wrapping an in-memory Python callable."""
        return cls(key=key, kind="callable", source=fn, value_fn=value_fn)

    @classmethod
    def from_module(
        cls,
        module: nn.Module,
        *,
        key: str,
        device: str = "cpu",
    ) -> TeacherSpec:
        """Create a `TeacherSpec` wrapping a PyTorch `nn.Module`."""
        return cls(key=key, kind="torch", source=module, device=device)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        kind: TeacherKind | Literal["auto"] = "auto",
        key: str | None = None,
        act_dim: int | None = None,
        sb3_algo: str | None = None,
        device: str = "cpu",
    ) -> TeacherSpec:
        """Create a `TeacherSpec` from a checkpoint file path with canonical key."""
        resolved = Path(path).resolve()
        inferred: TeacherKind
        if kind == "auto":
            suffix = resolved.suffix.lower()
            if suffix == ".onnx":
                inferred = "onnx"
            elif suffix == ".zip":
                inferred = "sb3"
            elif suffix in {".ts", ".jit"}:
                inferred = "torchscript"
            else:
                inferred = "rslrl"
        else:
            inferred = kind
        canonical_key = key if key is not None else f"{inferred}:{resolved}"
        return cls(
            key=canonical_key,
            kind=inferred,
            source=resolved,
            act_dim=act_dim,
            sb3_algo=sb3_algo,
            device=device,
        )


@dataclass(frozen=True)
class SwapPair:
    """One `(teacher, reset_options)` trajectory-swap entry.

    Repeating the same `SwapPair` `K` times in a schedule gives it `K` times the
    chunk selection weight while `TeacherRegistry` loads `teacher` once.

    Attributes:
        teacher: `TeacherSpec` identifying the active teacher for the chunk.
        reset_options: Optional dictionary forwarded to `env.reset(options=...)`.
    """

    teacher: TeacherSpec
    reset_options: dict[str, Any] | None = None


def repeat_pair(pair: SwapPair, count: int) -> list[SwapPair]:
    """Repeat `pair` `count` times to give `count`x sampling weight in a schedule."""
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}.")
    return [pair for _ in range(count)]


class LoadedTeacher:
    """Uniform inference wrapper around a cached teacher policy and optional critic."""

    def __init__(
        self,
        key: str,
        act_fn: Callable[[np.ndarray | torch.Tensor], np.ndarray | torch.Tensor],
        value_fn: Callable[[np.ndarray | torch.Tensor], np.ndarray | torch.Tensor]
        | None = None,
    ) -> None:
        self.key = key
        self._act_fn = act_fn
        self._value_fn = value_fn

    @property
    def supports_value(self) -> bool:
        """Whether this teacher exposes state-value predictions `V_teacher(s)`."""
        return self._value_fn is not None

    def act(self, obs: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Return deterministic teacher actions for `obs`."""
        return self._act_fn(obs)

    def predict_with_value(
        self, obs: np.ndarray | torch.Tensor
    ) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor | None]:
        """Return `(teacher_action, teacher_value)` for `obs`."""
        actions = self._act_fn(obs)
        values = self._value_fn(obs) if self._value_fn is not None else None
        return actions, values


class TeacherRegistry:
    """Deduplicating cache of specialist teachers keyed by `TeacherSpec.key`.

    Tracks `load_counts` per key so schedules that repeat a teacher across many
    `(teacher, reset_options)` pairs instantiate or deserialize that teacher once.
    """

    def __init__(self) -> None:
        self._cache: dict[str, LoadedTeacher] = {}
        self.load_counts: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def get_or_load(self, spec: TeacherSpec) -> LoadedTeacher:
        """Return cached `LoadedTeacher` for `spec.key`, loading it on first access."""
        cached = self._cache.get(spec.key)
        if cached is not None:
            return cached
        loaded = self._load_spec(spec)
        self._cache[spec.key] = loaded
        self.load_counts[spec.key] = self.load_counts.get(spec.key, 0) + 1
        return loaded

    def preload_schedule(self, pairs: Sequence[SwapPair]) -> None:
        """Eagerly load every unique teacher referenced in `pairs`."""
        for pair in pairs:
            self.get_or_load(pair.teacher)

    def _load_spec(self, spec: TeacherSpec) -> LoadedTeacher:
        dev = torch.device(spec.device)

        if spec.kind == "callable":
            if not callable(spec.source):
                raise TypeError(f"TeacherSpec {spec.key!r} source must be callable.")
            return LoadedTeacher(spec.key, spec.source, spec.value_fn)

        if spec.kind == "torch":
            mod = self._resolve_torch_module(spec.source, dev)
            return self._wrap_torch_module(spec.key, mod, dev, spec.value_fn)

        if spec.kind == "torchscript":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                script_mod = torch.jit.load(
                    str(Path(spec.source)), map_location=dev
                ).eval()
            return self._wrap_torch_module(spec.key, script_mod, dev, spec.value_fn)

        if spec.kind == "rslrl":
            return self._load_rslrl(spec, dev)

        if spec.kind == "sb3":
            return self._load_sb3(spec, dev)

        if spec.kind == "onnx":
            return self._load_onnx(spec)

        raise ValueError(f"Unsupported TeacherSpec kind: {spec.kind!r}.")

    @staticmethod
    def _resolve_torch_module(source: Any, dev: torch.device) -> nn.Module:
        if isinstance(source, nn.Module):
            return source.to(dev).eval()
        path = Path(source)
        obj = torch.load(path, map_location=dev, weights_only=False)
        if isinstance(obj, nn.Module):
            return obj.to(dev).eval()
        if isinstance(obj, dict) and "student_config" in obj and "state_dict" in obj:
            cfg = obj["student_config"]
            student = StudentActorCritic(**cfg)
            student.load_state_dict(obj["state_dict"])
            return student.to(dev).eval()
        raise ValueError(
            f"Cannot reconstruct nn.Module from {path}; use kind='rslrl' or 'torchscript'."
        )

    @staticmethod
    def _wrap_torch_module(
        key: str,
        mod: nn.Module,
        dev: torch.device,
        explicit_value_fn: Callable[
            [np.ndarray | torch.Tensor], np.ndarray | torch.Tensor
        ]
        | None,
    ) -> LoadedTeacher:
        @torch.no_grad()
        def _act(obs: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
            is_np = isinstance(obs, np.ndarray)
            x = torch.as_tensor(obs, dtype=torch.float32, device=dev)
            single = x.ndim == 1
            if single:
                x = x.unsqueeze(0)
            if isinstance(mod, StudentActorCritic):
                y = mod(x)
            else:
                y = mod(x)
            if single:
                y = y.squeeze(0)
            if is_np:
                return y.detach().cpu().numpy().astype(np.float32, copy=False)
            return y

        val_fn = explicit_value_fn
        if (
            val_fn is None
            and isinstance(mod, StudentActorCritic)
            and mod.use_value_head
        ):

            @torch.no_grad()
            def _val(obs: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
                is_np = isinstance(obs, np.ndarray)
                x = torch.as_tensor(obs, dtype=torch.float32, device=dev)
                v = mod.value(x)
                if is_np:
                    return v.detach().cpu().numpy().astype(np.float32, copy=False)
                return v

            val_fn = _val

        return LoadedTeacher(key, _act, val_fn)

    @staticmethod
    def _load_rslrl(spec: TeacherSpec, dev: torch.device) -> LoadedTeacher:
        from myosuite.utils.rslrl_policy import load_rslrl_policy

        ckpt = Path(spec.source)
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        if (
            isinstance(state, dict)
            and "student_config" in state
            and "state_dict" in state
        ):
            student = StudentActorCritic(**state["student_config"])
            student.load_state_dict(state["state_dict"])
            student.to(dev).eval()
            return TeacherRegistry._wrap_torch_module(
                spec.key, student, dev, spec.value_fn
            )

        if spec.act_dim is None:
            raise ValueError(
                f"TeacherSpec {spec.key!r} with kind='rslrl' requires act_dim."
            )
        policy = load_rslrl_policy(ckpt, spec.act_dim).to(dev).eval()
        critic_mod = _try_build_rslrl_critic(state, dev)

        @torch.no_grad()
        def _act(obs: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
            is_np = isinstance(obs, np.ndarray)
            x = torch.as_tensor(obs, dtype=torch.float32, device=dev)
            y = policy(x)
            if is_np:
                return y.detach().cpu().numpy().astype(np.float32, copy=False)
            return y

        val_fn = spec.value_fn
        if val_fn is None and critic_mod is not None:

            @torch.no_grad()
            def _val(obs: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
                is_np = isinstance(obs, np.ndarray)
                x = torch.as_tensor(obs, dtype=torch.float32, device=dev)
                v = critic_mod(x).squeeze(-1)
                if is_np:
                    return v.detach().cpu().numpy().astype(np.float32, copy=False)
                return v

            val_fn = _val

        return LoadedTeacher(spec.key, _act, val_fn)

    @staticmethod
    def _load_sb3(spec: TeacherSpec, dev: torch.device) -> LoadedTeacher:
        import stable_baselines3 as sb3

        if hasattr(spec.source, "predict") and hasattr(spec.source, "policy"):
            model = spec.source
        else:
            path = Path(spec.source)
            algos = (
                [getattr(sb3, spec.sb3_algo)]
                if spec.sb3_algo is not None
                else [sb3.PPO, sb3.SAC, sb3.TD3]
            )
            model = None
            last_err: Exception | None = None
            for cls in algos:
                try:
                    model = cls.load(str(path), device=str(dev))
                    break
                except Exception as exc:
                    last_err = exc
            if model is None:
                raise ValueError(
                    f"Could not load SB3 checkpoint {path}: {last_err}"
                ) from last_err

        def _act(obs: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
            is_tensor = isinstance(obs, torch.Tensor)
            obs_np = (
                obs.detach().cpu().numpy().astype(np.float32, copy=False)
                if is_tensor
                else np.asarray(obs, dtype=np.float32)
            )
            act_np, _ = model.predict(obs_np, deterministic=True)
            act_f32 = np.asarray(act_np, dtype=np.float32)
            if is_tensor:
                assert isinstance(obs, torch.Tensor)
                return torch.as_tensor(act_f32, dtype=torch.float32, device=obs.device)
            return act_f32

        val_fn = spec.value_fn
        if val_fn is None and hasattr(model.policy, "predict_values"):

            @torch.no_grad()
            def _val(obs: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
                is_tensor = isinstance(obs, torch.Tensor)
                obs_t, _ = model.policy.obs_to_tensor(
                    obs.detach().cpu().numpy() if is_tensor else np.asarray(obs)
                )
                v = model.policy.predict_values(obs_t).squeeze(-1)
                if is_tensor:
                    assert isinstance(obs, torch.Tensor)
                    return v.to(device=obs.device, dtype=torch.float32)
                return v.detach().cpu().numpy().astype(np.float32, copy=False)

            val_fn = _val

        return LoadedTeacher(spec.key, _act, val_fn)

    @staticmethod
    def _load_onnx(spec: TeacherSpec) -> LoadedTeacher:
        import onnxruntime as ort

        sess = ort.InferenceSession(
            str(Path(spec.source)), providers=["CPUExecutionProvider"]
        )
        in_name = sess.get_inputs()[0].name

        def _act(obs: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
            is_tensor = isinstance(obs, torch.Tensor)
            x_np = (
                obs.detach().cpu().numpy().astype(np.float32, copy=False)
                if is_tensor
                else np.asarray(obs, dtype=np.float32)
            )
            single = x_np.ndim == 1
            batch_np = x_np[None, :] if single else x_np
            out_np = np.asarray(
                sess.run(None, {in_name: batch_np})[0], dtype=np.float32
            )
            if single:
                out_np = out_np[0]
            if is_tensor:
                assert isinstance(obs, torch.Tensor)
                return torch.as_tensor(out_np, dtype=torch.float32, device=obs.device)
            return out_np

        return LoadedTeacher(spec.key, _act, spec.value_fn)


def _try_build_rslrl_critic(
    state: dict[str, Any], dev: torch.device
) -> nn.Module | None:
    """Extract an optional RSL-RL MLP critic from v3/v4/v5 checkpoints if present."""
    from myosuite.utils.rslrl_policy import _build_mlp as _rsl_mlp
    from myosuite.utils.rslrl_policy import _build_normalizer as _rsl_norm

    critic_state: dict[str, torch.Tensor] = {}
    if "critic_state_dict" in state and isinstance(state["critic_state_dict"], dict):
        critic_state = dict(state["critic_state_dict"])
    else:
        raw = state.get("model_state_dict") or state.get("actor_critic_state_dict")
        if isinstance(raw, dict):
            for k, v in raw.items():
                if k.startswith("critic."):
                    critic_state[f"mlp.{k.removeprefix('critic.')}"] = v
                elif k.startswith("critic_obs_normalizer."):
                    critic_state[
                        f"obs_normalizer.{k.removeprefix('critic_obs_normalizer.')}"
                    ] = v

    mlp_state = {
        k.removeprefix("mlp."): v
        for k, v in critic_state.items()
        if k.startswith("mlp.")
    }
    linear = sorted(
        (k for k in mlp_state if k.endswith(".weight") and k.split(".")[0].isdigit()),
        key=lambda k: int(k.split(".")[0]),
    )
    if not linear:
        return None
    shapes = [tuple(mlp_state[k].shape) for k in linear]
    in_dim, out_dim = shapes[0][1], shapes[-1][0]
    if out_dim != 1:
        return None
    hidden = [s[0] for s in shapes[:-1]]
    mlp = _rsl_mlp(in_dim, 1, hidden, "elu")
    mlp.load_state_dict(mlp_state)
    if any(k.startswith("obs_normalizer.") for k in critic_state):
        norm: nn.Module = _rsl_norm(in_dim)
        norm_sd = {
            k.removeprefix("obs_normalizer."): v
            for k, v in critic_state.items()
            if k.startswith("obs_normalizer.")
        }
        norm.load_state_dict(norm_sd, strict=False)
    else:
        norm = nn.Identity()
    return nn.Sequential(norm, mlp).to(dev).eval()


class SyncEnvPool:
    """Seeded synchronous Gymnasium environment pool with per-chunk `reset_options`."""

    def __init__(
        self, env_fns: Sequence[Callable[[], gym.Env]], *, seed: int = 0
    ) -> None:
        if not env_fns:
            raise ValueError("env_fns must be non-empty.")
        self.envs: list[gym.Env] = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self._rng = np.random.default_rng(seed)

        obs_space = self.envs[0].observation_space
        act_space = self.envs[0].action_space
        if not isinstance(obs_space, gym.spaces.Box) or len(obs_space.shape) != 1:
            raise TypeError(f"Expected 1-D Box observation space, got {obs_space}.")
        if not isinstance(act_space, gym.spaces.Box) or len(act_space.shape) != 1:
            raise TypeError(f"Expected 1-D Box action space, got {act_space}.")
        self.obs_dim = int(obs_space.shape[0])
        self.act_dim = int(act_space.shape[0])
        self.act_low = act_space.low.astype(np.float32)
        self.act_high = act_space.high.astype(np.float32)

    def reset_all(self, options: dict[str, Any] | None = None) -> np.ndarray:
        """Reset every environment cleanly (discarding previous episode carry)."""
        obs_list: list[np.ndarray] = []
        for env in self.envs:
            ep_seed = int(self._rng.integers(0, 2**31 - 1))
            opts = copy.deepcopy(options) if options is not None else None
            obs, _ = env.reset(seed=ep_seed, options=opts)
            obs_list.append(np.asarray(obs, dtype=np.float32))
        return np.stack(obs_list, axis=0)

    def step(
        self,
        actions: np.ndarray,
        *,
        reset_options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Step all environments and auto-reset done episodes with `reset_options`."""
        clipped = np.clip(
            np.asarray(actions, dtype=np.float32), self.act_low, self.act_high
        )
        next_obs_list: list[np.ndarray] = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict[str, Any]] = []

        for i, env in enumerate(self.envs):
            obs, rwd, terminated, truncated, info = env.step(clipped[i])
            done = bool(terminated or truncated)
            rewards[i] = float(rwd)
            dones[i] = done
            if done:
                info = dict(info)
                info["terminal_observation"] = np.asarray(obs, dtype=np.float32)
                ep_seed = int(self._rng.integers(0, 2**31 - 1))
                opts = (
                    copy.deepcopy(reset_options) if reset_options is not None else None
                )
                obs, _ = env.reset(seed=ep_seed, options=opts)
            next_obs_list.append(np.asarray(obs, dtype=np.float32))
            infos.append(info)

        return np.stack(next_obs_list, axis=0), rewards, dones, infos

    def close(self) -> None:
        """Close all underlying Gymnasium environments."""
        for env in self.envs:
            env.close()


@dataclass(frozen=True)
class ChunkMetrics:
    """Training diagnostics emitted after each trajectory-swap chunk."""

    chunk_idx: int
    env_steps: int
    pair_idx: int
    teacher_key: str
    beta: float
    teacher_step_fraction: float
    policy_nll: float
    policy_mse: float
    value_loss: float
    buffer_size: int
    ring_size: int
    reservoir_size: int


@dataclass
class DaggerConfig:
    """Hyperparameters for `TrajSwapDagger`."""

    steps_per_chunk: int = 1024
    beta_start: float = 1.0
    beta_end: float = 0.05
    beta_anneal_steps: int | None = None
    beta_schedule: Literal["linear", "exponential"] = "linear"
    sticky_h_min: int = 5
    sticky_h_max: int = 15
    ring_buffer_size: int = 65_536
    long_term_buffer_size: int = 16_384
    min_include_prob: float = 0.05
    batch_size: int = 256
    epochs_per_chunk: int = 4
    max_batches_per_epoch: int | None = None
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    value_loss_coef: float = 0.5
    update_obs_rms: bool = True
    pair_sampling: Literal["weighted_random", "round_robin"] = "weighted_random"
    student_rollout_stochastic: bool = False
    seed: int = 0
    device: str = "cpu"


def compute_beta(
    step: int,
    *,
    beta_start: float,
    beta_end: float,
    anneal_steps: int,
    schedule: Literal["linear", "exponential"] = "linear",
) -> float:
    """Compute teacher mixing probability `beta` at environment step `step`."""
    if anneal_steps <= 0:
        return float(beta_end)
    progress = min(max(float(step) / float(anneal_steps), 0.0), 1.0)
    if schedule == "linear":
        return float(beta_start + (beta_end - beta_start) * progress)
    if schedule == "exponential":
        start = max(float(beta_start), 1e-6)
        end = max(float(beta_end), 1e-6)
        val = start * ((end / start) ** progress)
        return float(0.0 if beta_end == 0.0 and progress >= 1.0 else val)
    raise ValueError(f"Unknown beta_schedule: {schedule!r}.")


def update_student_on_buffer(
    student: StudentActorCritic,
    optimizer: torch.optim.Optimizer,
    buffer: PersistentDaggerBuffer,
    *,
    batch_size: int,
    epochs: int,
    value_loss_coef: float,
    max_grad_norm: float,
    max_batches_per_epoch: int | None = None,
) -> tuple[float, float, float]:
    """Optimize `student` on `buffer` with Gaussian policy NLL + optional value MSE.

    Returns:
        Tuple `(mean_nll, mean_action_mse, mean_value_mse)`.
    """
    student.train()
    nll_vals: list[float] = []
    mse_vals: list[float] = []
    val_vals: list[float] = []

    for _ in range(max(1, int(epochs))):
        for batch in buffer.iter_minibatches(
            batch_size,
            shuffle=True,
            drop_last=False,
            max_batches=max_batches_per_epoch,
        ):
            nll = student.policy_nll(batch.obs, batch.teacher_action)
            with torch.no_grad():
                pred_mean, _ = student.distribution_params(batch.obs)
                act_mse = F.mse_loss(pred_mean, batch.teacher_action)

            if (
                value_loss_coef > 0.0
                and student.use_value_head
                and bool(batch.has_value.any())
            ):
                pred_val = student.value(batch.obs[batch.has_value])
                v_loss = F.mse_loss(pred_val, batch.teacher_value[batch.has_value])
                total_loss = nll + float(value_loss_coef) * v_loss
                v_scalar = float(v_loss.detach().item())
            else:
                total_loss = nll
                v_scalar = 0.0

            optimizer.zero_grad()
            total_loss.backward()
            if max_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
            optimizer.step()

            nll_vals.append(float(nll.detach().item()))
            mse_vals.append(float(act_mse.detach().item()))
            val_vals.append(v_scalar)

    student.eval()
    return (
        float(np.mean(nll_vals)) if nll_vals else 0.0,
        float(np.mean(mse_vals)) if mse_vals else 0.0,
        float(np.mean(val_vals)) if val_vals else 0.0,
    )


class TrajSwapDagger:
    """Multi-teacher trajectory-swap DAgger trainer.

    At each chunk, selects a `(teacher, reset_options)` pair from `pairs`, resets
    the environment carry cleanly so no stale state crosses task/teacher boundaries,
    preserves `PersistentDaggerBuffer` and `StudentActorCritic` optimizer state,
    collects sticky-`beta` rollouts labeled by the active teacher, and minimizes
    Gaussian policy NLL plus optional critic MSE.
    """

    def __init__(
        self,
        env_pool: SyncEnvPool,
        student: StudentActorCritic,
        pairs: Sequence[SwapPair],
        cfg: DaggerConfig | None = None,
        *,
        registry: TeacherRegistry | None = None,
    ) -> None:
        if not pairs:
            raise ValueError("pairs schedule must contain at least one SwapPair.")
        self.env_pool = env_pool
        self.cfg = cfg if cfg is not None else DaggerConfig()
        self.device = torch.device(self.cfg.device)
        self.student = student.to(self.device)
        self.pairs = list(pairs)
        self.registry = registry if registry is not None else TeacherRegistry()
        self._rng = np.random.default_rng(self.cfg.seed)
        self._torch_rng = torch.Generator(device=self.device)
        self._torch_rng.manual_seed(int(self.cfg.seed))

        self.optimizer = torch.optim.Adam(
            self.student.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        self.buffer = PersistentDaggerBuffer(
            obs_dim=self.env_pool.obs_dim,
            act_dim=self.env_pool.act_dim,
            ring_buffer_size=self.cfg.ring_buffer_size,
            long_term_buffer_size=self.cfg.long_term_buffer_size,
            min_include_prob=self.cfg.min_include_prob,
            seed=self.cfg.seed + 17,
            device=self.device,
        )
        self.rollout_state = DaggerRolloutState.create(
            num_envs=self.env_pool.num_envs,
            beta=self.cfg.beta_start,
            h_min=self.cfg.sticky_h_min,
            h_max=self.cfg.sticky_h_max,
            backend="numpy",
            np_rng=np.random.default_rng(self.cfg.seed + 31),
        )
        self.total_env_steps = 0
        self.chunk_count = 0
        self.history: list[ChunkMetrics] = []

    def select_pair_index(self) -> int:
        """Sample or cycle the next `(teacher, reset_options)` pair index."""
        if self.cfg.pair_sampling == "round_robin":
            return self.chunk_count % len(self.pairs)
        return int(self._rng.integers(0, len(self.pairs)))

    def beta_for_step(self, step: int, *, total_steps: int) -> float:
        """Compute `beta` at `step` using `cfg.beta_anneal_steps` or `total_steps`."""
        anneal = (
            self.cfg.beta_anneal_steps
            if self.cfg.beta_anneal_steps is not None
            else total_steps
        )
        return compute_beta(
            step,
            beta_start=self.cfg.beta_start,
            beta_end=self.cfg.beta_end,
            anneal_steps=anneal,
            schedule=self.cfg.beta_schedule,
        )

    def run_chunk(
        self,
        *,
        max_env_steps: int | None = None,
        total_steps_for_anneal: int | None = None,
        pair_index: int | None = None,
    ) -> ChunkMetrics:
        """Execute one `(teacher, reset_options)` chunk swap, rollout, and update."""
        budget = (
            self.cfg.steps_per_chunk
            if max_env_steps is None
            else min(self.cfg.steps_per_chunk, max_env_steps)
        )
        horizon = max(1, math.ceil(budget / self.env_pool.num_envs))
        anneal_ref = (
            total_steps_for_anneal
            if total_steps_for_anneal is not None
            else max(self.cfg.steps_per_chunk, self.total_env_steps + budget)
        )

        p_idx = self.select_pair_index() if pair_index is None else int(pair_index)
        pair = self.pairs[p_idx]
        teacher = self.registry.get_or_load(pair.teacher)

        beta = self.beta_for_step(self.total_env_steps, total_steps=anneal_ref)
        obs = self.env_pool.reset_all(options=pair.reset_options)
        self.rollout_state.reset_all(beta)

        teacher_flags: list[float] = []
        collected = 0
        self.student.eval()

        for _ in range(horizon):
            if max_env_steps is not None and collected >= max_env_steps:
                break
            beta = self.beta_for_step(self.total_env_steps, total_steps=anneal_ref)
            t_act_raw, t_val_raw = teacher.predict_with_value(obs)
            t_act = np.asarray(t_act_raw, dtype=np.float32)
            t_val = (
                np.asarray(t_val_raw, dtype=np.float32).reshape(-1)
                if t_val_raw is not None
                else None
            )

            s_act = np.asarray(
                self.student.act(
                    obs,
                    deterministic=not self.cfg.student_rollout_stochastic,
                    generator=self._torch_rng,
                ),
                dtype=np.float32,
            )
            exec_act = np.asarray(
                self.rollout_state.select_action(t_act, s_act), dtype=np.float32
            )
            assert isinstance(self.rollout_state.using_teacher, np.ndarray)
            teacher_flags.append(float(np.mean(self.rollout_state.using_teacher)))

            take = (
                self.env_pool.num_envs
                if max_env_steps is None
                else min(self.env_pool.num_envs, max_env_steps - collected)
            )
            obs_slice = obs[:take]
            act_slice = t_act[:take]
            val_slice = t_val[:take] if t_val is not None else None
            if self.cfg.update_obs_rms and isinstance(
                self.student.obs_rms, RunningMeanStd
            ):
                self.student.obs_rms.update(obs_slice)
            self.buffer.add(obs_slice, act_slice, val_slice, source_id=p_idx)

            obs, _, dones, _ = self.env_pool.step(
                exec_act, reset_options=pair.reset_options
            )
            self.rollout_state.advance(dones, beta)

            self.total_env_steps += take
            collected += take

        nll, mse, v_loss = update_student_on_buffer(
            self.student,
            self.optimizer,
            self.buffer,
            batch_size=self.cfg.batch_size,
            epochs=self.cfg.epochs_per_chunk,
            value_loss_coef=self.cfg.value_loss_coef,
            max_grad_norm=self.cfg.max_grad_norm,
            max_batches_per_epoch=self.cfg.max_batches_per_epoch,
        )
        metrics = ChunkMetrics(
            chunk_idx=self.chunk_count,
            env_steps=self.total_env_steps,
            pair_idx=p_idx,
            teacher_key=pair.teacher.key,
            beta=beta,
            teacher_step_fraction=float(np.mean(teacher_flags))
            if teacher_flags
            else 0.0,
            policy_nll=nll,
            policy_mse=mse,
            value_loss=v_loss,
            buffer_size=self.buffer.size,
            ring_size=self.buffer.ring_size,
            reservoir_size=self.buffer.reservoir_size,
        )
        self.chunk_count += 1
        self.history.append(metrics)
        return metrics

    def train(
        self,
        total_steps: int,
        *,
        eval_interval: int | None = None,
        eval_callback: Callable[[int, StudentActorCritic], None] | None = None,
    ) -> list[ChunkMetrics]:
        """Run trajectory-swap DAgger up to `total_steps` interactive env steps."""
        if total_steps < 1:
            raise ValueError(f"total_steps must be >= 1, got {total_steps}.")

        target_total = self.total_env_steps + int(total_steps)
        next_eval = (
            self.total_env_steps + int(eval_interval)
            if eval_interval is not None and eval_interval > 0
            else None
        )

        if eval_callback is not None and self.total_env_steps == 0:
            eval_callback(0, self.student)

        new_metrics: list[ChunkMetrics] = []
        while self.total_env_steps < target_total:
            step_cap = target_total - self.total_env_steps
            if next_eval is not None:
                step_cap = min(step_cap, next_eval - self.total_env_steps)
            m = self.run_chunk(
                max_env_steps=step_cap, total_steps_for_anneal=target_total
            )
            new_metrics.append(m)
            if (
                eval_callback is not None
                and next_eval is not None
                and self.total_env_steps >= next_eval
            ):
                eval_callback(self.total_env_steps, self.student)
                assert eval_interval is not None
                next_eval += int(eval_interval)

        return new_metrics


@dataclass
class OfflineBCConfig:
    """Hyperparameters for the offline Behavior Cloning (`beta=1.0`) baseline."""

    steps_per_chunk: int = 1024
    batch_size: int = 256
    bc_epochs: int = 8
    max_batches_per_epoch: int | None = None
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    value_loss_coef: float = 0.5
    update_obs_rms: bool = True
    pair_sampling: Literal["weighted_random", "round_robin"] = "round_robin"
    seed: int = 0
    device: str = "cpu"


class OfflineBC:
    """Offline Behavior Cloning baseline (`beta=1.0` teacher-only rollouts).

    Collects demonstrations exclusively by executing the active specialist teacher
    (`beta=1.0`, no interactive student rollouts or sticky mixture) across the same
    `(teacher, reset_options)` schedule and fits `StudentActorCritic` with the
    identical Gaussian policy NLL (+ optional value MSE) loss.
    """

    def __init__(
        self,
        env_pool: SyncEnvPool,
        student: StudentActorCritic,
        pairs: Sequence[SwapPair],
        cfg: OfflineBCConfig | None = None,
        *,
        registry: TeacherRegistry | None = None,
        max_buffer_size: int = 262_144,
    ) -> None:
        if not pairs:
            raise ValueError("pairs schedule must contain at least one SwapPair.")
        self.env_pool = env_pool
        self.cfg = cfg if cfg is not None else OfflineBCConfig()
        self.device = torch.device(self.cfg.device)
        self.student = student.to(self.device)
        self.pairs = list(pairs)
        self.registry = registry if registry is not None else TeacherRegistry()
        self._rng = np.random.default_rng(self.cfg.seed)

        self.optimizer = torch.optim.Adam(
            self.student.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        self.buffer = PersistentDaggerBuffer(
            obs_dim=self.env_pool.obs_dim,
            act_dim=self.env_pool.act_dim,
            ring_buffer_size=max(int(max_buffer_size), self.cfg.steps_per_chunk),
            long_term_buffer_size=0,
            min_include_prob=0.0,
            seed=self.cfg.seed + 19,
            device=self.device,
        )
        self.total_env_steps = 0
        self.chunk_count = 0

    def collect_teacher_steps(self, num_steps: int) -> int:
        """Collect `num_steps` transitions executing only teacher actions (`beta=1`)."""
        collected = 0
        while collected < num_steps:
            chunk_budget = min(self.cfg.steps_per_chunk, num_steps - collected)
            if self.cfg.pair_sampling == "round_robin":
                p_idx = self.chunk_count % len(self.pairs)
            else:
                p_idx = int(self._rng.integers(0, len(self.pairs)))
            pair = self.pairs[p_idx]
            teacher = self.registry.get_or_load(pair.teacher)

            obs = self.env_pool.reset_all(options=pair.reset_options)
            chunk_done = 0
            while chunk_done < chunk_budget:
                t_act_raw, t_val_raw = teacher.predict_with_value(obs)
                t_act = np.asarray(t_act_raw, dtype=np.float32)
                t_val = (
                    np.asarray(t_val_raw, dtype=np.float32).reshape(-1)
                    if t_val_raw is not None
                    else None
                )
                take = min(self.env_pool.num_envs, chunk_budget - chunk_done)
                obs_slice = obs[:take]
                act_slice = t_act[:take]
                val_slice = t_val[:take] if t_val is not None else None

                if self.cfg.update_obs_rms and isinstance(
                    self.student.obs_rms, RunningMeanStd
                ):
                    self.student.obs_rms.update(obs_slice)
                self.buffer.add(obs_slice, act_slice, val_slice, source_id=p_idx)

                obs, _, _, _ = self.env_pool.step(
                    t_act, reset_options=pair.reset_options
                )
                chunk_done += take
                collected += take
                self.total_env_steps += take

            self.chunk_count += 1
        return collected

    def fit(self, epochs: int | None = None) -> tuple[float, float, float]:
        """Fit `student` offline on all collected teacher transitions."""
        ep = self.cfg.bc_epochs if epochs is None else int(epochs)
        return update_student_on_buffer(
            self.student,
            self.optimizer,
            self.buffer,
            batch_size=self.cfg.batch_size,
            epochs=ep,
            value_loss_coef=self.cfg.value_loss_coef,
            max_grad_norm=self.cfg.max_grad_norm,
            max_batches_per_epoch=self.cfg.max_batches_per_epoch,
        )

    def train(
        self,
        total_steps: int,
        *,
        eval_interval: int | None = None,
        eval_callback: Callable[[int, StudentActorCritic], None] | None = None,
    ) -> tuple[float, float, float]:
        """Collect `total_steps` teacher transitions (`beta=1`) and fit BC policy.

        Because teacher rollouts never execute the student (`beta=1`), collecting
        in causal `eval_interval` increments yields the exact offline teacher
        trajectory distribution while recording step-matched evaluation curves.
        """
        if total_steps < 1:
            raise ValueError(f"total_steps must be >= 1, got {total_steps}.")

        if eval_callback is not None and self.total_env_steps == 0:
            eval_callback(0, self.student)

        if eval_interval is None or eval_interval <= 0 or eval_callback is None:
            self.collect_teacher_steps(total_steps)
            losses = self.fit()
            if eval_callback is not None:
                eval_callback(self.total_env_steps, self.student)
            return losses

        target_total = self.total_env_steps + int(total_steps)
        losses = (0.0, 0.0, 0.0)
        while self.total_env_steps < target_total:
            inc = min(int(eval_interval), target_total - self.total_env_steps)
            self.collect_teacher_steps(inc)
            losses = self.fit()
            eval_callback(self.total_env_steps, self.student)
        return losses
