# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Standalone PyTorch, TorchScript, and ONNX export for DreamerV3 checkpoints.

Reconstructs DreamerV3's proprioceptive encoder, block-GRU RSSM posterior, and
bounded-normal actor directly from saved checkpoint NumPy arrays so trained
policies can be deployed and verified on MyoSuite without JAX or ``dreamerv3``
installed at inference time.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from myosuite_dreamerv3.compat import import_myosuite
from myosuite_dreamerv3.hf import resolve_dreamerv3_checkpoint

_SQRT_2_OVER_PI = 0.7978845608028654


def _apply_act(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == "gelu":
        return (
            0.5 * x * (1.0 + torch.tanh(_SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)))
        )
    if name == "silu":
        return F.silu(x)
    if name == "elu":
        return F.elu(x)
    if name == "relu":
        return F.relu(x)
    if name == "tanh":
        return torch.tanh(x)
    if name == "none":
        return x
    raise ValueError(f"Unsupported activation {name!r}")


class _RMSNorm(nn.Module):
    """Mirror of ``embodied.jax.nets.Norm('rms', eps=1e-4)``."""

    def __init__(self, scale: np.ndarray, eps: float = 1e-4) -> None:
        super().__init__()
        self.eps = float(eps)
        flat = np.asarray(scale, dtype=np.float32).reshape(-1)
        self.scale = nn.Parameter(torch.from_numpy(flat.copy()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean2 = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(mean2 + self.eps) * self.scale


class _Linear(nn.Module):
    """Mirror of ``embodied.jax.nets.Linear``."""

    def __init__(self, kernel: np.ndarray, bias: np.ndarray | None = None) -> None:
        super().__init__()
        k = np.asarray(kernel, dtype=np.float32)
        self.linear = nn.Linear(int(k.shape[0]), int(k.shape[1]), bias=bias is not None)
        with torch.no_grad():
            self.linear.weight.copy_(torch.from_numpy(k.T.copy()))
            if bias is not None:
                b = np.asarray(bias, dtype=np.float32).reshape(-1)
                self.linear.bias.copy_(torch.from_numpy(b.copy()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _BlockLinear(nn.Module):
    """Mirror of ``embodied.jax.nets.BlockLinear`` using ONNX-friendly batched matmul."""

    def __init__(self, kernel: np.ndarray, bias: np.ndarray | None = None) -> None:
        super().__init__()
        k = np.asarray(kernel, dtype=np.float32)
        self.blocks = int(k.shape[0])
        self.in_per_block = int(k.shape[1])
        self.out_per_block = int(k.shape[2])
        self.units = self.blocks * self.out_per_block
        self.kernel = nn.Parameter(torch.from_numpy(k.copy()))
        if bias is not None:
            b = np.asarray(bias, dtype=np.float32).reshape(-1)
            self.bias: nn.Parameter | None = nn.Parameter(torch.from_numpy(b.copy()))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        x_blk = x.reshape(bsz, self.blocks, self.in_per_block).permute(1, 0, 2)
        out = torch.matmul(x_blk, self.kernel).permute(1, 0, 2).reshape(bsz, self.units)
        if self.bias is not None:
            out = out + self.bias
        return out


def _locate_agent_pkl_and_meta(
    checkpoint: str | Path,
    *,
    revision: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Locate ``agent.pkl`` and optional metadata from a logdir, ``ckpt`` dir, or file."""
    resolved = resolve_dreamerv3_checkpoint(checkpoint, revision=revision)
    meta: dict[str, Any] = {}

    if resolved.is_file():
        agent_pkl = resolved
        search_dirs = [
            resolved.parent,
            resolved.parent.parent,
            resolved.parent.parent.parent,
        ]
    else:
        search_dirs = [resolved, resolved.parent]
        if (resolved / "agent.pkl").is_file():
            agent_pkl = resolved / "agent.pkl"
        elif (resolved / "latest").is_file():
            latest = (resolved / "latest").read_text(encoding="utf-8").strip()
            agent_pkl = resolved / latest / "agent.pkl"
        elif (resolved / "ckpt" / "latest").is_file():
            latest = (resolved / "ckpt" / "latest").read_text(encoding="utf-8").strip()
            agent_pkl = resolved / "ckpt" / latest / "agent.pkl"
        else:
            candidates = sorted(resolved.rglob("agent.pkl"))
            if not candidates:
                raise FileNotFoundError(
                    f"Could not find 'agent.pkl' inside checkpoint directory {resolved}"
                )
            agent_pkl = candidates[-1]

    for d in search_dirs:
        meta_file = d / "policy_meta.json"
        if meta_file.is_file():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            break
        cfg_file = d / "config.yaml"
        if cfg_file.is_file():
            try:
                import ruamel.yaml as yaml

                cfg = yaml.YAML(typ="safe").load(cfg_file.read_text(encoding="utf-8"))
                rssm_cfg = cfg.get("agent", {}).get("dyn", {}).get("rssm", {})
                enc_cfg = cfg.get("agent", {}).get("enc", {}).get("simple", {})
                pol_cfg = cfg.get("agent", {}).get("policy", {})
                meta = {
                    "stoch": int(rssm_cfg.get("stoch", 32)),
                    "classes": int(rssm_cfg.get("classes", 32)),
                    "symlog": bool(enc_cfg.get("symlog", True)),
                    "enc_act": str(enc_cfg.get("act", "gelu")),
                    "dyn_act": str(rssm_cfg.get("act", "gelu")),
                    "pol_act": str(pol_cfg.get("act", "silu")),
                }
                break
            except Exception:
                pass

    return agent_pkl, meta


def _infer_stoch_and_classes(
    stoch_flat: int,
    deter_dim: int,
    meta: dict[str, Any],
) -> tuple[int, int]:
    if "stoch" in meta and "classes" in meta:
        s, c = int(meta["stoch"]), int(meta["classes"])
        if s * c == stoch_flat:
            return s, c
    if stoch_flat == 8 and deter_dim == 8:
        return 2, 4
    if stoch_flat % 32 == 0:
        return 32, stoch_flat // 32
    for s in (16, 8, 4, 2, 1):
        if stoch_flat % s == 0:
            return s, stoch_flat // s
    return 1, stoch_flat


class DreamerV3TorchPolicy(nn.Module):
    """Pure-PyTorch deterministic DreamerV3 world-model actor for CPU/ONNX deployment.

    Mirrors ``Encoder`` + ``RSSM._observe`` (using posterior ``argmax`` mode) +
    ``MLPHead.bounded_normal`` (using ``tanh(mean)`` mode).
    """

    def __init__(
        self,
        params: dict[str, np.ndarray],
        meta: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        meta = dict(meta or {})
        if any(k.startswith("enc/cnn") for k in params):
            raise ValueError(
                "DreamerV3TorchPolicy supports proprioceptive checkpoints; "
                "CNN image checkpoints are not supported by vector ONNX export."
            )

        self.symlog = bool(meta.get("symlog", True))
        self.enc_act = str(meta.get("enc_act", "silu"))
        self.dyn_act = str(meta.get("dyn_act", "silu"))
        self.pol_act = str(meta.get("pol_act", "silu"))
        norm_eps = float(meta.get("norm_eps", 1e-4))

        enc_idx = 0
        enc_linears: list[nn.Module] = []
        enc_norms: list[nn.Module] = []
        while f"enc/mlp{enc_idx}/kernel" in params:
            enc_linears.append(
                _Linear(
                    params[f"enc/mlp{enc_idx}/kernel"],
                    params.get(f"enc/mlp{enc_idx}/bias"),
                )
            )
            enc_norms.append(
                _RMSNorm(params[f"enc/mlp{enc_idx}norm/scale"], eps=norm_eps)
            )
            enc_idx += 1
        if not enc_linears:
            raise ValueError("Checkpoint params missing 'enc/mlp0/kernel'.")
        self.enc_linears = nn.ModuleList(enc_linears)
        self.enc_norms = nn.ModuleList(enc_norms)

        self.obs_dim = int(np.asarray(params["enc/mlp0/kernel"]).shape[0])
        self.deter_dim = int(np.asarray(params["dyn/dynin0/kernel"]).shape[0])
        stoch_flat = int(np.asarray(params["dyn/dynin1/kernel"]).shape[0])
        self.act_dim = int(np.asarray(params["dyn/dynin2/kernel"]).shape[0])
        self.stoch_dim, self.classes = _infer_stoch_and_classes(
            stoch_flat, self.deter_dim, meta
        )
        self.blocks = int(np.asarray(params["dyn/dyngru/kernel"]).shape[0])

        self.dynin0 = _Linear(
            params["dyn/dynin0/kernel"], params.get("dyn/dynin0/bias")
        )
        self.dynin0norm = _RMSNorm(params["dyn/dynin0norm/scale"], eps=norm_eps)
        self.dynin1 = _Linear(
            params["dyn/dynin1/kernel"], params.get("dyn/dynin1/bias")
        )
        self.dynin1norm = _RMSNorm(params["dyn/dynin1norm/scale"], eps=norm_eps)
        self.dynin2 = _Linear(
            params["dyn/dynin2/kernel"], params.get("dyn/dynin2/bias")
        )
        self.dynin2norm = _RMSNorm(params["dyn/dynin2norm/scale"], eps=norm_eps)

        dynhid_idx = 0
        dynhid_layers: list[nn.Module] = []
        dynhid_norms: list[nn.Module] = []
        while f"dyn/dynhid{dynhid_idx}/kernel" in params:
            dynhid_layers.append(
                _BlockLinear(
                    params[f"dyn/dynhid{dynhid_idx}/kernel"],
                    params.get(f"dyn/dynhid{dynhid_idx}/bias"),
                )
            )
            dynhid_norms.append(
                _RMSNorm(params[f"dyn/dynhid{dynhid_idx}norm/scale"], eps=norm_eps)
            )
            dynhid_idx += 1
        self.dynhid_layers = nn.ModuleList(dynhid_layers)
        self.dynhid_norms = nn.ModuleList(dynhid_norms)
        self.dyngru = _BlockLinear(
            params["dyn/dyngru/kernel"], params.get("dyn/dyngru/bias")
        )

        obs_idx = 0
        obs_layers: list[nn.Module] = []
        obs_norms: list[nn.Module] = []
        while f"dyn/obs{obs_idx}/kernel" in params:
            obs_layers.append(
                _Linear(
                    params[f"dyn/obs{obs_idx}/kernel"],
                    params.get(f"dyn/obs{obs_idx}/bias"),
                )
            )
            obs_norms.append(
                _RMSNorm(params[f"dyn/obs{obs_idx}norm/scale"], eps=norm_eps)
            )
            obs_idx += 1
        self.obs_layers = nn.ModuleList(obs_layers)
        self.obs_norms = nn.ModuleList(obs_norms)
        self.obslogit = _Linear(
            params["dyn/obslogit/kernel"], params.get("dyn/obslogit/bias")
        )

        pol_idx = 0
        pol_layers: list[nn.Module] = []
        pol_norms: list[nn.Module] = []
        while f"pol/mlp/linear{pol_idx}/kernel" in params:
            pol_layers.append(
                _Linear(
                    params[f"pol/mlp/linear{pol_idx}/kernel"],
                    params.get(f"pol/mlp/linear{pol_idx}/bias"),
                )
            )
            pol_norms.append(
                _RMSNorm(params[f"pol/mlp/norm{pol_idx}/scale"], eps=norm_eps)
            )
            pol_idx += 1
        self.pol_layers = nn.ModuleList(pol_layers)
        self.pol_norms = nn.ModuleList(pol_norms)

        mean_key = "pol/head/action/mean"
        if f"{mean_key}/kernel" not in params:
            candidates = [
                k[: -len("/kernel")]
                for k in params
                if k.startswith("pol/head/") and k.endswith("/mean/kernel")
            ]
            if not candidates:
                raise ValueError("Checkpoint params missing actor mean head kernel.")
            mean_key = candidates[0]
        self.pol_mean = _Linear(
            params[f"{mean_key}/kernel"], params.get(f"{mean_key}/bias")
        )

    def initial_state(
        self, batch_size: int = 1, device: torch.device | str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return zeroed ``(deter, stoch, prevact)`` tensors for ``batch_size`` envs."""
        deter = torch.zeros(
            batch_size, self.deter_dim, dtype=torch.float32, device=device
        )
        stoch = torch.zeros(
            batch_size, self.stoch_dim, self.classes, dtype=torch.float32, device=device
        )
        prevact = torch.zeros(
            batch_size, self.act_dim, dtype=torch.float32, device=device
        )
        return deter, stoch, prevact

    def forward(
        self,
        obs: torch.Tensor,
        deter: torch.Tensor,
        stoch: torch.Tensor,
        prevact: torch.Tensor,
        is_first: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Step world-model posterior and return ``(action, next_deter, next_stoch)``."""
        bsz = obs.size(0)
        deter = deter.expand(bsz, self.deter_dim)
        stoch = stoch.expand(bsz, self.stoch_dim, self.classes)
        prevact = prevact.expand(bsz, self.act_dim)
        keep = (1.0 - is_first.to(torch.float32).reshape(-1, 1).expand(bsz, 1)).clamp(
            0.0, 1.0
        )

        x = torch.sign(obs) * torch.log(1.0 + torch.abs(obs)) if self.symlog else obs
        for lin, norm in zip(self.enc_linears, self.enc_norms, strict=True):
            x = _apply_act(norm(lin(x)), self.enc_act)
        tokens = x

        deter_m = deter * keep
        stoch_m = (stoch * keep.unsqueeze(-1)).reshape(
            bsz, self.stoch_dim * self.classes
        )
        act_m = prevact * keep
        act_norm = act_m / torch.clamp(torch.abs(act_m), min=1.0)

        x0 = _apply_act(self.dynin0norm(self.dynin0(deter_m)), self.dyn_act)
        x1 = _apply_act(self.dynin1norm(self.dynin1(stoch_m)), self.dyn_act)
        x2 = _apply_act(self.dynin2norm(self.dynin2(act_norm)), self.dyn_act)
        x_shared = (
            torch.cat([x0, x1, x2], dim=-1).unsqueeze(1).expand(bsz, self.blocks, -1)
        )
        deter_g = deter_m.reshape(bsz, self.blocks, self.deter_dim // self.blocks)
        h = torch.cat([deter_g, x_shared], dim=-1).reshape(bsz, -1)

        for blk_lin, blk_norm in zip(
            self.dynhid_layers, self.dynhid_norms, strict=True
        ):
            h = _apply_act(blk_norm(blk_lin(h)), self.dyn_act)

        gru_raw = self.dyngru(h).reshape(
            bsz, self.blocks, 3, self.deter_dim // self.blocks
        )
        reset_g = gru_raw[:, :, 0, :].reshape(bsz, self.deter_dim)
        cand_g = gru_raw[:, :, 1, :].reshape(bsz, self.deter_dim)
        update_g = gru_raw[:, :, 2, :].reshape(bsz, self.deter_dim)
        reset_gate = torch.sigmoid(reset_g)
        cand = torch.tanh(reset_gate * cand_g)
        update_gate = torch.sigmoid(update_g - 1.0)
        next_deter = update_gate * cand + (1.0 - update_gate) * deter_m

        post_in = torch.cat([next_deter, tokens], dim=-1)
        for obs_lin, obs_norm in zip(self.obs_layers, self.obs_norms, strict=True):
            post_in = _apply_act(obs_norm(obs_lin(post_in)), self.dyn_act)
        logit = self.obslogit(post_in).reshape(bsz, self.stoch_dim, self.classes)
        next_stoch = F.one_hot(
            torch.argmax(logit, dim=-1), num_classes=self.classes
        ).to(torch.float32)

        feat = torch.cat(
            [next_deter, next_stoch.reshape(bsz, self.stoch_dim * self.classes)], dim=-1
        )
        p = feat
        for pol_lin, pol_norm in zip(self.pol_layers, self.pol_norms, strict=True):
            p = _apply_act(pol_norm(pol_lin(p)), self.pol_act)
        action = torch.tanh(self.pol_mean(p))
        return action, next_deter, next_stoch


def load_dreamerv3_torch_policy(
    checkpoint_or_params: str | Path | dict[str, Any],
    *,
    meta: dict[str, Any] | None = None,
    revision: str | None = None,
) -> DreamerV3TorchPolicy:
    """Load a standalone ``DreamerV3TorchPolicy`` from disk, ``hf://``, or a params dict."""
    if isinstance(checkpoint_or_params, dict):
        raw_params = checkpoint_or_params.get("params", checkpoint_or_params)
        params = {k: np.asarray(v) for k, v in raw_params.items()}
        policy = DreamerV3TorchPolicy(params, meta=meta)
        policy.eval()
        return policy

    agent_pkl, disk_meta = _locate_agent_pkl_and_meta(
        checkpoint_or_params, revision=revision
    )
    merged_meta = {**disk_meta, **(meta or {})}
    payload = pickle.loads(agent_pkl.read_bytes())
    raw_params = (
        payload["params"]
        if isinstance(payload, dict) and "params" in payload
        else payload
    )
    params = {k: np.asarray(v) for k, v in raw_params.items()}
    policy = DreamerV3TorchPolicy(params, meta=merged_meta)
    policy.eval()
    return policy


def export_dreamerv3_torchscript(
    policy_or_checkpoint: DreamerV3TorchPolicy | str | Path | dict[str, Any],
    output: str | Path | None = None,
    *,
    meta: dict[str, Any] | None = None,
    revision: str | None = None,
) -> torch.jit.ScriptModule:
    """Trace a standalone stateful TorchScript DreamerV3 policy."""
    policy = (
        policy_or_checkpoint
        if isinstance(policy_or_checkpoint, DreamerV3TorchPolicy)
        else load_dreamerv3_torch_policy(
            policy_or_checkpoint, meta=meta, revision=revision
        )
    )
    policy = policy.cpu().eval()
    obs = torch.zeros(1, policy.obs_dim, dtype=torch.float32)
    deter, stoch, prevact = policy.initial_state(1)
    is_first = torch.ones(1, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(
            policy, (obs, deter, stoch, prevact, is_first), check_trace=False
        )
    if output is not None:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        traced.save(str(out_path))
    return traced


def write_onnx_deployment_manifest(
    onnx_path: str | Path,
    env_id: str,
    *,
    features: Sequence[Any] = (),
    symlog: bool = True,
) -> Path:
    """Write ``<onnx_path>.manifest.json`` physical deployment provenance sidecar."""
    myosuite = import_myosuite()
    from myosuite.utils.deployment_manifest import (
        env_deployment_manifest,
        write_deployment_manifest,
    )

    onnx_file = Path(onnx_path)
    if features:
        from myosuite.core.config import EnvConfig
        from myosuite.utils.feature_cli import feature_spec

        parsed = tuple(feature_spec(f) if isinstance(f, str) else f for f in features)
        env = myosuite.make_env(EnvConfig(env_id=env_id, features=parsed))
    else:
        env = myosuite.make_env(env_id)
    try:
        manifest = env_deployment_manifest(
            env,
            artifacts={onnx_file.name: onnx_file},
            normalization={"kind": "symlog" if symlog else "none"},
        )
        return write_deployment_manifest(onnx_file, manifest)
    finally:
        env.close()


def export_dreamerv3_onnx(
    policy_or_checkpoint: DreamerV3TorchPolicy | str | Path | dict[str, Any],
    output: str | Path,
    *,
    opset: int = 17,
    env_id: str = "",
    features: Sequence[Any] = (),
    write_manifest: bool = True,
    meta: dict[str, Any] | None = None,
    revision: str | None = None,
) -> Path:
    """Export a deterministic stateful DreamerV3 policy to a single ``.onnx`` file.

    Embeds zero default initializers for ``(deter, stoch, prevact, is_first=1)`` so
    the exported graph works both for stateful recurrent rollouts and for stateless
    single-input callers such as ``myosuite.utils.export_onnx.verify_onnx_on_cpu``.
    When ``env_id`` is provided (or recorded in ``policy_meta.json``), also writes
    ``<output>.manifest.json`` so ``verify_onnx_on_cpu`` validates strictly without
    requiring ``allow_provenance_mismatch=True``.
    """
    resolved_env_id = env_id
    if isinstance(policy_or_checkpoint, DreamerV3TorchPolicy):
        policy = policy_or_checkpoint
        if not resolved_env_id and meta:
            resolved_env_id = str(meta.get("env_id", ""))
    else:
        if not resolved_env_id and isinstance(policy_or_checkpoint, (str, Path)):
            try:
                _, disk_meta = _locate_agent_pkl_and_meta(
                    policy_or_checkpoint, revision=revision
                )
                resolved_env_id = str(disk_meta.get("env_id", ""))
            except Exception:
                pass
        policy = load_dreamerv3_torch_policy(
            policy_or_checkpoint, meta=meta, revision=revision
        )
    policy = policy.cpu().eval()
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    obs = torch.zeros(1, policy.obs_dim, dtype=torch.float32)
    deter, stoch, prevact = policy.initial_state(1)
    is_first = torch.ones(1, dtype=torch.float32)

    torch.onnx.export(
        policy,
        (obs, deter, stoch, prevact, is_first),
        str(out_path),
        opset_version=opset,
        input_names=["obs", "deter", "stoch", "prevact", "is_first"],
        output_names=["action", "next_deter", "next_stoch"],
        dynamic_axes={
            "obs": {0: "batch"},
            "deter": {0: "batch"},
            "stoch": {0: "batch"},
            "prevact": {0: "batch"},
            "is_first": {0: "batch"},
            "action": {0: "batch"},
            "next_deter": {0: "batch"},
            "next_stoch": {0: "batch"},
        },
        do_constant_folding=True,
        dynamo=False,
    )

    try:
        import onnx
        from onnx import numpy_helper

        model_proto = onnx.load(str(out_path))
        existing = {init.name for init in model_proto.graph.initializer}
        defaults = {
            "deter": np.zeros((1, policy.deter_dim), dtype=np.float32),
            "stoch": np.zeros((1, policy.stoch_dim, policy.classes), dtype=np.float32),
            "prevact": np.zeros((1, policy.act_dim), dtype=np.float32),
            "is_first": np.ones((1,), dtype=np.float32),
        }
        for name, arr in defaults.items():
            if name not in existing:
                model_proto.graph.initializer.append(numpy_helper.from_array(arr, name))
        onnx.save(model_proto, str(out_path))
    except ImportError:
        pass

    if write_manifest and resolved_env_id:
        write_onnx_deployment_manifest(
            out_path,
            resolved_env_id,
            features=features,
            symlog=policy.symlog,
        )

    return out_path


def evaluate_dreamer_torch_policy(
    policy: DreamerV3TorchPolicy,
    env_id: str,
    episodes: int = 10,
    seed: int = 0,
    features: Sequence[Any] = (),
) -> dict[str, float]:
    """Evaluate a ``DreamerV3TorchPolicy`` deterministically in memory on CPU ``make_env``."""
    myosuite = import_myosuite()
    policy = policy.cpu().eval()
    # Batch-1 recurrent inference is latency-bound: PyTorch's default of one thread
    # per core oversubscribes the CPU (70x slower next to a running trainer).
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(max(1, min(prev_threads, 4)))
    try:
        return _evaluate_dreamer_torch_policy(
            policy, myosuite, env_id, episodes, seed, features
        )
    finally:
        torch.set_num_threads(prev_threads)


def _evaluate_dreamer_torch_policy(
    policy: DreamerV3TorchPolicy,
    myosuite: Any,
    env_id: str,
    episodes: int,
    seed: int,
    features: Sequence[Any],
) -> dict[str, float]:
    def _make_env() -> Any:
        if features:
            from myosuite.core.config import EnvConfig
            from myosuite.utils.feature_cli import feature_spec

            parsed = tuple(
                feature_spec(f) if isinstance(f, str) else f for f in features
            )
            return myosuite.make_env(EnvConfig(env_id=env_id, features=parsed))
        return myosuite.make_env(env_id)

    returns: list[float] = []
    solved_flags: list[float] = []
    final_flags: list[float] = []
    lengths: list[int] = []

    with torch.no_grad():
        for ep in range(episodes):
            env = _make_env()
            try:
                obs, _ = env.reset(seed=int(seed + ep))
                deter, stoch, prevact = policy.initial_state(1)
                is_first = torch.ones(1, dtype=torch.float32)
                lo = np.asarray(env.action_space.low, dtype=np.float32)
                hi = np.asarray(env.action_space.high, dtype=np.float32)

                done = False
                ep_ret = 0.0
                ep_solved = False
                ep_len = 0
                while not done:
                    obs_t = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
                    action, deter, stoch = policy(
                        obs_t, deter, stoch, prevact, is_first
                    )
                    prevact = action
                    is_first = torch.zeros(1, dtype=torch.float32)
                    act_vec = np.clip(action.squeeze(0).cpu().numpy(), -1.0, 1.0)
                    # Same linear [-1, 1] -> action_space map as NormalizeAction.
                    obs, rew, term, trunc, info = env.step(
                        lo + (act_vec + 1.0) * 0.5 * (hi - lo)
                    )
                    ep_ret += float(rew)
                    rwd_dict = info.get("rwd_dict", {})
                    solved = info.get("solved", rwd_dict.get("solved", 0.0))
                    step_solved = float(np.asarray(solved).mean()) > 0.5
                    ep_solved = ep_solved or step_solved
                    ep_len += 1
                    done = bool(term or trunc)
                returns.append(ep_ret)
                solved_flags.append(1.0 if ep_solved else 0.0)
                final_flags.append(1.0 if step_solved else 0.0)
                lengths.append(ep_len)
            finally:
                env.close()

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "solved_rate": float(np.mean(solved_flags)),
        "solved_final_rate": float(np.mean(final_flags)),
        "mean_length": float(np.mean(lengths)),
    }


def evaluate_exported_onnx_stateful(
    onnx_path: str | Path,
    env_id: str,
    episodes: int = 10,
    seed: int = 0,
    features: Sequence[Any] = (),
) -> dict[str, float]:
    """Evaluate a stateful DreamerV3 ``.onnx`` policy on CPU ``myosuite.make_env``."""
    from myosuite.utils.onnxruntime_utils import require_onnxruntime

    ort = require_onnxruntime()
    myosuite = import_myosuite()
    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3
    sess = ort.InferenceSession(
        str(onnx_path), sess_options=sess_opts, providers=["CPUExecutionProvider"]
    )

    out_shapes = {out.name: out.shape for out in sess.get_outputs()}
    act_dim = int(out_shapes["action"][1])
    deter_dim = int(out_shapes["next_deter"][1])
    stoch_dim = int(out_shapes["next_stoch"][1])
    classes = int(out_shapes["next_stoch"][2])

    def _make_env() -> Any:
        if features:
            from myosuite.core.config import EnvConfig
            from myosuite.utils.feature_cli import feature_spec

            parsed = tuple(
                feature_spec(f) if isinstance(f, str) else f for f in features
            )
            return myosuite.make_env(EnvConfig(env_id=env_id, features=parsed))
        return myosuite.make_env(env_id)

    returns: list[float] = []
    solved_flags: list[float] = []
    final_flags: list[float] = []
    lengths: list[int] = []

    for ep in range(episodes):
        env = _make_env()
        try:
            obs, _ = env.reset(seed=int(seed + ep))
            deter = np.zeros((1, deter_dim), dtype=np.float32)
            stoch = np.zeros((1, stoch_dim, classes), dtype=np.float32)
            prevact = np.zeros((1, act_dim), dtype=np.float32)
            is_first = np.ones((1,), dtype=np.float32)

            done = False
            ep_ret = 0.0
            ep_solved = False
            ep_len = 0
            while not done:
                action, deter, stoch = sess.run(
                    ["action", "next_deter", "next_stoch"],
                    {
                        "obs": np.asarray(obs, dtype=np.float32).reshape(1, -1),
                        "deter": deter,
                        "stoch": stoch,
                        "prevact": prevact,
                        "is_first": is_first,
                    },
                )
                act_vec = np.asarray(action[0], dtype=np.float32)
                # DreamerV3 acts in [-1, 1]; map linearly onto the env's action space
                # exactly like NormalizeAction (identity for [-1, 1] pose/reach envs,
                # [0, 1] activations for the leg-walk envs).
                lo = np.asarray(env.action_space.low, dtype=np.float32)
                hi = np.asarray(env.action_space.high, dtype=np.float32)
                act_vec = lo + (np.clip(act_vec, -1.0, 1.0) + 1.0) * 0.5 * (hi - lo)
                obs, rew, term, trunc, info = env.step(act_vec)
                prevact = action.astype(np.float32, copy=False)
                is_first = np.zeros((1,), dtype=np.float32)
                ep_ret += float(rew)
                rwd_dict = info.get("rwd_dict", {})
                solved = info.get("solved", rwd_dict.get("solved", 0.0))
                step_solved = float(np.asarray(solved).mean()) > 0.5
                ep_solved = ep_solved or step_solved
                ep_len += 1
                done = bool(term or trunc)
            returns.append(ep_ret)
            solved_flags.append(1.0 if ep_solved else 0.0)
            final_flags.append(1.0 if step_solved else 0.0)
            lengths.append(ep_len)
        finally:
            env.close()

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "solved_rate": float(np.mean(solved_flags)),
        "solved_final_rate": float(np.mean(final_flags)),
        "mean_length": float(np.mean(lengths)),
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """CLI entrypoint for ``dreamerv3-export``."""
    parser = argparse.ArgumentParser(
        description="Export a DreamerV3 checkpoint to stateful ONNX and/or TorchScript."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to DreamerV3 logdir, ckpt directory, agent.pkl, or pinned hf:// URI.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Pinned commit SHA or tag when --checkpoint uses hf://.",
    )
    parser.add_argument("--onnx", type=str, default="", help="Output .onnx path.")
    parser.add_argument(
        "--torchscript", type=str, default="", help="Output TorchScript .pt path."
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--verify-env",
        type=str,
        default="",
        help="Optional MyoSuite env_id for CPU playback verification.",
    )
    parser.add_argument("--verify-episodes", type=int, default=2)
    args = parser.parse_args(argv)

    if not args.onnx and not args.torchscript:
        raise ValueError("Specify at least one of --onnx or --torchscript.")

    policy = load_dreamerv3_torch_policy(args.checkpoint, revision=args.revision)
    result: dict[str, Any] = {
        "obs_dim": policy.obs_dim,
        "act_dim": policy.act_dim,
        "deter_dim": policy.deter_dim,
        "stoch_shape": (policy.stoch_dim, policy.classes),
    }
    if args.torchscript:
        export_dreamerv3_torchscript(policy, args.torchscript)
        result["torchscript"] = str(Path(args.torchscript).resolve())
    if args.onnx:
        onnx_path = export_dreamerv3_onnx(policy, args.onnx, opset=args.opset)
        result["onnx"] = str(onnx_path.resolve())
        if args.verify_env:
            result["eval"] = evaluate_exported_onnx_stateful(
                onnx_path, env_id=args.verify_env, episodes=args.verify_episodes
            )
    return result


if __name__ == "__main__":
    main()
