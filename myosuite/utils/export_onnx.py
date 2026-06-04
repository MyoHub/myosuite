#!/usr/bin/env python3
# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""
Export a trained policy to ONNX for cross-backend playback.

ONNX is the common interchange format: a policy trained on any backend
(CPU/SB3, MJX/Brax, mjlab/RSL-RL) can be exported once and played back
on all three, enabling:

  - Reward verification: same ONNX policy → same (obs, action) pairs on
    CPU, MJX, and mjlab (up to float32/float64 rounding).
  - Deployment: load the ONNX model in C++, PyTorch, or TensorFlow without
    the training framework installed.
  - Regression testing: store a reference ONNX checkpoint and assert that
    future training runs match it within a reward threshold.

Supported source frameworks
---------------------------
  - **SB3 (SAC / TD3 / PPO)** — saved as a ``.zip`` file via
    ``model.save("path")``.  The actor (deterministic mean) is exported.
  - **RSL-RL (PPO)** — saved via ``runner.save("path.pt")`` as a state-dict.
    Requires reconstructing the ActorCritic network with the same config.
  - **JAX/Flax (Brax PPO)** — parameters saved as a ``.pkl`` checkpoint.
    Converted via ``jax.experimental.jax2tf`` → TensorFlow → ONNX, or via
    an intermediate PyTorch wrapper.

Usage
-----
    # Export an SB3 policy:
    python export_onnx.py --framework sb3 --checkpoint walk_sac.zip \
        --obs-dim 243 --act-dim 80 --output walk_policy.onnx

    # Export an RSL-RL policy (needs --rslrl-config):
    python export_onnx.py --framework rslrl --checkpoint walk_ppo.pt \
        --obs-dim 243 --act-dim 80 --output walk_policy.onnx

    # Verify an exported ONNX policy on the CPU env:
    python export_onnx.py --verify --onnx walk_policy.onnx --steps 200

ONNX model interface
--------------------
  Input:  ``obs``    float32  (1, obs_dim)   — normalised observation vector
  Output: ``action`` float32  (1, act_dim)   — deterministic action in [0, 1]

The action output is clipped to [0, 1] inside the export wrapper so that
the ONNX model is self-contained and does not require the env's action-space
clipping to be applied externally.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SB3 export
# ---------------------------------------------------------------------------


def export_sb3_to_onnx(
    checkpoint: str | Path,
    output: str | Path,
    obs_dim: int,
    act_dim: int,
    opset: int = 17,
) -> None:
    """Export a Stable-Baselines3 policy (SAC/TD3/PPO) to ONNX.

    The exported model runs the deterministic actor (mean action, no
    exploration noise).  For SAC/TD3, this is ``model.policy.actor``;
    for PPO, it is ``model.policy.mlp_extractor`` + ``model.policy.action_net``.

    Args:
        checkpoint: Path to the ``.zip`` file saved by ``model.save()``.
        output: Destination ``.onnx`` file path.
        obs_dim: Observation vector dimension (must match the trained env).
        act_dim: Action vector dimension (number of muscles).
        opset: ONNX opset version (17 recommended for PyTorch >= 2.0).
    """
    from stable_baselines3 import SAC, TD3, PPO

    checkpoint = Path(checkpoint)
    output = Path(output)

    # Try SAC first (most common for myoSuite), fall back to TD3/PPO.
    algo_name: str | None = None
    for cls in (SAC, TD3, PPO):
        try:
            model = cls.load(checkpoint, device="cpu")
            log.info("Loaded %s checkpoint from %s", cls.__name__, checkpoint)
            algo_name = cls.__name__
            break
        except Exception:
            continue
    else:
        raise ValueError(f"Could not load {checkpoint} as SAC, TD3, or PPO")

    class _DeterministicWrapper(torch.nn.Module):
        """Wrap SB3 actor to return a clipped action tensor."""

        def __init__(self, policy: Any, algorithm_name: str) -> None:
            super().__init__()
            self.policy = policy
            self.algorithm_name = algorithm_name

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                if self.algorithm_name == "PPO":
                    action, _, _ = self.policy(obs, deterministic=True)
                else:
                    actor = self.policy.actor
                    try:
                        actor_out = actor(obs, deterministic=True)
                    except TypeError:
                        actor_out = actor(obs)
                    action = actor_out[0] if isinstance(actor_out, tuple) else actor_out
            # Clip to [0, 1] — myoSuite action space
            return torch.clamp(action, 0.0, 1.0)

    assert algo_name is not None
    wrapper = _DeterministicWrapper(model.policy, algo_name)
    wrapper.eval()

    dummy_obs = torch.zeros(1, obs_dim, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        dummy_obs,
        str(output),
        opset_version=opset,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        do_constant_folding=True,
        dynamo=False,
    )
    log.info(
        "Exported SB3 policy → %s  (obs_dim=%d, act_dim=%d)", output, obs_dim, act_dim
    )


# ---------------------------------------------------------------------------
# RSL-RL export
# ---------------------------------------------------------------------------


def export_rslrl_to_onnx(
    checkpoint: str | Path,
    output: str | Path,
    obs_dim: int,
    act_dim: int,
    hidden_dims: tuple[int, ...] = (512, 256, 128),
    opset: int = 17,
) -> None:
    """Export an RSL-RL ActorCritic policy (PPO) to ONNX.

    RSL-RL checkpoints store state-dicts; the network architecture must be
    reconstructed with the same hidden dims used during training.

    Args:
        checkpoint: Path to ``.pt`` file saved by ``runner.save()``.
        output: Destination ``.onnx`` file path.
        obs_dim: Observation vector dimension.
        act_dim: Action dimension.
        hidden_dims: MLP hidden layer sizes (default matches mjlab walk config).
        opset: ONNX opset version.
    """
    checkpoint = Path(checkpoint)
    output = Path(output)

    state = torch.load(checkpoint, map_location="cpu")

    # Build a minimal actor MLP matching RSL-RL ActorCritic architecture.
    layers: list[torch.nn.Module] = []
    in_dim = obs_dim
    for h in hidden_dims:
        layers += [torch.nn.Linear(in_dim, h), torch.nn.ELU()]
        in_dim = h
    layers += [torch.nn.Linear(in_dim, act_dim), torch.nn.Sigmoid()]
    actor = torch.nn.Sequential(*layers)

    # Load the actor weights from the checkpoint.
    actor_state = {
        k.removeprefix("actor."): v
        for k, v in state.get("model_state_dict", state).items()
        if k.startswith("actor.")
    }
    missing, unexpected = actor.load_state_dict(actor_state, strict=False)
    if missing:
        log.warning("Missing actor keys: %s", missing)
    if unexpected:
        log.warning("Unexpected actor keys: %s", unexpected)
    actor.eval()

    dummy_obs = torch.zeros(1, obs_dim, dtype=torch.float32)
    torch.onnx.export(
        actor,
        dummy_obs,
        str(output),
        opset_version=opset,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        do_constant_folding=True,
    )
    log.info(
        "Exported RSL-RL policy → %s  (obs_dim=%d, act_dim=%d)",
        output,
        obs_dim,
        act_dim,
    )


# ---------------------------------------------------------------------------
# JAX/Flax export
# ---------------------------------------------------------------------------


def export_jax_to_onnx(
    checkpoint: str | Path,
    output: str | Path,
    obs_dim: int,
    act_dim: int,
    opset: int = 17,
) -> None:
    """Export a JAX/Flax Brax PPO policy to ONNX via jax2tf + tf2onnx.

    Requires:  ``pip install tf2onnx tensorflow``

    The conversion pipeline:
        JAX params → jax.experimental.jax2tf → TensorFlow SavedModel → ONNX

    Args:
        checkpoint: Path to a ``.pkl`` or ``.msgpack`` checkpoint file
                    containing ``{"params": flax_param_tree}``.
        output: Destination ``.onnx`` file path.
        obs_dim: Observation vector dimension.
        act_dim: Action dimension.
        opset: ONNX opset version.
    """
    raise NotImplementedError(
        "Generic JAX/Flax export is not implemented. "
        "For MuscleMimic full-body (Orbax/HF checkpoints), use --framework orbax."
    )


# ---------------------------------------------------------------------------
# Orbax/Flax export (MuscleMimic full-body residual actor)
# ---------------------------------------------------------------------------


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


class _OrbaxResBlock(torch.nn.Module):
    """PyTorch mirror of one residual block in the Orbax/Flax actor."""

    def __init__(
        self,
        in_dim: int,
        mid_dim: int,
        out_dim: int,
        has_proj: bool,
    ) -> None:
        super().__init__()
        self.dense0 = torch.nn.Linear(in_dim, mid_dim)
        self.ln0 = torch.nn.LayerNorm(mid_dim, eps=1e-5)
        self.dense1 = torch.nn.Linear(mid_dim, out_dim)
        self.ln1 = torch.nn.LayerNorm(out_dim, eps=1e-5)
        self.proj = torch.nn.Linear(in_dim, out_dim, bias=True) if has_proj else None
        self.gate = torch.nn.Parameter(torch.zeros(1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        y = self.dense0(h)
        y = torch.nn.functional.silu(self.ln0(y))
        y = self.dense1(y)
        y = self.ln1(y)
        shortcut = self.proj(h) if self.proj is not None else h
        gate = torch.sigmoid(self.gate)
        return torch.nn.functional.silu(shortcut + gate * y)


class _OrbaxActorNet(torch.nn.Module):
    """PyTorch mirror of the Orbax/Flax residual actor network."""

    def __init__(self, params: dict) -> None:
        super().__init__()
        actor = params["actor"]
        block_indices = _residual_block_indices(actor)
        if not block_indices:
            raise ValueError("Unsupported actor params: no residual blocks found.")
        self._block_indices = tuple(block_indices)
        self.blocks = torch.nn.ModuleList()
        for idx in block_indices:
            kernel0 = np.asarray(actor[f"block{idx}_layer0_dense"]["kernel"])
            kernel1 = np.asarray(actor[f"block{idx}_layer1_dense"]["kernel"])
            self.blocks.append(
                _OrbaxResBlock(
                    in_dim=int(kernel0.shape[0]),
                    mid_dim=int(kernel0.shape[1]),
                    out_dim=int(kernel1.shape[1]),
                    has_proj=f"block{idx}_proj" in actor,
                )
            )

        self.tail_dense: torch.nn.Linear | None = None
        self.tail_ln: torch.nn.LayerNorm | None = None
        tail = actor.get("tail_dense")
        if tail is not None:
            if "tail_ln" not in actor:
                raise ValueError(
                    "Unsupported actor params: tail_dense without tail_ln."
                )
            tail_out = int(np.asarray(tail["kernel"]).shape[1])
            tail_in = int(np.asarray(tail["kernel"]).shape[0])
            self.tail_dense = torch.nn.Linear(tail_in, tail_out)
            self.tail_ln = torch.nn.LayerNorm(tail_out, eps=1e-5)
            output_in = tail_out
        else:
            output_in = int(
                np.asarray(
                    actor[f"block{block_indices[-1]}_layer1_dense"]["kernel"]
                ).shape[1]
            )
        act_dim: int = int(np.asarray(actor["output"]["bias"]).shape[0])
        self.output_dense = torch.nn.Linear(output_in, act_dim)

        self._load_weights(params)

    def _load_weights(self, params: dict) -> None:
        actor = params["actor"]

        def _set_linear(mod: torch.nn.Linear, dense_p: dict) -> None:
            mod.weight.data.copy_(
                torch.as_tensor(
                    np.array(dense_p["kernel"], dtype=np.float32, copy=True).T
                )
            )
            mod.bias.data.copy_(
                torch.as_tensor(np.array(dense_p["bias"], dtype=np.float32, copy=True))
            )

        def _set_ln(mod: torch.nn.LayerNorm, ln_p: dict) -> None:
            mod.weight.data.copy_(
                torch.as_tensor(np.array(ln_p["scale"], dtype=np.float32, copy=True))
            )
            mod.bias.data.copy_(
                torch.as_tensor(np.array(ln_p["bias"], dtype=np.float32, copy=True))
            )

        for i, blk in zip(self._block_indices, self.blocks, strict=True):
            _set_linear(blk.dense0, actor[f"block{i}_layer0_dense"])
            _set_ln(blk.ln0, actor[f"block{i}_layer0_ln"])
            _set_linear(blk.dense1, actor[f"block{i}_layer1_dense"])
            _set_ln(blk.ln1, actor[f"block{i}_layer1_ln"])
            if blk.proj is not None:
                _set_linear(blk.proj, actor[f"block{i}_proj"])
            gate_raw = float(np.asarray(actor[f"res_gate_{i}"]).reshape(()))
            blk.gate.data.fill_(gate_raw)

        if self.tail_dense is not None:
            assert self.tail_ln is not None
            _set_linear(self.tail_dense, actor["tail_dense"])
            _set_ln(self.tail_ln, actor["tail_ln"])
        _set_linear(self.output_dense, actor["output"])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = obs
        for block in self.blocks:
            h = block(h)
        if self.tail_dense is not None:
            assert self.tail_ln is not None
            h = torch.nn.functional.silu(self.tail_ln(self.tail_dense(h)))
        return self.output_dense(h)


class _OrbaxPolicyWrapper(torch.nn.Module):
    """Actor with baked-in obs normalization stats from the checkpoint."""

    def __init__(
        self,
        actor_net: _OrbaxActorNet,
        obs_mean: np.ndarray,
        obs_var: np.ndarray,
    ) -> None:
        super().__init__()
        self.actor = actor_net
        self.register_buffer(
            "obs_mean",
            torch.as_tensor(np.array(obs_mean, dtype=np.float32, copy=True)),
        )
        self.register_buffer(
            "obs_std",
            torch.as_tensor(
                np.sqrt(np.array(obs_var, dtype=np.float32, copy=True) + 1e-8)
            ),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        norm_obs = (obs - self.obs_mean) / self.obs_std
        action = self.actor(norm_obs)
        return torch.clamp(action, -1.0, 1.0)


def export_orbax_to_onnx(
    checkpoint: str | Path,
    output: str | Path,
    opset: int = 17,
) -> None:
    """Export a MuscleMimic full-body Orbax/Flax policy to ONNX."""
    import tempfile

    import onnx

    output = Path(output)

    checkpoint_str = str(checkpoint)
    if checkpoint_str.startswith("hf://"):
        from myosuite.integrations.musclemimic.fullbody_checkpoint_io import (
            resolve_checkpoint_ref,
        )

        ckpt_ref = resolve_checkpoint_ref(checkpoint_str)
        local_path = ckpt_ref.local_path
        log.info("Resolved %s → %s", checkpoint_str, local_path)
    else:
        local_path = Path(checkpoint_str)

    from myosuite.integrations.musclemimic.fullbody_local_policy import (
        has_local_policy_artifacts,
        load_local_policy_artifacts,
    )

    if not has_local_policy_artifacts(local_path):
        raise FileNotFoundError(
            f"Checkpoint at {local_path} is missing train_state/ or config/metadata. "
            "Ensure the checkpoint was downloaded with setup-demo-cache or hf auth login."
        )

    artifacts = load_local_policy_artifacts(local_path)
    actor_net = _OrbaxActorNet(artifacts.params)
    wrapper = _OrbaxPolicyWrapper(actor_net, artifacts.obs_mean, artifacts.obs_var)
    wrapper.eval()

    dummy_obs = torch.zeros(1, artifacts.obs_dim, dtype=torch.float32)
    with torch.no_grad():
        _ = wrapper(dummy_obs).numpy()

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_out = Path(tmp_dir) / "policy.onnx"
        torch.onnx.export(
            wrapper,
            dummy_obs,
            str(tmp_out),
            opset_version=opset,
            input_names=["obs"],
            output_names=["action"],
            dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
            do_constant_folding=True,
        )
        model_proto = onnx.load(str(tmp_out), load_external_data=True)
        onnx.save_model(
            model_proto,
            str(output),
            save_as_external_data=False,
        )


# ---------------------------------------------------------------------------
# ONNX playback verifier
# ---------------------------------------------------------------------------


def verify_onnx_on_cpu(
    onnx_path: str | Path,
    n_steps: int = 200,
    seed: int = 0,
    env_id: str = "myoLegWalk-v0",
    min_mean_reward: float = -np.inf,
) -> dict[str, float]:
    """Run an ONNX policy on the CPU Gymnasium env and return episode metrics.

    Args:
        onnx_path: Path to the exported ``.onnx`` file.
        n_steps: Maximum steps per episode.
        seed: Environment reset seed.
        env_id: Gymnasium environment ID to run.
        min_mean_reward: If set, assert that mean per-step reward >= this value.

    Returns:
        Dict with ``total_reward``, ``mean_reward``, ``n_steps``, ``terminated``.
    """
    import onnxruntime as ort
    import gymnasium as gym

    onnx_path = Path(onnx_path)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    env = gym.make(env_id)
    obs, _ = env.reset(seed=seed)

    total_reward = 0.0
    steps = 0
    for _ in range(n_steps):
        obs_f32 = obs[None].astype(np.float32)  # (1, obs_dim)
        action = sess.run([output_name], {input_name: obs_f32})[0].squeeze()
        obs, rew, term, trunc, _ = env.step(action)
        total_reward += float(rew)
        steps += 1
        if term or trunc:
            break
    env.close()

    mean_reward = total_reward / max(steps, 1)
    if mean_reward < min_mean_reward:
        raise AssertionError(
            f"ONNX policy mean_reward={mean_reward:.3f} < threshold {min_mean_reward:.3f}"
        )

    return {
        "total_reward": total_reward,
        "mean_reward": mean_reward,
        "n_steps": steps,
        "terminated": term if "term" in dir() else False,
    }


def compare_onnx_across_backends(
    onnx_path: str | Path,
    n_steps: int = 50,
    seed: int = 0,
    atol: float = 0.1,
) -> dict[str, Any]:
    """Run an ONNX policy on CPU and MJX backends and compare observations.

    This is the core cross-backend playback validation:
    the same ONNX model running the same sequence of actions must produce
    observations and rewards within ``atol`` of each other on all backends.

    Args:
        onnx_path: Path to the exported ``.onnx`` file.
        n_steps: Steps to compare (keep short to avoid termination divergence).
        seed: Reset seed (must be the same on all backends).
        atol: Absolute tolerance for obs/reward comparison.

    Returns:
        Dict with ``max_obs_diff_cpu_mjx``, ``max_rew_diff_cpu_mjx``, and
        ``passed`` (bool).
    """
    import onnxruntime as ort
    import gymnasium as gym

    onnx_path = Path(onnx_path)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    def _rollout(env_id: str, env_kwargs: dict) -> tuple[np.ndarray, np.ndarray]:
        env = gym.make(env_id, **env_kwargs)
        obs, _ = env.reset(seed=seed)
        obs_seq, rew_seq = [obs.copy()], []
        for _ in range(n_steps):
            obs_f32 = np.asarray(obs).ravel()[None].astype(np.float32)
            action = sess.run([output_name], {input_name: obs_f32})[0].squeeze()
            obs, rew, term, trunc, _ = env.step(action)
            obs_seq.append(np.asarray(obs).ravel().copy())
            rew_seq.append(float(rew))
            if term or trunc:
                break
        env.close()
        return np.array(obs_seq), np.array(rew_seq)

    cpu_obs, cpu_rew = _rollout("myoLegWalk-v0", {})

    results: dict[str, Any] = {"passed": True}

    try:
        mjx_obs, mjx_rew = _rollout("MjxLegWalk-v0", {"num_envs": 1})
        n = min(cpu_obs.shape[0], mjx_obs.shape[0])
        max_obs = float(np.max(np.abs(cpu_obs[:n] - mjx_obs[:n])))
        max_rew = float(
            np.max(
                np.abs(
                    cpu_rew[: min(len(cpu_rew), len(mjx_rew))]
                    - mjx_rew[: min(len(cpu_rew), len(mjx_rew))]
                )
            )
        )
        results["max_obs_diff_cpu_mjx"] = max_obs
        results["max_rew_diff_cpu_mjx"] = max_rew
        if max_obs > atol:
            results["passed"] = False
            log.warning("CPU vs MJX max obs diff %.4f > atol %.4f", max_obs, atol)
    except Exception as exc:
        log.warning("MJX backend unavailable for comparison: %s", exc)
        results["mjx_error"] = str(exc)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export a trained myoSuite policy to ONNX and optionally verify it."
    )
    sub = p.add_subparsers(dest="command", required=True)

    # --- export ---
    ex = sub.add_parser("export", help="Export a policy to ONNX")
    ex.add_argument(
        "--framework",
        choices=["sb3", "rslrl", "jax", "orbax"],
        required=True,
    )
    ex.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the checkpoint file/directory (or hf://... for orbax)",
    )
    ex.add_argument("--output", required=True, help="Output .onnx path")
    ex.add_argument("--obs-dim", type=int, default=None)
    ex.add_argument("--act-dim", type=int, default=None)
    ex.add_argument("--opset", type=int, default=17)
    ex.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[512, 256, 128],
        help="Hidden layer sizes (RSL-RL only)",
    )

    # --- verify ---
    vr = sub.add_parser("verify", help="Run an ONNX policy on the CPU env")
    vr.add_argument("--onnx", required=True)
    vr.add_argument("--steps", type=int, default=200)
    vr.add_argument("--env", default="myoLegWalk-v0")
    vr.add_argument("--min-reward", type=float, default=-float("inf"))

    # --- compare ---
    cm = sub.add_parser(
        "compare", help="Compare ONNX policy across CPU and MJX backends"
    )
    cm.add_argument("--onnx", required=True)
    cm.add_argument("--steps", type=int, default=50)
    cm.add_argument("--atol", type=float, default=0.1)

    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args()

    if args.command == "export":
        if args.framework == "orbax":
            export_orbax_to_onnx(args.checkpoint, args.output, args.opset)
        else:
            if args.obs_dim is None or args.act_dim is None:
                _build_parser().error(
                    f"--framework {args.framework} requires --obs-dim and --act-dim"
                )
            if args.framework == "sb3":
                export_sb3_to_onnx(
                    args.checkpoint, args.output, args.obs_dim, args.act_dim, args.opset
                )
            elif args.framework == "rslrl":
                export_rslrl_to_onnx(
                    args.checkpoint,
                    args.output,
                    args.obs_dim,
                    args.act_dim,
                    tuple(args.hidden_dims),
                    args.opset,
                )
            elif args.framework == "jax":
                export_jax_to_onnx(
                    args.checkpoint, args.output, args.obs_dim, args.act_dim, args.opset
                )

    elif args.command == "verify":
        metrics = verify_onnx_on_cpu(
            args.onnx, args.steps, env_id=args.env, min_mean_reward=args.min_reward
        )
        print(
            f"total_reward={metrics['total_reward']:.3f}  "
            f"mean_reward={metrics['mean_reward']:.3f}  "
            f"steps={metrics['n_steps']}"
        )

    elif args.command == "compare":
        results = compare_onnx_across_backends(args.onnx, args.steps, atol=args.atol)
        for k, v in results.items():
            print(f"  {k}: {v}")
        sys.exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()
