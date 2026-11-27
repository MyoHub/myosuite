# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Orbax actor checkpoint parameter and normalization-statistics loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from myosuite.policies.actor_tree import sorted_prefix_indices as _sorted_prefix_indices


def _to_numpy_tree(tree: Any) -> Any:
    if isinstance(tree, dict):
        return {k: _to_numpy_tree(v) for k, v in tree.items()}
    return np.asarray(tree, dtype=np.float32)


@dataclass(frozen=True)
class LocalPolicyArtifacts:
    """Local policy artifacts parsed from an Orbax checkpoint."""

    params: dict[str, Any]
    obs_mean: np.ndarray
    obs_var: np.ndarray
    obs_count: np.ndarray
    obs_dim: int
    action_dim: int


def has_local_policy_artifacts(checkpoint_root: Path) -> bool:
    """Return ``True`` if checkpoint root looks loadable for local policy."""
    p = Path(checkpoint_root)
    return (p / "train_state").is_dir() and (p / "config" / "metadata").exists()


def _restore_on_cpu(checkpointer: Any, path: str) -> Any:
    """Restore an Orbax checkpoint onto this process's CPU device.

    Without a target tree Orbax rebuilds each array's sharding from the device
    name saved in the checkpoint (``TFRT_CPU_0`` for ``mm-10m-2``), which JAX
    >= 0.10 no longer reports. A target on ``jax.devices("cpu")[0]`` restores
    on any JAX version. The target also turns scalar leaves into JAX arrays and,
    with x64 disabled, narrows 64-bit leaves to 32-bit; callers here cast to
    float32 anyway.
    """
    if not hasattr(checkpointer, "metadata"):
        return checkpointer.restore(path)
    import jax

    meta = checkpointer.metadata(path)
    tree = getattr(meta, "item_metadata", meta)
    tree = getattr(tree, "tree", tree)
    if not isinstance(tree, dict):
        return checkpointer.restore(path)
    cpu = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])

    def abstract(leaf: Any) -> Any:
        if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
            return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=cpu)
        return leaf

    return checkpointer.restore(path, jax.tree_util.tree_map(abstract, tree))


def load_local_policy_artifacts(
    checkpoint_root: Path, seed: int = 0
) -> LocalPolicyArtifacts:
    """Load policy parameters and normalization stats from Orbax artifacts."""
    p = Path(checkpoint_root).absolute()
    train_state_dir = p / "train_state"
    if not train_state_dir.is_dir():
        raise FileNotFoundError(f"train_state directory missing: {train_state_dir}")
    try:
        import orbax.checkpoint as ocp
    except ImportError as err:
        raise ImportError(
            "orbax.checkpoint is required for local policy inference."
        ) from err

    restored = _restore_on_cpu(ocp.StandardCheckpointer(), str(train_state_dir))
    params_raw = restored["params"]
    run_stats = restored["run_stats"]["RunningMeanStd_0"]
    params = _to_numpy_tree(params_raw)
    obs_mean = np.asarray(run_stats["mean"], dtype=np.float32)
    obs_var = np.asarray(run_stats["var"], dtype=np.float32)
    obs_count = np.asarray(run_stats.get("count", 1e-6), dtype=np.float32)
    if obs_mean.ndim == 2:

        def select(tree: Any) -> Any:
            if isinstance(tree, dict):
                return {key: select(value) for key, value in tree.items()}
            return tree[seed]

        params = select(params)
        obs_mean, obs_var = obs_mean[seed], obs_var[seed]
        if obs_count.ndim:
            obs_count = obs_count[seed]
    actor = params["actor"]
    if "output" in actor:
        action_dim = int(np.asarray(actor["output"]["bias"]).shape[0])
        obs_dim = int(np.asarray(actor["block0_layer0_dense"]["kernel"]).shape[0])
    else:
        dense_indices = _sorted_prefix_indices(actor, "Dense_")
        if not dense_indices:
            raise ValueError("Unsupported actor params: no Dense_* layers found.")
        first_dense = actor[f"Dense_{dense_indices[0]}"]
        last_dense = actor[f"Dense_{dense_indices[-1]}"]
        obs_dim = int(np.asarray(first_dense["kernel"]).shape[0])
        action_dim = int(np.asarray(last_dense["bias"]).shape[0])
    return LocalPolicyArtifacts(
        params=params,
        obs_mean=obs_mean,
        obs_var=obs_var,
        obs_count=obs_count,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )


__all__ = [
    "LocalPolicyArtifacts",
    "_restore_on_cpu",
    "_to_numpy_tree",
    "has_local_policy_artifacts",
    "load_local_policy_artifacts",
]
