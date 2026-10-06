# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Helpers for bundling resumable training checkpoints into ONNX files.

The exported ONNX graph remains directly usable for inference, while a compressed
copy of the framework-native training checkpoint is embedded in the model
metadata so scripts can resume training from the same ``.onnx`` file.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from myosuite.integrations.musclemimic.actor_onnx import load_onnx_session

try:
    import wandb
except ImportError:  # wandb is optional; only get_wandb_onnx_checkpoint_path needs it.
    wandb = None

_BUNDLE_META_KEY = "myosuite.checkpoint_bundle.v1.meta"
_BUNDLE_PAYLOAD_KEY = "myosuite.checkpoint_bundle.v1.payload_gzip_base64"
_FATIGUE_STATE_KEY = "fatigue_state"
_VEC_NORMALIZE_KEY = "vec_normalize"
_MODEL_STEP_ONNX_RE = re.compile(r"^model_(\d+)\.onnx$")
_MODEL_STEP_PT_RE = re.compile(r"^model_(\d+)\.pt$")


def _load_model_props(onnx_path: Path) -> tuple[Any, dict[str, str]]:
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    props = {entry.key: entry.value for entry in model.metadata_props}
    return model, props


def bundle_onnx_with_checkpoint(
    onnx_path: str | Path,
    checkpoint_path: str | Path,
    *,
    framework: str,
    metadata: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Embed a framework-native checkpoint inside an ONNX file.

    Args:
        onnx_path: Existing ONNX model path.
        checkpoint_path: Native checkpoint file to embed.
        framework: Training stack identifier, e.g. ``"sb3-ppo"``.
        metadata: Extra JSON-serializable metadata to persist.
        output_path: Optional destination path. Defaults to in-place update.

    Returns:
        Resolved path to the bundled ONNX file.
    """
    import onnx

    source_onnx = Path(onnx_path)
    checkpoint = Path(checkpoint_path)
    target = Path(output_path) if output_path is not None else source_onnx
    target.parent.mkdir(parents=True, exist_ok=True)

    model, props = _load_model_props(source_onnx)
    payload = checkpoint.read_bytes()
    meta = {
        "framework": framework,
        "checkpoint_name": checkpoint.name,
        "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
        "metadata": metadata or {},
    }
    props[_BUNDLE_META_KEY] = json.dumps(meta, sort_keys=True)
    props[_BUNDLE_PAYLOAD_KEY] = base64.b64encode(gzip.compress(payload)).decode(
        "ascii"
    )
    onnx.helper.set_model_props(model, props)
    onnx.save_model(model, str(target), save_as_external_data=False)
    return target.resolve()


def read_onnx_checkpoint_metadata(onnx_path: str | Path) -> dict[str, Any]:
    """Return embedded checkpoint metadata from an ONNX file."""
    _, props = _load_model_props(Path(onnx_path))
    if _BUNDLE_META_KEY not in props:
        raise ValueError(f"ONNX checkpoint bundle metadata missing in {onnx_path}.")
    return json.loads(props[_BUNDLE_META_KEY])


def extract_checkpoint_from_onnx(
    onnx_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> tuple[Path, dict[str, Any], tempfile.TemporaryDirectory[str] | None]:
    """Extract the embedded native checkpoint from an ONNX bundle.

    Args:
        onnx_path: ONNX bundle created by :func:`bundle_onnx_with_checkpoint`.
        output_path: Optional extraction destination. When omitted, a temporary
            file is created and its owning temp directory is returned.

    Returns:
        Tuple of ``(checkpoint_path, metadata, temp_dir)``. ``temp_dir`` is only
        non-``None`` when *output_path* was omitted.
    """
    props_path = Path(onnx_path)
    meta = read_onnx_checkpoint_metadata(props_path)
    _, props = _load_model_props(props_path)
    if _BUNDLE_PAYLOAD_KEY not in props:
        raise ValueError(f"ONNX checkpoint payload missing in {onnx_path}.")

    payload = gzip.decompress(
        base64.b64decode(props[_BUNDLE_PAYLOAD_KEY].encode("ascii"))
    )
    payload_sha = hashlib.sha256(payload).hexdigest()
    if payload_sha != meta["checkpoint_sha256"]:
        raise ValueError(
            f"ONNX checkpoint payload hash mismatch for {onnx_path}: "
            f"expected {meta['checkpoint_sha256']}, got {payload_sha}."
        )

    if output_path is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="myosuite-onnx-ckpt-")
        destination = Path(temp_dir.name) / str(meta["checkpoint_name"])
    else:
        temp_dir = None
        destination = Path(output_path)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return destination.resolve(), meta, temp_dir


def normalize_onnx_checkpoint_name(checkpoint_name: str) -> str:
    """Normalize legacy checkpoint names to their ONNX bundle equivalents."""
    if checkpoint_name == "model_final.pt":
        return "model_final.onnx"
    match = _MODEL_STEP_PT_RE.fullmatch(checkpoint_name)
    if match is not None:
        return f"model_{match.group(1)}.onnx"
    return checkpoint_name


def is_onnx_checkpoint_name(checkpoint_name: str) -> bool:
    """Return whether *checkpoint_name* matches the ONNX checkpoint convention."""
    return checkpoint_name == "model_final.onnx" or (
        _MODEL_STEP_ONNX_RE.fullmatch(checkpoint_name) is not None
    )


def onnx_checkpoint_sort_key(checkpoint_name: str) -> tuple[int, int]:
    """Sort numeric checkpoints by step and keep ``model_final.onnx`` last."""
    if checkpoint_name == "model_final.onnx":
        return (1, 0)
    match = _MODEL_STEP_ONNX_RE.fullmatch(checkpoint_name)
    if match is None:
        return (-1, -1)
    return (0, int(match.group(1)))


def get_wandb_onnx_checkpoint_path(
    log_path: Path,
    run_path: Path,
    checkpoint_name: str | None = None,
) -> tuple[Path, bool]:
    if wandb is None:
        raise ImportError("wandb is required for get_wandb_onnx_checkpoint_path")

    run_path_str = Path(run_path).as_posix()  # W&B paths use "/" on every OS
    run_id = run_path_str.split("/")[-1]
    download_dir = log_path / "wandb_checkpoints" / run_id
    api = wandb.Api()
    wandb_run = api.run(run_path_str)
    files = [
        file.name for file in wandb_run.files() if is_onnx_checkpoint_name(file.name)
    ]
    if not files:
        raise FileNotFoundError(f"No ONNX checkpoints found in W&B run {run_path}.")
    if checkpoint_name is None:
        checkpoint_file = max(files, key=onnx_checkpoint_sort_key)
    else:
        checkpoint_name = normalize_onnx_checkpoint_name(checkpoint_name)
        if checkpoint_name not in files:
            raise ValueError(
                f"Checkpoint '{checkpoint_name}' not found in run {run_path}. "
                f"Available: {files}"
            )
        checkpoint_file = checkpoint_name
    checkpoint_path = download_dir / checkpoint_file
    was_cached = checkpoint_path.exists()
    if not was_cached:
        download_dir.mkdir(parents=True, exist_ok=True)
        wandb_run.file(checkpoint_file).download(str(download_dir), replace=True)
    return checkpoint_path, was_cached


class OnnxPolicy:
    def __init__(self, onnx_path: Path, device: str) -> None:
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device.startswith("cuda")
            else ["CPUExecutionProvider"]
        )
        self._session = load_onnx_session(onnx_path, providers=providers)
        self._device = device
        try:
            meta = read_onnx_checkpoint_metadata(onnx_path)
            self._obs_key: str | None = meta.get("metadata", {}).get("obs_key", None)
        except Exception:
            self._obs_key = None

    def __call__(self, obs):
        if hasattr(obs, "keys"):
            # TensorDict: try stored key, then "actor", then match by obs_dim
            if self._obs_key is not None and self._obs_key in obs.keys():
                actor_obs = obs[self._obs_key]
            elif "actor" in obs.keys():
                actor_obs = obs["actor"]
            else:
                obs_dim = self._session.obs_dim
                actor_obs = next(
                    (
                        v
                        for v in obs.values()
                        if hasattr(v, "shape") and v.shape[-1] == obs_dim
                    ),
                    obs,
                )
        else:
            actor_obs = obs
        return self._session.act_torch(actor_obs).to(device=self._device)

    def reset(self) -> None:
        return None


def _cpu_fatigue_models(env: Any) -> list[Any]:
    """Fatigue models of the CPU envs behind *env*, one entry (or ``None``) per env.

    Looks through gymnasium wrappers and SB3 VecEnvs: ``VecEnvWrapper``s
    (``VecNormalize``, ...) via ``.venv``, then a ``DummyVecEnv``'s ``.envs``. A
    ``SubprocVecEnv`` keeps its envs in other processes and yields none.
    ``MyoGymnasiumEnv`` tasks keep the model in the ``muscle_fatigue`` attribute
    of their ``FatigueWrapper``, the experimental ``ModularTaskEnv`` in
    ``_fatigue_model``.
    """
    while hasattr(env, "venv"):
        env = env.venv
    models = []
    for sub_env in getattr(env, "envs", None) or [env]:
        unwrapped = getattr(sub_env, "unwrapped", sub_env)
        get_attr = getattr(sub_env, "get_wrapper_attr", None)
        try:
            model = get_attr("muscle_fatigue") if get_attr else None
        except AttributeError:
            model = None
        if model is None:
            model = getattr(unwrapped, "_fatigue_model", None)
        models.append(model if hasattr(model, "state_dict") else None)
    return models


def get_env_fatigue_state(env: Any) -> dict[str, Any] | None:
    """Extract the current fatigue compartment state from an env's action terms.

    Works for the mjlab path (``ManagerBasedRlEnv`` with an action manager whose
    terms expose ``_fatigue``) and the CPU path (``muscle_fatigue`` of a
    ``MyoGymnasiumEnv`` task, ``_fatigue_model`` of ``ModularTaskEnv``), also
    behind wrappers and in-process SB3 VecEnvs.

    Args:
        env: Gymnasium-compatible env, possibly wrapped, or an SB3 VecEnv.

    Returns:
        Dict mapping action-term name → ``{"MA": ..., "MR": ..., "MF": ...}``
        (CPU: ``"cpu"`` → that dict, or a list with one per VecEnv sub-env), or
        ``None`` when no active fatigue model is found.
    """
    unwrapped = getattr(env, "unwrapped", env)
    action_manager = getattr(unwrapped, "action_manager", None)
    if action_manager is not None:
        states: dict[str, Any] = {}
        for term_name, term in action_manager._terms.items():
            fatigue = getattr(term, "_fatigue", None)
            if fatigue is not None:
                states[term_name] = fatigue.state_dict()
        return states or None
    # CPU path
    models = _cpu_fatigue_models(env)
    if not models or any(model is None for model in models):
        return None
    cpu_states = [model.state_dict() for model in models]
    return {"cpu": cpu_states[0] if len(cpu_states) == 1 else cpu_states}


def set_env_fatigue_state(env: Any, state: dict[str, Any]) -> None:
    """Restore fatigue compartment state into an env's action terms.

    A CPU state saved from a VecEnv is matched to sub-envs by index (cycled when
    the env counts differ); a single-env state is loaded into every sub-env.

    Args:
        env: Gymnasium-compatible env, possibly wrapped, or an SB3 VecEnv.
        state: Dict previously returned by :func:`get_env_fatigue_state`.
    """
    unwrapped = getattr(env, "unwrapped", env)
    action_manager = getattr(unwrapped, "action_manager", None)
    if action_manager is not None:
        for term_name, term_state in state.items():
            try:
                term = action_manager.get_term(term_name)
                fatigue = getattr(term, "_fatigue", None)
                if fatigue is not None:
                    fatigue.load_state_dict(term_state)
            except (KeyError, AttributeError):
                pass
        return
    # CPU path
    cpu_state = state.get("cpu")
    if cpu_state is None:
        return
    cpu_states = cpu_state if isinstance(cpu_state, list) else [cpu_state]
    for i, model in enumerate(_cpu_fatigue_models(env)):
        if model is not None:
            model.load_state_dict(cpu_states[i % len(cpu_states)])


def _running_mean_std_state(rms: Any) -> dict[str, Any] | None:
    """JSON data of an SB3 ``RunningMeanStd`` (float64 values round-trip exactly)."""
    if rms is None:
        return None
    return {
        "mean": np.asarray(rms.mean, dtype=np.float64).tolist(),
        "var": np.asarray(rms.var, dtype=np.float64).tolist(),
        "count": float(rms.count),
    }


def get_vec_normalize_state(vec_normalize: Any) -> dict[str, Any]:
    """Return the statistics and settings of an SB3 ``VecNormalize`` as JSON data.

    Args:
        vec_normalize: The ``VecNormalize`` a policy is trained behind (Box
            observations).

    Returns:
        Everything :func:`vec_normalize_from_state` needs to resume training.
    """
    obs_rms = getattr(vec_normalize, "obs_rms", None)
    if isinstance(obs_rms, dict):
        raise ValueError("VecNormalize of a Dict observation space is not supported.")
    return {
        "norm_obs": bool(vec_normalize.norm_obs),
        "norm_reward": bool(vec_normalize.norm_reward),
        "clip_obs": float(vec_normalize.clip_obs),
        "clip_reward": float(vec_normalize.clip_reward),
        "gamma": float(vec_normalize.gamma),
        "epsilon": float(vec_normalize.epsilon),
        "obs_rms": _running_mean_std_state(obs_rms),
        "ret_rms": _running_mean_std_state(vec_normalize.ret_rms),
    }


def vec_normalize_from_state(state: dict[str, Any], venv: Any) -> Any:
    """Wrap *venv* in an SB3 ``VecNormalize`` restored from :func:`get_vec_normalize_state`.

    Args:
        state: Output of :func:`get_vec_normalize_state`, e.g. the
            ``"vec_normalize"`` entry of an SB3 bundle's metadata.
        venv: The VecEnv to normalize.

    Returns:
        A ``VecNormalize`` in training mode with the saved statistics.
    """
    from stable_baselines3.common.vec_env import VecNormalize  # noqa: PLC0415

    vec_normalize = VecNormalize(
        venv,
        norm_obs=state["norm_obs"],
        norm_reward=state["norm_reward"],
        clip_obs=state["clip_obs"],
        clip_reward=state["clip_reward"],
        gamma=state["gamma"],
        epsilon=state["epsilon"],
    )
    for name in ("obs_rms", "ret_rms"):
        saved = state[name]
        if saved is None:
            continue
        rms = getattr(vec_normalize, name)
        rms.mean = np.asarray(saved["mean"], dtype=np.float64).reshape(rms.mean.shape)
        rms.var = np.asarray(saved["var"], dtype=np.float64).reshape(rms.var.shape)
        rms.count = saved["count"]
    return vec_normalize


__all__ = [
    "bundle_onnx_with_checkpoint",
    "extract_checkpoint_from_onnx",
    "read_onnx_checkpoint_metadata",
    "get_wandb_onnx_checkpoint_path",
    "is_onnx_checkpoint_name",
    "normalize_onnx_checkpoint_name",
    "onnx_checkpoint_sort_key",
    "get_env_fatigue_state",
    "set_env_fatigue_state",
    "get_vec_normalize_state",
    "vec_normalize_from_state",
    "OnnxPolicy",
]
