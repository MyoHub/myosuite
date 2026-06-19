# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""mjlab-compatible fullbody checkpoint policy wrappers.

mjlab advances MuJoCo-Warp simulation on torch tensors.  The MuscleMimic
fullbody checkpoint, however, was trained against the upstream MuJoCo CPU
observation builder.  These wrappers bridge that gap by:

1. Copying the selected mjlab env state into CPU ``mujoco.MjData`` buffers.
2. Calling ``mujoco.mj_forward`` so actuator, site, contact, and sensor fields
   are consistent with the copied state.
3. Building the checkpoint observation with ``FullbodyObsAdapter``.
4. Running ONNX or Orbax/Torch actor inference.
5. Returning a torch action tensor on the mjlab device.

For faithful checkpoint playback, register the mjlab task with
``action_mode="direct"`` and use ``FullbodyOrbaxMjlabPolicy``.  The direct
action term clamps policy outputs to MuJoCo's muscle-control range rather than
passing them through the training-time sigmoid action normalizer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import mujoco
import numpy as np
import torch

from myosuite.integrations.musclemimic.running_stats import (
    torch_running_mean_std_update,
)

if TYPE_CHECKING:
    from myosuite.integrations.musclemimic.fullbody_local_policy import (
        FullbodyObsAdapter,
        LocalPolicyArtifacts,
    )
    from myosuite.core.trajectory_io import MotionClip

logger = logging.getLogger(__name__)

_NormalizationMode = Literal["frozen", "running"]


def _unwrap_env(env: Any) -> Any:
    """Return the underlying mjlab env through common wrapper attributes."""
    current = env
    seen: set[int] = set()
    while True:
        seen.add(id(current))
        nxt = getattr(current, "unwrapped", None)
        if nxt is not None and nxt is not current and id(nxt) not in seen:
            current = nxt
            continue
        nxt = getattr(current, "env", None)
        if nxt is not None and nxt is not current and id(nxt) not in seen:
            current = nxt
            continue
        return current


def _env_device(env: Any, device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device(getattr(_unwrap_env(env), "device", "cpu"))


def _to_numpy(value: Any) -> np.ndarray:
    """Convert torch/TorchArray/numpy/scalar values to a CPU numpy array."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
        if isinstance(value, torch.Tensor):
            return value.detach().numpy()
    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        return np.asarray(numpy())
    return np.asarray(value)


def _batched_row(value: Any, env_idx: int) -> np.ndarray:
    """Return ``value[env_idx]`` as numpy, supporting mjlab TorchArray fields."""
    try:
        return _to_numpy(value[env_idx])
    except (AttributeError, IndexError, TypeError):
        return _to_numpy(value)[env_idx]


def _copy_field_row(dst: np.ndarray, src_container: Any, env_idx: int) -> None:
    """Copy one batched field row into ``dst`` when the shapes are compatible."""
    src = np.asarray(_batched_row(src_container, env_idx), dtype=dst.dtype)
    if src.shape == dst.shape:
        dst[...] = src
        return
    if src.size == dst.size:
        dst[...] = src.reshape(dst.shape)
        return
    raise ValueError(
        f"Cannot copy mjlab field row with shape {src.shape} into CPU field "
        f"with shape {dst.shape}."
    )


def _artifact_tensor(value: Any, device: torch.device) -> torch.Tensor:
    """Convert checkpoint stats to writable float32 tensors on ``device``."""
    return torch.as_tensor(
        np.array(value, dtype=np.float32, copy=True),
        dtype=torch.float32,
        device=device,
    )


class _BatchedObservationHistoryBuffer:
    """Batched equivalent of upstream single-env observation history."""

    def __init__(
        self,
        n_steps: int,
        *,
        split_goal: bool = False,
        state_indices: np.ndarray | None = None,
        goal_indices: np.ndarray | None = None,
    ) -> None:
        self.n_steps = int(n_steps)
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}.")
        self.split_goal = bool(split_goal)
        self.state_indices = (
            None if state_indices is None else np.asarray(state_indices, dtype=int)
        )
        self.goal_indices = (
            None if goal_indices is None else np.asarray(goal_indices, dtype=int)
        )
        if self.split_goal and (
            self.state_indices is None or self.goal_indices is None
        ):
            raise ValueError("split_goal=True requires state_indices and goal_indices.")
        self._buffer: np.ndarray | None = None

    def clear(self) -> None:
        self._buffer = None

    def reset(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if self.split_goal:
            assert self.state_indices is not None
            assert self.goal_indices is not None
            state_obs = obs[:, self.state_indices]
            goal_obs = obs[:, self.goal_indices]
            self._buffer = np.zeros(
                (obs.shape[0], self.n_steps, state_obs.shape[1]),
                dtype=obs.dtype,
            )
            self._buffer[:, -1, :] = state_obs
            return np.concatenate(
                [self._buffer.reshape(obs.shape[0], -1), goal_obs],
                axis=1,
            ).astype(np.float32)

        self._buffer = np.zeros(
            (obs.shape[0], self.n_steps, obs.shape[1]),
            dtype=obs.dtype,
        )
        self._buffer[:, -1, :] = obs
        return self._buffer.reshape(obs.shape[0], -1).astype(np.float32)

    def step(self, obs: np.ndarray) -> np.ndarray:
        if self._buffer is None:
            return self.reset(obs)
        obs = np.asarray(obs, dtype=np.float32)
        if self.split_goal:
            assert self.state_indices is not None
            assert self.goal_indices is not None
            state_obs = obs[:, self.state_indices]
            goal_obs = obs[:, self.goal_indices]
            self._buffer = np.roll(self._buffer, shift=-1, axis=1)
            self._buffer[:, -1, :] = state_obs
            return np.concatenate(
                [self._buffer.reshape(obs.shape[0], -1), goal_obs],
                axis=1,
            ).astype(np.float32)

        self._buffer = np.roll(self._buffer, shift=-1, axis=1)
        self._buffer[:, -1, :] = obs
        return self._buffer.reshape(obs.shape[0], -1).astype(np.float32)


def reset_mjlab_env_to_clip_frame(
    env: Any,
    clip: MotionClip,
    *,
    frame_idx: int = 0,
    entity_name: str = "mimic_fullbody_robot",
    variant: str = "fullbody",
    ctrl_dt: float | None = None,
    env_indices: tuple[int, ...] | list[int] | None = None,
    zero_ctrl: bool = True,
    zero_act: bool = True,
) -> int:
    """Set an mjlab env to the same clip state used by CPU playback.

    mjlab trajectory tasks normally use RSI and assign each env a random start
    frame.  That is good for training, but it is not equivalent to the CPU
    checkpoint playback path, which starts from a specific motion frame and
    advances deterministically.  This helper overwrites the mjlab state and the
    shared :class:`ClipTrajectorySource` phase so the next policy call builds
    the same fullbody observation as ``FullbodyObsAdapter`` on CPU.

    Args:
        env: mjlab env instance.
        clip: Motion clip with ``qpos`` and optionally ``qvel``.
        frame_idx: Clip frame to install.  ``0`` matches the native CPU playback
            default.
        entity_name: mjlab scene entity key.
        variant: Mimic variant passed to the mjlab cache resolver.
        ctrl_dt: Control timestep.  Defaults to ``env.physics_dt * decimation``.
        env_indices: Env rows to overwrite.  Defaults to every env.
        zero_ctrl: Whether to clear MuJoCo controls.
        zero_act: Whether to clear muscle activation state.

    Returns:
        The wrapped frame index that was installed.
    """
    if clip.qpos is None:
        raise ValueError("reset_mjlab_env_to_clip_frame requires clip.qpos.")

    unwrapped = _unwrap_env(env)
    sim_data = unwrapped.scene[entity_name].data.data
    n_envs = int(sim_data.qpos.shape[0])
    if env_indices is None:
        env_ids = torch.arange(n_envs, device=sim_data.qpos.device, dtype=torch.long)
    else:
        env_ids = torch.as_tensor(
            [int(i) for i in env_indices],
            device=sim_data.qpos.device,
            dtype=torch.long,
        )
    if env_ids.numel() == 0:
        raise ValueError("At least one env index is required.")

    frame = int(frame_idx) % int(clip.qpos.shape[0])
    qpos = torch.as_tensor(
        np.asarray(clip.qpos[frame], dtype=np.float32),
        device=sim_data.qpos.device,
    )
    sim_data.qpos[env_ids] = qpos.unsqueeze(0).expand(env_ids.numel(), -1)

    if clip.qvel is not None:
        qvel_np = np.asarray(clip.qvel[frame], dtype=np.float32)
    else:
        qvel_np = np.zeros((sim_data.qvel.shape[1],), dtype=np.float32)
    qvel = torch.as_tensor(qvel_np, device=sim_data.qvel.device)
    sim_data.qvel[env_ids] = qvel.unsqueeze(0).expand(env_ids.numel(), -1)

    if zero_ctrl and hasattr(sim_data, "ctrl"):
        sim_data.ctrl[env_ids] = 0.0
    if zero_act and hasattr(sim_data, "act"):
        sim_data.act[env_ids] = 0.0
    if hasattr(sim_data, "time"):
        sim_data.time[env_ids] = 0.0

    if hasattr(unwrapped, "sim"):
        unwrapped.sim.forward()

    if ctrl_dt is None:
        physics_dt = getattr(unwrapped, "physics_dt", None)
        cfg = getattr(unwrapped, "cfg", None)
        decimation = getattr(cfg, "decimation", None)
        if physics_dt is not None and decimation is not None:
            ctrl_dt = float(physics_dt) * float(decimation)

    if ctrl_dt is not None:
        from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
            _resolve_mimic_mjlab_ids,
        )

        cache = _resolve_mimic_mjlab_ids(
            unwrapped,
            entity_name,
            variant,
            clip,
            float(ctrl_dt),
        )
        source = cache.get("clip_source")
        if source is not None:
            source._ensure_device(sim_data.qpos.device, n_envs)
            source._start_offsets[env_ids] = frame
            if source._last_t is not None:
                source._last_t[env_ids] = 0.0

    return frame


class _FullbodyMjlabPolicyBridge:
    """Shared state-sync and observation-building logic for mjlab policies."""

    def __init__(
        self,
        *,
        env: Any,
        cpu_model: mujoco.MjModel,
        obs_adapter: FullbodyObsAdapter,
        clip: MotionClip,
        device: str | torch.device | None = None,
        env_indices: tuple[int, ...] | list[int] | None = None,
        entity_name: str = "mimic_fullbody_robot",
        variant: str = "fullbody",
        ctrl_dt: float | None = None,
        output_ctrl: bool = False,
        broadcast_single_env: bool = False,
        len_obs_history: int = 1,
        split_goal: bool = False,
        goal_indices: np.ndarray | None = None,
        state_indices: np.ndarray | None = None,
    ) -> None:
        self._env = env
        self._unwrapped = _unwrap_env(env)
        self._cpu_model = cpu_model
        self._obs_adapter = obs_adapter
        self._clip = clip
        self._entity_name = entity_name
        self._variant = variant
        self._ctrl_dt = self._resolve_ctrl_dt(ctrl_dt)
        self._output_ctrl = bool(output_ctrl)
        self._broadcast_single_env = bool(broadcast_single_env)
        self._device = _env_device(env, device)
        self._clip_source: Any | None = None
        self._clip_source_failed = False
        self._frame_idx = 0
        self._len_obs_history = int(len_obs_history)
        if self._len_obs_history < 1:
            raise ValueError(f"len_obs_history must be >= 1, got {len_obs_history}.")
        self._split_goal = bool(split_goal)
        self._state_indices = (
            None if state_indices is None else np.asarray(state_indices, dtype=int)
        )
        self._goal_indices = (
            None if goal_indices is None else np.asarray(goal_indices, dtype=int)
        )
        self._history: _BatchedObservationHistoryBuffer | None = None
        self._history_started = False

        n_envs = self._num_envs()
        if env_indices is None:
            self._env_indices = tuple(range(n_envs))
        else:
            self._env_indices = tuple(int(i) for i in env_indices)
            bad = [i for i in self._env_indices if i < 0 or i >= n_envs]
            if bad:
                raise ValueError(f"env_indices out of range for {n_envs} envs: {bad}")
        if not self._env_indices:
            raise ValueError("At least one env index is required for mjlab inference.")

        self._cpu_data = [mujoco.MjData(cpu_model) for _ in self._env_indices]
        self._selected_all = self._env_indices == tuple(range(n_envs))

        if len(self._env_indices) > 16:
            logger.warning(
                "%s will CPU-sync %d mjlab envs per policy call; this path is "
                "intended for playback/debug inference rather than large-batch PPO.",
                type(self).__name__,
                len(self._env_indices),
            )

    def reset(self) -> None:
        """Reset fallback frame counter and running normalizer state if present."""
        self._frame_idx = 0
        self._history_started = False
        if self._history is not None:
            self._history.clear()

    def _ensure_history(
        self, raw_obs_dim: int
    ) -> _BatchedObservationHistoryBuffer | None:
        """Create the history buffer once the raw fullbody obs width is known."""
        if self._len_obs_history <= 1:
            return None
        if self._history is not None:
            return self._history

        state_indices = self._state_indices
        goal_indices = self._goal_indices
        if self._split_goal and (state_indices is None or goal_indices is None):
            goal_indices = self._obs_adapter.goal_indices_for_obs_dim(raw_obs_dim)
            state_indices = self._obs_adapter.state_indices_for_obs_dim(raw_obs_dim)
            self._goal_indices = goal_indices
            self._state_indices = state_indices

        self._history = _BatchedObservationHistoryBuffer(
            self._len_obs_history,
            split_goal=self._split_goal,
            state_indices=state_indices,
            goal_indices=goal_indices,
        )
        return self._history

    def _num_envs(self) -> int:
        return int(getattr(self._unwrapped, "num_envs", 1))

    def _resolve_ctrl_dt(self, explicit: float | None) -> float | None:
        if explicit is not None:
            return float(explicit)
        physics_dt = getattr(self._unwrapped, "physics_dt", None)
        cfg = getattr(self._unwrapped, "cfg", None)
        decimation = getattr(cfg, "decimation", None)
        if physics_dt is not None and decimation is not None:
            return float(physics_dt) * float(decimation)
        return None

    def _sim_data(self) -> Any:
        scene = getattr(self._unwrapped, "scene", None)
        if scene is not None and self._entity_name:
            try:
                return scene[self._entity_name].data.data
            except (KeyError, AttributeError, TypeError):
                pass
        return self._unwrapped.sim.data

    def _frame_count(self) -> int:
        qpos = getattr(self._clip, "qpos", None)
        if qpos is not None:
            return int(qpos.shape[0])
        site_xpos = getattr(self._clip, "site_xpos", None)
        if site_xpos is not None:
            return int(site_xpos.shape[0])
        raise ValueError("MotionClip must provide qpos or site_xpos for frame count.")

    def _trajectory_source(self) -> Any | None:
        if self._clip_source is not None:
            return self._clip_source
        if self._clip_source_failed or self._ctrl_dt is None:
            return None
        try:
            from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
                _resolve_mimic_mjlab_ids,
            )

            cache = _resolve_mimic_mjlab_ids(
                self._unwrapped,
                self._entity_name,
                self._variant,
                self._clip,
                self._ctrl_dt,
            )
            self._clip_source = cache.get("clip_source")
        except Exception as err:  # pragma: no cover - diagnostic fallback path
            self._clip_source_failed = True
            logger.debug("Could not resolve mjlab ClipTrajectorySource: %s", err)
        return self._clip_source

    def _current_frame_indices(self, sim_data: Any) -> np.ndarray:
        source = self._trajectory_source()
        if source is not None:
            from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
                _mjlab_sim_time_to_tensor,
            )

            t = _mjlab_sim_time_to_tensor(sim_data.time)
            source.update(t)
            frames = source.frame_indices(t).detach().cpu().numpy().astype(np.int64)
            return frames[np.asarray(self._env_indices, dtype=np.int64)]

        frame = self._frame_idx % self._frame_count()
        return np.full((len(self._env_indices),), frame, dtype=np.int64)

    def _sync_env_to_cpu(
        self, cpu_data: mujoco.MjData, sim_data: Any, env_idx: int
    ) -> None:
        for field in ("qpos", "qvel", "ctrl", "act"):
            if not hasattr(sim_data, field) or not hasattr(cpu_data, field):
                continue
            dst = getattr(cpu_data, field)
            if dst.size:
                _copy_field_row(dst, getattr(sim_data, field), env_idx)

        if self._cpu_model.nmocap > 0:
            if hasattr(sim_data, "mocap_pos"):
                _copy_field_row(cpu_data.mocap_pos, sim_data.mocap_pos, env_idx)
            if hasattr(sim_data, "mocap_quat"):
                _copy_field_row(cpu_data.mocap_quat, sim_data.mocap_quat, env_idx)

        mujoco.mj_forward(self._cpu_model, cpu_data)

    def _build_fullbody_obs_batch(self) -> np.ndarray:
        sim_data = self._sim_data()
        frame_indices = self._current_frame_indices(sim_data)

        obs_batch: list[np.ndarray] = []
        for cpu_data, env_idx, frame_idx in zip(
            self._cpu_data, self._env_indices, frame_indices, strict=True
        ):
            self._sync_env_to_cpu(cpu_data, sim_data, env_idx)
            obs = self._obs_adapter.build(cpu_data, int(frame_idx))
            obs_batch.append(np.asarray(obs, dtype=np.float32))

        if self._trajectory_source() is None:
            self._frame_idx += 1
        raw_obs = np.stack(obs_batch, axis=0).astype(np.float32)
        history = self._ensure_history(raw_obs.shape[1])
        if history is None:
            return raw_obs
        if not self._history_started:
            self._history_started = True
            return history.reset(raw_obs)
        return history.step(raw_obs)

    def _actions_to_tensor(self, action_np: np.ndarray) -> torch.Tensor:
        action_np = np.asarray(action_np, dtype=np.float32)
        if action_np.ndim == 1:
            action_np = action_np[None, :]
        if action_np.shape[0] != len(self._env_indices):
            raise ValueError(
                "Policy returned "
                f"{action_np.shape[0]} actions for {len(self._env_indices)} synced envs."
            )

        action_np = np.clip(action_np, -1.0, 1.0)
        if self._output_ctrl:
            ctrl_range = np.asarray(
                self._cpu_model.actuator_ctrlrange, dtype=np.float32
            )
            if ctrl_range.shape == (action_np.shape[1], 2):
                action_np = np.clip(action_np, ctrl_range[:, 0], ctrl_range[:, 1])
            else:
                action_np = np.clip(action_np, 0.0, 1.0)

        action_t = torch.as_tensor(action_np, dtype=torch.float32, device=self._device)
        n_envs = self._num_envs()
        if self._broadcast_single_env and action_t.shape[0] == 1:
            return action_t.expand(n_envs, -1)
        if self._selected_all:
            return action_t

        full = torch.zeros(
            (n_envs, action_t.shape[1]), dtype=action_t.dtype, device=self._device
        )
        full[list(self._env_indices)] = action_t
        return full


class FullbodyOnnxMjlabPolicy(_FullbodyMjlabPolicyBridge):
    """Policy wrapper that bridges mjlab GPU state to a full-body ONNX model.

    This class preserves the previous ONNX wrapper's behaviour by syncing
    ``env_idx`` and broadcasting that action across all mjlab envs.  Pass
    ``output_ctrl=True`` only when the mjlab task was registered with
    ``action_mode="direct"``.
    """

    def __init__(
        self,
        env: Any,
        cpu_model: mujoco.MjModel,
        obs_adapter: FullbodyObsAdapter,
        onnx_path: str | Path,
        clip: MotionClip,
        device: str | torch.device | None = None,
        env_idx: int = 0,
        *,
        entity_name: str = "mimic_fullbody_robot",
        variant: str = "fullbody",
        ctrl_dt: float | None = None,
        output_ctrl: bool = False,
        len_obs_history: int = 1,
        split_goal: bool = False,
        goal_indices: np.ndarray | None = None,
        state_indices: np.ndarray | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as err:
            raise ImportError(
                "onnxruntime is required. Install with: pip install onnxruntime"
            ) from err

        super().__init__(
            env=env,
            cpu_model=cpu_model,
            obs_adapter=obs_adapter,
            clip=clip,
            device=device,
            env_indices=(int(env_idx),),
            entity_name=entity_name,
            variant=variant,
            ctrl_dt=ctrl_dt,
            output_ctrl=output_ctrl,
            broadcast_single_env=True,
            len_obs_history=len_obs_history,
            split_goal=split_goal,
            goal_indices=goal_indices,
            state_indices=state_indices,
        )

        onnx_path = Path(onnx_path)
        self._session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        input_shape = self._session.get_inputs()[0].shape
        self._expected_obs_dim = input_shape[1] if len(input_shape) > 1 else None
        self._action_dim = int(self._session.get_outputs()[0].shape[1])

        logger.info(
            "FullbodyOnnxMjlabPolicy: onnx=%s action_dim=%d device=%s output_ctrl=%s",
            onnx_path.name,
            self._action_dim,
            self._device,
            self._output_ctrl,
        )

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        """Build fullbody obs from mjlab state, run ONNX, return env actions."""
        del obs
        obs_np = self._build_fullbody_obs_batch()
        if (
            isinstance(self._expected_obs_dim, int)
            and obs_np.shape[1] != self._expected_obs_dim
        ):
            raise ValueError(
                f"Fullbody obs dim {obs_np.shape[1]} != ONNX input {self._expected_obs_dim}."
            )
        action_np = self._session.run(
            [self._output_name],
            {self._input_name: obs_np},
        )[0]
        return self._actions_to_tensor(action_np)


class FullbodyOrbaxMjlabPolicy(_FullbodyMjlabPolicyBridge):
    """mjlab policy wrapper for MuscleMimic Orbax checkpoints.

    Args:
        env: mjlab ``ManagerBasedRlEnv`` (wrapped or unwrapped).
        cpu_model: CPU ``mujoco.MjModel`` matching the mjlab MJCF.
        obs_adapter: ``FullbodyObsAdapter`` configured from checkpoint goal params.
        clip: Motion clip used by the mjlab trajectory task.
        checkpoint_path: Local checkpoint directory or ``hf://owner/repo[/subdir]``.
        artifacts: Optional preloaded ``LocalPolicyArtifacts``.  When supplied,
            ``checkpoint_path`` is only used for logging.
        device: Device for returned actions.  Defaults to ``env.device``.
        actor_device: Device for Torch actor inference.  Defaults to ``device``.
        env_indices: Env rows to sync.  Defaults to all envs so ``env.step`` can
            consume the returned action tensor directly.
        normalization_mode: ``"running"`` updates RunningMeanStd before
            inference and matches the CPU ``LocalPolicyRunner`` path.
            ``"frozen"`` uses checkpoint stats as fixed buffers.
        output_ctrl: If ``True`` clips actions through ``cpu_model.ctrlrange`` so
            the returned tensor is already in MuJoCo muscle-control space.
    """

    def __init__(
        self,
        env: Any,
        cpu_model: mujoco.MjModel,
        obs_adapter: FullbodyObsAdapter,
        clip: MotionClip,
        checkpoint_path: str | Path | None = None,
        *,
        artifacts: LocalPolicyArtifacts | None = None,
        device: str | torch.device | None = None,
        actor_device: str | torch.device | None = None,
        env_indices: tuple[int, ...] | list[int] | None = None,
        entity_name: str = "mimic_fullbody_robot",
        variant: str = "fullbody",
        ctrl_dt: float | None = None,
        normalization_mode: _NormalizationMode = "running",
        output_ctrl: bool = True,
        len_obs_history: int = 1,
        split_goal: bool = False,
        goal_indices: np.ndarray | None = None,
        state_indices: np.ndarray | None = None,
    ) -> None:
        if normalization_mode not in ("frozen", "running"):
            raise ValueError(
                "normalization_mode must be 'frozen' or 'running', "
                f"got {normalization_mode!r}."
            )

        if artifacts is None:
            if checkpoint_path is None:
                raise ValueError("checkpoint_path or artifacts must be provided.")
            from myosuite.integrations.musclemimic.fullbody_checkpoint_io import (
                resolve_checkpoint_ref,
            )
            from myosuite.integrations.musclemimic.fullbody_local_policy import (
                load_local_policy_artifacts,
            )

            checkpoint_root = resolve_checkpoint_ref(str(checkpoint_path)).local_path
            artifacts = load_local_policy_artifacts(checkpoint_root)

        super().__init__(
            env=env,
            cpu_model=cpu_model,
            obs_adapter=obs_adapter,
            clip=clip,
            device=device,
            env_indices=env_indices,
            entity_name=entity_name,
            variant=variant,
            ctrl_dt=ctrl_dt,
            output_ctrl=output_ctrl,
            broadcast_single_env=False,
            len_obs_history=len_obs_history,
            split_goal=split_goal,
            goal_indices=goal_indices,
            state_indices=state_indices,
        )

        from myosuite.integrations.musclemimic.actor_torch import (
            make_actor_module,
        )

        self._artifacts = artifacts
        self._normalization_mode = normalization_mode
        self._actor_device = (
            torch.device(actor_device) if actor_device is not None else self._device
        )
        self._actor = make_actor_module(artifacts).to(self._actor_device).eval()
        self._run_mean = _artifact_tensor(artifacts.obs_mean, self._actor_device)
        self._run_var = _artifact_tensor(artifacts.obs_var, self._actor_device)
        self._run_count = _artifact_tensor(artifacts.obs_count, self._actor_device)

        logger.info(
            "FullbodyOrbaxMjlabPolicy: obs_dim=%d action_dim=%d actor_device=%s "
            "action_device=%s normalization=%s output_ctrl=%s",
            artifacts.obs_dim,
            artifacts.action_dim,
            self._actor_device,
            self._device,
            self._normalization_mode,
            self._output_ctrl,
        )

    def reset(self) -> None:
        """Reset fallback frame counter and running-normalizer state."""
        super().reset()
        self._run_mean = _artifact_tensor(self._artifacts.obs_mean, self._actor_device)
        self._run_var = _artifact_tensor(self._artifacts.obs_var, self._actor_device)
        self._run_count = _artifact_tensor(
            self._artifacts.obs_count, self._actor_device
        )

    def reset_env_to_clip_frame(
        self,
        frame_idx: int = 0,
        *,
        env_indices: tuple[int, ...] | list[int] | None = None,
        zero_ctrl: bool = True,
        zero_act: bool = True,
    ) -> int:
        """Reset mjlab state and policy normalizer to a deterministic clip frame."""
        frame = reset_mjlab_env_to_clip_frame(
            self._unwrapped,
            self._clip,
            frame_idx=frame_idx,
            entity_name=self._entity_name,
            variant=self._variant,
            ctrl_dt=self._ctrl_dt,
            env_indices=env_indices,
            zero_ctrl=zero_ctrl,
            zero_act=zero_act,
        )
        self.reset()
        if self._trajectory_source() is None:
            self._frame_idx = frame
        return frame

    def _normalize_running(self, obs: torch.Tensor) -> torch.Tensor:
        """RunningMeanStd update compatible with the CPU local runner."""
        normalized, self._run_mean, self._run_var, self._run_count = (
            torch_running_mean_std_update(
                obs, self._run_mean, self._run_var, self._run_count
            )
        )
        return normalized

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        """Build fullbody obs from mjlab state, run Orbax/Torch, return actions."""
        del obs
        obs_np = self._build_fullbody_obs_batch()
        if obs_np.shape[1] != self._artifacts.obs_dim:
            raise ValueError(
                f"Fullbody obs dim {obs_np.shape[1]} != checkpoint {self._artifacts.obs_dim}."
            )

        with torch.no_grad():
            obs_t = torch.as_tensor(
                obs_np, dtype=torch.float32, device=self._actor_device
            )
            if self._normalization_mode == "running":
                action = self._actor.forward_normalized(self._normalize_running(obs_t))
            else:
                action = self._actor(obs_t)
            action_np = action.detach().cpu().numpy()
        return self._actions_to_tensor(action_np)


__all__ = [
    "FullbodyOnnxMjlabPolicy",
    "FullbodyOrbaxMjlabPolicy",
    "reset_mjlab_env_to_clip_frame",
]
