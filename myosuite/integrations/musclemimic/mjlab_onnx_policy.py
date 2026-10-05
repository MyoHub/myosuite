# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""mjlab-compatible fullbody checkpoint policy wrappers.

mjlab advances MuJoCo-Warp simulation on torch tensors.  The MuscleMimic
fullbody checkpoint, however, was trained against the upstream MuJoCo CPU
observation builder (``FullbodyObsAdapter``).  These wrappers bridge that gap:

1. Build the checkpoint observation of every selected env in one batch with
   ``TorchFullbodyObsAdapter``, from mjlab's sim data on its device.  The bridge
   is called after ``env.step``/``env.reset``, whose final ``sim.forward()``
   leaves actuator, site, contact and sensor fields consistent with the state
   (also for envs that were just reset).  The result is float32: it matches
   the CPU builder fed the same arrays to float32 rounding, not bitwise, and
   differs from a CPU ``mj_forward`` of the same state where MuJoCo Warp's
   own results differ (some wrapped-tendon lengths, contact/touch forces).
2. Run ONNX (one bulk host copy of the batch) or Orbax/Torch actor inference
   (stays on device).
3. Return a torch action tensor on the mjlab device.

``obs_backend="cpu"`` keeps the reference path for debugging and parity checks:
copy each env's state into a CPU ``mujoco.MjData``, ``mj_forward`` it and build
its observation with ``FullbodyObsAdapter`` (cost linear in the number of envs).

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
    torch_running_mean_std_update_per_env,
)

if TYPE_CHECKING:
    from myosuite.integrations.musclemimic.fullbody_local_policy import (
        FullbodyObsAdapter,
        LocalPolicyArtifacts,
    )
    from myosuite.integrations.musclemimic.fullbody_obs_torch import (
        TorchFullbodyObsAdapter,
    )
    from myosuite.core.trajectory_io import MotionClip

logger = logging.getLogger(__name__)

_NormalizationMode = Literal["frozen", "running"]
# "torch": batched on the mjlab device; "cpu": per-env MjData + mj_forward;
# "auto": "torch" for a FullbodyObsAdapter, "cpu" for any other (duck-typed) adapter.
ObsBackend = Literal["auto", "torch", "cpu"]
# Model fields whose equality makes CPU-model ids valid indices into mjlab sim data.
_SIM_LAYOUT_FIELDS = (
    "nq",
    "nv",
    "nu",
    "na",
    "nsite",
    "nbody",
    "nsensordata",
    "jnt_qposadr",
    "jnt_dofadr",
    "site_bodyid",
    "body_rootid",
    "sensor_adr",
)


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


def _artifact_tensor(value: Any, device: torch.device) -> torch.Tensor:
    """Convert checkpoint stats to writable float32 tensors on ``device``."""
    return torch.as_tensor(
        np.array(value, dtype=np.float32, copy=True),
        dtype=torch.float32,
        device=device,
    )


class _RowView:
    """Selected env rows of batched sim data, gathered per field on access."""

    def __init__(self, data: Any, rows: torch.Tensor) -> None:
        self._data = data
        self._rows = rows

    def __getattr__(self, name: str) -> torch.Tensor:
        return getattr(self._data, name)[self._rows]


class _BatchedObservationHistoryBuffer:
    """Batched equivalent of upstream single-env observation history (torch)."""

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
        self._buffer: torch.Tensor | None = None
        self._split_idx: tuple[torch.Tensor, torch.Tensor] | None = None

    def clear(self) -> None:
        self._buffer = None

    def _split(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``(state, goal)`` columns of *obs*; goal is ``None`` without split_goal."""
        if not self.split_goal:
            return obs, None
        if self._split_idx is None or self._split_idx[0].device != obs.device:
            self._split_idx = (
                torch.as_tensor(self.state_indices, device=obs.device),
                torch.as_tensor(self.goal_indices, device=obs.device),
            )
        return obs[:, self._split_idx[0]], obs[:, self._split_idx[1]]

    def _output(self, goal: torch.Tensor | None) -> torch.Tensor:
        assert self._buffer is not None
        flat = self._buffer.reshape(self._buffer.shape[0], -1)
        return flat if goal is None else torch.cat([flat, goal], dim=1)

    def reset(self, obs: torch.Tensor | np.ndarray) -> torch.Tensor:
        state, goal = self._split(torch.as_tensor(obs, dtype=torch.float32))
        self._buffer = state.new_zeros((state.shape[0], self.n_steps, state.shape[1]))
        self._buffer[:, -1] = state
        return self._output(goal)

    def step(
        self,
        obs: torch.Tensor | np.ndarray,
        new_episode: torch.Tensor | np.ndarray | None = None,
    ) -> torch.Tensor:
        """Append *obs*; envs flagged in *new_episode* (``(N,)`` bool) restart their history."""
        if self._buffer is None:
            return self.reset(obs)
        state, goal = self._split(torch.as_tensor(obs, dtype=torch.float32))
        self._buffer = torch.roll(self._buffer, shifts=-1, dims=1)
        self._buffer[:, -1] = state
        if new_episode is not None:
            # Drop the older frames of the flagged envs (as reset does), sync-free.
            fresh = torch.as_tensor(
                new_episode, dtype=torch.bool, device=self._buffer.device
            )
            self._buffer[:, :-1].masked_fill_(fresh[:, None, None], 0.0)
        return self._output(goal)


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
            _mimic_episode_steps,
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
            # Clip frame = min(episode step + offset, T - 1): choose the offset that
            # puts each env's current step on `frame` (negative mid-episode), so
            # the clip end is reached after T - frame steps.
            step = _mimic_episode_steps(unwrapped)[env_ids]
            source._start_offsets[env_ids] = frame - step

    return frame


class _FullbodyMjlabPolicyBridge:
    """Shared observation-building logic for mjlab policies.

    Observations, history and per-env episode masks are torch tensors on the sim
    device; only the ``obs_backend="cpu"`` path goes through host memory.
    """

    # Device index of the selected envs; None selects every env in order.
    _env_rows: torch.Tensor | None = None

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
        obs_backend: ObsBackend = "auto",
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
        self._last_steps: torch.Tensor | None = None
        self._sim_device = torch.device(
            getattr(self._sim_data().qpos, "device", self._device)
        )
        self._new_episode = torch.zeros(0, dtype=torch.bool, device=self._sim_device)

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
        self._selected_all = self._env_indices == tuple(range(n_envs))
        if not self._selected_all:
            self._env_rows = torch.as_tensor(
                self._env_indices, dtype=torch.long, device=self._sim_device
            )
        ctrl_range = torch.as_tensor(
            np.asarray(cpu_model.actuator_ctrlrange, dtype=np.float32).reshape(-1, 2),
            device=self._device,
        )
        self._ctrl_lo, self._ctrl_hi = ctrl_range[:, 0], ctrl_range[:, 1]

        self._obs_backend = self._resolve_obs_backend(obs_backend)
        self._torch_obs: TorchFullbodyObsAdapter | None = None
        self._cpu_data: list[mujoco.MjData] = []
        if self._obs_backend == "torch":
            from myosuite.integrations.musclemimic.fullbody_obs_torch import (
                TorchFullbodyObsAdapter,
            )

            self._check_sim_layout()
            self._torch_obs = TorchFullbodyObsAdapter(
                obs_adapter, device=self._sim_device
            )
        else:
            self._cpu_data = [mujoco.MjData(cpu_model) for _ in self._env_indices]
            if len(self._env_indices) > 16:
                logger.warning(
                    "%s will CPU-sync %d mjlab envs per policy call "
                    "(obs_backend='cpu'); this path is meant for debugging and "
                    "parity checks, not large-batch inference.",
                    type(self).__name__,
                    len(self._env_indices),
                )

    def _resolve_obs_backend(self, obs_backend: str) -> str:
        from myosuite.integrations.musclemimic.fullbody_local_policy import (
            FullbodyObsAdapter,
        )

        if obs_backend not in ("auto", "torch", "cpu"):
            raise ValueError(
                f"obs_backend must be 'auto', 'torch' or 'cpu', got {obs_backend!r}."
            )
        is_fullbody = isinstance(self._obs_adapter, FullbodyObsAdapter)
        if obs_backend == "auto":
            return "torch" if is_fullbody else "cpu"
        if obs_backend == "torch" and not is_fullbody:
            raise TypeError(
                "obs_backend='torch' needs a FullbodyObsAdapter, got "
                f"{type(self._obs_adapter).__name__}; use obs_backend='cpu'."
            )
        return obs_backend

    def _check_sim_layout(self) -> None:
        """Fail if CPU-model ids would not index mjlab's sim arrays correctly."""
        sim_model = getattr(getattr(self._unwrapped, "sim", None), "mj_model", None)
        if sim_model is None:
            return
        bad = [
            name
            for name in _SIM_LAYOUT_FIELDS
            if not np.array_equal(
                np.asarray(getattr(self._cpu_model, name)),
                np.asarray(getattr(sim_model, name)),
            )
        ]
        if bad:
            raise ValueError(
                "cpu_model and the mjlab sim model differ in "
                f"{', '.join(bad)}; the batched observation indexes sim data with "
                "cpu_model ids. Pass the scene's model or use obs_backend='cpu'."
            )

    def reset(self) -> None:
        """Reset fallback frame counter and running normalizer state if present."""
        self._frame_idx = 0
        self._history_started = False
        self._last_steps = None
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

    def _rows(self, value: Any) -> Any:
        """The selected envs' rows of a batched ``(num_envs, ...)`` field."""
        return value if self._env_rows is None else value[self._env_rows]

    def _current_frame_indices(self) -> torch.Tensor:
        """``(n_selected,)`` int64 clip frame of each selected env, on the sim device."""
        source = self._trajectory_source()
        if source is not None:
            from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
                _mimic_episode_steps,
            )

            # Same integer step counter as the env's clip terms (not float time).
            step = _mimic_episode_steps(self._unwrapped)
            source.update(step)
            return self._rows(source.frame_indices(step))

        frame = self._frame_idx % self._frame_count()
        return torch.full(
            (len(self._env_indices),), frame, dtype=torch.long, device=self._sim_device
        )

    def _episode_start_mask(self) -> torch.Tensor:
        """``(n_selected,)`` bool: envs whose episode began since the last call.

        Read from each env's own step counter, so an env that resets mid-run
        restarts its history and normalizer without touching the others.
        """
        n = len(self._env_indices)
        steps = getattr(self._unwrapped, "episode_length_buf", None)
        if steps is None:
            return torch.zeros(n, dtype=torch.bool, device=self._sim_device)
        steps = self._rows(steps).clone()
        last, self._last_steps = self._last_steps, steps
        if last is None:
            return torch.ones(n, dtype=torch.bool, device=steps.device)
        return (steps < last) | (steps == 0)

    def _raw_obs_batch(self, frames: torch.Tensor) -> torch.Tensor:
        """``(n_selected, obs_dim)`` float32 observation of the current sim state."""
        if self._torch_obs is not None:
            # Raw MuJoCo-layout sim arrays (all entities): the adapter indexes them
            # with cpu_model ids, checked equal to the sim model's at construction.
            data = self._sim_data()
            if self._env_rows is not None:
                data = _RowView(data, self._env_rows)
            with torch.no_grad():
                return self._torch_obs.build(data, frames)
        return self._cpu_obs_batch(frames)

    def _cpu_obs_batch(self, frames: torch.Tensor) -> torch.Tensor:
        """Reference path: per-env CPU ``mj_forward`` + ``FullbodyObsAdapter``."""
        sim_data = self._sim_data()
        fields = ["qpos", "qvel", "ctrl", "act"]
        if self._cpu_model.nmocap > 0:
            fields += ["mocap_pos", "mocap_quat"]
        # One host copy per field for all selected envs.
        rows = {
            field: _to_numpy(self._rows(getattr(sim_data, field)))
            for field in fields
            if hasattr(sim_data, field) and getattr(self._cpu_data[0], field).size
        }
        obs_batch: list[np.ndarray] = []
        for i, (cpu_data, frame) in enumerate(
            zip(self._cpu_data, _to_numpy(frames), strict=True)
        ):
            for field, values in rows.items():
                dst = getattr(cpu_data, field)
                dst[...] = np.asarray(values[i], dtype=dst.dtype).reshape(dst.shape)
            mujoco.mj_forward(self._cpu_model, cpu_data)
            obs = self._obs_adapter.build(cpu_data, int(frame))
            obs_batch.append(np.asarray(obs, dtype=np.float32))
        return torch.as_tensor(np.stack(obs_batch, axis=0), device=self._sim_device)

    def _build_fullbody_obs_batch(self) -> torch.Tensor:
        """Policy observation of the selected envs (with history), on the sim device."""
        raw_obs = self._raw_obs_batch(self._current_frame_indices())
        if self._trajectory_source() is None:
            self._frame_idx += 1
        self._new_episode = self._episode_start_mask()
        history = self._ensure_history(int(raw_obs.shape[1]))
        if history is None:
            return raw_obs
        if not self._history_started:
            self._history_started = True
            return history.reset(raw_obs)
        return history.step(raw_obs, self._new_episode)

    def _actions_to_tensor(self, action: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Clip policy actions and scatter them into an ``(num_envs, act_dim)`` tensor."""
        action_t = torch.as_tensor(action, dtype=torch.float32, device=self._device)
        if action_t.ndim == 1:
            action_t = action_t[None, :]
        if action_t.shape[0] != len(self._env_indices):
            raise ValueError(
                "Policy returned "
                f"{action_t.shape[0]} actions for {len(self._env_indices)} synced envs."
            )

        action_t = action_t.clamp(-1.0, 1.0)
        if self._output_ctrl:
            if self._ctrl_lo.shape[0] == action_t.shape[1]:
                action_t = torch.clamp(action_t, self._ctrl_lo, self._ctrl_hi)
            else:
                action_t = action_t.clamp(0.0, 1.0)

        n_envs = self._num_envs()
        if self._broadcast_single_env and action_t.shape[0] == 1:
            return action_t.expand(n_envs, -1)
        if self._selected_all:
            return action_t

        assert self._env_rows is not None
        full = torch.zeros(
            (n_envs, action_t.shape[1]), dtype=action_t.dtype, device=self._device
        )
        full[self._env_rows.to(self._device)] = action_t
        return full


class FullbodyOnnxMjlabPolicy(_FullbodyMjlabPolicyBridge):
    """Policy wrapper that bridges mjlab GPU state to a full-body ONNX model.

    By default this preserves the previous ONNX wrapper's behaviour: it builds
    the observation of ``env_idx`` and broadcasts that action across all mjlab
    envs.  Pass ``env_indices`` to run those envs as one batch with one action
    each (the ONNX graph needs a dynamic batch axis).  onnxruntime runs on the
    CPU, so each call makes one host copy of the observation batch and one
    device copy of the actions.  Pass ``output_ctrl=True`` only when the mjlab
    task was registered with ``action_mode="direct"``.  ``obs_backend`` selects
    the observation builder (see :data:`ObsBackend`).
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
        env_indices: tuple[int, ...] | list[int] | None = None,
        entity_name: str = "mimic_fullbody_robot",
        variant: str = "fullbody",
        ctrl_dt: float | None = None,
        output_ctrl: bool = False,
        len_obs_history: int = 1,
        split_goal: bool = False,
        goal_indices: np.ndarray | None = None,
        state_indices: np.ndarray | None = None,
        obs_backend: ObsBackend = "auto",
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
            env_indices=(int(env_idx),) if env_indices is None else env_indices,
            entity_name=entity_name,
            variant=variant,
            ctrl_dt=ctrl_dt,
            output_ctrl=output_ctrl,
            broadcast_single_env=env_indices is None,
            len_obs_history=len_obs_history,
            split_goal=split_goal,
            goal_indices=goal_indices,
            state_indices=state_indices,
            obs_backend=obs_backend,
        )

        onnx_path = Path(onnx_path)
        # onnxruntime's spinning intra-op pool starves the torch/warp threads that
        # share the CPU (CPU torch: 94 -> 29 ms per 256-env call); a batch of one
        # env gains nothing from extra threads.
        session_options = ort.SessionOptions()
        session_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        if len(self._env_indices) == 1:
            session_options.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(onnx_path), session_options, providers=["CPUExecutionProvider"]
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
        policy_obs = self._build_fullbody_obs_batch()
        if (
            isinstance(self._expected_obs_dim, int)
            and policy_obs.shape[1] != self._expected_obs_dim
        ):
            raise ValueError(
                f"Fullbody obs dim {policy_obs.shape[1]} != ONNX input {self._expected_obs_dim}."
            )
        # onnxruntime (CPU provider) reads host memory: one bulk copy of the batch.
        obs_np = policy_obs.detach().cpu().numpy()
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
        obs_backend: Observation builder (see :data:`ObsBackend`).  The default
            builds a ``FullbodyObsAdapter`` observation batched on the mjlab
            device; ``"cpu"`` uses one CPU ``mj_forward`` per env (reference).
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
        obs_backend: ObsBackend = "auto",
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
            obs_backend=obs_backend,
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
        self._init_running_stats()

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

    def _init_running_stats(self) -> None:
        """Per-env running statistics, all starting from the checkpoint's."""
        n = len(self._env_indices)
        mean = _artifact_tensor(self._artifacts.obs_mean, self._actor_device)
        var = _artifact_tensor(self._artifacts.obs_var, self._actor_device)
        count = _artifact_tensor(self._artifacts.obs_count, self._actor_device)
        self._init_stats = (mean, var, count.reshape(()))
        self._run_mean = mean.expand(n, -1).clone()
        self._run_var = var.expand(n, -1).clone()
        self._run_count = count.reshape(()).expand(n).clone()

    def reset(self) -> None:
        """Reset fallback frame counter and running-normalizer state."""
        super().reset()
        self._init_running_stats()

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
        """Per-env RunningMeanStd update, equal to one CPU ``LocalPolicyRunner`` per env.

        Envs that started a new episode first restart from the checkpoint statistics.
        """
        fresh = self._new_episode.to(self._actor_device)
        mean, var, count = self._init_stats
        # torch.where instead of a boolean-mask write: no host sync.
        self._run_mean = torch.where(fresh[:, None], mean, self._run_mean)
        self._run_var = torch.where(fresh[:, None], var, self._run_var)
        self._run_count = torch.where(fresh, count, self._run_count)
        normalized, self._run_mean, self._run_var, self._run_count = (
            torch_running_mean_std_update_per_env(
                obs, self._run_mean, self._run_var, self._run_count
            )
        )
        return normalized

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        """Build fullbody obs from mjlab state, run Orbax/Torch, return actions."""
        del obs
        policy_obs = self._build_fullbody_obs_batch()
        if policy_obs.shape[1] != self._artifacts.obs_dim:
            raise ValueError(
                f"Fullbody obs dim {policy_obs.shape[1]} != checkpoint {self._artifacts.obs_dim}."
            )

        with torch.no_grad():
            obs_t = policy_obs.to(self._actor_device)
            if self._normalization_mode == "running":
                action = self._actor.forward_normalized(self._normalize_running(obs_t))
            else:
                action = self._actor(obs_t)
        return self._actions_to_tensor(action)


__all__ = [
    "FullbodyOnnxMjlabPolicy",
    "FullbodyOrbaxMjlabPolicy",
    "ObsBackend",
    "reset_mjlab_env_to_clip_frame",
]
