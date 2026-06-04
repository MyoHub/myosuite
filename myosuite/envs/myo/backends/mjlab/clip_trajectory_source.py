# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Per-environment trajectory source backed by a :class:`~myosuite.core.trajectory_io.MotionClip`.

Each of the N parallel mjlab environments gets its own random starting frame
that is resampled independently on episode reset.  Frame indices advance in
lock-step with the per-env simulation time:

    frame[i] = (floor(t[i] / ctrl_dt) + start_offset[i]) % T

This gives a diverse distribution of motion phases across the batch while
keeping each individual episode's targets coherent with the reference clip.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

    from myosuite.core.trajectory_io import MotionClip


@dataclass
class ClipTrajectorySource:
    """Manages per-env frame tracking for batched MotionClip playback in mjlab.

    Instances are held inside the per-env mjlab cache dict so that all
    observation and reward closures share the same source object.

    Args:
        clip: Loaded motion trajectory.  Must have ``site_xpos`` populated
              (shape ``(T, n_model_sites, 3)``).
        tracked_site_ids: Indices into the model's full site array selecting
                          the sites used for tracking (shape ``(n_tracked,)``).
        ctrl_dt: Control timestep in seconds (``sim_dt × decimation``).
                 Determines how fast frames advance.

    Raises:
        ValueError: If ``clip.site_xpos`` is ``None``.
    """

    clip: MotionClip
    tracked_site_ids: np.ndarray
    ctrl_dt: float

    # Lazily initialised once the device / batch size are known.
    _site_tensor: torch.Tensor | None = field(default=None, repr=False, init=False)
    _qpos_tensor: torch.Tensor | None = field(default=None, repr=False, init=False)
    _qvel_tensor: torch.Tensor | None = field(default=None, repr=False, init=False)
    _start_offsets: torch.Tensor | None = field(default=None, repr=False, init=False)
    _last_t: torch.Tensor | None = field(default=None, repr=False, init=False)
    _device: object = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        if self.clip.site_xpos is None:
            raise ValueError(
                "ClipTrajectorySource requires clip.site_xpos; "
                "the loaded MotionClip does not contain site positions."
            )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _ensure_device(self, device: torch.device, n_envs: int) -> None:
        """Upload clip tensors and initialise per-env state for *device*."""
        import torch

        if (
            self._device == device
            and self._start_offsets is not None
            and self._start_offsets.shape[0] == n_envs
        ):
            return

        self._device = device
        n_frames = self.n_frames

        # Site positions: select only tracked sites, upload to device
        tracked = self.clip.site_xpos[:, self.tracked_site_ids, :]  # (T, n_tracked, 3)
        self._site_tensor = torch.as_tensor(
            np.asarray(tracked, dtype=np.float32), dtype=torch.float32, device=device
        )

        if self.clip.qpos is not None:
            self._qpos_tensor = torch.as_tensor(
                np.asarray(self.clip.qpos, dtype=np.float32),
                dtype=torch.float32,
                device=device,
            )
        else:
            self._qpos_tensor = None

        if self.clip.qvel is not None:
            self._qvel_tensor = torch.as_tensor(
                np.asarray(self.clip.qvel, dtype=np.float32),
                dtype=torch.float32,
                device=device,
            )
        else:
            self._qvel_tensor = None

        # Each env starts at a random frame in [0, T)
        self._start_offsets = torch.randint(
            0, n_frames, (n_envs,), device=device, dtype=torch.long
        )
        self._last_t = torch.full((n_envs,), -1.0, device=device, dtype=torch.float32)

    def _detect_and_resample_resets(self, t: torch.Tensor) -> None:
        """Resample start offsets for envs whose time regressed (new episode)."""
        import torch

        assert self._last_t is not None and self._start_offsets is not None

        reset_mask = t < (self._last_t - 1e-6)  # (N,) bool
        if reset_mask.any():
            new_offsets = torch.randint(
                0,
                self.n_frames,
                self._start_offsets.shape,
                device=self._start_offsets.device,
                dtype=torch.long,
            )
            self._start_offsets = torch.where(
                reset_mask, new_offsets, self._start_offsets
            )
        self._last_t = t.clone()

    def _frame_indices(self, t: torch.Tensor) -> torch.Tensor:
        """Return ``(N,)`` frame indices from per-env simulation time."""
        assert self._start_offsets is not None
        step = (t / self.ctrl_dt).long()  # (N,) floor
        return (step + self._start_offsets) % self.n_frames  # (N,)

    def frame_indices(self, t: torch.Tensor) -> torch.Tensor:
        """Return current ``(N,)`` clip frame indices for each environment.

        This is the public equivalent of :meth:`_frame_indices` for consumers
        that need to stay phase-aligned with the mjlab trajectory source, such
        as checkpoint policy inference wrappers.
        """
        return self._frame_indices(t)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def n_frames(self) -> int:
        """Number of frames in the clip."""
        assert self.clip.site_xpos is not None
        return int(self.clip.site_xpos.shape[0])

    @property
    def n_tracked(self) -> int:
        """Number of tracked sites."""
        return int(self.tracked_site_ids.shape[0])

    def update(self, t: torch.Tensor) -> None:
        """Synchronise internal state with the current per-env simulation time.

        Must be called once per step before querying :meth:`site_targets`,
        :meth:`ref_qpos`, or :meth:`phase`.

        Args:
            t: Per-env simulation time, shape ``(N,)``, ``float32``.
        """
        n_envs = int(t.shape[0])
        self._ensure_device(t.device, n_envs)
        self._detect_and_resample_resets(t)

    def site_targets(self, t: torch.Tensor) -> torch.Tensor:
        """Return ``(N, n_tracked, 3)`` site targets at the current frame.

        Args:
            t: Per-env simulation time, shape ``(N,)``.  Must match the ``t``
               passed to the most recent :meth:`update` call.

        Returns:
            World-frame site positions from the clip, shape ``(N, n_tracked, 3)``.
        """
        assert self._site_tensor is not None
        idx = self._frame_indices(t)  # (N,)
        return self.site_targets_at_frames(idx)  # (N, n_tracked, 3)

    def site_targets_at_frames(self, frame_idx: torch.Tensor) -> torch.Tensor:
        """Return tracked-site targets for explicit frame indices."""
        assert self._site_tensor is not None
        return self._site_tensor[frame_idx]

    def ref_qpos(self, t: torch.Tensor) -> torch.Tensor | None:
        """Return ``(N, nq)`` reference joint positions, or ``None`` if unavailable.

        Args:
            t: Per-env simulation time, shape ``(N,)``.

        Returns:
            Reference qpos tensor or ``None`` when ``clip.qpos`` is absent.
        """
        if self._qpos_tensor is None:
            return None
        idx = self._frame_indices(t)
        return self.ref_qpos_at_frames(idx)  # (N, nq)

    def ref_qpos_at_frames(self, frame_idx: torch.Tensor) -> torch.Tensor | None:
        """Return reference qpos for explicit frame indices."""
        if self._qpos_tensor is None:
            return None
        return self._qpos_tensor[frame_idx]

    def ref_qvel(self, t: torch.Tensor) -> torch.Tensor | None:
        """Return ``(N, nv)`` reference joint velocities, or ``None`` if unavailable.

        Args:
            t: Per-env simulation time, shape ``(N,)``.

        Returns:
            Reference qvel tensor or ``None`` when ``clip.qvel`` is absent.
        """
        if self._qvel_tensor is None:
            return None
        idx = self._frame_indices(t)
        return self.ref_qvel_at_frames(idx)  # (N, nv)

    def ref_qvel_at_frames(self, frame_idx: torch.Tensor) -> torch.Tensor | None:
        """Return reference qvel for explicit frame indices."""
        if self._qvel_tensor is None:
            return None
        return self._qvel_tensor[frame_idx]

    def phase(self, t: torch.Tensor) -> torch.Tensor:
        """Return ``(N, 1)`` normalised phase in ``[0, 1]`` along the clip.

        Phase reaches 1.0 at the last frame and wraps back to 0.0, giving the
        RL policy a continuous signal of progress through the motion cycle.

        Args:
            t: Per-env simulation time, shape ``(N,)``.

        Returns:
            Phase tensor, shape ``(N, 1)``, ``float32``.
        """
        idx = self._frame_indices(t).float()
        return (idx / float(self.n_frames)).unsqueeze(-1)  # (N, 1)

    def clip_lengths(self, t: torch.Tensor) -> torch.Tensor:
        """Return the active clip length for each environment."""
        import torch

        return torch.full_like(self._frame_indices(t), self.n_frames, dtype=torch.long)

    def initial_qpos(self) -> torch.Tensor | None:
        """Return ``(N, nq)`` qpos for each env at its assigned start offset.

        Called after :meth:`update` — uses the current ``start_offsets`` so
        results match the first call to :meth:`ref_qpos` at ``t=0``.

        Returns:
            Tensor of shape ``(N, nq)`` or ``None`` when ``clip.qpos`` is absent.
        """
        if self._qpos_tensor is None or self._start_offsets is None:
            return None
        return self._qpos_tensor[self._start_offsets]  # (N, nq)

    def initial_qvel(self) -> torch.Tensor | None:
        """Return ``(N, nv)`` qvel for each env at its assigned start offset.

        Returns:
            Tensor of shape ``(N, nv)`` or ``None`` when ``clip.qvel`` is absent.
        """
        if self._qvel_tensor is None or self._start_offsets is None:
            return None
        return self._qvel_tensor[self._start_offsets]  # (N, nv)

    def make_init_state_fn(
        self,
    ) -> Callable[[int, torch.device], tuple[torch.Tensor, torch.Tensor | None]]:
        """Return a factory that produces initial qpos/qvel tensors on demand.

        The returned callable accepts ``(n_envs, device)`` and returns
        ``(qpos_init, qvel_init)`` drawn from random frames in the clip.
        Useful for wiring into an mjlab reset callback outside the normal
        step-observation loop.

        Returns:
            A callable ``(n_envs, device) -> (qpos, qvel | None)`` that
            samples a fresh set of random starting offsets each time it is
            invoked.
        """
        import torch as _torch

        clip_qpos = self.clip.qpos
        clip_qvel = self.clip.qvel
        n_frames = self.n_frames

        def _init_fn(
            n_envs: int,
            device: _torch.device,
        ) -> tuple[_torch.Tensor, _torch.Tensor | None]:
            offsets = _torch.randint(
                0, n_frames, (n_envs,), device=device, dtype=_torch.long
            )
            qpos = None
            if clip_qpos is not None:
                qpos_t = _torch.as_tensor(
                    np.asarray(clip_qpos, dtype=np.float32),
                    dtype=_torch.float32,
                    device=device,
                )
                qpos = qpos_t[offsets]  # (n_envs, nq)
            qvel = None
            if clip_qvel is not None:
                qvel_t = _torch.as_tensor(
                    np.asarray(clip_qvel, dtype=np.float32),
                    dtype=_torch.float32,
                    device=device,
                )
                qvel = qvel_t[offsets]  # (n_envs, nv)
            return qpos, qvel

        return _init_fn


@dataclass
class MultiClipTrajectorySource:
    """Per-env trajectory source backed by a bank of motion clips.

    Each environment samples a clip index and a start frame independently on
    reset, while keeping the single-clip public API used by the mjlab mimic
    observation and reward closures.
    """

    clips: tuple[MotionClip, ...]
    tracked_site_ids: np.ndarray
    ctrl_dt: float

    _site_tensors: tuple[torch.Tensor, ...] | None = field(
        default=None, repr=False, init=False
    )
    _qpos_tensors: tuple[torch.Tensor | None, ...] | None = field(
        default=None, repr=False, init=False
    )
    _qvel_tensors: tuple[torch.Tensor | None, ...] | None = field(
        default=None, repr=False, init=False
    )
    _clip_lengths: torch.Tensor | None = field(default=None, repr=False, init=False)
    _clip_indices: torch.Tensor | None = field(default=None, repr=False, init=False)
    _start_offsets: torch.Tensor | None = field(default=None, repr=False, init=False)
    _last_t: torch.Tensor | None = field(default=None, repr=False, init=False)
    _device: object = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        if not self.clips:
            raise ValueError("MultiClipTrajectorySource requires at least one clip.")
        has_qpos = self.clips[0].qpos is not None
        has_qvel = self.clips[0].qvel is not None
        site_count = (
            int(self.clips[0].site_xpos.shape[1])
            if self.clips[0].site_xpos is not None
            else -1
        )
        qpos_width = int(self.clips[0].qpos.shape[1]) if has_qpos else None
        qvel_width = int(self.clips[0].qvel.shape[1]) if has_qvel else None
        for clip in self.clips:
            if clip.site_xpos is None:
                raise ValueError(
                    "MultiClipTrajectorySource requires clip.site_xpos for every clip."
                )
            if int(clip.site_xpos.shape[1]) != site_count:
                raise ValueError("All clips must expose the same tracked site layout.")
            if (clip.qpos is not None) != has_qpos:
                raise ValueError("All clips must agree on qpos availability.")
            if (clip.qvel is not None) != has_qvel:
                raise ValueError("All clips must agree on qvel availability.")
            if has_qpos and int(clip.qpos.shape[1]) != qpos_width:
                raise ValueError("All clips must share the same qpos width.")
            if has_qvel and int(clip.qvel.shape[1]) != qvel_width:
                raise ValueError("All clips must share the same qvel width.")

    def _sample_assignments(
        self,
        n_envs: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import torch

        clip_indices = torch.randint(
            0,
            len(self.clips),
            (n_envs,),
            device=device,
            dtype=torch.long,
        )
        lengths = self._clip_lengths.index_select(0, clip_indices)  # type: ignore[union-attr]
        offsets = torch.floor(
            torch.rand(n_envs, device=device, dtype=torch.float32) * lengths.float()
        ).to(dtype=torch.long)
        return clip_indices, offsets

    def _ensure_device(self, device: torch.device, n_envs: int) -> None:
        import torch

        if (
            self._device == device
            and self._start_offsets is not None
            and self._start_offsets.shape[0] == n_envs
        ):
            return

        self._device = device
        self._site_tensors = tuple(
            torch.as_tensor(
                np.asarray(
                    clip.site_xpos[:, self.tracked_site_ids, :], dtype=np.float32
                ),
                dtype=torch.float32,
                device=device,
            )
            for clip in self.clips
        )
        self._qpos_tensors = tuple(
            None
            if clip.qpos is None
            else torch.as_tensor(
                np.asarray(clip.qpos, dtype=np.float32),
                dtype=torch.float32,
                device=device,
            )
            for clip in self.clips
        )
        self._qvel_tensors = tuple(
            None
            if clip.qvel is None
            else torch.as_tensor(
                np.asarray(clip.qvel, dtype=np.float32),
                dtype=torch.float32,
                device=device,
            )
            for clip in self.clips
        )
        self._clip_lengths = torch.as_tensor(
            [int(clip.site_xpos.shape[0]) for clip in self.clips],
            dtype=torch.long,
            device=device,
        )
        self._clip_indices, self._start_offsets = self._sample_assignments(
            n_envs, device
        )
        self._last_t = torch.full((n_envs,), -1.0, device=device, dtype=torch.float32)

    def _detect_and_resample_resets(self, t: torch.Tensor) -> None:
        import torch

        assert self._last_t is not None
        assert self._clip_indices is not None
        assert self._start_offsets is not None

        reset_mask = t < (self._last_t - 1e-6)
        if reset_mask.any():
            new_clip_indices, new_offsets = self._sample_assignments(
                int(t.shape[0]), t.device
            )
            self._clip_indices = torch.where(
                reset_mask, new_clip_indices, self._clip_indices
            )
            self._start_offsets = torch.where(
                reset_mask, new_offsets, self._start_offsets
            )
        self._last_t = t.clone()

    def _frame_indices(self, t: torch.Tensor) -> torch.Tensor:
        assert self._clip_indices is not None
        assert self._start_offsets is not None
        assert self._clip_lengths is not None
        step = (t / self.ctrl_dt).long()
        lengths = self._clip_lengths.index_select(0, self._clip_indices)
        return (step + self._start_offsets) % lengths

    def _gather_from_bank(
        self,
        bank: tuple[torch.Tensor | None, ...],
        frame_idx: torch.Tensor,
    ) -> torch.Tensor | None:
        import torch

        assert self._clip_indices is not None
        template = next((tensor for tensor in bank if tensor is not None), None)
        if template is None:
            return None
        out = torch.zeros(
            (frame_idx.shape[0],) + tuple(template.shape[1:]),
            dtype=template.dtype,
            device=template.device,
        )
        for clip_idx, tensor in enumerate(bank):
            if tensor is None:
                return None
            mask = self._clip_indices == clip_idx
            if mask.any():
                out[mask] = tensor[frame_idx[mask]]
        return out

    @property
    def n_frames(self) -> int:
        """Return the maximum clip length in the bank."""
        return max(int(clip.site_xpos.shape[0]) for clip in self.clips)

    @property
    def n_tracked(self) -> int:
        """Number of tracked sites."""
        return int(self.tracked_site_ids.shape[0])

    def update(self, t: torch.Tensor) -> None:
        """Synchronise the clip bank state with per-env simulation time."""
        self._ensure_device(t.device, int(t.shape[0]))
        self._detect_and_resample_resets(t)

    def frame_indices(self, t: torch.Tensor) -> torch.Tensor:
        """Return the current frame index within each env's active clip."""
        return self._frame_indices(t)

    def clip_lengths(self, t: torch.Tensor) -> torch.Tensor:
        """Return the active clip length for each environment."""
        assert self._clip_lengths is not None
        assert self._clip_indices is not None
        return self._clip_lengths.index_select(0, self._clip_indices)

    def site_targets(self, t: torch.Tensor) -> torch.Tensor:
        """Return tracked-site targets at the current frame."""
        return self.site_targets_at_frames(self._frame_indices(t))

    def site_targets_at_frames(self, frame_idx: torch.Tensor) -> torch.Tensor:
        """Return tracked-site targets for explicit frame indices."""
        result = self._gather_from_bank(self._site_tensors, frame_idx)  # type: ignore[arg-type]
        assert result is not None
        return result

    def ref_qpos(self, t: torch.Tensor) -> torch.Tensor | None:
        """Return reference qpos at the current frame."""
        return self.ref_qpos_at_frames(self._frame_indices(t))

    def ref_qpos_at_frames(self, frame_idx: torch.Tensor) -> torch.Tensor | None:
        """Return reference qpos for explicit frame indices."""
        return self._gather_from_bank(self._qpos_tensors, frame_idx)  # type: ignore[arg-type]

    def ref_qvel(self, t: torch.Tensor) -> torch.Tensor | None:
        """Return reference qvel at the current frame."""
        return self.ref_qvel_at_frames(self._frame_indices(t))

    def ref_qvel_at_frames(self, frame_idx: torch.Tensor) -> torch.Tensor | None:
        """Return reference qvel for explicit frame indices."""
        return self._gather_from_bank(self._qvel_tensors, frame_idx)  # type: ignore[arg-type]

    def phase(self, t: torch.Tensor) -> torch.Tensor:
        """Return normalised phase within the active clip for each environment."""
        idx = self._frame_indices(t).float()
        lengths = self.clip_lengths(t).float().clamp_min(1.0)
        return (idx / lengths).unsqueeze(-1)

    def initial_qpos(self) -> torch.Tensor | None:
        """Return qpos sampled from each env's assigned clip/frame."""
        if self._start_offsets is None:
            return None
        return self.ref_qpos_at_frames(self._start_offsets)

    def initial_qvel(self) -> torch.Tensor | None:
        """Return qvel sampled from each env's assigned clip/frame."""
        if self._start_offsets is None:
            return None
        return self.ref_qvel_at_frames(self._start_offsets)

    def make_init_state_fn(
        self,
    ) -> Callable[[int, torch.device], tuple[torch.Tensor | None, torch.Tensor | None]]:
        """Return a factory that samples qpos/qvel from a random clip bank entry."""
        import torch as _torch

        clips = self.clips
        tracked_site_ids = np.asarray(self.tracked_site_ids, dtype=np.int64)
        ctrl_dt = float(self.ctrl_dt)

        def _init_fn(
            n_envs: int,
            device: _torch.device,
        ) -> tuple[_torch.Tensor | None, _torch.Tensor | None]:
            source = MultiClipTrajectorySource(
                clips=clips,
                tracked_site_ids=tracked_site_ids,
                ctrl_dt=ctrl_dt,
            )
            source._ensure_device(device, n_envs)
            return source.initial_qpos(), source.initial_qvel()

        return _init_fn


__all__ = ["ClipTrajectorySource", "MultiClipTrajectorySource"]
