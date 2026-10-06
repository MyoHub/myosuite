# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Per-environment trajectory source backed by a :class:`~myosuite.core.trajectory_io.MotionClip`.

Each of the N parallel mjlab environments gets its own random starting frame
that is resampled independently on episode reset.  Frames advance one per
control step:

    frame[i] = min(step[i] + start_offset[i], T - 1)

where ``step`` is the integer number of control steps since each env's last
reset (mjlab's ``env.episode_length_buf``), the counterpart of the CPU twin's
step counter.  Float32 simulation time is not used: it drifts, and
``floor(time / ctrl_dt)`` lags the step counter on most steps.  The step that
plays past the last frame is truncated (:meth:`ClipTrajectorySource.clip_end`)
and scored against the last frame, not a wrapped frame 0.

This gives a diverse distribution of motion phases across the batch while
keeping each individual episode's targets coherent with the reference clip.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from myosuite.core.trajectory_io import check_clip_rate

if TYPE_CHECKING:
    import torch

    from myosuite.core.trajectory_io import MotionClip


def _as_steps(step: torch.Tensor) -> torch.Tensor:
    """Return the per-env step counter as int64, rejecting float tensors."""
    if step.is_floating_point():
        raise TypeError(
            "Clip frames are indexed by the integer per-env control-step counter "
            "(mjlab env.episode_length_buf), not by float simulation time; "
            f"got dtype {step.dtype}."
        )
    return step.long()


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
        ctrl_dt: Control timestep in seconds (``sim_dt × decimation``); the
                 clip plays back one frame per control step.

    The per-step methods take ``step``: the ``(N,)`` integer control-step
    counter since each env's last reset (mjlab ``env.episode_length_buf``).

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
    _last_step: torch.Tensor | None = field(default=None, repr=False, init=False)
    _device: object = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        if self.clip.site_xpos is None:
            raise ValueError(
                "ClipTrajectorySource requires clip.site_xpos; "
                "the loaded MotionClip does not contain site positions."
            )
        check_clip_rate(self.clip.frequency_hz, self.ctrl_dt)

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
        self._last_step = torch.full((n_envs,), -1, device=device, dtype=torch.long)

    def _detect_and_resample_resets(self, step: torch.Tensor) -> None:
        """Resample start offsets for envs whose step counter regressed (new episode)."""
        import torch

        assert self._last_step is not None and self._start_offsets is not None

        reset_mask = step < self._last_step  # (N,) bool
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
        self._last_step = step.clone()

    def _frame_indices(self, step: torch.Tensor) -> torch.Tensor:
        """Return ``(N,)`` frame indices from the per-env step counter.

        Held at the last frame past the clip end (see :meth:`clip_end`).
        """
        assert self._start_offsets is not None
        frames = _as_steps(step) + self._start_offsets
        return frames.clamp(max=self.n_frames - 1)  # (N,)

    def clip_end(self, step: torch.Tensor) -> torch.Tensor:
        """``(N,)`` bool: the episode has played past the last frame of its clip."""
        assert self._start_offsets is not None
        return _as_steps(step) + self._start_offsets >= self.n_frames

    def frame_indices(self, step: torch.Tensor) -> torch.Tensor:
        """Return current ``(N,)`` clip frame indices for each environment.

        This is the public equivalent of :meth:`_frame_indices` for consumers
        that need to stay phase-aligned with the mjlab trajectory source, such
        as checkpoint policy inference wrappers.
        """
        return self._frame_indices(step)

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

    def update(self, step: torch.Tensor, *, check_resets: bool = True) -> None:
        """Synchronise internal state with the current per-env step counter.

        Must be called once per step before querying :meth:`site_targets`,
        :meth:`ref_qpos`, or :meth:`phase`.  A counter that went backwards
        marks a new episode and resamples that env's start offset.

        Args:
            step: Control steps since each env's last reset, shape ``(N,)``,
                integer dtype (mjlab ``env.episode_length_buf``).
            check_resets: ``False`` when *step* is known not to have decreased
                since the last update (e.g. one ``+= 1``): skips the reset
                check, a host sync on a GPU.

        Raises:
            TypeError: If *step* is a floating-point tensor.
        """
        step = _as_steps(step)
        n_envs = int(step.shape[0])
        self._ensure_device(step.device, n_envs)
        if check_resets:
            self._detect_and_resample_resets(step)
        else:
            self._last_step = step.clone()

    def site_targets(self, step: torch.Tensor) -> torch.Tensor:
        """Return ``(N, n_tracked, 3)`` site targets at the current frame.

        Args:
            step: Per-env step counter, shape ``(N,)``.  Must match the
                ``step`` passed to the most recent :meth:`update` call.

        Returns:
            World-frame site positions from the clip, shape ``(N, n_tracked, 3)``.
        """
        assert self._site_tensor is not None
        idx = self._frame_indices(step)  # (N,)
        return self.site_targets_at_frames(idx)  # (N, n_tracked, 3)

    def site_targets_at_frames(self, frame_idx: torch.Tensor) -> torch.Tensor:
        """Return tracked-site targets for explicit frame indices."""
        assert self._site_tensor is not None
        return self._site_tensor[frame_idx]

    def ref_qpos(self, step: torch.Tensor) -> torch.Tensor | None:
        """Return ``(N, nq)`` reference joint positions, or ``None`` if unavailable.

        Args:
            step: Per-env step counter, shape ``(N,)``.

        Returns:
            Reference qpos tensor or ``None`` when ``clip.qpos`` is absent.
        """
        if self._qpos_tensor is None:
            return None
        idx = self._frame_indices(step)
        return self.ref_qpos_at_frames(idx)  # (N, nq)

    def ref_qpos_at_frames(
        self, frame_idx: torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        """Return reference qpos for explicit frame indices (*env_ids* is unused)."""
        if self._qpos_tensor is None:
            return None
        return self._qpos_tensor[frame_idx]

    def ref_qvel(self, step: torch.Tensor) -> torch.Tensor | None:
        """Return ``(N, nv)`` reference joint velocities, or ``None`` if unavailable.

        Args:
            step: Per-env step counter, shape ``(N,)``.

        Returns:
            Reference qvel tensor or ``None`` when ``clip.qvel`` is absent.
        """
        if self._qvel_tensor is None:
            return None
        idx = self._frame_indices(step)
        return self.ref_qvel_at_frames(idx)  # (N, nv)

    def ref_qvel_at_frames(
        self, frame_idx: torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        """Return reference qvel for explicit frame indices (*env_ids* is unused)."""
        if self._qvel_tensor is None:
            return None
        return self._qvel_tensor[frame_idx]

    def phase(self, step: torch.Tensor) -> torch.Tensor:
        """Return ``(N, 1)`` normalised phase ``frame / T`` in ``[0, 1)`` along the clip.

        Phase grows by ``1 / T`` per step and holds at ``(T - 1) / T`` from the
        last frame on, giving the RL policy a signal of progress through the motion.

        Args:
            step: Per-env step counter, shape ``(N,)``.

        Returns:
            Phase tensor, shape ``(N, 1)``, ``float32``.
        """
        idx = self._frame_indices(step).float()
        return (idx / float(self.n_frames)).unsqueeze(-1)  # (N, 1)

    def clip_lengths(self, step: torch.Tensor) -> torch.Tensor:
        """Return the active clip length for each environment."""
        import torch

        return torch.full_like(
            self._frame_indices(step), self.n_frames, dtype=torch.long
        )

    def initial_qpos(self) -> torch.Tensor | None:
        """Return ``(N, nq)`` qpos for each env at its assigned start offset.

        Called after :meth:`update` — uses the current ``start_offsets`` so
        results match the first call to :meth:`ref_qpos` at ``step=0``.

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
    observation and reward closures (per-step methods take the integer
    per-env step counter, see :class:`ClipTrajectorySource`).
    """

    clips: tuple[MotionClip, ...]
    tracked_site_ids: np.ndarray
    ctrl_dt: float

    # Frames of all clips concatenated along time; clip c starts at row
    # _clip_starts[c], so a gather reads row start + frame (no per-clip mask).
    _site_bank: torch.Tensor | None = field(default=None, repr=False, init=False)
    _qpos_bank: torch.Tensor | None = field(default=None, repr=False, init=False)
    _qvel_bank: torch.Tensor | None = field(default=None, repr=False, init=False)
    _clip_starts: torch.Tensor | None = field(default=None, repr=False, init=False)
    # Per-clip views of the banks.
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
    _last_step: torch.Tensor | None = field(default=None, repr=False, init=False)
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
        lengths = [int(clip.site_xpos.shape[0]) for clip in self.clips]

        def upload(arrays: list[np.ndarray] | None) -> torch.Tensor | None:
            if arrays is None:
                return None
            flat = np.concatenate([np.asarray(a, dtype=np.float32) for a in arrays])
            return torch.as_tensor(flat, dtype=torch.float32, device=device)

        self._site_bank = upload(
            [clip.site_xpos[:, self.tracked_site_ids, :] for clip in self.clips]
        )
        # __post_init__ checked that all clips agree on qpos / qvel availability.
        has_qpos = self.clips[0].qpos is not None
        has_qvel = self.clips[0].qvel is not None
        self._qpos_bank = upload([c.qpos for c in self.clips] if has_qpos else None)
        self._qvel_bank = upload([c.qvel for c in self.clips] if has_qvel else None)
        self._clip_starts = torch.as_tensor(
            np.cumsum([0] + lengths[:-1]), dtype=torch.long, device=device
        )
        self._site_tensors = tuple(self._site_bank.split(lengths))
        self._qpos_tensors = (
            tuple(self._qpos_bank.split(lengths))
            if self._qpos_bank is not None
            else (None,) * len(self.clips)
        )
        self._qvel_tensors = (
            tuple(self._qvel_bank.split(lengths))
            if self._qvel_bank is not None
            else (None,) * len(self.clips)
        )
        self._clip_lengths = torch.as_tensor(lengths, dtype=torch.long, device=device)
        self._clip_indices, self._start_offsets = self._sample_assignments(
            n_envs, device
        )
        self._last_step = torch.full((n_envs,), -1, device=device, dtype=torch.long)

    def _detect_and_resample_resets(self, step: torch.Tensor) -> None:
        import torch

        assert self._last_step is not None
        assert self._clip_indices is not None
        assert self._start_offsets is not None

        reset_mask = step < self._last_step
        if reset_mask.any():
            new_clip_indices, new_offsets = self._sample_assignments(
                int(step.shape[0]), step.device
            )
            self._clip_indices = torch.where(
                reset_mask, new_clip_indices, self._clip_indices
            )
            self._start_offsets = torch.where(
                reset_mask, new_offsets, self._start_offsets
            )
        self._last_step = step.clone()

    def _frame_indices(self, step: torch.Tensor) -> torch.Tensor:
        assert self._clip_indices is not None
        assert self._start_offsets is not None
        assert self._clip_lengths is not None
        lengths = self._clip_lengths.index_select(0, self._clip_indices)
        # Held at each clip's last frame past its end (see :meth:`clip_end`).
        return (_as_steps(step) + self._start_offsets).minimum(lengths - 1)

    def clip_end(self, step: torch.Tensor) -> torch.Tensor:
        """``(N,)`` bool: the episode has played past the last frame of its clip."""
        assert self._clip_indices is not None
        assert self._start_offsets is not None
        assert self._clip_lengths is not None
        lengths = self._clip_lengths.index_select(0, self._clip_indices)
        return _as_steps(step) + self._start_offsets >= lengths

    def _gather_from_bank(
        self,
        bank: torch.Tensor | None,
        frame_idx: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Gather rows of *bank* at *frame_idx* from each env's clip.

        *frame_idx* has one entry per env, or per env of *env_ids* when only those
        envs are gathered (e.g. the envs of a partial reset).  Each frame must lie
        in its env's clip (``[0, clip length)``), as :meth:`frame_indices`
        returns; a frame outside it would read a neighbouring clip's rows, so
        it is checked on the device (raises at once on CPU, a device-side
        assert on CUDA).  One gather, no host sync.
        """
        if bank is None:
            return None
        import torch

        assert self._clip_indices is not None and self._clip_starts is not None
        assert self._clip_lengths is not None
        clip_indices = (
            self._clip_indices if env_ids is None else self._clip_indices[env_ids]
        )
        lengths = self._clip_lengths.index_select(0, clip_indices)
        torch._assert_async(
            ((frame_idx >= 0) & (frame_idx < lengths)).all(),
            "clip bank read: a frame lies outside its env's clip",
        )
        rows = self._clip_starts.index_select(0, clip_indices) + frame_idx
        return bank.index_select(0, rows)

    @property
    def n_frames(self) -> int:
        """Return the maximum clip length in the bank."""
        return max(int(clip.site_xpos.shape[0]) for clip in self.clips)

    @property
    def n_tracked(self) -> int:
        """Number of tracked sites."""
        return int(self.tracked_site_ids.shape[0])

    def update(self, step: torch.Tensor, *, check_resets: bool = True) -> None:
        """Synchronise the clip bank state with the per-env step counter.

        See :meth:`ClipTrajectorySource.update` for *check_resets*.
        """
        step = _as_steps(step)
        self._ensure_device(step.device, int(step.shape[0]))
        if check_resets:
            self._detect_and_resample_resets(step)
        else:
            self._last_step = step.clone()

    def frame_indices(self, step: torch.Tensor) -> torch.Tensor:
        """Return the current frame index within each env's active clip."""
        return self._frame_indices(step)

    def clip_lengths(self, step: torch.Tensor) -> torch.Tensor:
        """Return the active clip length for each environment."""
        assert self._clip_lengths is not None
        assert self._clip_indices is not None
        return self._clip_lengths.index_select(0, self._clip_indices)

    def site_targets(self, step: torch.Tensor) -> torch.Tensor:
        """Return tracked-site targets at the current frame."""
        return self.site_targets_at_frames(self._frame_indices(step))

    def site_targets_at_frames(self, frame_idx: torch.Tensor) -> torch.Tensor:
        """Return tracked-site targets for explicit frame indices."""
        result = self._gather_from_bank(self._site_bank, frame_idx)
        assert result is not None
        return result

    def ref_qpos(self, step: torch.Tensor) -> torch.Tensor | None:
        """Return reference qpos at the current frame."""
        return self.ref_qpos_at_frames(self._frame_indices(step))

    def ref_qpos_at_frames(
        self, frame_idx: torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        """Return reference qpos for explicit frame indices (of *env_ids*, if given)."""
        return self._gather_from_bank(self._qpos_bank, frame_idx, env_ids)

    def ref_qvel(self, step: torch.Tensor) -> torch.Tensor | None:
        """Return reference qvel at the current frame."""
        return self.ref_qvel_at_frames(self._frame_indices(step))

    def ref_qvel_at_frames(
        self, frame_idx: torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        """Return reference qvel for explicit frame indices (of *env_ids*, if given)."""
        return self._gather_from_bank(self._qvel_bank, frame_idx, env_ids)

    def phase(self, step: torch.Tensor) -> torch.Tensor:
        """Return normalised phase within the active clip for each environment."""
        idx = self._frame_indices(step).float()
        lengths = self.clip_lengths(step).float().clamp_min(1.0)
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
