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

A bank of clips (:class:`MultiClipTrajectorySource`) keeps every frame on the
device; :class:`ClipBankCfg` (passed with :class:`MotionClipBank`) trades exact
storage for memory on large motion datasets.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from myosuite.core.trajectory_io import check_clip_rate

if TYPE_CHECKING:
    import torch

    from myosuite.core.trajectory_io import MotionClip

_FLOAT16_MAX_ABS = 4.0
"""In a float16 bank, qpos columns larger than this stay float32 (float16 error <= 4 * 2**-11)."""


@dataclass(frozen=True)
class ClipBankCfg:
    """How a bank of clips is stored on the device (the default is exact).

    Args:
        dtype: ``"float32"`` stores every frame as loaded (exact). ``"float16"``
            halves the bank: site targets as a float32 per-frame centroid plus
            float16 offsets, qpos columns larger than 4 in magnitude (the root
            position) in float32 and the rest in float16, qvel in float16. With
            ``store_qvel=False`` the qpos stays float32 (the derived qvel would
            amplify its float16 error by ``2 / ctrl_dt``), which gives the same
            size with exact qpos. The measured error is
            :attr:`MultiClipTrajectorySource.storage_error`.
        store_qvel: Keep the clips' qvel on the device. ``False`` drops the qvel
            bank and derives the reference qvel from consecutive qpos frames as
            MuJoCo's ``mj_differentiatePos`` over ``ctrl_dt``: the backward
            difference ``qpos[t-1] -> qpos[t]`` (forward at a clip's first
            frame), the convention of the MuscleMimic clips. Check a dataset
            with :func:`qvel_derivation_error` first.
    """

    dtype: Literal["float32", "float16"] = "float32"
    store_qvel: bool = True

    def __post_init__(self) -> None:
        if self.dtype not in ("float32", "float16"):
            raise ValueError(
                f"dtype must be 'float32' or 'float16', got {self.dtype!r}."
            )

    @property
    def exact(self) -> bool:
        """Whether the bank returns the loaded frames unchanged."""
        return self.dtype == "float32" and self.store_qvel


class MotionClipBank(tuple):
    """Motion clips plus the storage of their mjlab bank.

    A tuple of :class:`~myosuite.core.trajectory_io.MotionClip`, so it goes
    wherever the mjlab Mimic tasks take a clip bank (``clip=``); the bank reads
    :attr:`bank_cfg`.

    Args:
        clips: The motion clips.
        bank_cfg: Storage of the bank (default: exact).
    """

    bank_cfg: ClipBankCfg

    def __new__(
        cls, clips: Iterable[MotionClip], bank_cfg: ClipBankCfg | None = None
    ) -> MotionClipBank:
        bank = super().__new__(cls, tuple(clips))
        bank.bank_cfg = bank_cfg or ClipBankCfg()
        return bank


@dataclass(frozen=True)
class ClipJointLayout:
    """Joint layout of the model the clips drive (to derive qvel from qpos).

    Args:
        jnt_type: ``mjtJoint`` of each joint.
        jnt_qposadr: First qpos column of each joint.
        jnt_dofadr: First qvel column of each joint.
        nv: Width of qvel.
    """

    jnt_type: tuple[int, ...]
    jnt_qposadr: tuple[int, ...]
    jnt_dofadr: tuple[int, ...]
    nv: int

    @classmethod
    def from_model(cls, mj_model: Any) -> ClipJointLayout:
        """Layout of a compiled ``MjModel``."""
        return cls(
            tuple(int(t) for t in mj_model.jnt_type),
            tuple(int(a) for a in mj_model.jnt_qposadr),
            tuple(int(a) for a in mj_model.jnt_dofadr),
            int(mj_model.nv),
        )


def qvel_derivation_error(
    clips: Iterable[MotionClip], mj_model: Any, ctrl_dt: float
) -> float:
    """Largest ``|stored qvel - qvel derived from qpos|`` over *clips*.

    The derivation is the one of ``ClipBankCfg(store_qvel=False)``: MuJoCo's
    ``mj_differentiatePos`` from the previous frame to this one over *ctrl_dt*
    (forward at a clip's first frame). Clips must be in model layout
    (:func:`~myosuite.core.trajectory_io.expand_motion_clip_to_model`).

    Args:
        clips: Clips with qpos and qvel.
        mj_model: The model the clips drive.
        ctrl_dt: Control step the clips play at.

    Returns:
        The largest absolute difference (qvel units), 0 for clips without qvel.
    """
    import mujoco

    worst = 0.0
    derived = np.zeros(int(mj_model.nv))
    for clip in clips:
        if clip.qpos is None or clip.qvel is None:
            continue
        qpos = np.asarray(clip.qpos, dtype=np.float64)
        n = int(qpos.shape[0])
        for t in range(n):
            a, b = (t - 1, t) if t > 0 else (0, min(1, n - 1))
            mujoco.mj_differentiatePos(mj_model, derived, ctrl_dt, qpos[a], qpos[b])
            worst = max(worst, float(np.abs(derived - clip.qvel[t]).max()))
    return worst


class _QvelFromQpos:
    """``mj_differentiatePos`` between two batches of qpos rows, on the device."""

    def __init__(self, layout: ClipJointLayout, dt: float, device: Any) -> None:
        import mujoco
        import torch
        from mjlab.utils.lab_api import math as lab_math

        self._math = lab_math
        lin_q: list[int] = []
        lin_v: list[int] = []
        quat_q: list[list[int]] = []
        quat_v: list[list[int]] = []
        for jtype, qadr, vadr in zip(
            layout.jnt_type, layout.jnt_qposadr, layout.jnt_dofadr, strict=True
        ):
            if jtype == mujoco.mjtJoint.mjJNT_FREE:  # world position, then a quaternion
                lin_q += [qadr, qadr + 1, qadr + 2]
                lin_v += [vadr, vadr + 1, vadr + 2]
                qadr, vadr = qadr + 3, vadr + 3
            if jtype in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
                quat_q.append([qadr + i for i in range(4)])
                quat_v.append([vadr + i for i in range(3)])
            else:  # hinge / slide
                lin_q.append(qadr)
                lin_v.append(vadr)

        def _long(values: list[Any]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.long, device=device)

        self._lin_q, self._lin_v = _long(lin_q), _long(lin_v)
        self._quat_q = _long(quat_q).reshape(-1, 4)
        self._quat_v = _long(quat_v).reshape(-1, 3)
        self._nv, self._inv_dt = layout.nv, 1.0 / dt

    def __call__(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        import torch

        m = self._math
        qvel = torch.zeros(q1.shape[0], self._nv, dtype=q1.dtype, device=q1.device)
        qvel[:, self._lin_v] = (q2[:, self._lin_q] - q1[:, self._lin_q]) * self._inv_dt
        if self._quat_q.shape[0]:
            # mju_subQuat: the rotation q1^-1 q2, in the body frame of q1.
            rel = m.quat_mul(m.quat_conjugate(q1[:, self._quat_q]), q2[:, self._quat_q])
            qvel[:, self._quat_v] = m.axis_angle_from_quat(rel) * self._inv_dt
        return qvel


class _Bank:
    """One field of every clip on the device: float32 rows, or float16 with float32 parts.

    Args:
        arrays: Per-clip arrays ``(T_c, *shape)``.
        device: Torch device.
        half: Store float16 (see :class:`ClipBankCfg`).
        centroid: (half) Store the per-frame mean over axis 1 in float32 and the
            offsets from it in float16 (site positions).
        max_abs: (half) Columns larger than this stay float32; ``None``: none.
    """

    def __init__(
        self,
        arrays: list[np.ndarray],
        device: Any,
        half: bool = False,
        centroid: bool = False,
        max_abs: float | None = None,
    ) -> None:
        import torch

        self.shape = tuple(int(s) for s in arrays[0].shape[1:])
        self.lengths = [int(a.shape[0]) for a in arrays]
        starts = np.cumsum([0, *self.lengths[:-1]])
        total, width = sum(self.lengths), int(np.prod(self.shape))
        self.error = 0.0
        self.full = self.centroid = self.part32 = self.part16 = None

        def _put(dst: torch.Tensor, start: int, values: np.ndarray) -> None:
            dst[start : start + values.shape[0]].copy_(torch.from_numpy(values))

        if not half:
            # One device copy per clip: no host copy of the whole bank.
            self.full = torch.empty(
                (total, *self.shape), dtype=torch.float32, device=device
            )
            for start, a in zip(starts, arrays, strict=True):
                _put(self.full, start, np.ascontiguousarray(a, dtype=np.float32))
            return

        def _flat(a: np.ndarray) -> np.ndarray:
            return np.ascontiguousarray(a, dtype=np.float32).reshape(a.shape[0], width)

        if centroid:
            self.centroid = torch.empty((total, 3), dtype=torch.float32, device=device)
            self.part16 = torch.empty(
                (total, width), dtype=torch.float16, device=device
            )
            for start, a in zip(starts, arrays, strict=True):
                x = np.ascontiguousarray(a, dtype=np.float32)
                mid = x.mean(axis=1)
                off = (x - mid[:, None]).astype(np.float16)
                self._track_error(off.astype(np.float32) + mid[:, None], x)
                _put(self.centroid, start, mid)
                _put(self.part16, start, off.reshape(off.shape[0], width))
            return

        col_max = np.zeros(width, dtype=np.float32)
        for a in arrays:
            col_max = np.maximum(col_max, np.abs(_flat(a)).max(axis=0, initial=0.0))
        wide = col_max > (np.inf if max_abs is None else max_abs)
        self.cols32 = torch.as_tensor(np.flatnonzero(wide), device=device)
        self.cols16 = torch.as_tensor(np.flatnonzero(~wide), device=device)
        self.part32 = torch.empty(
            (total, int(wide.sum())), dtype=torch.float32, device=device
        )
        self.part16 = torch.empty(
            (total, int((~wide).sum())), dtype=torch.float16, device=device
        )
        for start, a in zip(starts, arrays, strict=True):
            x = _flat(a)
            low = x[:, ~wide].astype(np.float16)
            self._track_error(low.astype(np.float32), x[:, ~wide])
            _put(self.part32, start, np.ascontiguousarray(x[:, wide]))
            _put(self.part16, start, low)

    def _track_error(self, stored: np.ndarray, exact: np.ndarray) -> None:
        if not np.isfinite(stored).all():
            raise ValueError("A clip value does not fit float16; use dtype='float32'.")
        if stored.size:
            self.error = max(self.error, float(np.abs(stored - exact).max()))

    def gather(self, rows: torch.Tensor) -> torch.Tensor:
        """Float32 rows *rows*, shape ``(len(rows), *shape)``."""
        import torch

        if self.full is not None:
            return self.full.index_select(0, rows)
        n = rows.shape[0]
        if self.centroid is not None:
            off = self.part16.index_select(0, rows).float().view(n, *self.shape)
            return off + self.centroid.index_select(0, rows).unsqueeze(1)
        width = self.part16.shape[1] + self.part32.shape[1]
        out = torch.empty((n, width), dtype=torch.float32, device=rows.device)
        out[:, self.cols32] = self.part32.index_select(0, rows)
        out[:, self.cols16] = self.part16.index_select(0, rows).float()
        return out.view(n, *self.shape)

    def per_clip(self) -> tuple[torch.Tensor, ...]:
        """Per-clip float32 views (exact banks only; empty otherwise)."""
        return tuple(self.full.split(self.lengths)) if self.full is not None else ()

    @property
    def nbytes(self) -> int:
        """Device bytes of the bank."""
        parts = (self.full, self.centroid, self.part32, self.part16)
        return sum(t.numel() * t.element_size() for t in parts if t is not None)


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

    Args:
        clips: The motion clips (model layout, same site / qpos / qvel widths).
        tracked_site_ids: Columns of ``clip.site_xpos`` that are tracked.
        ctrl_dt: Control timestep in seconds (one frame per control step).
        bank_cfg: Device storage of the bank (default: exact float32).
        joint_layout: The model's joint layout; required for
            ``bank_cfg.store_qvel=False`` on clips with qvel.
    """

    clips: tuple[MotionClip, ...]
    tracked_site_ids: np.ndarray
    ctrl_dt: float
    bank_cfg: ClipBankCfg = field(default_factory=ClipBankCfg)
    joint_layout: ClipJointLayout | None = None

    # Frames of all clips concatenated along time; clip c starts at row
    # _clip_starts[c], so a gather reads row start + frame (no per-clip mask).
    _site_bank: _Bank | None = field(default=None, repr=False, init=False)
    _qpos_bank: _Bank | None = field(default=None, repr=False, init=False)
    _qvel_bank: _Bank | None = field(default=None, repr=False, init=False)
    # qvel derived from the qpos bank (bank_cfg.store_qvel=False).
    _qvel_from_qpos: _QvelFromQpos | None = field(default=None, repr=False, init=False)
    _clip_starts: torch.Tensor | None = field(default=None, repr=False, init=False)
    # Per-clip float32 views of an exact bank (empty for a float16 / derived one);
    # not None once the bank is on the device.
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
    _max_frames: int = field(default=0, repr=False, init=False)

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
        if not self.bank_cfg.store_qvel and has_qvel:
            if not has_qpos or self.joint_layout is None:
                raise ValueError(
                    "ClipBankCfg(store_qvel=False) derives qvel from qpos: the clips "
                    "need qpos and the source a joint_layout."
                )
            if self.joint_layout.nv != qvel_width:
                raise ValueError(
                    f"joint_layout.nv={self.joint_layout.nv} does not match the "
                    f"clips' qvel width {qvel_width}."
                )
        # Once: n_frames is read on every reset, a loop over a large bank.
        self._max_frames = max(int(clip.site_xpos.shape[0]) for clip in self.clips)

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
        if self._device != device or self._clip_lengths is None:
            self._upload(device)  # the bank only; a new env count reuses it
        self._clip_indices, self._start_offsets = self._sample_assignments(
            n_envs, device
        )
        self._last_step = torch.full((n_envs,), -1, device=device, dtype=torch.long)

    def _upload(self, device: torch.device) -> None:
        """Put the bank on *device*, stored as :attr:`bank_cfg` says."""
        import torch

        self._device = device
        lengths = [int(clip.site_xpos.shape[0]) for clip in self.clips]
        half = self.bank_cfg.dtype == "float16"
        ids = np.asarray(self.tracked_site_ids)
        # A contiguous id range is a view of each clip, not a copy.
        sites = (
            slice(int(ids[0]), int(ids[-1]) + 1)
            if ids.size and np.array_equal(ids, np.arange(ids[0], ids[0] + ids.size))
            else ids
        )
        self._site_bank = _Bank(
            [clip.site_xpos[:, sites, :] for clip in self.clips],
            device,
            half=half,
            centroid=True,
        )
        # __post_init__ checked that all clips agree on qpos / qvel availability.
        has_qpos = self.clips[0].qpos is not None
        has_qvel = self.clips[0].qvel is not None
        derive = has_qvel and not self.bank_cfg.store_qvel
        # A derived qvel differences qpos over ctrl_dt, which would turn float16
        # qpos error into ~2 err / ctrl_dt of qvel error: qpos stays float32 then.
        self._qpos_bank = (
            _Bank(
                [c.qpos for c in self.clips],
                device,
                half and not derive,
                max_abs=_FLOAT16_MAX_ABS,
            )
            if has_qpos
            else None
        )
        self._qvel_bank = (
            _Bank([c.qvel for c in self.clips], device, half)
            if has_qvel and not derive
            else None
        )
        self._qvel_from_qpos = (
            _QvelFromQpos(self.joint_layout, self.ctrl_dt, device) if derive else None
        )
        self._clip_starts = torch.as_tensor(
            np.cumsum([0] + lengths[:-1]), dtype=torch.long, device=device
        )
        self._site_tensors = self._site_bank.per_clip()
        self._qpos_tensors = (
            self._qpos_bank.per_clip()
            if self._qpos_bank is not None
            else (None,) * len(self.clips)
        )
        self._qvel_tensors = (
            self._qvel_bank.per_clip()
            if self._qvel_bank is not None
            else (None,) * len(self.clips)
        )
        self._clip_lengths = torch.as_tensor(lengths, dtype=torch.long, device=device)

    @property
    def bank_nbytes(self) -> int:
        """Device bytes of the uploaded bank (0 before the first update)."""
        banks = (self._site_bank, self._qpos_bank, self._qvel_bank)
        return sum(b.nbytes for b in banks if b is not None)

    @property
    def storage_error(self) -> dict[str, float]:
        """Largest absolute storage error of each uploaded bank (0 when exact)."""
        banks = {"site_xpos": self._site_bank, "qpos": self._qpos_bank}
        banks["qvel"] = self._qvel_bank
        return {name: b.error for name, b in banks.items() if b is not None}

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
        bank: _Bank | None,
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
        return bank.gather(rows)

    @property
    def n_frames(self) -> int:
        """Return the maximum clip length in the bank."""
        return self._max_frames

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
        """Return reference qvel for explicit frame indices (of *env_ids*, if given).

        With ``bank_cfg.store_qvel=False`` it is derived from the qpos of the
        previous frame and this one (this one and the next at a clip's first frame).
        """
        derive = self._qvel_from_qpos
        if derive is None:
            return self._gather_from_bank(self._qvel_bank, frame_idx, env_ids)
        import torch

        assert self._clip_indices is not None and self._clip_lengths is not None
        clip_indices = (
            self._clip_indices if env_ids is None else self._clip_indices[env_ids]
        )
        last = self._clip_lengths.index_select(0, clip_indices) - 1
        first = frame_idx == 0
        prev = torch.where(first, frame_idx, frame_idx - 1)
        cur = torch.where(first, torch.minimum(frame_idx + 1, last), frame_idx)
        q1 = self._gather_from_bank(self._qpos_bank, prev, env_ids)
        q2 = self._gather_from_bank(self._qpos_bank, cur, env_ids)
        return derive(q1, q2)

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
        """Return a factory that samples qpos/qvel from a random clip bank entry.

        It reuses this source's bank when that is already on the requested device
        (a large bank is not uploaded twice).
        """
        import torch as _torch

        tracked_site_ids = np.asarray(self.tracked_site_ids, dtype=np.int64)

        def _init_fn(
            n_envs: int,
            device: _torch.device,
        ) -> tuple[_torch.Tensor | None, _torch.Tensor | None]:
            source = MultiClipTrajectorySource(
                clips=self.clips,
                tracked_site_ids=tracked_site_ids,
                ctrl_dt=float(self.ctrl_dt),
                bank_cfg=self.bank_cfg,
                joint_layout=self.joint_layout,
            )
            if self._device == device and self._clip_lengths is not None:
                source._share_bank(self)
            source._ensure_device(device, n_envs)
            return source.initial_qpos(), source.initial_qvel()

        return _init_fn

    def _share_bank(self, other: MultiClipTrajectorySource) -> None:
        """Use *other*'s uploaded bank (same clips and storage); state stays own."""
        for name in (
            "_device",
            "_site_bank",
            "_qpos_bank",
            "_qvel_bank",
            "_qvel_from_qpos",
            "_clip_starts",
            "_site_tensors",
            "_qpos_tensors",
            "_qvel_tensors",
            "_clip_lengths",
        ):
            setattr(self, name, getattr(other, name))


__all__ = [
    "ClipBankCfg",
    "ClipJointLayout",
    "ClipTrajectorySource",
    "MotionClipBank",
    "MultiClipTrajectorySource",
    "qvel_derivation_error",
]
