# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Composable kinematic and EMG trial contracts for MyoSuite motion datasets.

Supports kinematics-only trajectories (:class:`KinematicTrial`), standalone
surface/intramuscular **EMG-only** recordings (:class:`EMGTrial` with
``kinematics=None``, e.g. NinaPro or GRABMyo), and synchronized multimodal
mocap + EMG trials (:class:`EMGTrial` with ``kinematics`` attached).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from myosuite.motion.io import MotionClip


@dataclass(frozen=True)
class KinematicTrial:
    """Model-coordinate kinematic trial with explicit frame, timing, and validity metadata.

    Args:
        clip: Underlying :class:`~myosuite.motion.io.MotionClip` trajectory.
        trial_id: Unique identifier within a dataset or session.
        frequency_hz: Sampling frequency in Hz when ``clip.frequency_hz`` is unset;
            must agree with ``clip.frequency_hz`` when both are provided.
        world_frame: World-frame convention (default ``"z_up"``).
        quat_order: Quaternion component ordering (default ``"wxyz"``).
        valid_mask: Optional boolean mask of shape ``(T,)`` marking valid frames.
        subject_id: Optional subject identifier.
        license: Optional SPDX or upstream dataset license identifier.
        metadata: Additional provenance or diagnostic metadata.
    """

    clip: MotionClip
    trial_id: str = ""
    frequency_hz: float | None = None
    world_frame: str = "z_up"
    quat_order: str = "wxyz"
    valid_mask: np.ndarray | None = None
    subject_id: str | None = None
    license: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            self.frequency_hz is not None
            and self.clip.frequency_hz is not None
            and not np.isclose(
                self.frequency_hz, self.clip.frequency_hz, rtol=1e-6, atol=1e-9
            )
        ):
            raise ValueError(
                f"frequency_hz ({self.frequency_hz!r}) contradicts "
                f"clip.frequency_hz ({self.clip.frequency_hz!r})"
            )
        resolved_hz = (
            self.frequency_hz
            if self.frequency_hz is not None
            else self.clip.frequency_hz
        )
        if resolved_hz is not None and (
            not np.isfinite(resolved_hz) or resolved_hz <= 0.0
        ):
            raise ValueError(
                f"frequency_hz must be positive and finite, got {resolved_hz!r}"
            )
        object.__setattr__(self, "frequency_hz", resolved_hz)

        n_frames = self.clip.num_frames
        if self.valid_mask is not None:
            mask = np.asarray(self.valid_mask, dtype=bool)
            if mask.shape != (n_frames,):
                raise ValueError(
                    f"valid_mask shape {mask.shape} does not match n_frames ({n_frames},)"
                )
            object.__setattr__(self, "valid_mask", mask)

    @property
    def n_frames(self) -> int:
        """Number of kinematic frames ``T``."""
        return self.clip.num_frames

    @property
    def duration_s(self) -> float | None:
        """Trial duration in seconds when ``frequency_hz`` is known."""
        if self.frequency_hz is None or self.n_frames <= 1:
            return 0.0 if self.n_frames <= 1 and self.frequency_hz else None
        return float(self.n_frames - 1) / float(self.frequency_hz)


@dataclass(frozen=True)
class EMGTrial:
    """Composable EMG recording supporting both EMG-only and synchronized kinematic trials.

    Args:
        signals: Array of shape ``(T_emg, C)`` holding raw or envelope EMG samples.
        channels: Channel names of length ``C`` matching ``signals.shape[1]``.
        sampling_rate_hz: Positive EMG sampling frequency in Hz.
        trial_id: Unique identifier within a dataset or session.
        kinematics: Optional synchronized :class:`KinematicTrial`; ``None`` for
            standalone **EMG-only** trials (e.g. isometric or un-mocapped sEMG).
        units: Signal unit convention (default ``"unknown"``; e.g. ``"mV"``,
            ``"uV"``, ``"peak_normalized"``, ``"mvc_normalized"``).
        muscle_map: Optional mapping from EMG channel name to model actuator/muscle name.
        valid_mask: Optional boolean mask of shape ``(T_emg,)`` or ``(T_emg, C)``.
        time_offset_s: Start time of EMG sample 0 relative to ``kinematics`` frame 0.
        subject_id: Optional subject identifier.
        license: Optional SPDX or upstream dataset license identifier.
        metadata: Additional provenance or calibration metadata.
    """

    signals: np.ndarray
    channels: tuple[str, ...]
    sampling_rate_hz: float
    trial_id: str = ""
    kinematics: KinematicTrial | None = None
    units: str = "unknown"
    muscle_map: dict[str, str] | None = None
    valid_mask: np.ndarray | None = None
    time_offset_s: float = 0.0
    subject_id: str | None = None
    license: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arr = np.asarray(self.signals, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError(
                f"signals must have non-empty 2D shape (T, C), got {arr.shape}"
            )
        ch_tuple = tuple(self.channels)
        if len(ch_tuple) != arr.shape[1]:
            raise ValueError(
                f"len(channels)={len(ch_tuple)} does not match signals.shape[1]={arr.shape[1]}"
            )
        if len(set(ch_tuple)) != len(ch_tuple) or not all(ch_tuple):
            raise ValueError("EMG channel names must be non-empty and unique.")
        if not np.isfinite(self.sampling_rate_hz) or self.sampling_rate_hz <= 0.0:
            raise ValueError(
                f"sampling_rate_hz must be positive and finite, got {self.sampling_rate_hz!r}"
            )
        if not np.isfinite(self.time_offset_s):
            raise ValueError(
                f"time_offset_s must be finite, got {self.time_offset_s!r}"
            )

        object.__setattr__(self, "signals", arr)
        object.__setattr__(self, "channels", ch_tuple)

        if self.valid_mask is not None:
            mask = np.asarray(self.valid_mask, dtype=bool)
            if mask.shape not in {(arr.shape[0],), arr.shape}:
                raise ValueError(
                    f"valid_mask shape {mask.shape} must be ({arr.shape[0]},) or {arr.shape}"
                )
            object.__setattr__(self, "valid_mask", mask)

    @property
    def is_emg_only(self) -> bool:
        """True when no kinematic trajectory is attached to this trial."""
        return self.kinematics is None

    @property
    def n_samples(self) -> int:
        """Number of EMG time samples ``T_emg``."""
        return int(self.signals.shape[0])

    @property
    def n_channels(self) -> int:
        """Number of EMG channels ``C``."""
        return int(self.signals.shape[1])

    @property
    def duration_s(self) -> float:
        """Duration of the EMG recording in seconds."""
        if self.n_samples <= 1:
            return 0.0
        return float(self.n_samples - 1) / float(self.sampling_rate_hz)

    @property
    def timestamps_s(self) -> np.ndarray:
        """Sample timestamps in seconds relative to ``kinematics`` frame 0."""
        return self.time_offset_s + np.arange(self.n_samples, dtype=np.float64) / float(
            self.sampling_rate_hz
        )

    def select_channels(self, names: Sequence[str]) -> EMGTrial:
        """Return a new ``EMGTrial`` restricted to *names* in the requested order."""
        index = {ch: idx for idx, ch in enumerate(self.channels)}
        missing = [ch for ch in names if ch not in index]
        if missing:
            raise KeyError(f"Unknown EMG channel(s): {missing}")
        cols = [index[ch] for ch in names]
        sub_mask = (
            self.valid_mask[:, cols]
            if self.valid_mask is not None and self.valid_mask.ndim == 2
            else self.valid_mask
        )
        sub_map = (
            {ch: self.muscle_map[ch] for ch in names if ch in self.muscle_map}
            if self.muscle_map is not None
            else None
        )
        return EMGTrial(
            signals=self.signals[:, cols],
            channels=tuple(names),
            sampling_rate_hz=self.sampling_rate_hz,
            trial_id=self.trial_id,
            kinematics=self.kinematics,
            units=self.units,
            muscle_map=sub_map,
            valid_mask=sub_mask,
            time_offset_s=self.time_offset_s,
            subject_id=self.subject_id,
            license=self.license,
            metadata=dict(self.metadata),
        )


__all__ = ["EMGTrial", "KinematicTrial"]
