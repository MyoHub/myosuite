# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Buffered colored-noise process (the scripted challenge opponents' velocities)."""

from __future__ import annotations

from collections.abc import Iterable

import colorednoise
import numpy as np


class ColoredNoiseProcess:
    """Infinite colored-noise process, sampled a few time steps at a time.

    Same semantics as ``ColoredNoiseProcess`` of pink-noise-rl 2.0.1 (Eberhard
    et al., "Pink Noise Is All You Need", ICLR 2023; MIT license): a buffer holds
    one series of shape ``size`` whose last axis is time, and a new series is
    drawn whenever it is used up. The series come from
    :func:`colorednoise.powerlaw_psd_gaussian` (F. Patzelt, colorednoise; MIT
    license), which draws exactly as pink's bundled copy, so seeded streams are
    bit-identical. Importing ``pink`` itself loads stable-baselines3 and torch
    whenever they are installed.

    Args:
        beta: Exponent of the power-law spectrum (1: pink, 2: Brownian noise).
        size: Shape of a buffered series; the last axis is time.
        scale: Factor applied to the samples, read at sampling time.
        max_period: Maximum correlation length (1 / low-frequency cutoff);
            ``None`` uses ``size[-1]``.
        rng: Generator for the draws; ``None`` seeds a fresh one per series.
    """

    def __init__(
        self,
        beta: float,
        size: int | Iterable[int],
        scale: float = 1,
        max_period: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.beta = beta
        self.minimum_frequency = 0 if max_period is None else 1 / max_period
        self.scale = scale
        self.rng = rng
        self.size = list(size) if np.iterable(size) else [size]
        self.time_steps = self.size[-1]
        self.reset()

    def reset(self) -> None:
        """Draw a new series into the buffer and rewind to its start."""
        self.buffer = colorednoise.powerlaw_psd_gaussian(
            self.beta, self.size, fmin=self.minimum_frequency, random_state=self.rng
        )
        self.idx = 0

    def sample(self, T: int = 1) -> np.ndarray:
        """Return the next ``T`` time steps, drawing new series as needed.

        Args:
            T: Number of time steps.

        Returns:
            Array of shape ``(*size[:-1], T)``, or ``size[:-1]`` when ``T == 1``.
        """
        n, chunks = 0, []
        while n < T:
            if self.idx >= self.time_steps:
                self.reset()
            m = min(T - n, self.time_steps - self.idx)
            chunks.append(self.buffer[..., self.idx : self.idx + m])
            n += m
            self.idx += m
        out = self.scale * np.concatenate(chunks, axis=-1)
        return out if n > 1 else out[..., 0]
