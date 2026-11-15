# Copyright 2026 The muscle-supermimic Authors. Modified for MyoSuite by the MyoSuite Authors.
#
# Licensed under the Apache License, Version 2.0. See NOTICE-muscle-supermimic.txt in
# this package for the notices of the code it includes.
"""Reset-frame sampling over 1 s bins of the clips, for one CPU env.

The same distribution as the mjlab twin's motion command (which runs it batched and
CUDA-graph safe in torch): ``"uniform"`` over the bins, or SuperMimic's return-based
mixture (``"return"``), with ``fail_reset`` of it moved to the bins before recent
failures (InstantTrack). The statistics are training state that persists across
episodes; :meth:`ResetSampler.clear` restarts them.

Ported from muscle-supermimic (github.com/kaibohex/muscle-supermimic, Apache-2.0) at
commit 8a6244b, ``src/muscle_supermimic/tasks/tracking/mdp/commands.py``.
"""

from __future__ import annotations

import numpy as np

from .config import ResetCfg
from .motion import MotionLib


class ResetSampler:
    """Bin statistics and reset draws of one env.

    Args:
        motion: The motion library.
        cfg: Reset settings.
        max_step_reward: Largest reward of one step (normalizes the bin returns).
    """

    def __init__(self, motion: MotionLib, cfg: ResetCfg, max_step_reward: float):
        self.motion, self.cfg = motion, cfg
        self.G_max = max_step_reward / (1 - cfg.gamma)
        nb = motion.num_bins
        prev = np.arange(nb) - 1
        same = (prev >= 0) & (motion.bin_clip[np.maximum(prev, 0)] == motion.bin_clip)
        self.bin_prev = np.where(same, prev, prev + 1)
        self.clear()

    def clear(self) -> None:
        """Forget every statistic."""
        nb = self.motion.num_bins
        self.bin_G = np.zeros(nb)
        self.bin_w_avg = np.full(nb, 1.0 / nb)
        self.bin_fail = np.zeros(nb)

    def finish_episode(
        self, start_bin: int, end_frame: int, steps: int, disc_ret: float,
        ret: float, terminated: bool,
    ) -> None:  # fmt: skip
        """Fold one finished episode into its start bin (truncated ones bootstrap)."""
        if steps <= 0:
            return
        g = self.cfg.gamma
        boot = 0.0 if terminated else (g**steps) * (ret / steps) / (1 - g)
        self.bin_G[start_bin] = 0.8 * self.bin_G[start_bin] + 0.2 * (disc_ret + boot)
        if self.cfg.fail_reset > 0:
            fb = self.motion.frame_bin[min(end_frame, self.motion.num_frames - 1)]
            self.bin_fail *= 0.999
            self.bin_fail[fb] = 0.9 * self.bin_fail[fb] + 0.1 * float(terminated)

    def probs(self) -> np.ndarray:
        """The reset distribution over bins (advances the running average)."""
        nb = self.motion.num_bins
        z = -25.0 * self.bin_G / self.G_max
        soft = np.exp(z - z.max())
        soft /= soft.sum()
        self.bin_w_avg = 0.999 * self.bin_w_avg + 0.001 * soft
        if self.cfg.mode == "uniform":
            p = np.full(nb, 1.0 / nb)
        else:
            p = 0.5 / nb + 0.4 * soft + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
        if self.cfg.fail_reset > 0:
            fail = np.bincount(self.bin_prev, weights=self.bin_fail, minlength=nb)
            if fail.sum() > 0:
                p = (
                    1 - self.cfg.fail_reset
                ) * p + self.cfg.fail_reset * fail / fail.sum()
        return p

    def sample(self, rng: np.random.Generator) -> tuple[int, int]:
        """``(frame, bin)`` of a reset: a bin from :meth:`probs`, then a uniform frame
        in it at least one step before the clip end."""
        m = self.motion
        p = self.probs()
        b = int(rng.choice(m.num_bins, p=p / p.sum()))
        start = int(m.bin_start[b])
        end = min(start + m.bin_frames, int(m.clip_end[m.bin_clip[b]]))
        return start + int(rng.random() * (end - start)), b
