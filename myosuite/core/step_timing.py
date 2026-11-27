# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Control-step timing of the CPU envs, shared by the mjlab and MJX twins (pure Python)."""

from __future__ import annotations


def first_step_after(time_s: float, timestep: float, frame_skip: int) -> int:
    """First control step at which the CPU ``data.time`` exceeds *time_s*.

    CPU envs gate some terms on ``data.time > t`` where ``data.time`` is a
    float64 sum of ``timestep`` increments; replaying that sum reproduces the
    exact boundary step instead of relying on float equality.

    Args:
        time_s: Threshold time in seconds.
        timestep: Physics timestep.
        frame_skip: Substeps per control step.

    Returns:
        Smallest control-step count ``k`` with accumulated time ``> time_s``.
    """
    t = 0.0
    step = 0
    while t <= time_s:
        for _ in range(frame_skip):
            t += timestep
        step += 1
    return step
