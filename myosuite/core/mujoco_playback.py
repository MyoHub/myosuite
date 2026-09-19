# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Reusable MuJoCo playback loop primitives."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import mujoco
import mujoco.viewer

if TYPE_CHECKING:
    pass

StepCallback = Callable[[int, mujoco.MjModel, mujoco.MjData], None]
MetricCallback = Callable[[int, mujoco.MjModel, mujoco.MjData], float | None]
VizCallback = Callable[[int, mujoco.MjModel, mujoco.MjData, mujoco.MjvScene], None]


def run_passive_viewer_loop(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    n_steps: int,
    step_fn: StepCallback,
    metric_fn: MetricCallback | None = None,
    viz_fn: VizCallback | None = None,
) -> list[float]:
    """Run passive MuJoCo viewer loop with callbacks.

    Args:
        model: Compiled MuJoCo model.
        data: MuJoCo simulation data.
        n_steps: Maximum number of steps to run.
        step_fn: Called each step to advance the simulation.
        metric_fn: Optional scalar metric to collect each step.
        viz_fn: Optional callback receiving ``(step, model, data, user_scn)``
                to add custom geometry to the viewer's user scene each frame.

    Returns:
        List of non-None metric values collected from ``metric_fn``.
    """
    values: list[float] = []
    step_limit = max(1, int(n_steps))
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step_idx = 0
        while viewer.is_running() and step_idx < step_limit:
            step_fn(step_idx, model, data)
            if metric_fn is not None:
                value = metric_fn(step_idx, model, data)
                if value is not None:
                    values.append(float(value))
            if viz_fn is not None:
                with viewer.lock():
                    viz_fn(step_idx, model, data, viewer.user_scn)
            viewer.sync()
            time.sleep(max(0.0, float(model.opt.timestep)))
            step_idx += 1
    return values


__all__ = ["MetricCallback", "StepCallback", "VizCallback", "run_passive_viewer_loop"]
