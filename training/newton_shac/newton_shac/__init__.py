# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Differentiable Short-Horizon Actor-Critic (SHAC) for MyoSuite musculoskeletal tasks."""

from __future__ import annotations

from newton_shac.diff_env import DifferentiablePoseEnv
from newton_shac.networks import ActorCritic, RunningMeanStd, export_shac_onnx
from newton_shac.shac import (
    ShacConfig,
    TrainRunMetrics,
    evaluate_on_cpu_env,
    train_shac,
)

__all__ = [
    "ActorCritic",
    "DifferentiablePoseEnv",
    "RunningMeanStd",
    "ShacConfig",
    "TrainRunMetrics",
    "evaluate_on_cpu_env",
    "export_shac_onnx",
    "train_shac",
]
