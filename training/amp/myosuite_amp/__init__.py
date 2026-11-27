# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Adversarial Motion Priors (AMP) and GAIL for MyoSuite tasks."""

from __future__ import annotations

from typing import Any

from myosuite_amp.discriminator import (
    DiscriminatorLossStats,
    DiscriminatorMode,
    MotionDiscriminator,
    RunningMeanStd,
)
from myosuite_amp.ppo_amp import (
    ActorCritic,
    AMPConfig,
    AMPTrainer,
    BCConfig,
    BehaviorCloningTrainer,
    CpuVecEnv,
    EvalStats,
    ExpertDataset,
    TrainStepStats,
    collect_expert_dataset,
    evaluate_policy_cpu,
    export_policy_onnx,
    export_policy_torchscript,
    train_bc,
)

__all__ = [
    "AMPConfig",
    "AMPTrainer",
    "ActorCritic",
    "BCConfig",
    "BehaviorCloningTrainer",
    "CpuVecEnv",
    "DiscriminatorLossStats",
    "DiscriminatorMode",
    "EvalStats",
    "ExpertDataset",
    "MotionDiscriminator",
    "RunningMeanStd",
    "TrainStepStats",
    "collect_expert_dataset",
    "evaluate_policy_cpu",
    "export_policy_onnx",
    "export_policy_torchscript",
    "format_benchmark_markdown",
    "run_amp_benchmark",
    "train_bc",
]


def __getattr__(name: str) -> Any:
    if name in ("format_benchmark_markdown", "run_amp_benchmark"):
        from myosuite_amp import benchmark

        return getattr(benchmark, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
