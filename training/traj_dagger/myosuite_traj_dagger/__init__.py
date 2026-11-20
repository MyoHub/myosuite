# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Multi-teacher trajectory-swap DAgger distillation for MyoSuite."""

from __future__ import annotations

from typing import Any

from myosuite_traj_dagger.buffer import (
    DaggerBatch,
    DaggerRolloutState,
    PersistentDaggerBuffer,
)
from myosuite_traj_dagger.dagger import (
    ChunkMetrics,
    DaggerConfig,
    LoadedTeacher,
    OfflineBC,
    OfflineBCConfig,
    RunningMeanStd,
    StudentActorCritic,
    SwapPair,
    SyncEnvPool,
    TeacherRegistry,
    TeacherSpec,
    TrajSwapDagger,
    compute_beta,
    export_onnx,
    export_torchscript,
    repeat_pair,
    update_student_on_buffer,
)

__all__ = [
    "ChunkMetrics",
    "DEFAULT_BENCHMARK_ENVS",
    "DaggerBatch",
    "DaggerConfig",
    "DaggerRolloutState",
    "EvalSnapshot",
    "LoadedTeacher",
    "OfflineBC",
    "OfflineBCConfig",
    "PPOConfig",
    "PersistentDaggerBuffer",
    "RunningMeanStd",
    "StudentActorCritic",
    "SwapPair",
    "SyncEnvPool",
    "TargetHalfSpaceWrapper",
    "TeacherRegistry",
    "TeacherSpec",
    "TrajSwapDagger",
    "compute_beta",
    "evaluate_student_policy",
    "export_onnx",
    "export_torchscript",
    "format_markdown_summary",
    "repeat_pair",
    "run_dagger_benchmark",
    "train_online_ppo",
    "update_student_on_buffer",
]

_BENCHMARK_EXPORTS = frozenset(
    {
        "DEFAULT_BENCHMARK_ENVS",
        "EvalSnapshot",
        "PPOConfig",
        "TargetHalfSpaceWrapper",
        "evaluate_student_policy",
        "format_markdown_summary",
        "run_dagger_benchmark",
        "train_online_ppo",
    }
)


def __getattr__(name: str) -> Any:
    if name in _BENCHMARK_EXPORTS:
        from myosuite_traj_dagger import benchmark

        return getattr(benchmark, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
