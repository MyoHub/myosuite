# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Stochastic Stateful Policy Gradient PPO (S2PG-PPO) for MyoSuite."""

from __future__ import annotations

from typing import Any

from myosuite_s2pg.networks import (
    VALID_RNN_TYPES,
    ActorCriticFeedforward,
    ActorCriticS2PG,
    IndependentJointDist,
    RNNType,
    RunningMeanStd,
    StatefulS2PGExportWrapper,
    export_onnx_stateful,
    export_torchscript,
    stored_hidden_dim,
)
from myosuite_s2pg.s2pg_ppo import (
    CpuVectorEnv,
    FeedforwardPPOTrainer,
    FeedforwardRolloutBatch,
    S2PGConfig,
    S2PGPPOTrainer,
    S2PGRolloutBatch,
    VectorEnvProtocol,
    baseline_feedforward_ppo,
    compute_gae,
)

__all__ = [
    "ActorCriticFeedforward",
    "ActorCriticS2PG",
    "BenchmarkConfig",
    "CpuVectorEnv",
    "DEFAULT_BENCHMARK_ENVS",
    "FeedforwardPPOTrainer",
    "FeedforwardRolloutBatch",
    "IndependentJointDist",
    "RNNType",
    "RunningMeanStd",
    "S2PGConfig",
    "S2PGPPOTrainer",
    "S2PGRolloutBatch",
    "StatefulS2PGExportWrapper",
    "VALID_RNN_TYPES",
    "VectorEnvProtocol",
    "baseline_feedforward_ppo",
    "compute_gae",
    "evaluate_ppo_deterministic",
    "evaluate_s2pg_deterministic",
    "export_onnx_stateful",
    "export_torchscript",
    "format_markdown_summary",
    "run_s2pg_benchmark",
    "stored_hidden_dim",
]

_BENCHMARK_EXPORTS = frozenset(
    {
        "DEFAULT_BENCHMARK_ENVS",
        "BenchmarkConfig",
        "evaluate_ppo_deterministic",
        "evaluate_s2pg_deterministic",
        "format_markdown_summary",
        "run_s2pg_benchmark",
    }
)


def __getattr__(name: str) -> Any:
    if name in _BENCHMARK_EXPORTS:
        from myosuite_s2pg import benchmark

        return getattr(benchmark, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
