# Copyright (c) MyoSuite Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the LambdaHold training package."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from myosuite_lambda_hold.benchmark_sb3 import run_ppo_comparison

pytestmark = pytest.mark.tier2


def test_ppo_comparison_smoke() -> None:
    pytest.importorskip("stable_baselines3")
    summary = run_ppo_comparison(
        env_ids=("myoElbowPose1D6MRandom-v0",),
        seeds=(0,),
        total_steps=512,
        eval_interval=512,
        n_envs=2,
        n_eval_episodes=2,
    )
    assert len(summary.arms) == 3
    assert [a.method for a in summary.arms] == [
        "PPO",
        "LambdaHold+PPO",
        "CurriculumLambdaHold+PPO",
    ]
    assert all(len(a.curves) == 2 for a in summary.arms)
