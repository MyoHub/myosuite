# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for experimental batched Newton SolverMuJoCo rollouts."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from myosuite import make_env
from myosuite.core.protocols import EnvAccessor
from myosuite.envs.muscle_stages import LowPassStage
from myosuite.terms.base_reward import pose_reward, reach_reward

pytestmark = pytest.mark.tier2

_REPO = Path(__file__).resolve().parents[2]


def _default_newton_device() -> str:
    if os.environ.get("MYOSUITE_REQUIRE_NEWTON") == "1":
        importlib.import_module("newton")
        wp = importlib.import_module("warp")
    else:
        pytest.importorskip("newton")
        wp = pytest.importorskip("warp")
    return "cuda:0" if wp.is_cuda_available() else "cpu"


def test_newton_batch_sim_lazy_export_skips_heavy_dependencies() -> None:
    script = (
        "import json, sys\n"
        "import myosuite.integrations.newton as pkg\n"
        "assert 'NewtonBatchSim' in dir(pkg)\n"
        "heavy = [m for m in ('newton', 'warp', 'mujoco_warp', 'torch') if m in sys.modules]\n"
        "print(json.dumps(heavy))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == []


@pytest.mark.parametrize(
    "env_id",
    ["myoElbowPose1D6MRandom-v0", "myoHandPoseRandom-v0"],
)
def test_newton_batch_rollout_matches_cpu_env(env_id: str) -> None:
    device = _default_newton_device()
    from myosuite.integrations.newton import NewtonBatchSim

    num_envs = 4
    sim = NewtonBatchSim(env_id, num_envs=num_envs, device=device, max_episode_steps=5)
    assert isinstance(sim.accessor, EnvAccessor)

    obs0, info0 = sim.reset(seed=7)
    assert obs0.shape == (num_envs, sim.nq + sim.nv + sim.na)
    assert np.all(info0["step_count"] == 0)

    cpu_envs = [make_env(env_id) for _ in range(num_envs)]
    try:
        init_qpos = np.zeros((num_envs, sim.nq), dtype=np.float32)
        init_qvel = np.zeros((num_envs, sim.nv), dtype=np.float32)
        for idx, env in enumerate(cpu_envs):
            env.reset(seed=100 + idx)
            init_qpos[idx] = env.unwrapped.data.qpos
            init_qvel[idx] = env.unwrapped.data.qvel

        sim.reset(qpos=init_qpos, qvel=init_qvel)
        np.testing.assert_allclose(sim.accessor.joint_pos(), init_qpos, atol=1e-5)

        rng = np.random.default_rng(42)
        for step_idx in range(4):
            actions = rng.uniform(-0.8, 0.8, size=(num_envs, sim.nu)).astype(np.float32)
            obs, reward, terminated, truncated, info = sim.step(actions)
            assert obs.shape == (num_envs, sim.nq + sim.nv + sim.na)
            assert reward.shape == (num_envs,)
            assert terminated.shape == (num_envs,)
            assert truncated.shape == (num_envs,)
            assert np.all(truncated == (step_idx + 1 >= 5))
            assert np.all(info["step_count"] == step_idx + 1)

            for idx, env in enumerate(cpu_envs):
                env.step(actions[idx])
                np.testing.assert_allclose(
                    sim.accessor.joint_pos()[idx],
                    env.unwrapped.data.qpos,
                    rtol=1e-4,
                    atol=1e-4,
                )
                np.testing.assert_allclose(
                    sim.accessor.muscle_act()[idx],
                    env.unwrapped.data.act,
                    rtol=1e-4,
                    atol=1e-4,
                )
                np.testing.assert_allclose(
                    sim.accessor.muscle_force()[idx],
                    env.unwrapped.data.actuator_force,
                    rtol=1e-3,
                    atol=0.05,
                )
                unchanged_sites = np.flatnonzero(
                    np.all(
                        np.isclose(
                            env.unwrapped.model.site_pos,
                            sim.reference_model.site_pos,
                        ),
                        axis=-1,
                    )
                )
                np.testing.assert_allclose(
                    sim.accessor.site_xpos(unchanged_sites)[idx],
                    env.unwrapped.data.site_xpos[unchanged_sites],
                    rtol=1e-4,
                    atol=1e-4,
                )
    finally:
        for env in cpu_envs:
            env.close()


def test_newton_masked_reset_and_pure_terms() -> None:
    device = _default_newton_device()
    from myosuite.integrations.newton import NewtonBatchSim

    stage = LowPassStage(alpha=0.4)
    target_angles = np.array([0.8], dtype=np.float32)
    sim = NewtonBatchSim(
        "myoElbowPose1D6MRandom-v0",
        num_envs=4,
        device=device,
        excitation_stages=(stage,),
        reward_fn=lambda acc: pose_reward(acc, {"target_angles": target_angles})[
            "dense"
        ],
    )
    sim.reset(seed=0)
    actions = np.full((4, sim.nu), 0.5, dtype=np.float32)
    sim.step(actions)
    _, rwd, _, _, _ = sim.step(actions)
    assert rwd.shape == (4,) and np.all(np.isfinite(rwd))

    qpos_before = sim.accessor.joint_pos().copy()
    time_before = sim.accessor.time().copy()

    sim.reset(env_ids=[1, 3])
    qpos_after = sim.accessor.joint_pos()
    time_after = sim.accessor.time()

    np.testing.assert_allclose(qpos_after[[0, 2]], qpos_before[[0, 2]], atol=1e-6)
    np.testing.assert_allclose(time_after[[0, 2]], time_before[[0, 2]], atol=1e-6)
    np.testing.assert_allclose(
        qpos_after[[1, 3]],
        np.broadcast_to(sim.reference_model.qpos0, (2, sim.nq)),
        atol=1e-5,
    )
    np.testing.assert_allclose(time_after[[1, 3]], 0.0, atol=1e-7)
    assert np.array_equal(sim.step_count, np.array([2, 0, 2, 0], dtype=np.int32))

    wrist_sid = sim.accessor.site_id("wrist")
    reach_out = reach_reward(
        sim.accessor,
        {"target_pos": np.zeros(3, dtype=np.float32), "tip_site_ids": [wrist_sid]},
    )
    assert reach_out["dense"].shape == (4,)
