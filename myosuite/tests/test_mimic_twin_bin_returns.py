# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""``myoMimic-v0`` twin: an ended episode enters the reset statistics as on the CPU.

The twin's reset can run inside ``step()`` (``auto_reset``), before the motion
command's update of that step, or after it (the caller resets). Either way the bin
return holds every step's reward and the failure lands in the bin of the frame the
episode ended on.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from myosuite import make_env
from myosuite.core.config import EnvConfig
from myosuite.envs.myo.tasks.mimic.tracking import MIMIC_ENV_ID
from myosuite.envs.myo.tasks.mimic.tracking.models import compiled_body
from myosuite.envs.myo.tasks.mimic.tracking.motion import write_synthetic_clip

pytestmark = pytest.mark.tier2

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")

_NUM_ENVS = 2
_ACTION = np.full(354, 0.1, np.float32)
_METRIC = "Metrics/motion/task_reward"


@pytest.fixture(scope="module")
def envs(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Any, Any]]:
    """The twin and the CPU env on one synthetic clip, one reset bin per frame."""
    clip = str(tmp_path_factory.mktemp("clips") / "synthetic.npz")
    write_synthetic_clip(clip, compiled_body("myofullbody"), 1.5, 1.0, 0.3, 0)
    kwargs = {
        "overrides": {
            "motion.clips": clip,
            "reset.mode": "start",
            "motion.bin_s": 0.02,
        }
    }
    twin = make_env(
        EnvConfig(
            MIMIC_ENV_ID, backend="mjlab", num_envs=_NUM_ENVS, task_kwargs=kwargs
        ),
        device="cpu",
    )
    cpu = make_env(EnvConfig(MIMIC_ENV_ID, task_kwargs=kwargs)).unwrapped
    yield twin, cpu
    twin.close()


def _cpu_episode(cpu: Any, steps: int, fail: bool) -> tuple[Any, float]:
    """Play one CPU episode; return its sampler and the sum of its rewards."""
    cpu.reset(seed=0)
    cpu._sampler.clear()
    if fail:
        cpu.data.qpos[0] += 3.0
    total = sum(cpu.step(_ACTION)[1] for _ in range(steps))
    cpu.reset(seed=0)
    return cpu._sampler, total


def _twin_episode(twin: Any, steps: int, fail: bool, auto_reset: bool) -> Any:
    """Play the same episode in every env of the twin; return the motion command."""
    env = twin.unwrapped
    cmd = env.command_manager.get_term("motion")
    env.cfg.auto_reset = auto_reset
    env.cfg.episode_length_s = 1e9 if fail else (steps - 0.5) * env.step_dt
    twin.reset()
    cmd.bin_G.zero_()
    cmd.bin_fail.zero_()
    if fail:
        root = cmd.robot.data
        pose = torch.cat((root.root_link_pos_w, root.root_link_quat_w), -1)
        pose[:, 0] += 3.0
        cmd.robot.write_root_link_pose_to_sim(pose)
    action = torch.tensor(_ACTION).repeat(_NUM_ENVS, 1)
    for _ in range(steps):
        _, _, terminated, truncated, _ = twin.step(action)
    assert bool(terminated.all()) == fail and bool(truncated.all()) != fail
    if not auto_reset:
        env.reset(env_ids=torch.arange(_NUM_ENVS))
    return cmd


@pytest.mark.parametrize("auto_reset", [True, False])
@pytest.mark.parametrize(
    ("steps", "fail"),
    [(1, False), (3, False), (1, True)],
    ids=["1-step-time-out", "3-step-time-out", "1-step-failure"],
)
def test_bin_statistics_match_the_cpu_sampler(
    envs: tuple[Any, Any], steps: int, fail: bool, auto_reset: bool
) -> None:
    twin, cpu = envs
    sampler, total = _cpu_episode(cpu, steps, fail)
    cmd = _twin_episode(twin, steps, fail, auto_reset)

    start = int(cpu.motion.frame_bin[cpu.motion.clip_start[0]])
    assert sampler.bin_G[start] > 0
    np.testing.assert_allclose(cmd.bin_G.numpy(), sampler.bin_G, rtol=1e-3, atol=1e-7)
    np.testing.assert_allclose(
        cmd.bin_fail.numpy(), sampler.bin_fail, rtol=1e-3, atol=1e-7
    )
    if fail:
        assert sampler.bin_fail[start + 1] == pytest.approx(0.1)
    logged = twin.unwrapped.extras["log"][_METRIC]
    assert logged == pytest.approx(total, rel=1e-3)
