# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The steered reference and the CPU tracker observation on synthetic clips (no AMASS data, no checkpoint)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from chasetag_teacher.reference import (
    ClipLibrary,
    Steer,
    SteeredReference,
    pose_features,
    rotz_batch,
    wrap,
)
from chasetag_teacher.tracker import CpuTracker

pytestmark = pytest.mark.tier2


@pytest.fixture(scope="module", name="task_env")
def task_env_fixture(clips: Path) -> Any:
    """CPU ``myoMimic-v0`` on the synthetic clips with the tracker's options (clip starts)."""
    supermimic_env = pytest.importorskip("supermimic.env")
    from myosuite import make_env
    from myosuite.core.config import EnvConfig
    from myosuite.envs.myo.tasks.mimic.tracking import MIMIC_ENV_ID

    cfg = supermimic_env.TaskEnvCfg(
        motion_file=str(clips / "clips.txt"), sampling_mode="start"
    )
    env = make_env(EnvConfig(MIMIC_ENV_ID, ctrl_dt=cfg.ctrl_dt,
                             task_kwargs={"preset": "supermimic", "overrides": supermimic_env.task_overrides(cfg)}))  # fmt: skip
    yield env
    env.close()


@pytest.fixture(scope="module", name="lib")
def lib_fixture(task_env: Any) -> ClipLibrary:
    from myosuite.envs.myo.tasks.mimic.tracking.models import root_offset

    u = task_env.unwrapped
    return ClipLibrary(u.motion, root_offset(u.model))


def test_tracker_observation_and_goal_match_the_task(
    task_env: Any, lib: ClipLibrary
) -> None:
    """At a clip start the CPU tracker's observation and the unwarped reference goal are the env's."""
    u = task_env.unwrapped
    tracker = CpuTracker(None, u.model, "")
    ref = SteeredReference(lib)
    for clip in range(len(u.motion.names)):
        obs, _ = task_env.reset(seed=clip, options={"clip": clip})
        tracker.reset()
        mine = tracker.observation(tracker.state(u.data))[0].numpy()
        n = len(mine)
        ref.f, ref.off, ref.omega = u.frame, 0.0, 0.0
        goal = ref.goal()[0].numpy()
        assert n + len(goal) == len(obs)
        np.testing.assert_allclose(mine, obs[:n], atol=2e-3)
        np.testing.assert_allclose(goal, obs[n:], atol=1e-4)


def test_steering_turns_the_reference(lib: ClipLibrary) -> None:
    """Asked for a heading 90 deg to the left, the yaw warp turns the reference toward it."""
    ref = SteeredReference(lib, mm_every=10**6)  # no clip switches: only the warp turns
    f0 = 0
    H = rotz_batch(-lib.yaw[f0 : f0 + 1])
    pose = pose_features(H, lib.rpos[f0 : f0 + 1], lib.v[f0 : f0 + 1])[0]
    start = float(lib.yaw[f0])
    steer = Steer(start + math.pi / 2, 1.0)
    ref.reset(pose, start, steer)
    for _ in range(100):  # 2 s
        ref.update(pose, steer)
        g = ref.goal()
        assert g.shape == (
            1,
            lib.rpos.shape[1] * 3
            + (lib.lookahead - 1) * (9 + lib.rpos.shape[1] * 3)
            + 1,
        )
        assert bool(g.isfinite().all())
        ref.step()
    assert abs(wrap(ref.ref_heading() - steer.heading)) < 0.3
