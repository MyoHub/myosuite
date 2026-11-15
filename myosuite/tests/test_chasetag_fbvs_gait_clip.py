# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The scored 1v1 game loads its gait clip only for a nonzero gait/steering shaping weight."""

from __future__ import annotations

import numpy as np
import pytest

from myosuite import make_env
from myosuite.tests.support.hf_clip import resolve_walk_clip

pytestmark = pytest.mark.tier1

ENV_ID = "myoChallengeChaseTagFBVs-v0"


def _no_shaping_env():
    from myosuite.envs.myo.tasks.challenge.chase_tag_vs.chase_tag_vs_task_config import (
        ChaseTagVsFullbodySteeredTaskConfig,
    )

    cfg = ChaseTagVsFullbodySteeredTaskConfig(
        w_gait_jpos=0.0, w_gait_jvel=0.0, w_steering=0.0
    )
    return make_env(ENV_ID, task_config=cfg)


def _play(env, steps: int = 5) -> list[tuple]:
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=3)
    out = [(obs["agent_0"].copy(), obs["agent_1"].copy(), None, None)]
    for _ in range(steps):
        acts = {
            a: rng.uniform(0.0, 1.0, 354).astype(np.float32)
            for a in ("agent_0", "agent_1")
        }
        obs, rwd, _, _, info = env.step(acts)
        out.append((obs["agent_0"].copy(), obs["agent_1"].copy(), info["score"], rwd))
    return out


def test_zero_shaping_weights_never_download(monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    def refuse(*args, **kwargs):
        raise AssertionError("gait clip download attempted")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", refuse)
    env = _no_shaping_env()
    try:
        _play(env)
    finally:
        env.close()


@pytest.mark.skipif(
    resolve_walk_clip() is None,
    reason="gait clip (gated Hugging Face dataset) unavailable",
)
def test_default_weights_keep_obs_and_score() -> None:
    from myosuite.envs.myo.tasks.challenge.chase_tag_vs.chase_tag_vs_task_config import (
        ChaseTagVsFullbodySteeredTaskConfig,
    )

    default = ChaseTagVsFullbodySteeredTaskConfig()
    assert (default.w_gait_jpos, default.w_gait_jvel, default.w_steering) == (
        0.15,
        0.05,
        0.6,
    )
    a, b = make_env(ENV_ID), _no_shaping_env()
    try:
        ra, rb = _play(a), _play(b)
    finally:
        a.close()
        b.close()
    shaped = False
    for (oa0, oa1, sa, wa), (ob0, ob1, sb, wb) in zip(ra, rb):
        np.testing.assert_array_equal(oa0, ob0)
        np.testing.assert_array_equal(oa1, ob1)
        assert sa == sb
        if wa is not None:
            shaped |= any(wa[k] != wb[k] for k in wa)
    assert shaped, "the default weights should still add the shaping reward"
