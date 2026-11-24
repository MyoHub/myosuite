# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Judged single events and combined events (pentathlon) on the public myofullbody."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from myosuite.envs.myo.athletic.combined import COMBINED_EVENTS, make_combined
from myosuite.envs.myo.athletic.events import EVENTS, make_event

pytestmark = pytest.mark.tier1


@pytest.mark.parametrize(
    ("event", "performance", "points"),
    [
        # World Athletics combined-events tables (men)
        ("100m", 10.00, 1096),
        ("long_jump", 8.00, 1061),
        ("110m_hurdles", 13.80, 1000),
        ("high_jump", 2.20, 992),
        ("100m", 18.5, 0),  # slower than B: no points
        ("long_jump", None, 0),  # no valid mark
    ],
)
def test_world_athletics_points(
    event: str, performance: float | None, points: int
) -> None:
    assert EVENTS[event].score(performance) == points


def test_pentathlon_events_and_total() -> None:
    pent = COMBINED_EVENTS["tf_pentathlon"]
    assert pent.events == ("100m", "long_jump", "high_jump", "110m_hurdles", "1500m")
    s = pent.score(
        {"100m": 10.0, "long_jump": 8.0, "high_jump": 2.2, "110m_hurdles": 13.8}
    )
    assert s["points"]["1500m"] == 0
    assert s["total"] == 1096 + 1061 + 992 + 1000


def test_decathlon_needs_hands_and_implements() -> None:
    assert COMBINED_EVENTS["decathlon"].missing(frozenset({"lanes"})) == [
        "shot_put",
        "discus",
        "pole_vault",
        "javelin",
    ]
    with pytest.raises(NotImplementedError, match="shot_put"):
        make_combined("decathlon", model="myofullbody")


def test_hurdles_are_colliders_at_the_official_marks() -> None:
    env = make_event("110m_hurdles", model="myofullbody")
    m = env.base.model
    names = [
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)
    ]
    ids = [g for g, n in enumerate(names) if n.startswith("event_hurdle")]
    assert len(ids) == 10
    fwd = np.sort(-m.geom_pos[ids, 0])  # myofullbody runs towards -x
    np.testing.assert_allclose(fwd, 13.72 + 9.14 * np.arange(10), atol=1e-9)
    top = m.geom_pos[ids, 2] + m.geom_size[ids, 2]
    np.testing.assert_allclose(top, 1.067, atol=1e-9)


def test_control_step_spans_ctrl_dt() -> None:
    env = make_event("100m", model="myofullbody")
    u = env.base
    assert u.frame_skip * u.model.opt.timestep == pytest.approx(u._ctrl_dt, abs=1e-12)


def test_track_finish_time_is_interpolated() -> None:
    env = make_event("100m", model="myofullbody")
    env.reset(seed=0)
    d = env.base.data
    start = env._record["start_fwd"]
    # carry the athlete to 100.5 m during the first step: finish inside that step
    d.qpos[0] = -(start + 100.5)
    _, reward, terminated, _, info = env.step(np.zeros(env.action_space.shape))
    ev = info["event"]
    assert terminated and ev["outcome"] == "finished"
    assert 0.0 < ev["performance"] <= env._dt
    assert reward > 0.0


def test_jump_judge_sees_the_plant() -> None:
    env = make_event("long_jump", model="myofullbody")
    env.reset(seed=0)
    feet = np.bincount(env._judge.foot, minlength=3)
    assert feet[1] > 0 and feet[2] > 0 and feet[1] == feet[2]
    _, _, _, _, info = env.step(np.zeros(env.action_space.shape))
    assert info["event"]["event"] == "long_jump"


def test_pentathlon_sequence_runs_every_event() -> None:
    env = make_combined("tf_pentathlon", model="myofullbody")
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    seen = [info["event_index"]]
    zero = np.zeros(env.action_space.shape)
    for _ in range(2000):  # a limp athlete falls in every event
        obs, _, terminated, truncated, info = env.step(zero)
        seen.append(info["event_index"])
        if terminated or truncated:
            break
    assert terminated or truncated
    assert sorted(set(seen)) == [0, 1, 2, 3, 4]
    summary = info["combined"]
    assert set(summary["points"]) == set(COMBINED_EVENTS["tf_pentathlon"].events)
    assert summary["total"] == 0


def test_pentathlon_random_mode_draws_events() -> None:
    env = make_combined("tf_pentathlon", model="myofullbody", mode="random")
    drawn = {env.reset(seed=s)[1]["event_index"] for s in range(12)}
    assert len(drawn) > 1


def test_event_ids_follow_the_athletic_scheme() -> None:
    from myosuite.envs.myo.athletic.catalog import PLANTS
    from myosuite.envs.myo.athletic.events import event_env_id

    fb, fbs = PLANTS["myofullbody"], PLANTS["myofullbody_simple"]
    assert event_env_id(fb, "100m") == "myoFullbody100m-v0"
    assert event_env_id(fb, "tf_pentathlon") == "myoFullbodyTFPentathlon-v0"
    assert (
        event_env_id(fbs, "tf_pentathlon", "elastic")
        == "myoFullbodySimpleTFPentathlonElastic-v0"
    )


def test_elastic_tendons_are_refused_until_applied() -> None:
    import gymnasium as gym

    import myosuite.envs.myo.athletic.register_cpu  # noqa: F401

    with pytest.raises(NotImplementedError, match="elastic"):
        make_event("100m", model="myofullbody", tendons="elastic")
    with pytest.raises(NotImplementedError, match="elastic"):
        make_combined("tf_pentathlon", model="myofullbody", tendons="elastic")
    assert "myoFullbodyTFPentathlon-v0" in gym.registry
    assert "myoFullbodyTFPentathlonElastic-v0" not in gym.registry
