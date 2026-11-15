# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Success indicators: one convention on the base env, used by every task that shows one (#586)."""

from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from myosuite import make_env
from myosuite.envs.gymnasium_env import MyoGymnasiumEnv

pytestmark = pytest.mark.tier1

SITE = """<mujoco><worldbody><body><geom size=".1"/>
  <site name="indicator" size=".05" rgba=".2 1.7 .3 .1"/>
</body></worldbody></mujoco>"""

# Every registered task family that colours a success indicator.
TASKS = (
    "myoChallengeRelocateP1-v0",
    "myoChallengeDieReorientDemo-v0",
    "myoHandPenTwirlFixed-v0",
    "myoChallengeChaseTagP1-v0",
    "myoHandReorient8-v0",
)
GREEN, RED = (0.0, 2.0), (2.0, 0.0)


def test_indicator_is_green_when_solved_and_red_otherwise() -> None:
    model = mujoco.MjModel.from_xml_string(SITE)
    env = SimpleNamespace(model=model)  # the method only needs the model
    size = model.site_size[0].copy()
    MyoGymnasiumEnv._set_success_indicator(env, 0, True)
    np.testing.assert_allclose(model.site_rgba[0], [*GREEN, 0.3, 0.1])
    MyoGymnasiumEnv._set_success_indicator(env, 0, False)
    np.testing.assert_allclose(model.site_rgba[0], [*RED, 0.3, 0.1])  # blue, alpha kept
    MyoGymnasiumEnv._set_success_indicator(env, -1, True)  # no such site: no change
    np.testing.assert_allclose(model.site_rgba[0], [*RED, 0.3, 0.1])
    np.testing.assert_allclose(model.site_size[0], size)


@pytest.mark.parametrize("env_id", TASKS)
def test_tasks_show_success_through_the_base_env(
    env_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = make_env(env_id)
    try:
        env.reset(seed=0)
        base = env.unwrapped
        calls: list[tuple[int, bool]] = []
        show = base._set_success_indicator

        def spy(site_id: int, solved: bool) -> None:
            calls.append((site_id, bool(solved)))
            show(site_id, solved)

        monkeypatch.setattr(base, "_set_success_indicator", spy)
        *_, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        sid = base.success_indicator_sid
        solved = bool(info["rwd_dict"]["solved"])
        assert calls and calls[-1] == (sid, solved)
        assert tuple(base.model.site_rgba[sid, :2]) == (GREEN if solved else RED)
        if env_id.startswith("myoChallengeChaseTag"):  # shown only on a tag
            assert base.model.site_rgba[sid, 3] == (0.2 if solved else 0.0)
        if env_id.startswith("myoChallengeRelocate"):  # grows when solved
            assert base.model.site_size[sid, 0] == (0.25 if solved else 0.1)
    finally:
        env.close()


def test_sar_reorient_indicator_stays_green_until_reset() -> None:
    env = make_env("myoHandReorient8-v0")
    try:
        env.reset(seed=0)
        base = env.unwrapped
        sid = base.success_indicator_sid
        aligned = {
            "obj_err_pos": np.zeros(3),
            "obj_rot": np.array([1.0, 0.0, 0.0]),
            "obj_des_rot": np.array([1.0, 0.0, 0.0]),
        }
        turned = {**aligned, "obj_des_rot": np.array([0.0, 1.0, 0.0])}
        rgb = lambda: tuple(base.model.site_rgba[sid, :2])  # noqa: E731
        assert not base.get_reward_dict(turned)["solved"] and rgb() == RED
        assert base.get_reward_dict(aligned)["solved"] and rgb() == GREEN
        assert not base.get_reward_dict(turned)["solved"] and rgb() == GREEN
        env.reset(seed=1)
        assert rgb() == RED
    finally:
        env.close()
