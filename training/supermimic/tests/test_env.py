# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The training env is MyoSuite's task through ``make_env`` with the recipe's options."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")

pytestmark = pytest.mark.tier2


def test_recipe_env_through_make_env(clips: Path) -> None:
    from supermimic.env import TaskEnvCfg, make_task_env, settings_cfg

    env = make_task_env(
        TaskEnvCfg(motion_file=str(clips / "clips.txt"), num_envs=3),
        device="cpu",
        seed=1,
    )
    u = env.unwrapped
    cmd = u.command_manager.get_term("motion")
    reset = cmd.task.reset
    assert (
        u.cfg.decimation,
        cmd.goal_stride,
        reset.mode,
        reset.bin_fix,
        reset.fail_reset,
    ) == (10, 10, "return", True, 0.3)
    assert not u.cfg.auto_reset and u.cfg.seed == 1
    assert "muscle_length" not in u.observation_manager.active_terms["policy"]
    obs = env.get_observations()
    assert set(obs.keys()) >= {"policy", "goal"}
    env.step(torch.full((3, u.action_manager.total_action_dim), 0.1))
    s = settings_cfg(
        {"decimation": 5, "stride": 20, "lookahead": 5, "muscle_obs": True},
        motion_file="x",
    )
    assert (s.ctrl_dt, s.goal_stride, s.muscle_obs) == (0.01, 20, True)


def test_the_benchmark_held_out_list_ships_with_the_project(
    monkeypatch, tmp_path
) -> None:
    """The 108 held-out clips live in the project, not in myosuite; training uses the task's default list."""
    from supermimic.env import TaskEnvCfg, held_out_clips

    from myosuite.envs.myo.tasks.mimic import tracking as tr

    monkeypatch.setenv(tr.KIT_ROOT_ENV, str(tmp_path))
    path, root = held_out_clips()
    clips = [
        ln
        for ln in open(path).read().splitlines()
        if ln.strip() and not ln.startswith("#")
    ]  # noqa: SIM115
    assert len(clips) == 108 and root == str(tmp_path)
    train = tr.resolve_clips("kit_train")[0]
    assert not set(clips) & set(open(train).read().splitlines())  # noqa: SIM115
    assert TaskEnvCfg().motion_file == "kit_train"
