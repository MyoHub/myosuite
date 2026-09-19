# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for rsl_rl episode-buffer monkey-patch (headless training)."""

from __future__ import annotations

from collections import deque

import pytest

pytestmark = pytest.mark.tier2


def test_episode_patch_updates_buffer_without_writer() -> None:
    rsl_rl = pytest.importorskip("rsl_rl")
    torch = pytest.importorskip("torch")

    from myosuite.envs.myo.backends.mjlab.rsl_rl_logger_episode_patch import (
        install_episode_reward_logging_patch,
        uninstall_episode_reward_logging_patch,
    )

    install_episode_reward_logging_patch()
    try:
        Logger = rsl_rl.utils.logger.Logger
        cfg = {
            "algorithm": {"rnd_cfg": None},
            "logger": "tensorboard",
        }
        log = Logger(
            log_dir=None,
            cfg=cfg,
            env_cfg=object(),
            num_envs=2,
            is_distributed=False,
            gpu_world_size=1,
            gpu_global_rank=0,
            device="cpu",
        )
        log.writer = None
        log.rewbuffer = deque(maxlen=10)
        log.lenbuffer = deque(maxlen=10)
        log.cur_reward_sum = torch.zeros(2)
        log.cur_episode_length = torch.zeros(2)
        log.ep_extras = []

        rewards = torch.tensor([0.5, 0.5])
        dones = torch.tensor([1, 0], dtype=torch.long)
        log.process_env_step(rewards, dones, {})
        assert len(log.rewbuffer) == 1
        assert log.rewbuffer[0] == pytest.approx(0.5)
    finally:
        uninstall_episode_reward_logging_patch()
