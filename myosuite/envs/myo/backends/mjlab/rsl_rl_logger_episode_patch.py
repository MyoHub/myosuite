# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Patch rsl_rl episode reward buffers when no SummaryWriter is active.

``rsl_rl.utils.logger.Logger.process_env_step`` only accumulates
``rewbuffer`` / ``lenbuffer`` inside ``if self.writer is not None``.
Headless runs with ``log_dir=None`` therefore never populate those
deques even though PPO still trains.  This module installs a
drop-in replacement that always updates episode statistics.

Call :func:`install_episode_reward_logging_patch` once before
``MjlabOnPolicyRunner.learn`` when using ``log_dir=None`` or any
setup where ``Logger.writer`` is unset.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

# Version beyond which the upstream bug is expected to be fixed.
_FIXED_VERSION = "2.3.0"


def _version_tuple(v: str) -> tuple[int, ...]:
    """Convert a version string to a comparable tuple of ints."""
    return tuple(int(x) for x in v.split(".")[:3])


_orig: Callable[..., Any] | None = None
_installed: bool = False


def install_episode_reward_logging_patch() -> None:
    """Monkey-patch ``Logger.process_env_step`` (idempotent).

    Skips the patch (with a warning) if ``rsl_rl.__version__`` is at or beyond
    ``_FIXED_VERSION``, indicating the upstream bug may have been resolved.
    Fails loudly if ``Logger.process_env_step`` is no longer present.
    """
    global _orig, _installed

    try:
        import rsl_rl

        rsl_rl_version = getattr(rsl_rl, "__version__", None)
        if rsl_rl_version is not None:
            if _version_tuple(rsl_rl_version) >= _version_tuple(_FIXED_VERSION):
                warnings.warn(
                    f"rsl_rl version {rsl_rl_version} is >= {_FIXED_VERSION}. "
                    "The episode-reward logging patch may no longer be needed. "
                    "Please verify and remove install_episode_reward_logging_patch() "
                    "if the upstream bug has been fixed.",
                    stacklevel=2,
                )
                return
    except ImportError:
        return

    from rsl_rl.utils.logger import Logger

    if not hasattr(Logger, "process_env_step"):
        raise AttributeError(
            "rsl_rl.utils.logger.Logger no longer has 'process_env_step'. "
            "The episode-reward logging patch is broken; please update or remove it."
        )

    if _installed:
        return

    _orig = Logger.process_env_step

    def process_env_step(
        self: Any,
        rewards: Any,
        dones: Any,
        extras: dict,
        intrinsic_rewards: Any | None = None,
    ) -> None:
        if "episode" in extras:
            self.ep_extras.append(extras["episode"])
        elif "log" in extras:
            self.ep_extras.append(extras["log"])

        if intrinsic_rewards is not None:
            self.cur_ereward_sum += rewards
            self.cur_ireward_sum += intrinsic_rewards
            self.cur_reward_sum += rewards + intrinsic_rewards
        else:
            self.cur_reward_sum += rewards
        self.cur_episode_length += 1

        new_ids = (dones > 0).nonzero(as_tuple=False)
        self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
        self.lenbuffer.extend(
            self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
        )
        self.cur_reward_sum[new_ids] = 0
        self.cur_episode_length[new_ids] = 0
        if intrinsic_rewards is not None:
            self.erewbuffer.extend(
                self.cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist()
            )
            self.irewbuffer.extend(
                self.cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist()
            )
            self.cur_ereward_sum[new_ids] = 0
            self.cur_ireward_sum[new_ids] = 0

    Logger.process_env_step = process_env_step  # type: ignore[method-assign]
    _installed = True


def uninstall_episode_reward_logging_patch() -> None:
    """Restore the original ``Logger.process_env_step``."""
    global _orig, _installed
    if not _installed or _orig is None:
        return
    from rsl_rl.utils.logger import Logger

    Logger.process_env_step = _orig  # type: ignore[method-assign]
    _orig = None
    _installed = False
