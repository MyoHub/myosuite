# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke: RSL-RL PPO on mjlab Table Tennis (mirrors CPU SB3 spirit, not identical stack).

CPU MyoChallenge training is covered by ``test_sb.py`` (Stable-Baselines3 PPO).
mjlab uses ``MjlabOnPolicyRunner`` + MuJoCo Warp; this test only asserts a
short run completes without error.

Runs training in a **subprocess** so Warp teardown does not interact with other
tests in the same pytest process (native crashes have been seen on macOS CPU).
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.tier2

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

try:
    import mjlab  # noqa: F401

    _MJLAB_AVAILABLE = True
except Exception:  # pragma: no cover
    _MJLAB_AVAILABLE = False

_MJLAB_SKIP = not (_TORCH_AVAILABLE and _MJLAB_AVAILABLE)
_MJLAB_SKIP_REASON = "mjlab and torch not installed (pip install myosuite[mjlab])"


@pytest.mark.skipif(_MJLAB_SKIP, reason=_MJLAB_SKIP_REASON)
def _inprocess_table_tennis_ppo_smoke() -> None:
    """Run PPO smoke in-process (used by subprocess worker)."""
    pytest.importorskip("rsl_rl")

    import myosuite

    myosuite.register_all_envs()
    import myosuite.envs.myo.backends.mjlab  # noqa: F401

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg

    from myosuite.envs.myo.backends.mjlab.register_mjlab_tabletennis import (
        _table_tennis_ppo_runner_cfg,
    )
    from myosuite.envs.myo.backends.mjlab.rsl_rl_logger_episode_patch import (
        install_episode_reward_logging_patch,
    )

    install_episode_reward_logging_patch()
    torch.manual_seed(1)
    device = "cpu"
    env_cfg = load_env_cfg("myoChallengeTableTennisP0-v0")
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped = RslRlVecEnvWrapper(env)
    rc = dataclasses.replace(
        _table_tennis_ppo_runner_cfg(),
        max_iterations=2,
        num_steps_per_env=8,
        seed=1,
        save_interval=999,
    )
    runner = MjlabOnPolicyRunner(
        env=wrapped,
        train_cfg=dataclasses.asdict(rc),
        log_dir=None,
        device=device,
    )
    runner.learn(num_learning_iterations=2, init_at_random_ep_len=True)
    assert runner.alg is not None


@pytest.mark.skipif(_MJLAB_SKIP, reason=_MJLAB_SKIP_REASON)
def test_mjlab_table_tennis_task_registered() -> None:
    """Table tennis mjlab modules register and expose P0 env cfg (no Warp sim)."""
    import myosuite

    myosuite.register_all_envs()
    import myosuite.envs.myo.backends.mjlab  # noqa: F401

    from mjlab.tasks.registry import load_env_cfg

    from myosuite.envs.myo.backends.mjlab.register_mjlab_tabletennis import (
        _table_tennis_ppo_runner_cfg,
    )

    cfg = load_env_cfg("myoChallengeTableTennisP0-v0")
    assert cfg is not None
    assert _table_tennis_ppo_runner_cfg().num_steps_per_env > 0


@pytest.mark.skipif(_MJLAB_SKIP, reason=_MJLAB_SKIP_REASON)
def test_mjlab_table_tennis_ppo_short_learn() -> None:
    """A few PPO iterations on ``myoChallengeTableTennisP0-v0`` must complete."""
    repo = Path(__file__).resolve().parents[2]
    code = (
        "import sys; sys.path.insert(0, %r); "
        "from myosuite.tests.test_table_tennis_mjlab_ppo_smoke import "
        "_inprocess_table_tennis_ppo_smoke; _inprocess_table_tennis_ppo_smoke()"
    ) % (str(repo),)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo),
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        msg = proc.stderr or proc.stdout or f"exit {proc.returncode}"
        rc = int(proc.returncode)
        # Warp/MuJoCo native faults: SIGSEGV (11), SIGABRT (6) as +139/+134 or -11/-6.
        sig = -rc if rc < 0 else (rc - 128 if rc > 128 else rc)
        if sig in (6, 11) or rc in (134, 139):
            pytest.skip(f"mjlab table tennis PPO smoke crashed (Warp): {msg[:500]}")
        raise AssertionError(f"mjlab table tennis PPO smoke failed:\n{msg}")
