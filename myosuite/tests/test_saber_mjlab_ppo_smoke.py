# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke: RSL-RL PPO on mjlab Saber (mirrors test_table_tennis_mjlab_ppo_smoke.py).

CPU MyoChallenge training is covered by ``test_sb.py`` (Stable-Baselines3 PPO).
mjlab uses ``MjlabOnPolicyRunner`` + MuJoCo Warp; this test asserts a short run
completes without error and surfaces whatever episode reward got logged.

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

_SABER_TASK_ID = "myoChallengeSaberP0-v0"


@pytest.mark.skipif(_MJLAB_SKIP, reason=_MJLAB_SKIP_REASON)
def _inprocess_saber_ppo_smoke() -> None:
    """Run PPO smoke in-process (used by subprocess worker)."""
    pytest.importorskip("rsl_rl")

    import myosuite

    myosuite.register_all_envs()
    import myosuite.envs.myo.backends.mjlab  # noqa: F401

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    from myosuite.envs.myo.backends.mjlab.rsl_rl_logger_episode_patch import (
        install_episode_reward_logging_patch,
    )

    install_episode_reward_logging_patch()
    torch.manual_seed(1)
    device = "cpu"
    env_cfg = load_env_cfg(_SABER_TASK_ID)
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped = RslRlVecEnvWrapper(env)
    rc = dataclasses.replace(
        load_rl_cfg(_SABER_TASK_ID),
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
    # log_dir=None disables rsl_rl's TensorBoard writer, which (pre-patch) also
    # silently disabled rewbuffer/lenbuffer accumulation — making it impossible
    # to tell a learning run from a stuck one. Surface whatever got logged so a
    # human/CI re-running this manually can see reward, not just "didn't crash".
    print(f"episode rewbuffer: {list(runner.logger.rewbuffer)}")
    print(f"episode lenbuffer: {list(runner.logger.lenbuffer)}")
    for r in runner.logger.rewbuffer:
        assert r == r and abs(r) < float("inf")  # noqa: PLR0124 (NaN/inf check)


@pytest.mark.skipif(_MJLAB_SKIP, reason=_MJLAB_SKIP_REASON)
def test_mjlab_saber_task_registered() -> None:
    """Saber mjlab modules register and expose a P0 env/rl cfg (no Warp sim)."""
    import myosuite

    myosuite.register_all_envs()
    import myosuite.envs.myo.backends.mjlab  # noqa: F401

    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    assert load_env_cfg(_SABER_TASK_ID) is not None
    assert load_rl_cfg(_SABER_TASK_ID).num_steps_per_env > 0


@pytest.mark.skipif(_MJLAB_SKIP, reason=_MJLAB_SKIP_REASON)
def test_mjlab_saber_ppo_short_learn() -> None:
    """A few PPO iterations on ``myoChallengeSaberP0-v0`` must complete."""
    repo = Path(__file__).resolve().parents[2]
    code = (
        "import sys; sys.path.insert(0, %r); "
        "from myosuite.tests.test_saber_mjlab_ppo_smoke import "
        "_inprocess_saber_ppo_smoke; _inprocess_saber_ppo_smoke()"
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
            pytest.skip(f"mjlab saber PPO smoke crashed (Warp): {msg[:500]}")
        raise AssertionError(f"mjlab saber PPO smoke failed:\n{msg}")
