# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""mjlab step hot paths of MyoSuite tasks make no avoidable host syncs.

Every host sync (a device-to-host read or a pageable host-to-device copy) stalls
the CUDA stream once per env step. The counter of ``support/host_sync`` sees them
on a CPU-only torch too. The walk twin used to upload the body masses and target
vectors in every reward term (34 syncs per step, 41 on terrain), and TableTennis
P1/P2 the ball launch ranges on every reset (4 per reset step); the ChaseTag check
guards its already sync-free step (#456).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

pytest.importorskip("mjlab")
torch = pytest.importorskip("torch")

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

from myosuite.tests.support.host_sync import HostSyncCounter  # noqa: E402

pytestmark = pytest.mark.tier2


def _make_env(task_id: str, num_envs: int = 2) -> ManagerBasedRlEnv:
    import myosuite.envs.myo.backends.mjlab  # noqa: F401, PLC0415 (registers twins)

    cfg = load_env_cfg(task_id)
    cfg.scene.num_envs = num_envs
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset(seed=0)
    return env


def _step_syncs(env: ManagerBasedRlEnv, n_steps: int) -> list[tuple[Any, bool]]:
    """Host syncs of each of *n_steps* random steps, with whether it reset an env.

    Env 0 is put one step before its time limit, so a reset step is included.
    """
    gen = torch.Generator().manual_seed(0)
    dim = int(env.action_manager.total_action_dim)
    env.episode_length_buf[0] = env.max_episode_length - 1
    steps = []
    for _ in range(n_steps):
        action = torch.rand(env.num_envs, dim, generator=gen) * 2.0 - 1.0
        with HostSyncCounter(package_only=True) as syncs:
            env.step(action)
        steps.append((syncs, bool(env.reset_buf.any())))
    assert any(reset for _, reset in steps), "no reset step was measured"
    return steps


# ---------------------------------------------------------------------------
# Leg walk twin (rough terrain: also the knee-height done condition)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def walk_env() -> Iterator[ManagerBasedRlEnv]:
    env = _make_env("myoLegRoughTerrainWalk-v0")
    yield env
    env.close()


def test_walk_twin_steps_without_host_syncs(walk_env: ManagerBasedRlEnv) -> None:
    for syncs, _ in _step_syncs(walk_env, 4):
        assert syncs.total == 0, syncs.report()


def test_walk_reward_dict_is_evaluated_once_per_step(
    walk_env: ManagerBasedRlEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reward terms, the done term and the metric share one evaluation."""
    from myosuite.envs.myo.backends.mjlab.tasks.leg import walk_mdp

    calls: list[int] = []
    real = walk_mdp.walk_components

    def counting(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(walk_env.common_step_counter)
        return real(*args, **kwargs)

    monkeypatch.setattr(walk_mdp, "walk_components", counting)
    action = torch.zeros(walk_env.num_envs, walk_env.action_manager.total_action_dim)
    for _ in range(3):
        walk_env.step(action)
    assert len(calls) == 3 and len(set(calls)) == 3


def test_walk_terms_read_a_fresh_evaluation(
    walk_env: ManagerBasedRlEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every read of the shared reward dict equals an evaluation at that moment."""
    from myosuite.envs.myo.backends.mjlab.tasks.leg import walk_mdp

    done = walk_env.termination_manager.get_term_cfg(walk_mdp.FALLEN_TERM).func
    real = type(done).components
    reads: list[str] = []

    def checked(self: Any, env: Any) -> dict[str, Any]:
        out = real(self, env)
        fresh = walk_mdp.walk_components(
            env,
            self._walk,
            self._asset_cfg,
            self.model,
            self._target_rot,
            self._target_vel,
        )
        assert out.keys() == fresh.keys()
        for key, value in fresh.items():
            assert torch.equal(out[key], value), key
        reads.append("read")
        return out

    monkeypatch.setattr(type(done), "components", checked)
    _step_syncs(walk_env, 3)
    # Per step: the done term, one read per reward term, the success metric.
    n_rewards = len(walk_env.reward_manager.active_terms)
    assert len(reads) == 3 * (n_rewards + 2)


# ---------------------------------------------------------------------------
# Challenge twins: TableTennis (ball relaunch on reset), ChaseTag (a guard)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "task_id",
    [
        "myoChallengeTableTennisP2-v0",
        pytest.param("myoChallengeChaseTagFBP2-v0", marks=pytest.mark.slow),
    ],
)
def test_challenge_twin_steps_without_host_syncs(task_id: str) -> None:
    env = _make_env(task_id)
    try:
        for syncs, _ in _step_syncs(env, 3):
            assert syncs.total == 0, syncs.report()
    finally:
        env.close()
