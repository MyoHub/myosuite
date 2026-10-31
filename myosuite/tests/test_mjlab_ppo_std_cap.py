# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""The mjlab PPO runner configs of the muscle tasks cap the policy's action std.

Without a cap, the entropy bonus drives the std of saturated muscles up by about
the learning rate per Adam step (MyoHub/myosuite#566).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

pytestmark = pytest.mark.tier2

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")
pytest.importorskip("rsl_rl")

from rsl_rl.utils import resolve_callable  # noqa: E402

import myosuite.envs.myo.backends.mjlab  # noqa: E402, F401
from mjlab.tasks.registry import list_tasks, load_rl_cfg  # noqa: E402
from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (  # noqa: E402
    default_mimic_clip_on_policy_runner_cfg,
)
from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (  # noqa: E402
    _elbow_ppo_runner_cfg,
)
from myosuite.envs.myo.backends.mjlab.tasks.rl import (  # noqa: E402
    MUSCLE_ENTROPY_COEF,
    MUSCLE_STD_RANGE,
    muscle_gaussian_distribution_cfg,
)

# Maximum learning rate of rsl_rl's adaptive schedule.
_MAX_ADAPTIVE_LR = 1e-2


def _registered_myo_tasks() -> list[str]:
    return sorted(t for t in list_tasks() if t.startswith("myo"))


def _assert_capped(rl_cfg: Any) -> None:
    dist = rl_cfg.actor.distribution_cfg
    assert dist["class_name"] == "GaussianDistribution"
    # "log" renames the checkpoint parameter; published checkpoints use "scalar".
    assert dist["std_type"] == "scalar"
    lo, hi = dist["std_range"]
    assert 0.0 < lo <= dist["init_std"] <= hi <= MUSCLE_STD_RANGE[1]
    assert rl_cfg.algorithm.entropy_coef <= MUSCLE_ENTROPY_COEF


def test_issue_tasks_are_registered() -> None:
    """The registry sweep below covers the Chase-Tag and table-tennis configs."""
    tasks = _registered_myo_tasks()
    assert "myoChallengeChaseTagFBP2-v0" in tasks
    assert "myoChallengeTableTennisP2-v0" in tasks


@pytest.mark.parametrize("task_id", _registered_myo_tasks())
def test_registered_runner_cfg_caps_std(task_id: str) -> None:
    _assert_capped(load_rl_cfg(task_id))


@pytest.mark.parametrize(
    "cfg_fn",
    [default_mimic_clip_on_policy_runner_cfg, _elbow_ppo_runner_cfg],
    ids=["mimic", "elbow"],
)
def test_unregistered_runner_cfg_caps_std(cfg_fn: Callable[[], Any]) -> None:
    """MuscleMimic tasks register only with a clip, so check the factories."""
    _assert_capped(cfg_fn())


def _build(distribution_cfg: dict[str, Any], dim: int = 8) -> Any:
    """Build the distribution from its config as rsl_rl's ``MLPModel`` does."""
    cfg = dict(distribution_cfg)
    return resolve_callable(cfg.pop("class_name"))(dim, **cfg)


def _std_after_entropy_only_updates(
    distribution_cfg: dict[str, Any], sign: float, steps: int = 3200
) -> tuple[float, float]:
    """(min, max) std after Adam steps on ``sign * entropy`` alone.

    This is the regime of saturated muscles: the return no longer depends on
    the std, so the entropy term is its only gradient.
    """
    dist = _build(distribution_cfg)
    opt = torch.optim.Adam(dist.parameters(), lr=_MAX_ADAPTIVE_LR)
    mean = torch.zeros(16, 8)
    for _ in range(steps):
        dist.update(mean)
        loss = -sign * MUSCLE_ENTROPY_COEF * dist.entropy.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    dist.update(mean)
    return dist.std.min().item(), dist.std.max().item()


def test_std_stays_in_range_under_entropy_only_updates() -> None:
    cfg = muscle_gaussian_distribution_cfg()
    lo, hi = cfg["std_range"]
    assert _std_after_entropy_only_updates(cfg, sign=1.0)[1] <= hi
    assert _std_after_entropy_only_updates(cfg, sign=-1.0)[0] >= lo


def test_std_grows_past_cap_without_std_range() -> None:
    """Control: rsl_rl's default range lets the same updates exceed the cap."""
    cfg = muscle_gaussian_distribution_cfg()
    del cfg["std_range"]
    assert _std_after_entropy_only_updates(cfg, sign=1.0)[1] > MUSCLE_STD_RANGE[1]


def test_uncapped_checkpoint_loads_into_capped_distribution() -> None:
    """Checkpoints trained before the cap load as is; resuming clamps their std."""
    old = _build({"class_name": "GaussianDistribution", "init_std": 1.0})
    old.std_param.data.fill_(14.0)
    new = _build(muscle_gaussian_distribution_cfg())
    new.load_state_dict(old.state_dict())
    new.update(torch.zeros(1, 8))
    torch.testing.assert_close(new.std_param, old.std_param)
    assert new.std.max().item() == MUSCLE_STD_RANGE[1]
