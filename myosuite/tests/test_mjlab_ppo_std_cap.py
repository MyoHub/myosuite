# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""The mjlab PPO runner configs of the muscle and motorFinger tasks bound the action std.

Once a control saturates, its return no longer depends on its std. The entropy bonus
then raises the std by up to the learning rate per Adam step (less once the noise of
PPO's surrogate gradient is included), without bound (MyoHub/myosuite#566). The
configs project the std onto ``std_range`` with ``ProjectedGaussianDistribution``,
which keeps it trainable at the bounds, and ``load_rslrl_policy`` samples with the
same bounded std.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.tier2

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")
pytest.importorskip("rsl_rl")

from rsl_rl.models import MLPModel  # noqa: E402
from rsl_rl.modules.distribution import GaussianDistribution  # noqa: E402
from rsl_rl.utils import resolve_callable  # noqa: E402
from tensordict import TensorDict  # noqa: E402

import myosuite.envs.myo.backends.mjlab  # noqa: E402, F401
from mjlab.tasks.registry import list_tasks, load_rl_cfg  # noqa: E402
from mjlab.utils.os import dump_yaml  # noqa: E402
from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (  # noqa: E402
    default_mimic_clip_on_policy_runner_cfg,
)
from myosuite.envs.myo.backends.mjlab.register_mjlab_tabletennis import (  # noqa: E402
    _table_tennis_ppo_runner_cfg,
)
from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (  # noqa: E402
    _chasetag_ppo_runner_cfg,
    _elbow_ppo_runner_cfg,
)
from myosuite.envs.myo.backends.mjlab.tasks.distribution import (  # noqa: E402
    ProjectedGaussianDistribution,
)
from myosuite.envs.myo.backends.mjlab.tasks.rl import (  # noqa: E402
    muscle_gaussian_distribution_cfg,
)
from myosuite.utils.rslrl_policy import load_rslrl_policy  # noqa: E402

# Literal values, so that a change of the module constants fails here.
_STD_MAX = 1.0
_ENTROPY_MAX = 1e-3
_INIT_STD = 0.5
# Maximum learning rate of rsl_rl's adaptive schedule.
_MAX_ADAPTIVE_LR = 1e-2
_DIM = 8

# Configs that #566 moved from std 1.0 / entropy 0.01 to the muscle defaults.
_ISSUE_FACTORIES: dict[str, Callable[[], Any]] = {
    "mimic": default_mimic_clip_on_policy_runner_cfg,
    "chasetag": _chasetag_ppo_runner_cfg,
    "table_tennis": _table_tennis_ppo_runner_cfg,
    "elbow": _elbow_ppo_runner_cfg,
}


def _registered_tasks() -> list[str]:
    """MyoSuite's mjlab ids: muscle tasks (``myo*``) and motor-driven fingers (``motor*``)."""
    return sorted(t for t in list_tasks() if t.startswith(("myo", "motor")))


def _assert_bounded(rl_cfg: Any) -> None:
    dist = rl_cfg.actor.distribution_cfg
    assert resolve_callable(dist["class_name"]) is ProjectedGaussianDistribution
    # "log" renames the checkpoint parameter; published checkpoints use "scalar".
    assert dist["std_type"] == "scalar"
    lo, hi = dist["std_range"]
    assert 0.0 < lo < dist["init_std"] < hi <= _STD_MAX
    assert rl_cfg.algorithm.entropy_coef <= _ENTROPY_MAX


def test_issue_tasks_are_registered() -> None:
    """The registry sweep below covers the ChaseTag, table-tennis and motorFinger configs."""
    tasks = _registered_tasks()
    for task_id in (
        "myoChallengeChaseTagFBP2-v0",
        "myoChallengeTableTennisP2-v0",
        "motorFingerPoseFixed-v0",
        "motorFingerReachRandom-v0",
    ):
        assert task_id in tasks


@pytest.mark.parametrize("task_id", _registered_tasks())
def test_registered_runner_cfg_bounds_std(task_id: str) -> None:
    _assert_bounded(load_rl_cfg(task_id))


@pytest.mark.parametrize("name", sorted(_ISSUE_FACTORIES))
def test_issue_factories_use_muscle_defaults(name: str) -> None:
    """MuscleMimic registers only with a clip, so its factory is checked directly."""
    rl_cfg = _ISSUE_FACTORIES[name]()
    _assert_bounded(rl_cfg)
    assert rl_cfg.actor.distribution_cfg["init_std"] == _INIT_STD


def _build(distribution_cfg: dict[str, Any], dim: int = _DIM) -> Any:
    """Build the distribution from its config as rsl_rl's ``MLPModel`` does."""
    cfg = dict(distribution_cfg)
    return resolve_callable(cfg.pop("class_name"))(dim, **cfg)


def test_init_std_must_lie_inside_range() -> None:
    lo, hi = muscle_gaussian_distribution_cfg()["std_range"]
    for init_std in (lo, hi, 2.0 * hi):
        with pytest.raises(ValueError):
            muscle_gaussian_distribution_cfg(init_std=init_std)
    # A CLI override bypasses the factory; the distribution rejects it too.
    cfg = muscle_gaussian_distribution_cfg()
    cfg["init_std"] = hi
    with pytest.raises(ValueError):
        _build(cfg)


def _adam_steps(
    dist: Any, steps: int, lr: float, loss_fn: Callable[[Any], Any]
) -> None:
    """``steps`` Adam updates of ``loss_fn(dist)``, then one more ``update()``."""
    opt = torch.optim.Adam(dist.parameters(), lr=lr)
    mean = torch.zeros(16, _DIM)
    for _ in range(steps):
        dist.update(mean)
        loss = loss_fn(dist)
        opt.zero_grad()
        loss.backward()
        opt.step()
    dist.update(mean)


def _entropy_bonus(dist: Any) -> Any:
    """Saturated controls: the entropy bonus is the only gradient on the std."""
    return -_ENTROPY_MAX * dist.entropy.mean()


def _smaller_std(dist: Any) -> Any:
    """A loss that favours a smaller std (precise control pays)."""
    return dist.std.sum()


def test_std_parameter_stays_in_range() -> None:
    """The raw parameter, not only the std rsl_rl samples with, stays in range."""
    cfg = muscle_gaussian_distribution_cfg()
    lo, hi = cfg["std_range"]
    up = _build(cfg)
    _adam_steps(up, 3200, _MAX_ADAPTIVE_LR, _entropy_bonus)
    assert up.std_param.max().item() <= hi
    down = _build(cfg)
    _adam_steps(down, 3200, _MAX_ADAPTIVE_LR, _smaller_std)
    assert down.std_param.min().item() >= lo


def test_std_grows_past_cap_without_std_range() -> None:
    """Control: without a range the same entropy updates exceed the cap."""
    cfg = muscle_gaussian_distribution_cfg()
    del cfg["std_range"]
    dist = _build(cfg)
    _adam_steps(dist, 3200, _MAX_ADAPTIVE_LR, _entropy_bonus)
    assert dist.std_param.max().item() > _STD_MAX


def test_std_has_gradient_at_bounds() -> None:
    """rsl_rl's own clamp has no gradient at a bound; the projection keeps it."""
    cfg = muscle_gaussian_distribution_cfg()
    for bound in cfg["std_range"]:
        dist = _build(cfg)
        with torch.no_grad():
            dist.std_param.fill_(bound)
        dist.update(torch.zeros(16, _DIM))
        dist.entropy.mean().backward()
        assert (dist.std_param.grad != 0).all()


def test_std_recovers_from_bounds() -> None:
    """A std pushed to a bound moves back once the loss favours it."""
    cfg = muscle_gaussian_distribution_cfg()
    lo, hi = cfg["std_range"]
    dist = _build(cfg)
    _adam_steps(dist, 200, _MAX_ADAPTIVE_LR, _entropy_bonus)
    assert dist.std.max().item() == pytest.approx(hi)
    _adam_steps(dist, 300, 1e-3, _smaller_std)
    assert dist.std.max().item() < hi - 0.1
    _adam_steps(dist, 300, _MAX_ADAPTIVE_LR, _smaller_std)
    assert dist.std.min().item() == pytest.approx(lo)
    _adam_steps(dist, 300, 1e-3, _entropy_bonus)
    assert dist.std.min().item() > lo + 0.1


def test_resume_from_uncapped_checkpoint() -> None:
    """A pre-cap checkpoint (std 14, as the #566 run) loads strictly and stays trainable."""
    old = GaussianDistribution(_DIM, init_std=1.0)
    with torch.no_grad():
        old.std_param.fill_(14.0)
    cfg = muscle_gaussian_distribution_cfg()
    hi = cfg["std_range"][1]
    new = _build(cfg)
    new.load_state_dict(old.state_dict(), strict=True)
    new.update(torch.zeros(1, _DIM))
    assert new.std_param.max().item() == pytest.approx(hi)
    _adam_steps(new, 300, 1e-3, _smaller_std)
    assert new.std.max().item() < hi - 0.1


def _save_actor(
    run_dir: Path, std: torch.Tensor, distribution_cfg: dict[str, Any] | None
) -> Path:
    """An rsl_rl actor checkpoint and, if a config is given, its ``params/agent.yaml``."""
    obs = TensorDict({"policy": torch.zeros(1, 4)}, batch_size=[1])
    actor = MLPModel(
        obs=obs,
        obs_groups={"actor": ["policy"]},
        obs_set="actor",
        output_dim=_DIM,
        hidden_dims=(16,),
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
    )
    state = actor.state_dict()
    state["distribution.std_param"] = std
    run_dir.mkdir(parents=True)
    path = run_dir / "model_1.pt"
    torch.save({"actor_state_dict": state}, path)
    if distribution_cfg is not None:
        # The writer scripts/train_mjlab.py uses for params/agent.yaml.
        dump_yaml(
            run_dir / "params" / "agent.yaml",
            {"actor": {"activation": "elu", "distribution_cfg": distribution_cfg}},
        )
    return path


def test_stochastic_eval_uses_training_std_range(tmp_path: Path) -> None:
    """``load_rslrl_policy`` clamps the std to the run's ``std_range``, else keeps it."""
    raw = torch.tensor([14.0, 0.01, 0.3, 1.0, 0.05, 2.0, 0.5, 0.02])
    cfg = muscle_gaussian_distribution_cfg()
    lo, hi = cfg["std_range"]
    bounded = _save_actor(tmp_path / "bounded", raw, cfg)
    torch.testing.assert_close(load_rslrl_policy(bounded, _DIM).std, raw.clamp(lo, hi))
    # Published checkpoints: no params/agent.yaml, or a run without std_range.
    for name, run_cfg in (
        ("bare", None),
        ("uncapped", {"class_name": "GaussianDistribution", "init_std": 1.0}),
    ):
        path = _save_actor(tmp_path / name, raw, run_cfg)
        torch.testing.assert_close(load_rslrl_policy(path, _DIM).std, raw)
