# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""PODS-style action targets and the median-relative action-gradient clip (pods.py), and the runner with them."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest
import torch

from supermimic.pods import MedianRowClip, live_median, pods_targets


def test_live_median_ignores_dead_rows() -> None:
    norm = torch.tensor([5.0, 0.0, 1.0, 3.0, 0.0, 2.0])
    torch.testing.assert_close(
        live_median(norm, norm > 0), torch.tensor(2.0)
    )  # lower median of 1, 2, 3, 5
    assert torch.isinf(live_median(norm, torch.zeros(6, dtype=torch.bool)))


def test_median_row_clip_in_the_backward() -> None:
    """Identity forward; rows above kappa x the median of the rows with gradient are scaled down to it, zero and NaN
    rows never set the threshold, and the probe's gradient carries the statistics."""
    x = torch.rand(5, 3, requires_grad=True)
    probe = torch.zeros(2, requires_grad=True)
    y = MedianRowClip.apply(x, probe, 2.0)
    assert torch.equal(y, x)
    g = torch.ones(5, 3)
    g[0] *= 100.0  # the outlier
    g[4] = 0.0  # a row without gradient
    (y * g).sum().backward()
    torch.testing.assert_close(x.grad[0].norm(), torch.tensor(2.0 * 3**0.5))
    torch.testing.assert_close(x.grad[1:], g[1:])
    frac, removed = probe.grad
    torch.testing.assert_close(frac, torch.tensor(0.25))
    torch.testing.assert_close(removed, torch.tensor(1 - (2 + 3) / (100 + 3)))
    x.grad = None
    g[2] = float("nan")
    MedianRowClip.apply(x, torch.zeros(2), 2.0).mul(g).sum().backward()
    torch.testing.assert_close(x.grad[0].norm(), torch.tensor(2.0 * 3**0.5))


def test_pods_targets_take_the_rms_step_over_used_rows() -> None:
    a = torch.full((3, 4, 5), 0.5)
    g = torch.randn(3, 4, 5)
    target, w, st = pods_targets(a, g, 0.03)
    step = target - a
    torch.testing.assert_close(step.square().mean().sqrt(), torch.tensor(0.03))
    torch.testing.assert_close(step, st["eta"] * g)
    assert bool((w == 1).all()) and float(st["used"]) == 1.0
    sat, _, _ = pods_targets(torch.ones(2, 3), torch.ones(2, 3), 0.1)
    assert float(sat.max()) == 1.0


def test_dead_and_nonfinite_rows_leave_the_valid_targets_unchanged() -> None:
    """Rows without gradient, with NaN / Inf entries or with invalid inputs get weight 0 and their own action as the
    target, and neither change eta nor the targets of the other rows."""
    gen = torch.Generator().manual_seed(0)
    a = torch.rand(6, 4, generator=gen)
    g = torch.randn(6, 4, generator=gen)
    ref, w_ref, st_ref = pods_targets(a, g, 0.03, kappa=3.0)
    extra_a = torch.rand(4, 4, generator=gen)
    extra_g = torch.randn(4, 4, generator=gen)
    extra_g[0] = 0.0
    extra_g[1, 2] = float("nan")
    extra_g[2, 0] = float("inf")
    valid = torch.ones(10, dtype=torch.bool)
    valid[9] = False  # e.g. a non-finite policy input
    target, w, st = pods_targets(
        torch.cat((a, extra_a)), torch.cat((g, extra_g)), 0.03, kappa=3.0, valid=valid
    )
    torch.testing.assert_close(target[:6], ref)
    torch.testing.assert_close(st["eta"], st_ref["eta"])
    assert torch.equal(w, torch.cat((w_ref, torch.zeros(4))))
    torch.testing.assert_close(target[6:], extra_a)
    assert float(st["nonfinite"]) == 2.0 and torch.isfinite(target).all()


@pytest.mark.parametrize("case", ["zero", "no_valid_row"])
def test_no_usable_row_keeps_the_actions(case: str) -> None:
    a = torch.rand(4, 3)
    g = torch.zeros(4, 3) if case == "zero" else torch.randn(4, 3)
    valid = None if case == "zero" else torch.zeros(4, dtype=torch.bool)
    target, w, st = pods_targets(a, g, 0.03, kappa=3.0, valid=valid)
    assert torch.equal(target, a) and float(w.sum()) == 0.0 and float(st["eta"]) == 0.0


def test_target_gradients_are_clipped_against_one_median_over_used_rows() -> None:
    """One median over all live rows (not per step): the outlier step is clipped, a uniformly larger step is not."""
    g = torch.ones(2, 4, 3)
    g[1] *= 2.0
    g[0, 0] *= 100.0
    g[1, 3] = 0.0  # dead: not part of the median
    target, _, st = pods_targets(torch.full_like(g, 0.5), g, 0.03, kappa=3.0)
    step = (target - 0.5) / st["eta"]
    torch.testing.assert_close(
        step[0, 0].norm(), torch.tensor(6.0 * 3**0.5)
    )  # 3 x the median, 2 sqrt 3
    torch.testing.assert_close(step[1, :3], g[1, :3])
    torch.testing.assert_close(st["clip"], torch.tensor(1.0 / 7.0))


# ------------------------------------------------------------------ the runner (CPU)
def _runner(clips: Path, **rl: Any) -> Any:
    from tests.test_runner import _make

    _, runner = _make(
        str(clips / "clip0.npz"),
        **{"wm_only_iters": 0, "pi_ema": 0.9, "pods_batch": 16, **rl},
    )
    return runner


@pytest.mark.tier2
def test_offset_gradient_is_the_action_gradient(clips: Path, monkeypatch: Any) -> None:
    """dJ/da_t from the zero offsets equals the gradient the default update's backward sends into the policy's
    actions: the offset is on the executed action, so it sees the dynamics, the reward (action terms included) and
    the history alike."""
    import supermimic.runner as rmod

    runner = _runner(clips, wm_only_iters=10**6)
    runner.learn(
        num_learning_iterations=1
    )  # fills the buffer, world-model updates only
    nets = {k: getattr(runner, k) for k in ("pi", "critic", "opt_pi", "opt_v")}
    saved = {k: copy_state(v) for k, v in nets.items()}
    grads: list[torch.Tensor] = []

    def record(m: Any, i: Any, o: torch.Tensor) -> None:
        o.register_hook(grads.append)

    hook = runner.pi.action_map.register_forward_hook(record)
    torch.manual_seed(1)
    runner.policy_update()
    hook.remove()
    for k, v in nets.items():
        v.load_state_dict(saved[k])
    seen: dict[str, torch.Tensor] = {}

    def spy(a: torch.Tensor, g: torch.Tensor, *args: Any, **kw: Any) -> Any:
        seen["g"] = g.clone()
        return pods_targets(a, g, *args, **kw)

    monkeypatch.setattr(rmod, "pods_targets", spy)
    runner.cfg.pods = True
    torch.manual_seed(1)
    runner.policy_update()
    ref = -torch.stack(grads[::-1]).flatten(0, 1)
    torch.testing.assert_close(seen["g"], ref, atol=1e-9, rtol=1e-5)
    assert float(ref.abs().sum()) > 0


def copy_state(m: Any) -> dict:
    import copy

    return copy.deepcopy(m.state_dict())


@pytest.mark.tier2
@pytest.mark.parametrize("clip", [0.0, 3.0], ids=["pods", "pods_clip"])
def test_runner_learns_with_pods_and_resumes(
    clips: Path, tmp_path: Path, clip: float
) -> None:
    runner = _runner(clips, pods=True, act_grad_clip=clip)
    runner.log_dir = str(tmp_path)
    before = copy_state(runner.pi)
    runner.learn(num_learning_iterations=3)
    after = runner.pi.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before)
    assert (
        all(torch.isfinite(v).all() for v in after.values())
        and int(runner.nonfinite) == 0
    )
    with open(tmp_path / "train_log.csv") as fh:
        row = list(csv.DictReader(fh))[-1]
    assert (
        float(row["pods_used"]) > 0
        and float(row["pods_dq_rms"]) > 0
        and float(row["pods_eta"]) > 0
    )
    assert ("act_clip_frac" in row) == (clip > 0)
    assert abs(float(row["pi_lr"]) - runner.lr_at(2) * runner.cfg.pods_lr_mult) < 1e-12
    runner.load(str(tmp_path / "model_000003.pt"))
    runner.learn(num_learning_iterations=1)
    assert runner.it == 4


@pytest.mark.tier2
def test_nonfinite_gradients_do_not_poison_the_update(
    clips: Path, monkeypatch: Any
) -> None:
    """NaN / Inf in some rows of dJ/da: those rows drop out, the policy still takes a finite update."""
    import supermimic.runner as rmod

    runner = _runner(clips, pods=True, act_grad_clip=3.0, wm_only_iters=10**6)
    runner.learn(num_learning_iterations=1)
    runner.cfg.wm_only_iters = 0
    seen: dict[str, Any] = {}

    def corrupt(a: torch.Tensor, g: torch.Tensor, *args: Any, **kw: Any) -> Any:
        g = g.clone()
        g[0, 0], g[1, 1], g[2] = float("nan"), float("inf"), float("nan")
        out = pods_targets(a, g, *args, **kw)
        seen.update(out[2])
        return out

    monkeypatch.setattr(rmod, "pods_targets", corrupt)
    before = copy_state(runner.pi)
    runner.policy_update()
    after = runner.pi.state_dict()
    assert float(seen["nonfinite"]) == 3.0 and 0 < float(seen["used"]) < 1
    assert all(torch.isfinite(v).all() for v in after.values())
    assert any(not torch.equal(before[k], after[k]) for k in before)


def test_pods_step_anneals_to_its_floor() -> None:
    from supermimic.config import SuperMimicRunnerCfg
    from supermimic.runner import SuperMimicRunner

    r = SuperMimicRunner.__new__(SuperMimicRunner)
    r.cfg = SuperMimicRunnerCfg(pods_step=0.03, pods_step_floor=0.3, lr_decay_iters=100)
    assert r.pods_step_at(0) == pytest.approx(0.03)
    assert r.pods_step_at(50) == pytest.approx(0.03 * 0.65)
    assert r.pods_step_at(500) == pytest.approx(0.009)
