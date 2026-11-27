# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The CUDA graphs of the SuperMimic runner: the capture wrapper replays what it captured, a tiny run
captures both updates and the collection step, keeps its learning-rate tensors on the schedule, advances the policy
EMA inside the replays, round-trips a checkpoint, overlaps the collection with the updates and stops on its
wall-clock budget. Skipped without a CUDA device.

Ported from muscle-supermimic's ``tests/test_cuda_graph.py`` (github.com/kaibohex/muscle-supermimic @ 8a6244b,
Apache-2.0), on synthetic clips (``tests/conftest.py``).
"""

from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")

pytestmark = [
    pytest.mark.tier2,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
]

GRAPH = dict(cuda_graph=True, collect_graph=False, overlap_collect=False)


def test_graphed_replays_the_captured_update() -> None:
    from torch import nn

    from supermimic.runner import Graphed, set_lr

    torch.manual_seed(0)
    dev = "cuda:0"
    x = torch.randn(64, 8, device=dev)
    nets = [nn.Linear(8, 4).to(dev) for _ in range(2)]
    nets[1].load_state_dict(nets[0].state_dict())
    opts = [
        torch.optim.Adam(
            n.parameters(),
            lr=torch.tensor(1e-2, device=dev),
            fused=True,
            capturable=True,
        )
        for n in nets
    ]

    def make_step(net: Any, opt: Any) -> Any:
        def step() -> tuple:
            opt.zero_grad(set_to_none=True)
            loss = (net(x) ** 2).mean()
            loss.backward()
            opt.step()
            return (loss.detach(),)

        return step

    eager, graphed = (
        make_step(nets[0], opts[0]),
        Graphed(make_step(nets[1], opts[1]), lambda: True),
    )
    for i in range(10):
        set_lr(opts[0], 1e-2 * (1 - 0.05 * i)), set_lr(opts[1], 1e-2 * (1 - 0.05 * i))
        (le,), (lg,) = eager(), graphed()
        assert abs(float(le) - float(lg)) < 1e-5
    assert graphed.graph is not None and graphed.calls == 10
    for p, q in zip(nets[0].parameters(), nets[1].parameters(), strict=True):
        torch.testing.assert_close(p, q, atol=1e-6, rtol=1e-5)


def test_updates_run_as_cuda_graphs(clips: Path, tmp_path: Path) -> None:
    from tests.test_runner import _make

    _, runner = _make(
        str(clips / "clip0.npz"), device="cuda:0", pi_ema=0.9, wm_only_iters=0, **GRAPH
    )
    runner.log_dir = str(tmp_path)
    runner.learn(num_learning_iterations=6)
    g = runner.graphs
    assert g["wm"].graph is not None and g["policy"].graph is not None
    assert g["wm"].calls == 12 and g["policy"].calls == 5
    with open(tmp_path / "train_log.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 6 and all(r["nonfinite"] == "0" for r in rows)
    for k in ("wm_loss", "pi_loss", "v_loss", "imag_rew", "pi_gradnorm"):
        assert all(abs(float(r[k])) < 1e6 for r in rows), k
    assert int(runner.pi_avg.n_averaged) == 6  # the EMA step ran inside the replays
    lr_v = runner.opt_v.param_groups[0]["lr"]
    assert torch.is_tensor(lr_v) and abs(float(lr_v) - runner.lr_at(5)) < 1e-9
    ck = tmp_path / "model_000006.pt"
    pi_before = copy.deepcopy(runner.pi.state_dict())
    runner.load(str(ck))
    assert runner.opt_v.param_groups[0]["lr"] is lr_v and runner.graphs is None
    runner.learn(num_learning_iterations=2)  # recaptures
    assert (
        runner.it == 8
        and int(runner.pi_avg.n_averaged) == 8
        and runner.graphs["policy"].calls == 2
    )
    assert any(
        not torch.equal(v, pi_before[k]) for k, v in runner.pi.state_dict().items()
    )


def test_the_recipe_collects_as_a_cuda_graph_concurrently(
    clips: Path, tmp_path: Path
) -> None:
    """The full recipe on a GPU: the collection step captured (episode ends and resets inside the graph) and
    overlapped with the update graphs, the host and device write heads together, and max_hours stopping training with
    a final checkpoint."""
    from tests.test_runner import _make

    def short(u: Any) -> None:
        u.cfg.episode_length_s = 0.3

    _, runner = _make(str(clips / "clips.txt"), device="cuda:0", num_envs=8, sampling="return", env_edit=short,
                      wm_only_iters=0, log_interval=2)  # fmt: skip
    c = runner.cfg
    assert c.cuda_graph and c.collect_graph and c.overlap_collect
    runner.log_dir = str(tmp_path)
    runner.learn(num_learning_iterations=12)
    col, buf = runner.collector, runner.buffer
    assert col.graph is not None and runner.graphs["policy"].graph is not None
    assert buf.ptr == int(buf.ptr_t) and runner.sim_steps == col.calls * 8
    with open(tmp_path / "train_log.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 6 and all(r["nonfinite"] == "0" for r in rows)
    assert any(float(r["ep_len"]) > 0 for r in rows)
    assert float(runner.cmd.bin_G.abs().sum()) > 0
    runner.cfg.max_hours = 1e-9
    runner.learn(num_learning_iterations=100)
    assert runner.it == 13 and (tmp_path / "model_000013.pt").exists()


@pytest.mark.parametrize("compile_", [False, True], ids=["eager", "compiled"])
def test_pods_runs_inside_the_policy_update_graph(
    clips: Path, tmp_path: Path, compile_: bool
) -> None:
    """pods and act_grad_clip in the captured policy update (a host sync would fail the capture): the annealed step
    and the statistics are read from device tensors, and the policy moves."""
    from tests.test_runner import _make

    _, runner = _make(str(clips / "clip0.npz"), device="cuda:0", pi_ema=0.9, wm_only_iters=0, pods=True,
                      act_grad_clip=3.0, pods_batch=16, lr_decay_iters=4, compile=compile_, **GRAPH)  # fmt: skip
    runner.log_dir = str(tmp_path)
    before = copy.deepcopy(runner.pi.state_dict())
    runner.learn(num_learning_iterations=6)
    assert (
        runner.graphs["policy"].graph is not None and runner.graphs["policy"].calls == 5
    )
    with open(tmp_path / "train_log.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert all(r["nonfinite"] == "0" and float(r["pods_used"]) > 0 for r in rows)
    assert (
        abs(
            float(rows[-1]["pods_step"])
            - runner.cfg.pods_step * runner.cfg.pods_step_floor
        )
        < 1e-9
    )
    assert abs(float(runner.pods_delta) - runner.pods_step_at(5)) < 1e-9
    assert any(not torch.equal(v, before[k]) for k, v in runner.pi.state_dict().items())
