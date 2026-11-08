# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""World model, policy and its action map, replay buffer, optimizers and exploration noise (no env needed).

Ported in part from muscle-supermimic's tests (github.com/kaibohex/muscle-supermimic @ 8a6244b, Apache-2.0).
"""

from __future__ import annotations

import copy
import math
from typing import Any

import pytest
import torch

pytestmark = pytest.mark.tier1


@pytest.mark.parametrize("structured", [False, True])
def test_world_model_integrate_inverts_targets(structured: bool) -> None:
    from supermimic.networks import WorldModel
    from myosuite.envs.myo.backends.mjlab.tasks.mimic_multiclip.rotations import (
        quat_to_mat,
    )

    gen = torch.Generator().manual_seed(0)
    rn = lambda *s: torch.randn(*s, generator=gen)  # noqa: E731
    nq = 7

    def state() -> tuple:
        return (
            rn(8, 3),
            quat_to_mat(rn(8, 4)),
            rn(8, 3),
            rn(8, 3),
            0.5 * rn(8, nq),
            rn(8, nq),
        )

    wm = WorldModel(nq, 5, 4, hidden=(16,), structured=structured, dt=0.02, substeps=10)
    S, S1 = state(), state()
    S1_hat = wm.integrate(S, wm.targets(S, S1))
    for a, b in zip(S1_hat, S1, strict=True):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    # the drift is the semi-implicit Euler position update of 10 substeps with a linearly changing velocity
    assert wm.drift_w == 11 / 20


def _policy(seed: int = 0) -> Any:
    from supermimic.networks import Policy

    torch.manual_seed(seed)
    return Policy(20, 30, 5, enc_hidden=(64, 32, 16), dec_hidden=(64, 32, 16))


def _set_grads(m: Any, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    for p in m.parameters():
        p.grad = torch.randn(p.shape, generator=g)


def test_muon_adam_splits_the_parameters_and_round_trips() -> None:
    from supermimic.optim import MuonAdam

    pi = _policy()
    opt = MuonAdam(pi, [pi.enc, pi.dec], lr=1e-2, fused=False)
    muon = [p for g in opt.muon.param_groups for p in g["params"]]
    adam = [p for g in opt.adam.param_groups for p in g["params"]]
    assert len(muon) == 4 and all(
        p.dim() == 2 for p in muon
    )  # two hidden Linear weights per MLP
    assert len(muon) + len(adam) == len(list(pi.parameters()))
    before = [p.detach().clone() for p in pi.parameters()]
    for i in range(3):
        _set_grads(pi, i)
        opt.step()
    assert all(not torch.equal(p, b) for p, b in zip(pi.parameters(), before))
    pi2 = _policy(1)
    opt2 = MuonAdam(pi2, [pi2.enc, pi2.dec], lr=1e-2, fused=False)
    opt2.load_state_dict(opt.state_dict())
    for g in opt2.param_groups:
        g["lr"] = 5e-3
    assert all(g["lr"] == 5e-3 for g in opt2.param_groups)


@pytest.mark.parametrize("kind", ["muon", "adam"])
def test_gated_step_is_the_plain_step_or_a_no_op(kind: str) -> None:
    from supermimic.optim import MuonAdam, gated_step

    a = _policy()
    b = copy.deepcopy(a)

    def make(m: Any) -> Any:
        if kind == "muon":
            return MuonAdam(m, [m.enc, m.dec], lr=1e-3, fused=True)
        return torch.optim.Adam(m.parameters(), lr=1e-3, fused=True)

    def snapshot(m: Any, opt: Any) -> list:
        t = [p.detach().clone() for p in m.parameters()]
        for sub in (opt.muon, opt.adam) if isinstance(opt, MuonAdam) else (opt,):
            for st in sub.state.values():
                t += [v.detach().clone() for v in st.values() if torch.is_tensor(v)]
        return t

    oa, ob = make(a), make(b)
    for i in range(3):
        _set_grads(a, i), _set_grads(b, i)
        oa.step(), gated_step(ob, torch.tensor(True))
    for x, y in zip(snapshot(a, oa), snapshot(b, ob), strict=True):
        assert torch.equal(x, y)
    before = snapshot(b, ob)
    _set_grads(b, 99)
    gated_step(ob, torch.tensor(False))
    for x, y in zip(before, snapshot(b, ob), strict=True):
        assert torch.equal(x, y)


def test_running_norm_no_sync_matches_the_host_side_freeze() -> None:
    from supermimic.networks import RunningNorm

    torch.manual_seed(0)
    a, b = RunningNorm(5, max_count=25), RunningNorm(5, max_count=25)
    b.no_sync = True
    for _ in range(6):
        x = torch.randn(8, 5) * 3 + 1
        a.update(x), b.update(x)
        torch.testing.assert_close(a.mean, b.mean, atol=0, rtol=0)
        torch.testing.assert_close(a.var, b.var, atol=0, rtol=0)
    assert float(a.count) == float(b.count) == 32


def test_exploration_noise() -> None:
    from supermimic.networks import explore

    torch.manual_seed(0)
    a = torch.tensor([0.0, 1e-4, 0.02, 0.2, 0.5, 0.9, 1.0]).repeat(40000, 1)
    x = explore(a, 0.1, "beta")
    assert torch.isfinite(x).all() and (x >= 0).all() and (x <= 1).all()
    k = 1 / (4 * 0.1**2) - 1
    torch.testing.assert_close(
        x.mean(0), a[0], atol=2e-3, rtol=0
    )  # mean-preserving, also next to the bounds
    torch.testing.assert_close(
        x.std(0), (a[0] * (1 - a[0]) / (k + 1)).sqrt(), atol=2e-3, rtol=0.02
    )
    assert torch.equal(x[:, 0], a[:, 0]) and torch.equal(x[:, -1], a[:, -1])
    assert (
        float(explore(a, 0.1, "gaussian")[:, 2].mean()) > 2 * 0.02
    )  # the clipped Gaussian is biased upwards
    r = explore(torch.tensor([0.02, 0.5]).repeat(100000, 1), 0.1, "reflect")
    assert (r >= 0).all() and (r <= 1).all() and abs(float(r[:, 1].std()) - 0.1) < 2e-3
    s = 0.1
    folded = s * math.sqrt(2 / math.pi) * math.exp(
        -(0.02**2) / (2 * s**2)
    ) + 0.02 * math.erf(0.02 / (s * math.sqrt(2)))
    assert abs(float(r[:, 0].mean()) - folded) < 2e-3


def test_sampling_from_a_snapshot_never_reaches_the_rows_being_written() -> None:
    from supermimic.buffer import ReplayBuffer

    torch.manual_seed(0)
    T, N, reserve = 10, 3, 3
    buf = ReplayBuffer(T, N, body_dim=4, num_muscles=2, num_touch=1, device="cpu")
    tr = {
        k: torch.zeros((N,) + shape, dtype=dt) for k, (shape, dt) in buf.fields.items()
    }
    for _ in range(12):
        buf.add(tr)
    buf.use_device_ptr()
    buf.use_sample_snapshot(reserve)
    rows: list = []
    row = buf._row
    buf._row = lambda age: rows.append(row(age).flatten()) or row(age)

    def sampled() -> set:
        rows.clear()
        buf.sample_windows(400, 2)
        buf.sample_starts(400, 1)
        return set(torch.cat(rows).tolist())

    ptr = buf.ptr
    first = sampled()
    assert first == set(range(T)) - {(ptr + i) % T for i in range(reserve)}
    for _ in range(reserve):
        buf.add_device(tr)
    assert sampled() == first
    buf.snapshot()
    assert sampled() == set(range(T)) - {
        (ptr + reserve + i) % T for i in range(reserve)
    }


def test_identity_action_map_keeps_the_policy_layout() -> None:
    """The action map has no state: a policy's checkpoint keys are those of the policy without one, and the map is
    recorded as its spec (DynSyn groups are planned as another kind)."""
    from supermimic.networks import ActionMap, Policy

    pi = _policy()
    assert all(not k.startswith("action_map") for k in pi.state_dict())
    assert pi.dec[-1].out_features == 5 and pi.action_map.spec() == {
        "kind": "identity",
        "num_muscles": 5,
    }
    obs, goal = torch.randn(3, 20), torch.randn(3, 30)
    torch.testing.assert_close(
        pi(obs, goal), torch.sigmoid(pi.logits(obs, goal)), atol=0, rtol=0
    )
    assert (
        ActionMap.from_spec(None, 5).spec() == pi.action_map.spec()
    )  # checkpoints from before action maps
    with pytest.raises(ValueError, match="muscles"):
        ActionMap.from_spec({"kind": "identity", "num_muscles": 4}, 5)
    with pytest.raises(NotImplementedError, match="groups"):
        ActionMap(5, "groups")
    with pytest.raises(ValueError, match="num_actions"):
        Policy(20, 30, 6, action_map=ActionMap(5))
