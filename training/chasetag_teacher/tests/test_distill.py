# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The student without data: yaw canonicalisation, the half-turn symmetry, the 546 -> 354 interface, ONNX export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from chasetag_teacher.augment import augment_half_turn, half_turn
from chasetag_teacher.distill import (
    ACT_DIM,
    OBS_DIM,
    CanonPlayerNet,
    StudentPlayer,
    canonicalise,
    canonicalise_inplace,
    export_canon_onnx,
    fold_normalisation,
    load_data,
    root_yaw,
    train,
    yaw_rotate,
)
from myosuite.envs.myo.tasks.mimic.policy import ActorCritic

pytestmark = pytest.mark.tier1


def test_interface_is_546_to_354() -> None:
    assert (OBS_DIM, ACT_DIM) == (546, 354)


def test_canonicalisation_is_exactly_yaw_invariant(fbvs_obs: torch.Tensor) -> None:
    theta = torch.rand(len(fbvs_obs), dtype=torch.float64) * 2 * np.pi
    d = (canonicalise(yaw_rotate(fbvs_obs, theta)) - canonicalise(fbvs_obs)).abs().max()
    assert float(d) < 1e-12


def test_canonical_observation_has_zero_yaw(fbvs_obs: torch.Tensor) -> None:
    c = canonicalise(fbvs_obs)
    assert float(root_yaw(c).abs().max()) < 1e-12
    assert bool((c[:, 3] >= 0).all())  # quaternion sign fixed
    # yaw-free entries (heights, joints, velocities in the body frame, muscles) pass unchanged
    keep = np.r_[2, 7:89, 91:531, 533, 536, 539, 542:546]
    assert torch.equal(c[:, keep], fbvs_obs[:, keep])


def test_canonicalise_is_scale_invariant_in_the_quaternion(
    fbvs_obs: torch.Tensor,
) -> None:
    scaled = fbvs_obs.clone()
    scaled[:, 3:7] *= 1.7
    assert torch.allclose(root_yaw(scaled), root_yaw(fbvs_obs), atol=1e-12)


def test_half_turn_is_the_yaw_rotation_by_pi(fbvs_obs: torch.Tensor) -> None:
    rotated = yaw_rotate(
        fbvs_obs, torch.full((len(fbvs_obs),), np.pi, dtype=torch.float64)
    )
    np.testing.assert_allclose(half_turn(fbvs_obs.numpy()), rotated.numpy(), atol=1e-12)


def test_canonicalise_inplace_matches(fbvs_obs: torch.Tensor) -> None:
    o = fbvs_obs.numpy().astype(np.float32)
    want = canonicalise(torch.as_tensor(o)).numpy()
    np.testing.assert_allclose(
        canonicalise_inplace(o.copy(), chunk=10), want, atol=1e-6
    )  # chunked float32


def test_fold_normalisation() -> None:
    torch.manual_seed(0)
    net = ActorCritic(OBS_DIM, ACT_DIM, hidden=16, layers=2)
    mean, std = torch.randn(OBS_DIM), torch.rand(OBS_DIM) + 0.5
    x = torch.randn(4, OBS_DIM)
    want = net.actor_mean(net.trunk((x - mean) / std))
    fold_normalisation(net, mean, std)
    torch.testing.assert_close(net.actor_mean(net.trunk(x)), want, atol=1e-5, rtol=1e-5)


def test_train_load_and_play(tmp_path: Path, fbvs_obs: torch.Tensor) -> None:
    """A student fitted on (546, 354) pairs loads as a plain ActorCritic and plays raw or canonicalised."""
    rng = np.random.default_rng(1)
    o = fbvs_obs.numpy().repeat(4, 0).astype(np.float16)
    y = rng.uniform(0, 1, (len(o), ACT_DIM)).astype(np.float16)
    res = train(o, y, epochs=1, batch=64, log=lambda s: None)
    assert np.isfinite(res["val_mse"])
    path = tmp_path / "student.pt"
    torch.save(res["state"], path)
    for canon in (False, True):
        a = StudentPlayer(path, canon=canon).act(
            None, "agent_0", o[0].astype(np.float32)
        )
        assert a.shape == (ACT_DIM,) and a.dtype == np.float32
        assert (a >= 0).all() and (a <= 1).all()


def test_canon_onnx_matches_torch(tmp_path: Path, fbvs_obs: torch.Tensor) -> None:
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    torch.manual_seed(0)
    net = ActorCritic(OBS_DIM, ACT_DIM)
    for p in net.actor_mean.parameters():
        torch.nn.init.normal_(p, std=0.05)
    path = tmp_path / "student.onnx"
    export_canon_onnx(net.state_dict(), path)
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [i.name for i in s.get_inputs()] == ["obs"] and [
        o.name for o in s.get_outputs()
    ] == ["action"]
    x = fbvs_obs.float().numpy()
    y = s.run(None, {"obs": x})[0]
    with torch.no_grad():
        t = CanonPlayerNet(net).eval()(torch.as_tensor(x)).numpy()
    assert y.shape == (len(x), ACT_DIM)
    np.testing.assert_allclose(y, t, atol=1e-5)
    # the canonicalisation is inside the graph: rotating the scene leaves the action unchanged
    rot = (
        yaw_rotate(fbvs_obs, torch.full((len(x),), 0.9, dtype=torch.float64))
        .float()
        .numpy()
    )
    np.testing.assert_allclose(s.run(None, {"obs": rot})[0], y, atol=1e-4)


def test_load_data(tmp_path: Path) -> None:
    for i, n in enumerate((3, 5)):
        np.savez(tmp_path / f"b{i:05d}.npz", obs=np.full((n, OBS_DIM), i, np.float16),
                 act=np.full((n, ACT_DIM), i, np.float16))  # fmt: skip
    o, y, held = load_data([tmp_path])
    assert o.shape == (8, OBS_DIM) and y.shape == (8, ACT_DIM) and o.dtype == np.float16
    assert held.shape == (8,) and held.dtype == bool
    assert float(o[-1, 0]) == 1.0


def _bouts(rng: np.random.Generator, n: int, steps: int = 400, dim: int = 32) -> list:
    """Bouts whose labels depend on bout context: a slowly drifting state and a constant target."""
    out = []
    for _ in range(n):
        drift = np.cumsum(rng.normal(scale=0.01, size=(steps, dim)), 0)
        y = np.clip(0.5 + 0.25 * rng.normal(size=ACT_DIM), 0, 1)
        out.append((rng.normal(size=dim) + drift, np.tile(y, (steps, 1))))
    return out


def test_val_mse_is_the_error_on_held_out_bouts(tmp_path: Path) -> None:
    """Consecutive steps of a bout are near-duplicates: the validation set holds out whole bouts."""
    rng = np.random.default_rng(0)
    for i, (o, a) in enumerate(_bouts(rng, 40)):
        np.savez(tmp_path / f"b{i:05d}.npz", obs=o.astype(np.float16), act=a.astype(np.float16))  # fmt: skip
    o, y, held = load_data([tmp_path], obs_dim=32, holdout=0.2)
    assert 0 < held.sum() < len(held)
    again = load_data([tmp_path], obs_dim=32, holdout=0.2)[2]
    np.testing.assert_array_equal(held, again)  # fixed per bout
    torch.manual_seed(0)
    res = train(o, y, epochs=15, batch=1024, obs_dim=32, held_out=held, log=lambda s: None)  # fmt: skip

    test = _bouts(rng, 40)
    xt = torch.as_tensor(np.concatenate([b[0] for b in test]), dtype=torch.float32)
    yt = np.concatenate([b[1] for b in test])
    net = ActorCritic(32, ACT_DIM)
    net.load_state_dict(res["state"])
    with torch.no_grad():
        pred = net.actor_mean(net.trunk(xt)).clamp(0, 1).numpy()
    unseen = float(((pred - yt) ** 2).mean())
    assert unseen / 2 < res["val_mse"] < unseen * 2


def test_half_turn_copies_of_held_out_bouts_stay_out_of_training(
    fbvs_obs: torch.Tensor,
) -> None:
    o = fbvs_obs.numpy()
    y = np.random.default_rng(0).uniform(0, 1, (len(o), ACT_DIM))
    held = np.arange(len(o)) < 10
    xa, ya, ha = augment_half_turn(o, y, held)
    assert len(xa) == len(o) + int((~held).sum())
    np.testing.assert_array_equal(ha[: len(o)], held)
    assert not ha[len(o) :].any()
    np.testing.assert_allclose(xa[len(o) :], half_turn(o[~held]))
    np.testing.assert_allclose(ya[len(o) :], y[~held])
