# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Online estimates of a policy's density (``myosuite.utils.density_estimator``)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy import stats

from myosuite.utils.density_estimator import KernelDensity, RunningGaussian

pytestmark = pytest.mark.tier1

MEAN = np.array([0.3, -1.0])
COV = np.array([[0.5, 0.2], [0.2, 0.8]])


def test_the_running_gaussian_learns_the_sample_moments() -> None:
    samples = np.random.default_rng(6).multivariate_normal(
        MEAN + 100.0, COV, size=5000
    )  # far from the origin
    est = RunningGaussian(2, decay=1.0, reg=0.0)
    for chunk in np.array_split(samples, 50):
        est.update(np, chunk)
    mean, cov = est.moments(np, samples)
    np.testing.assert_allclose(mean, samples.mean(0), atol=1e-9)
    np.testing.assert_allclose(cov, np.cov(samples.T, bias=True), atol=1e-9)
    x = samples[:20]
    expected = stats.multivariate_normal(
        samples.mean(0), np.cov(samples.T, bias=True)
    ).logpdf(x)
    np.testing.assert_allclose(est.log_prob(np, x), expected, atol=1e-8)


def test_the_running_gaussian_forgets_old_samples() -> None:
    est = RunningGaussian(1, decay=0.99)
    est.update(np, np.full((2000, 1), -5.0))
    est.update(np, np.random.default_rng(7).normal(3.0, 1.0, (2000, 1)))
    mean, _ = est.moments(np, np.zeros(1))
    assert abs(mean[0] - 3.0) < 0.1


def test_the_kernel_density_equals_scipy_in_one_dimension() -> None:
    samples = np.random.default_rng(8).normal(1.0, 2.0, (300, 1))
    est = KernelDensity(1, capacity=300)
    est.update(np, samples)
    x = np.linspace(-6, 8, 30)[:, None]
    np.testing.assert_allclose(
        est.log_prob(np, x),
        stats.gaussian_kde(samples[:, 0]).logpdf(x[:, 0]),
        atol=1e-10,
    )


def test_the_kernel_density_keeps_the_latest_samples() -> None:
    est = KernelDensity(1, capacity=5)
    est.update(np, np.arange(3.0)[:, None])
    est.update(np, np.arange(3.0, 7.0)[:, None])
    assert est.count == 5
    assert sorted(est.buffer[:, 0]) == [2.0, 3.0, 4.0, 5.0, 6.0]


@pytest.mark.parametrize(
    "make", [lambda: RunningGaussian(2), lambda: KernelDensity(2, capacity=64)]
)
def test_estimators_agree_on_torch_and_numpy(make: Any) -> None:
    torch = pytest.importorskip("torch")
    samples = np.random.default_rng(9).normal(size=(128, 2))
    on_np, on_torch = make(), make()
    for chunk in np.array_split(samples, 4):
        on_np.update(np, chunk)
        on_torch.update(torch, torch.as_tensor(chunk))
    x = samples[:10]
    np.testing.assert_allclose(
        on_torch.log_prob(torch, torch.as_tensor(x)).numpy(),
        on_np.log_prob(np, x),
        atol=1e-9,
    )


def test_a_frozen_estimator_does_not_change() -> None:
    est = KernelDensity(1, capacity=10, frozen=True)
    est.update(np, np.ones((3, 1)))
    assert est.count == 0
    est.unfreeze()
    est.update(np, np.ones((3, 1)))
    state = est.state_dict()
    other = KernelDensity(1, capacity=10)
    other.load_state_dict(state)
    np.testing.assert_array_equal(other.buffer, est.buffer)
    assert other.count == 3


def test_an_update_without_samples_changes_nothing() -> None:
    """A step whose frames are all masked out gives no samples."""
    gauss, kde = RunningGaussian(2), KernelDensity(2)
    for est in (gauss, kde):
        est.update(np, np.empty((0, 2)))
    assert gauss.seen == 0 and gauss.shift is None
    assert kde.count == 0 and kde.buffer is None
    kde.update(np, np.ones((3, 2)))
    assert kde.count == 3


@pytest.mark.parametrize("mean, sd", [(1.6, 1e-3), (100.0, 1e-2)])
def test_the_kernel_density_is_accurate_in_float32_away_from_zero(
    mean: float, sd: float
) -> None:
    """A narrow distribution far from zero (a collapsing policy): x / h is large, and the estimate must not cancel."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(4)
    data, x = rng.normal(mean, sd, 2048), rng.normal(mean, sd, 64)
    est = KernelDensity(1, capacity=2048)
    est.update(torch, torch.tensor(data[:, None], dtype=torch.float32))
    got = est.log_prob(torch, torch.tensor(x[:, None], dtype=torch.float32)).numpy()
    np.testing.assert_allclose(got, stats.gaussian_kde(data).logpdf(x), atol=0.02)


def test_a_batch_larger_than_the_capacity_is_sampled_evenly() -> None:
    """The rows of one update are envs: the buffer must not hold the last envs alone."""
    est = KernelDensity(1, capacity=8)
    est.update(np, np.arange(20.0)[:, None])
    assert sorted(est.buffer[:, 0]) == [0.0, 2.0, 5.0, 7.0, 10.0, 12.0, 15.0, 17.0]
    assert est.count == 8
    est.update(np, np.arange(20.0)[:, None])  # the next update shifts by one env
    assert sorted(est.buffer[:, 0]) == [1.0, 3.0, 6.0, 8.0, 11.0, 13.0, 16.0, 18.0]


def test_the_kernel_density_updates_after_inference_mode() -> None:
    """RSL-RL collects in inference mode; a later update outside it must not fail on the buffer made there."""
    torch = pytest.importorskip("torch")
    est = KernelDensity(1, capacity=16)
    with torch.inference_mode():
        est.update(torch, torch.ones(4, 1))
    est.update(torch, torch.zeros(4, 1))
    assert est.count == 8 and float(est.buffer.sum()) == 4.0


@pytest.mark.parametrize(
    "make", [lambda **k: RunningGaussian(2, **k), lambda **k: KernelDensity(2, **k)]
)
def test_a_loaded_state_keeps_the_estimators_settings_and_moves_once(
    make: Any,
) -> None:
    torch = pytest.importorskip("torch")
    samples = np.random.default_rng(5).normal(size=(64, 2))
    trained = make()
    trained.update(torch, torch.as_tensor(samples))
    state = trained.state_dict()
    assert "frozen" not in state and "dim" not in state
    other = make(frozen=True)
    other.load_state_dict(state)
    assert other.frozen
    x = torch.as_tensor(samples[:5])
    np.testing.assert_allclose(
        other.log_prob(torch, x).numpy(), trained.log_prob(torch, x).numpy()
    )
    arrays = [v for v in vars(other).values() if hasattr(v, "shape")]
    assert arrays and all(isinstance(v, torch.Tensor) for v in arrays)


def test_a_state_of_another_capacity_is_refused() -> None:
    est = KernelDensity(1, capacity=8)
    est.update(np, np.ones((3, 1)))
    with pytest.raises(ValueError, match="buffer"):
        KernelDensity(1, capacity=16).load_state_dict(est.state_dict())
    with pytest.raises(KeyError, match="buffer"):
        KernelDensity(1, capacity=8).load_state_dict({"count": 0, "next": 0})


def test_too_few_minimum_samples_are_refused() -> None:
    with pytest.raises(ValueError, match="min_samples"):
        KernelDensity(1, min_samples=1)
    with pytest.raises(ValueError, match="min_samples"):
        RunningGaussian(1, min_samples=0)


@pytest.mark.parametrize(
    "make", [lambda: RunningGaussian(1), lambda: KernelDensity(1, capacity=8)]
)
def test_a_saved_state_is_a_copy(make: Any) -> None:
    """Neither a later update nor a loaded estimator's update may change a saved state."""
    est = make()
    est.update(np, np.array([[1.0], [2.0], [3.0]]))
    state = est.state_dict()
    frozen_copy = {
        k: np.array(v) if hasattr(v, "shape") else v for k, v in state.items()
    }
    est.update(np, np.array([[7.0], [8.0]]))
    other = make()
    other.load_state_dict(state)
    other.update(np, np.array([[9.0]]))
    for k, v in frozen_copy.items():
        np.testing.assert_array_equal(state[k], v)


def test_the_estimate_keeps_its_dtype_when_queried_in_another() -> None:
    torch = pytest.importorskip("torch")
    samples = torch.as_tensor(np.random.default_rng(6).normal(size=(64, 1)))  # float64
    gauss, kde = RunningGaussian(1), KernelDensity(1, capacity=64)
    for est in (gauss, kde):
        est.update(torch, samples)
        assert est.log_prob(torch, samples[:4].float()).dtype == torch.float32
    assert kde.buffer.dtype == torch.float64
    assert gauss.sum2.dtype == torch.float64
    for est in (
        gauss,
        kde,
    ):  # a later update in another dtype is cast to the stored one
        est.update(torch, samples[:8].float())
    assert kde.buffer.dtype == torch.float64
    assert gauss.sum2.dtype == torch.float64


def test_logsumexp_refuses_an_unknown_array_module() -> None:
    from types import SimpleNamespace

    from myosuite.utils.density_estimator import logsumexp

    with pytest.raises(TypeError, match="logsumexp"):
        logsumexp(SimpleNamespace(__name__="cupy"), np.zeros(2))


def test_the_gaussian_waits_for_a_full_rank_covariance() -> None:
    """One sample per step (a CPU env): no density until the covariance of d dimensions has d + 1 samples."""
    est = RunningGaussian(3)
    rng = np.random.default_rng(8)
    for k in range(4):
        assert est.log_prob(np, rng.normal(size=3)) == 0.0, k
        est.update(np, rng.normal(size=3))
    assert est.log_prob(np, rng.normal(size=3)) != 0.0


def test_a_memory_shorter_than_min_samples_is_refused() -> None:
    """decay 0.5 keeps about 2 samples' weight, too few for the covariance of 3 dimensions."""
    with pytest.raises(ValueError, match="fewer than min_samples"):
        RunningGaussian(3, decay=0.5)
    with pytest.raises(ValueError, match="fewer than min_samples"):
        RunningGaussian(1, decay=0.4, min_samples=2)
    RunningGaussian(3, decay=0.75)  # a memory of 4 samples is enough for 3 dimensions
