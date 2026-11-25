# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The distribution-matching reward (``myosuite.terms.distribution_reward``)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scipy import stats
from sklearn.mixture import GaussianMixture

from myosuite.terms.distribution_reward import (
    DistributionMatchingCfg,
    GaussianMixtureSpec,
    density_ratio,
    distribution_matching_reward,
    mixture_log_density,
)
from myosuite.utils.density_estimator import KernelDensity, RunningGaussian

pytestmark = pytest.mark.tier1

MEAN = np.array([0.3, -1.0])
COV = np.array([[0.5, 0.2], [0.2, 0.8]])


def _bimodal(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [rng.normal(-2.0, 0.4, (n // 2, 2)), rng.normal(2.0, 0.6, (n - n // 2, 2))]
    )


def test_a_gaussian_matches_scipy() -> None:
    spec = GaussianMixtureSpec.from_gaussian(MEAN, COV)
    x = np.random.default_rng(1).normal(size=(50, 2))
    expected = stats.multivariate_normal(MEAN, COV).logpdf(x)
    np.testing.assert_allclose(
        mixture_log_density(np, x, spec), expected, rtol=0, atol=1e-10
    )
    assert np.ndim(mixture_log_density(np, x[0], spec)) == 0  # one env: a scalar
    np.testing.assert_allclose(spec.covariances[0], COV, atol=1e-12)


def test_a_fitted_mixture_matches_scikit_learn() -> None:
    samples = _bimodal(2000)
    spec = GaussianMixtureSpec.from_samples(samples, n_components=2, seed=3)
    fitted = GaussianMixture(
        2, covariance_type="full", reg_covar=1e-6, random_state=3
    ).fit(samples)
    x = np.random.default_rng(2).normal(scale=3.0, size=(100, 2))
    np.testing.assert_allclose(
        mixture_log_density(np, x, spec), fitted.score_samples(x), atol=1e-9
    )


def test_bic_finds_the_two_modes() -> None:
    assert (
        GaussianMixtureSpec.from_samples(_bimodal(2000), max_components=4).n_components
        == 2
    )


def test_bic_tries_no_more_components_than_samples() -> None:
    samples = np.random.default_rng(0).normal(size=5)
    assert 1 <= GaussianMixtureSpec.from_samples(samples).n_components <= 5


def test_torch_matches_numpy() -> None:
    torch = pytest.importorskip("torch")
    spec = GaussianMixtureSpec.from_samples(_bimodal(1000), n_components=2)
    x = np.random.default_rng(3).normal(scale=2.0, size=(64, 2))
    expected = mixture_log_density(np, x, spec)
    got64 = mixture_log_density(torch, torch.as_tensor(x, dtype=torch.float64), spec)
    got32 = mixture_log_density(torch, torch.as_tensor(x, dtype=torch.float32), spec)
    np.testing.assert_allclose(got64.numpy(), expected, atol=1e-10)
    np.testing.assert_allclose(got32.numpy(), expected, atol=1e-4)


def test_each_context_has_its_own_mixture() -> None:
    wide = GaussianMixtureSpec.from_gaussian(MEAN, COV)
    two = GaussianMixtureSpec.from_samples(_bimodal(1000), n_components=2)
    stacked = GaussianMixtureSpec.stack([wide, two])
    assert stacked.n_contexts == 2 and stacked.n_components == 2
    x = np.random.default_rng(4).normal(size=(6, 2))
    context = np.array([0, 1, 0, 1, 1, 0])
    expected = np.where(
        context == 0, mixture_log_density(np, x, wide), mixture_log_density(np, x, two)
    )
    np.testing.assert_allclose(
        mixture_log_density(np, x, stacked, context), expected, atol=1e-10
    )
    np.testing.assert_allclose(
        mixture_log_density(np, x[0], stacked, 0), expected[0], atol=1e-10
    )
    with pytest.raises(ValueError, match="context"):
        mixture_log_density(np, x, stacked)


def test_save_and_load_round_trip(tmp_path: Any) -> None:
    spec = GaussianMixtureSpec.from_gaussian(
        MEAN, COV, feature_names=("yaw rate (deg/s)", "pitch rate (deg/s)")
    )
    spec.save(tmp_path / "head.npz")
    loaded = GaussianMixtureSpec.load(tmp_path / "head.npz")
    assert loaded.feature_names == spec.feature_names
    x = np.random.default_rng(5).normal(size=(10, 2))
    np.testing.assert_array_equal(
        mixture_log_density(np, x, loaded), mixture_log_density(np, x, spec)
    )


def test_invalid_mixtures_are_refused() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        GaussianMixtureSpec(np.array([0.5, 0.2]), np.zeros((2, 1)), np.ones((2, 1, 1)))
    with pytest.raises(ValueError, match="shapes"):
        GaussianMixtureSpec(np.array([1.0]), np.zeros((1, 2)), np.ones((1, 1, 1)))
    with pytest.raises(ValueError, match="upper triangular"):
        GaussianMixtureSpec(
            np.array([1.0]), np.zeros((1, 2)), np.array([[[1.0, 0.0], [0.5, 1.0]]])
        )
    with pytest.raises(ValueError, match="positive diagonal"):
        GaussianMixtureSpec(np.array([1.0]), np.zeros((1, 1)), -np.ones((1, 1, 1)))
    with pytest.raises(ValueError, match="symmetric"):
        GaussianMixtureSpec.from_gaussian([0.0, 0.0], [[1.0, 0.5], [0.0, 1.0]])
    yaw = GaussianMixtureSpec.from_gaussian([0.0], [[1.0]], feature_names=("yaw rate",))
    pitch = GaussianMixtureSpec.from_gaussian(
        [0.0], [[1.0]], feature_names=("pitch rate",)
    )
    unnamed = GaussianMixtureSpec.from_gaussian([0.0], [[1.0]])
    with pytest.raises(ValueError, match="same features"):
        GaussianMixtureSpec.stack([yaw, pitch])
    assert GaussianMixtureSpec.stack([unnamed, yaw]).feature_names == ("yaw rate",)


def test_the_mean_reward_is_minus_the_kl_divergence() -> None:
    """Under the policy's distribution q the mean of ``log p* - log q`` is ``-KL(q || p*)``, 0 only at q = p*."""
    rng = np.random.default_rng(10)
    target = GaussianMixtureSpec.from_gaussian([0.0], [[1.0]])
    for sd in (0.2, 1.0, 2.0):
        x = rng.normal(0.0, sd, (200_000, 1))
        q = RunningGaussian(1, decay=1.0, reg=0.0)
        q.update(np, x)
        reward = density_ratio(np, x, target, q, clip=1e9)["dense"].mean()
        kl = np.log(1.0 / sd) + (sd**2) / 2.0 - 0.5
        assert reward == pytest.approx(-kl, abs=0.01)
    # without the estimator the narrow distribution wins: log p* alone rewards the mode, not the spread
    narrow, matched = (
        rng.normal(0.0, 0.2, (100_000, 1)),
        rng.normal(0.0, 1.0, (100_000, 1)),
    )
    assert (
        density_ratio(np, narrow, target)["dense"].mean()
        > density_ratio(np, matched, target)["dense"].mean()
    )


class _Accessor:
    def __init__(self, x: np.ndarray) -> None:
        self.x = x

    def array_module(self) -> Any:
        return np


def test_the_term_scores_before_it_learns() -> None:
    target = GaussianMixtureSpec.stack(
        [
            GaussianMixtureSpec.from_gaussian([0.0], [[1.0]]),
            GaussianMixtureSpec.from_gaussian([5.0], [[1.0]]),
        ]
    )
    ests = [RunningGaussian(1, min_samples=1) for _ in range(2)]
    cfg = DistributionMatchingCfg(
        feature=lambda a, state: a.x,
        target=target,
        estimator=ests,
        context_key="phase",
        clip=50.0,
    )
    x = np.array([[5.0], [0.0], [5.0]])
    out = distribution_matching_reward(
        _Accessor(x), {"phase": np.array([1, 0, 1])}, distribution=cfg, pose_thd=0.1
    )
    np.testing.assert_array_equal(
        out["log_q"], np.zeros(3)
    )  # nothing seen before this frame
    np.testing.assert_allclose(out["log_p"], np.full(3, -0.5 * np.log(2 * np.pi)))
    assert [e.seen for e in ests] == [1, 2]  # each context learns from its own frames
    assert not out["solved"].any() and not out["done"].any()


def test_a_non_finite_frame_is_paid_nothing_and_does_not_poison_the_estimator() -> None:
    """A diverged env's NaN frame pays 0; the estimator stays finite on every update path."""
    target = GaussianMixtureSpec.from_gaussian([0.0], [[1.0]])
    x = np.array([[0.5], [np.nan], [-1.0]])

    def run(xp: Any, frames: Any, state: dict[str, Any], **kwargs: Any) -> Any:
        accessor = SimpleNamespace(x=frames, array_module=lambda: xp)
        cfg = DistributionMatchingCfg(
            feature=lambda a, s: a.x, **{"target": target, **kwargs}
        )
        return distribution_matching_reward(accessor, state, distribution=cfg)["dense"]

    whole = RunningGaussian(1, decay=1.0, min_samples=1)
    dense = run(np, x, {}, estimator=whole)
    assert dense[1] == 0.0 and np.isfinite(dense).all()
    mean, _ = whole.moments(np, x[:1])
    assert whole.seen == 3 and mean[0] == pytest.approx(
        0.0
    )  # learnt 0.5, 0.5 (in NaN's place), -1
    masked = RunningGaussian(1, min_samples=1)
    run(np, x, {"m": np.array([True, True, False])}, estimator=masked, mask_key="m")
    assert masked.seen == 1  # the NaN frame and the masked-out frame are not learnt
    stacked = GaussianMixtureSpec.stack([target, target])
    per_context = [RunningGaussian(1, min_samples=1) for _ in range(2)]
    run(
        np,
        x,
        {"c": np.array([0, 0, 1])},
        target=stacked,
        estimator=per_context,
        context_key="c",
    )
    assert [e.seen for e in per_context] == [1, 1]
    one = RunningGaussian(1, min_samples=1)
    assert run(np, x[1], {}, estimator=one) == 0.0 and one.seen == 0
    alone = RunningGaussian(1, min_samples=1)  # one env, diverged: a zero frame instead
    run(np, x[1:2], {}, estimator=alone)
    assert np.isfinite(run(np, x[:1], {}, estimator=alone)).all()
    torch = pytest.importorskip("torch")
    kde = KernelDensity(1)
    dense_t = run(torch, torch.as_tensor(x), {}, estimator=kde)
    assert dense_t[1] == 0.0 and torch.isfinite(dense_t).all() and kde.count == 3
    assert torch.isfinite(kde.buffer[:3]).all()


def test_each_context_has_its_own_estimator() -> None:
    """Per context the mean reward is ``-KL(q(x|c) || p*(x|c))``; one estimator pooled over contexts is refused."""
    target = GaussianMixtureSpec.stack(
        [
            GaussianMixtureSpec.from_gaussian([0.0], [[1.0]]),
            GaussianMixtureSpec.from_gaussian([3.0], [[1.0]]),
        ]
    )
    rng = np.random.default_rng(11)
    sds, n = (0.5, 1.5), 100_000
    context = np.repeat([0, 1], n)
    x = np.concatenate([rng.normal(0.0, sds[0], n), rng.normal(3.0, sds[1], n)])[
        :, None
    ]
    q = [RunningGaussian(1, decay=1.0, reg=0.0) for _ in sds]
    for k, est in enumerate(q):
        est.update(np, x[context == k])
    dense = density_ratio(np, x, target, q, context=context, clip=1e9)["dense"]
    for k, sd in enumerate(sds):
        kl = np.log(1.0 / sd) + sd**2 / 2.0 - 0.5
        assert dense[context == k].mean() == pytest.approx(-kl, abs=0.01)
    torch = pytest.importorskip("torch")
    dense_t = density_ratio(
        torch, torch.as_tensor(x), target, q, context=torch.as_tensor(context), clip=1e9
    )["dense"]
    np.testing.assert_allclose(dense_t.numpy(), dense, atol=1e-6)
    with pytest.raises(ValueError, match="one estimator per context"):
        density_ratio(np, x, target, RunningGaussian(1), context=context)
    with pytest.raises(ValueError, match="1 estimators for 2 contexts"):
        DistributionMatchingCfg(
            feature=lambda a, s: a.x, target=target, estimator=q[:1], context_key="c"
        )
    with pytest.raises(ValueError, match="per-context target"):
        density_ratio(np, x, GaussianMixtureSpec.from_gaussian([0.0], [[1.0]]), q)


def test_masked_frames_are_paid_nothing_and_not_learned() -> None:
    target = GaussianMixtureSpec.from_gaussian([0.0], [[1.0]])
    kde = KernelDensity(1)
    cfg = DistributionMatchingCfg(
        feature=lambda a, state: a.x, target=target, estimator=kde, mask_key="counted"
    )
    x = np.array([[0.5], [3.0], [-1.0]])
    counted = np.array([True, False, True])
    out = distribution_matching_reward(
        _Accessor(x), {"counted": counted}, distribution=cfg
    )
    assert out["dense"][1] == 0.0 and out["dense"][0] != 0.0
    assert out["log_p"][1] == pytest.approx(stats.norm.logpdf(3.0))  # still reported
    assert kde.count == 2
    np.testing.assert_array_equal(kde.buffer[:2], x[counted])
    # no counted frame: nothing paid, nothing learned
    out = distribution_matching_reward(
        _Accessor(x), {"counted": np.zeros(3, dtype=bool)}, distribution=cfg
    )
    assert not out["dense"].any() and kde.count == 2


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_a_partial_mask_on_torch_matches_numpy(device: str) -> None:
    """The batched path of an mjlab term: some frames of each step masked out, one step entirely, on torch (CPU and
    GPU) as on numpy, with the estimator learning only the counted frames."""
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA device")

    class _XpAccessor:
        def __init__(self, x: Any, xp: Any) -> None:
            self.x, self.xp = x, xp

        def array_module(self) -> Any:
            return self.xp

    target = GaussianMixtureSpec.from_gaussian([0.0, 1.0], [[1.0, 0.2], [0.2, 2.0]])
    rng = np.random.default_rng(3)
    steps = [rng.normal(size=(8, 2)) for _ in range(4)]
    masks = [rng.random(8) < 0.5 for _ in range(4)]
    masks[2][:] = False
    masks[0][:2] = True  # the estimator has two samples after the first step

    def run(xp: Any, convert: Any) -> tuple[list[np.ndarray], int]:
        kde = KernelDensity(2, min_samples=2)
        cfg = DistributionMatchingCfg(
            feature=lambda a, state: a.x, target=target, estimator=kde, mask_key="m"
        )
        dense = []
        for x, m in zip(steps, masks):
            out = distribution_matching_reward(
                _XpAccessor(convert(x), xp), {"m": convert(m)}, distribution=cfg
            )
            dense.append(
                np.asarray(out["dense"].cpu() if xp is torch else out["dense"])
            )
        return dense, kde.count

    expected, count = run(np, lambda a: a)
    got, torch_count = run(torch, lambda a: torch.as_tensor(a, device=device))
    assert count == torch_count == sum(int(m.sum()) for m in masks)
    for e, g, m in zip(expected, got, masks):
        np.testing.assert_allclose(g, e, atol=1e-10)
        assert not g[~m].any()
    assert (
        np.abs(expected[3][masks[3]]).sum() > 0
    )  # the estimator corrects the later steps


def test_save_adds_the_suffix_that_load_reads(tmp_path: Any) -> None:
    spec = GaussianMixtureSpec.from_gaussian([0.5], [[0.04]])
    spec.save(tmp_path / "target")
    loaded = GaussianMixtureSpec.load(tmp_path / "target")
    np.testing.assert_array_equal(loaded.means, spec.means)
    assert (tmp_path / "target.npz").is_file()
