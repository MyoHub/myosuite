# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Rewards for matching a distribution of a movement feature.

The target distribution ``p*`` of a feature (for example the head's angular velocity) is
a Gaussian mixture, built from a mean and covariance or fitted to recorded samples.
:func:`distribution_matching_reward` pays ``log p*(x) - log q(x)``, where ``q`` is the
policy's own distribution of the feature, estimated online
(:mod:`myosuite.utils.density_estimator`). Its expectation under ``q`` is
``-KL(q || p*)``, so the return is largest when the policy's distribution of the feature
equals the target (state marginal matching; Lee et al. 2019, arXiv:1906.05274). Without
an estimator the reward is ``log p*(x)``, which is largest at the mode of ``p*``: the
policy then collapses onto the most likely movement.

A per-context target ``p*(x | c)`` takes one estimator per context, and the expectation
is then ``-E_c KL(q(x | c) || p*(x | c))``. One estimator pooled over the contexts would
pay ``log p*(x | c) - log q(x)``, whose expectation adds the mutual information
``I(x; c)`` and so rewards separating the contexts further than the target does; it is
refused.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from myosuite.terms._array import xp_asarray
from myosuite.utils.density_estimator import (
    LOG_2PI,
    DensityEstimator,
    logsumexp,
)

if TYPE_CHECKING:
    from myosuite.core.protocols import EnvAccessor


def _npz(path: str | Path) -> Path:
    """*path* with the ``.npz`` suffix numpy's ``savez`` adds when it is missing."""
    path = Path(path)
    return path if path.suffix == ".npz" else path.with_name(path.name + ".npz")


def _precision_cholesky(covariances: np.ndarray) -> np.ndarray:
    """scikit-learn's ``precisions_cholesky_``: ``P`` with ``P @ P.T`` the inverse of
    each ``(..., d, d)`` covariance."""
    return np.swapaxes(np.linalg.inv(np.linalg.cholesky(covariances)), -1, -2)


@dataclass(frozen=True, eq=False)
class GaussianMixtureSpec:
    """A Gaussian mixture over a feature vector of dimension ``d``, optionally one
    mixture per context.

    Build it with :meth:`from_gaussian`, :meth:`from_components` or
    :meth:`from_samples`, and one per context with :meth:`stack`.

    Attributes:
        weights: Component weights, ``(K,)``, or ``(C, K)`` with one mixture per
            context. Each mixture sums to 1.
        means: Component means, ``(K, d)`` or ``(C, K, d)``.
        precision_cholesky: ``P`` with ``P @ P.T`` the component's precision
            (scikit-learn's ``precisions_cholesky_``), ``(K, d, d)`` or
            ``(C, K, d, d)``.
        feature_names: Names of the ``d`` features with their units, or empty.
    """

    weights: np.ndarray
    means: np.ndarray
    precision_cholesky: np.ndarray
    feature_names: tuple[str, ...] = ()
    _cache: dict[tuple[str, ...], tuple[Any, Any, Any]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64)
        means = np.asarray(self.means, dtype=np.float64)
        prec = np.asarray(self.precision_cholesky, dtype=np.float64)
        d = means.shape[-1]
        if (
            means.shape != (*weights.shape, d)
            or prec.shape != (*weights.shape, d, d)
            or weights.ndim not in (1, 2)
        ):
            raise ValueError(
                f"shapes do not form a mixture: weights {weights.shape}, "
                f"means {means.shape}, precision {prec.shape}"
            )
        if (
            not np.allclose(prec, np.triu(prec))
            or not (np.diagonal(prec, axis1=-2, axis2=-1) > 0).all()
        ):
            raise ValueError(
                "precision_cholesky must be upper triangular with a positive diagonal "
                "(scikit-learn's precisions_cholesky_)"
            )
        if (weights < 0).any() or not np.allclose(weights.sum(-1), 1.0):
            raise ValueError("weights must be non-negative and sum to 1 per mixture")
        if self.feature_names and len(self.feature_names) != d:
            raise ValueError(
                f"{len(self.feature_names)} feature names for {d} features"
            )
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "precision_cholesky", prec)
        object.__setattr__(self, "feature_names", tuple(self.feature_names))

    @property
    def dim(self) -> int:
        """Feature dimension ``d``."""
        return int(self.means.shape[-1])

    @property
    def n_components(self) -> int:
        """Components per mixture ``K``."""
        return int(self.weights.shape[-1])

    @property
    def n_contexts(self) -> int | None:
        """Number of contexts ``C``, or ``None`` for a single mixture."""
        return int(self.weights.shape[0]) if self.weights.ndim == 2 else None

    @property
    def covariances(self) -> np.ndarray:
        """Component covariances, ``(K, d, d)`` or ``(C, K, d, d)``."""
        return np.linalg.inv(
            self.precision_cholesky @ np.swapaxes(self.precision_cholesky, -1, -2)
        )

    @classmethod
    def from_gaussian(
        cls,
        mean: Sequence[float] | np.ndarray,
        cov: Sequence[Sequence[float]] | np.ndarray,
        feature_names: Sequence[str] = (),
    ) -> GaussianMixtureSpec:
        """One Gaussian.

        Args:
            mean: Mean, ``(d,)``, or a number for one feature.
            cov: Covariance, ``(d, d)``, or a variance for one feature.
            feature_names: Names of the features with their units.

        Returns:
            The mixture of one component.

        Raises:
            ValueError: *cov* is not symmetric.
        """
        mean = np.atleast_1d(np.asarray(mean, dtype=np.float64))
        cov = np.asarray(cov, dtype=np.float64).reshape(mean.size, mean.size)
        return cls.from_components([1.0], mean[None], cov[None], feature_names)

    @classmethod
    def from_components(
        cls,
        weights: Sequence[float] | np.ndarray,
        means: np.ndarray,
        covariances: np.ndarray,
        feature_names: Sequence[str] = (),
    ) -> GaussianMixtureSpec:
        """A mixture of given components.

        Args:
            weights: Component weights, ``(K,)``, summing to 1.
            means: Component means, ``(K, d)``.
            covariances: Component covariances, ``(K, d, d)``, symmetric and positive
                definite.
            feature_names: Names of the features with their units.

        Returns:
            The mixture.

        Raises:
            ValueError: A covariance is not symmetric, or the arrays do not form a
                mixture.
        """
        covariances = np.asarray(covariances, dtype=np.float64)
        # the Cholesky factor reads only the lower triangle, so an asymmetric typo would
        # pass silently
        if not np.allclose(covariances, np.swapaxes(covariances, -1, -2)):
            raise ValueError("covariances must be symmetric")
        return cls(
            np.asarray(weights),
            np.asarray(means),
            _precision_cholesky(covariances),
            tuple(feature_names),
        )

    @classmethod
    def from_samples(
        cls,
        samples: np.ndarray,
        n_components: int | str = "bic",
        max_components: int = 8,
        seed: int = 0,
        reg_covar: float = 1e-6,
        feature_names: Sequence[str] = (),
    ) -> GaussianMixtureSpec:
        """A mixture fitted to recorded *samples* ``(n, d)`` with
        :class:`sklearn.mixture.GaussianMixture`.

        Args:
            samples: Recorded feature vectors, ``(n, d)`` or ``(n,)`` for one feature.
            n_components: Number of components, or ``"bic"`` for the number from 1 to
                *max_components* with the lowest Bayesian information criterion.
            max_components: Largest number of components tried with ``"bic"``.
            seed: Seed of the fit's initialisation.
            reg_covar: Added to each covariance's diagonal.
            feature_names: Names of the features with their units.

        Returns:
            The fitted mixture.

        Raises:
            ValueError: A given *n_components* exceeds the number of samples.
        """
        from sklearn.mixture import GaussianMixture

        x = np.asarray(samples, dtype=np.float64)
        x = x[:, None] if x.ndim == 1 else x
        # BIC tries at most one component per sample
        counts = (
            range(1, min(max_components, len(x)) + 1)
            if n_components == "bic"
            else [int(n_components)]
        )
        fits = [
            GaussianMixture(
                k, covariance_type="full", reg_covar=reg_covar, random_state=seed
            ).fit(x)
            for k in counts
        ]
        best = min(fits, key=lambda g: g.bic(x))
        return cls(
            best.weights_, best.means_, best.precisions_cholesky_, tuple(feature_names)
        )

    @classmethod
    def stack(cls, specs: Sequence[GaussianMixtureSpec]) -> GaussianMixtureSpec:
        """One mixture per context.

        Args:
            specs: Single mixtures of one feature dimension, in the order of the context
                index; mixtures with fewer components get zero-weight ones.

        Returns:
            The per-context mixture, ``n_contexts == len(specs)``.

        Raises:
            ValueError: *specs* is empty, holds a per-context mixture, mixes feature
                dimensions, or names different features.
        """
        if not specs or any(
            s.n_contexts is not None or s.dim != specs[0].dim for s in specs
        ):
            raise ValueError("stack takes single mixtures of one feature dimension")
        names = {s.feature_names for s in specs if s.feature_names}
        if len(names) > 1:
            raise ValueError(
                f"stack takes mixtures of the same features, got {sorted(names)}"
            )
        k, d = max(s.n_components for s in specs), specs[0].dim
        weights = np.zeros((len(specs), k))
        means = np.zeros((len(specs), k, d))
        prec = np.broadcast_to(np.eye(d), (len(specs), k, d, d)).copy()
        for c, s in enumerate(specs):
            weights[c, : s.n_components] = s.weights
            means[c, : s.n_components] = s.means
            prec[c, : s.n_components] = s.precision_cholesky
        return cls(weights, means, prec, next(iter(names), ()))

    def save(self, path: str | Path) -> None:
        """Write the mixture to *path* (``.npz``, added when missing)."""
        np.savez(
            _npz(path),
            weights=self.weights,
            means=self.means,
            precision_cholesky=self.precision_cholesky,
            feature_names=np.array(self.feature_names, dtype=str),
        )

    @classmethod
    def load(cls, path: str | Path) -> GaussianMixtureSpec:
        """Read a mixture written by :meth:`save`, without unpickling.

        Args:
            path: The ``.npz`` file; the suffix is added when missing.

        Returns:
            The mixture.

        Raises:
            ValueError: The file holds a pickled array, or the arrays do not form a
                mixture.
        """
        from myosuite.core.trajectory_io import read_npz_names

        with np.load(_npz(path), allow_pickle=False) as f:
            return cls(
                f["weights"],
                f["means"],
                f["precision_cholesky"],
                tuple(read_npz_names(f, "feature_names")),
            )

    def sample(
        self, n: int, rng: np.random.Generator, context: int | None = None
    ) -> np.ndarray:
        """Samples of the mixture.

        Args:
            n: Number of samples.
            rng: The random generator.
            context: The context index, for a per-context spec.

        Returns:
            Samples, ``(n, d)``.

        Raises:
            ValueError: A per-context spec without *context*.
        """
        weights, means, covs = self.weights, self.means, self.covariances
        if self.n_contexts is not None:
            if context is None:
                raise ValueError("a per-context spec needs the context")
            weights, means, covs = weights[context], means[context], covs[context]
        which = rng.choice(self.n_components, size=n, p=weights)
        chol = np.linalg.cholesky(covs)
        return means[which] + np.einsum(
            "nde,ne->nd", chol[which], rng.standard_normal((n, self.dim))
        )

    def backend_arrays(self, xp: Any, like: Any) -> tuple[Any, Any, Any]:
        """The arrays of the log density, cached per backend, device and dtype.

        Args:
            xp: The array module.
            like: An array of the backend, device and dtype to return.

        Returns:
            The precisions' Cholesky factors, ``means @ P`` and each component's log
            normaliser.
        """
        if ("numpy",) not in self._cache:
            # -inf for the zero-weight padding of stack
            with np.errstate(divide="ignore"):
                log_weights = np.log(self.weights)
            log_det = np.log(
                np.diagonal(self.precision_cholesky, axis1=-2, axis2=-1)
            ).sum(-1)
            self._cache[("numpy",)] = (
                self.precision_cholesky,
                np.einsum("...kd,...kde->...ke", self.means, self.precision_cholesky),
                log_weights + log_det - 0.5 * self.dim * LOG_2PI,
            )
        arrays = self._cache[("numpy",)]
        if getattr(xp, "__name__", "") == "numpy":
            return arrays
        key = (
            getattr(xp, "__name__", ""),
            str(getattr(like, "device", "")),
            str(like.dtype),
        )
        if key not in self._cache:
            self._cache[key] = tuple(
                xp_asarray(
                    xp, a, dtype=like.dtype, device=getattr(like, "device", None)
                )
                for a in arrays
            )
        return self._cache[key]


def mixture_log_density(
    xp: Any, x: Any, spec: GaussianMixtureSpec, context: Any = None
) -> Any:
    """Log density of the features *x* under *spec*.

    Args:
        xp: The array module (``accessor.array_module()``).
        x: Features, ``(d,)`` or ``(N, d)``.
        spec: The mixture.
        context: For a per-context spec, the context index: an int, or ``(N,)`` integers
            in the backend of *x*.

    Returns:
        Log densities, ``()`` or ``(N,)``.
    """
    prec, offsets, log_norm = spec.backend_arrays(xp, x)
    if spec.n_contexts is not None:
        if context is None:
            raise ValueError("a per-context spec needs the context")
        prec, offsets, log_norm = prec[context], offsets[context], log_norm[context]
    elif context is not None:
        raise ValueError("context given for a spec without contexts")
    y = xp.einsum("...d,...kde->...ke", x, prec) - offsets
    return logsumexp(xp, log_norm - 0.5 * (y * y).sum(-1))


Estimators = DensityEstimator | Sequence[DensityEstimator]


def _check_estimators(
    target: GaussianMixtureSpec, estimator: Estimators | None
) -> None:
    """One estimator for a single target, one per context for a per-context target (see
    the module docstring)."""
    if estimator is None:
        return
    if not isinstance(estimator, Sequence):
        if target.n_contexts is not None:
            raise ValueError(
                "a per-context target needs one estimator per context: "
                "one pooled estimator pays log p*(x|c) - log q(x), "
                "which also rewards separating the contexts"
            )
        return
    if target.n_contexts is None:
        raise ValueError("estimators per context need a per-context target")
    if len(estimator) != target.n_contexts:
        raise ValueError(
            f"{len(estimator)} estimators for {target.n_contexts} contexts"
        )


def _log_q(xp: Any, x: Any, estimator: Estimators, context: Any) -> Any:
    """``log q(x)``, or ``log q(x | c)`` from each frame's own context's estimator."""
    if not isinstance(estimator, Sequence):
        return estimator.log_prob(xp, x)
    if getattr(context, "ndim", 0) == 0:  # one frame
        return estimator[int(context)].log_prob(xp, x)
    log_q = xp.zeros_like(x[..., 0])
    # a few contexts, each scoring the whole batch: no row selection
    for k, est in enumerate(estimator):
        log_q = xp.where(context == k, est.log_prob(xp, x), log_q)
    return log_q


def density_ratio(
    xp: Any,
    x: Any,
    target: GaussianMixtureSpec,
    estimator: Estimators | None = None,
    *,
    context: Any = None,
    clip: float = 10.0,
) -> dict[str, Any]:
    """``log p*(x) - log q(x)``, clipped to ``[-clip, clip]``, without updating the
    estimator.

    Args:
        xp: The array module.
        x: Features, ``(d,)`` or ``(N, d)``.
        target: The target mixture ``p*``.
        estimator: The policy's own distribution ``q``, one per context for a
            per-context target; ``None`` pays ``log p*`` alone.
        context: The context index of a per-context target.
        clip: Bound of the reward's magnitude.

    Returns:
        ``dense`` (the clipped difference), ``log_p`` and ``log_q``, each ``()`` or
        ``(N,)``.
    """
    _check_estimators(target, estimator)
    log_p = mixture_log_density(xp, x, target, context)
    log_q = (
        xp.zeros_like(log_p) if estimator is None else _log_q(xp, x, estimator, context)
    )
    return {
        "dense": xp.clip(log_p - log_q, -clip, clip),
        "log_p": log_p,
        "log_q": log_q,
    }


@dataclass
class DistributionMatchingCfg:
    """What :func:`distribution_matching_reward` matches.

    Attributes:
        feature: ``feature(accessor, task_state)`` returns the feature ``(d,)`` or
            ``(N, d)``: from the state (joints, muscles, site positions, orientations
            and angular velocities) and from the task's state (for example a target, for
            a feature relative to it).
        target: The target mixture ``p*``.
        estimator: The policy's own distribution ``q``, updated by each reward call
            while not frozen; for a per-context target a sequence of one estimator per
            context, each learning from its context's frames. ``None`` pays ``log p*``
            alone, which favours the most likely movement over the spread.
        context_key: Key of ``task_state`` holding the context index of a per-context
            target.
        clip: Bound of the reward's magnitude.
        mask_key: Key of ``task_state`` holding a boolean ``(N,)`` of the frames that
            count, for a target fitted to some frames only (for example those inside a
            trial): the others are paid 0 and the estimator does not learn from them.
            Selecting the rows syncs the device once per call. Estimators per context
            select their context's rows with or without a mask: once per context.
    """

    feature: Callable[[EnvAccessor, dict[str, Any]], Any]
    target: GaussianMixtureSpec
    estimator: Estimators | None = None
    context_key: str | None = None
    clip: float = 10.0
    mask_key: str | None = None

    def __post_init__(self) -> None:
        _check_estimators(self.target, self.estimator)
        if (self.target.n_contexts is None) != (self.context_key is None):
            raise ValueError(
                "context_key is needed for a per-context target, and only for one"
            )


def distribution_matching_reward(
    accessor: EnvAccessor,
    task_state: dict[str, Any],
    *,
    distribution: DistributionMatchingCfg,
    **kwargs: Any,
) -> dict[str, Any]:
    """Reward for a feature distributed as the target: ``log p*(x) - log q(x)`` (see the
    module docstring).

    The estimator scores the frame before it learns from it. With an estimator that is
    not frozen the term therefore has a side effect, the only one among the terms: call
    it once per step (:func:`density_ratio` scores without learning).

    A frame whose feature is not finite (a diverged env, which mjlab scores before it
    resets it) is paid 0 and not learnt. One estimator without a mask learns the batch
    whole, so it learns a copy of a finite frame of the step in that frame's place, or a
    zero frame when no frame is finite: a negligible bias that keeps the term free of
    device syncs. Estimators per context select their context's rows, which syncs the
    device once per context per call.

    Args:
        accessor: The env's state.
        task_state: The task's state, passed to the feature; holds the context index
            under ``distribution.context_key`` and the frames that count under
            ``distribution.mask_key``.
        distribution: The feature, target and estimator.
        **kwargs: Ignored keyword arguments of other reward terms.

    Returns:
        ``dense``, ``log_p``, ``log_q``, and ``solved`` and ``done``, both false.
    """
    xp = accessor.array_module()
    x = distribution.feature(accessor, task_state)
    context: Any = (
        task_state[distribution.context_key] if distribution.context_key else None
    )
    out = density_ratio(
        xp,
        x,
        distribution.target,
        distribution.estimator,
        context=context,
        clip=distribution.clip,
    )
    mask = task_state[distribution.mask_key] if distribution.mask_key else None
    finite = xp.isfinite(x).all(-1)  # a diverged env's frame is neither paid nor learnt
    counted = finite if mask is None else mask & finite
    out["dense"] = xp.where(counted, out["dense"], xp.zeros_like(out["dense"]))
    estimator = distribution.estimator
    if estimator is None:
        pass
    elif x.ndim == 1:  # one frame
        one = estimator[int(context)] if isinstance(estimator, Sequence) else estimator
        one.update(xp, x[counted])
    elif isinstance(estimator, Sequence):
        # each context's estimator learns its own frames
        for k, est in enumerate(estimator):
            est.update(xp, x[(context == k) & counted])
    elif mask is not None:
        estimator.update(xp, x[counted])
    else:
        # the whole batch, without selecting rows (a device sync): a non-finite frame is
        # replaced by a copy of a finite one (zeros when none is finite), so that it
        # cannot poison the estimator
        fill = xp.nan_to_num(x[xp.argmax(finite * 1)], nan=0.0, posinf=0.0, neginf=0.0)
        estimator.update(xp, xp.where(finite[:, None], x, fill))
    false = xp.zeros_like(out["dense"], dtype=bool)
    return {**out, "solved": false, "done": false}
