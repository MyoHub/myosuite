# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Online estimates of a policy's own distribution of a feature, for a reward that
depends on it.

The estimators run on numpy and torch arrays, on the device of the data. The kernel
density writes its buffer in place, which JAX arrays do not allow. They keep their state
in the backend and dtype of the first update. They learn while not frozen;
:meth:`freeze` fixes them for evaluation, so a frozen estimator gives the same reward
after every ``reset(seed)``.
"""

from __future__ import annotations

import math
from typing import Any, Protocol

import numpy as np

from myosuite.terms._array import xp_asarray

LOG_2PI = math.log(2.0 * math.pi)


def logsumexp(xp: Any, a: Any) -> Any:
    """``log(sum(exp(a)))`` over the last axis, with each backend's library function.

    Args:
        xp: The array module (numpy, torch or jax.numpy).
        a: Values, ``(..., K)``.

    Returns:
        ``(...)`` values.

    Raises:
        TypeError: *xp* is none of the three.
    """
    name = getattr(xp, "__name__", "")
    if _is_torch(xp):
        return xp.logsumexp(a, dim=-1)
    if name == "numpy":
        from scipy.special import logsumexp as scipy_logsumexp

        return scipy_logsumexp(a, axis=-1)
    if name.startswith("jax"):
        from jax.scipy.special import logsumexp as jax_logsumexp

        return jax_logsumexp(a, axis=-1)
    raise TypeError(f"no logsumexp for the array module {name or xp!r}")


def _filled(xp: Any, shape: tuple[int, ...], like: Any, eye: bool = False) -> Any:
    """Zeros of *shape* (the identity with *eye*) in the backend of *like*, built on its
    device."""
    make = xp.eye if eye else xp.zeros
    size = shape[0] if eye else shape
    if _is_torch(xp):
        return make(size, dtype=like.dtype, device=getattr(like, "device", None))
    return make(size, dtype=like.dtype)


def _on_device(xp: Any, value: Any, like: Any) -> Any:
    """*value* in the backend (and, for torch, on the device) of *like*, keeping its own
    dtype."""
    if _is_torch(xp):
        return xp_asarray(xp, value, device=like.device)
    return value if isinstance(value, np.ndarray) else _to_numpy(value)


def _to_numpy(value: Any) -> np.ndarray:
    """A numpy copy of *value*, which later in-place updates of the estimate leave
    unchanged.

    ``model_bridge._to_numpy_array`` may return a view.
    """
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().copy()
    return np.array(value)


class DensityEstimator(Protocol):
    """An online estimate of a feature's density."""

    frozen: bool

    def log_prob(self, xp: Any, x: Any) -> Any:
        """Log density of ``x`` (``(d,)`` or ``(N, d)``), ``()`` or ``(N,)``."""
        ...

    def update(self, xp: Any, x: Any) -> None:
        """Add the samples ``x`` (``(d,)`` or ``(N, d)``); a no-op while frozen."""
        ...


class _Online:
    """Freezing and state handling shared by the estimators."""

    frozen: bool
    _STATE: tuple[str, ...] = ()

    def freeze(self) -> None:
        """Stop learning: :meth:`update` no longer changes the estimate."""
        self.frozen = True

    def unfreeze(self) -> None:
        """Learn again from :meth:`update`."""
        self.frozen = False

    def state_dict(self) -> dict[str, Any]:
        """The learnt estimate as numpy arrays and numbers, for a checkpoint (without
        the settings and ``frozen``)."""
        state = {k: getattr(self, k) for k in self._STATE}
        return {
            k: _to_numpy(v) if v is not None and not isinstance(v, (int, float)) else v
            for k, v in state.items()
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the learnt estimate from :meth:`state_dict`.

        The settings and ``frozen`` stay this estimator's; the arrays move to the
        backend of the next call, once.
        """
        missing = [k for k in self._STATE if k not in state]
        if missing:
            raise KeyError(f"state lacks {missing}")
        # copies: an update must not write into the caller's arrays
        for k in self._STATE:
            v = state[k]
            setattr(self, k, _to_numpy(v) if hasattr(v, "shape") else v)


class RunningGaussian(_Online):
    """A Gaussian with exponentially weighted moments of the samples seen so far.

    Each new sample multiplies the weight of the earlier ones by ``decay``, so the
    estimate covers about ``1 / (1 - decay)`` recent samples. An update of ``n`` samples
    (``n`` envs) keeps ``decay ** n`` of the earlier weight: at 0.999 and 4096 envs that
    is 0.017, so the estimate is the last step's batch. For a memory of ``s`` steps of
    ``n`` envs choose ``decay = 1 - 1 / (s * n)``. The moments are kept relative to the
    first batch's mean, which keeps the covariance accurate in float32 while the mean
    stays near it; a feature that drifts far from its first batch is better estimated in
    float64.

    Args:
        dim: Feature dimension ``d``.
        decay: Weight kept by the earlier samples per new sample, in ``(0, 1]``; 1
            weights all samples equally.
        reg: Added to the covariance's diagonal.
        min_samples: :meth:`log_prob` is 0 (no correction) until this many samples were
            seen; at least 1. The default ``dim + 1`` is the fewest samples whose
            covariance has full rank. A ``decay`` whose memory ``1 / (1 - decay)`` is
            shorter is refused: the estimate would rest on fewer samples than this.
        frozen: Start frozen.
    """

    _STATE = ("seen", "weight", "shift", "sum1", "sum2")

    def __init__(
        self,
        dim: int,
        decay: float = 0.999,
        reg: float = 1e-6,
        min_samples: int | None = None,
        frozen: bool = False,
    ) -> None:
        min_samples = dim + 1 if min_samples is None else min_samples
        if not 0.0 < decay <= 1.0:
            raise ValueError(f"decay must lie in (0, 1], got {decay}")
        if min_samples < 1:
            raise ValueError(f"min_samples must be at least 1, got {min_samples}")
        if decay < 1.0 and 1.0 / (1.0 - decay) < min_samples:
            raise ValueError(
                f"decay {decay} keeps about {1.0 / (1.0 - decay):.3g} samples' weight, "
                f"fewer than min_samples = "
                f"{min_samples}; raise decay or lower min_samples"
            )
        self.dim, self.decay, self.reg, self.min_samples, self.frozen = (
            dim,
            decay,
            reg,
            min_samples,
            frozen,
        )
        self.seen = 0
        self.weight = 0.0
        self.shift: Any = None
        self.sum1: Any = None
        self.sum2: Any = None

    def update(self, xp: Any, x: Any) -> None:
        """Add the samples ``x`` (``(d,)`` or ``(N, d)``); a no-op while frozen or
        without a sample."""
        rows = xp.reshape(x, (-1, self.dim))
        n = int(rows.shape[0])
        if self.frozen or n == 0:
            return
        if self.shift is None:
            self.shift = rows.mean(0)
            self.sum1 = xp.zeros_like(self.shift)
            self.sum2 = xp.zeros_like(rows[:1].T @ rows[:1])
        shift, sum1, sum2 = (
            _on_device(xp, v, rows) for v in (self.shift, self.sum1, self.sum2)
        )
        rows = xp_asarray(xp, rows, dtype=shift.dtype)  # the dtype of the first update
        keep = self.decay**n
        centred = rows - shift
        self.shift = shift
        self.sum1 = keep * sum1 + centred.sum(0)
        self.sum2 = keep * sum2 + centred.T @ centred
        self.weight = keep * self.weight + n
        self.seen += n

    def moments(self, xp: Any, like: Any) -> tuple[Any, Any]:
        """The estimate's mean ``(d,)`` and covariance ``(d, d)``, in the backend and
        dtype of *like*."""
        # kept on the device, in their own dtype, so that a loaded state moves once
        self.shift, self.sum1, self.sum2 = (
            _on_device(xp, v, like) for v in (self.shift, self.sum1, self.sum2)
        )
        shift, sum1, sum2 = (
            xp_asarray(xp, v, dtype=like.dtype, device=getattr(like, "device", None))
            for v in (self.shift, self.sum1, self.sum2)
        )
        centred_mean = sum1 / self.weight
        cov = sum2 / self.weight - centred_mean[:, None] * centred_mean[None, :]
        eye = _filled(xp, (self.dim,), like, eye=True)
        return shift + centred_mean, cov + self.reg * eye

    def log_prob(self, xp: Any, x: Any) -> Any:
        """Log density of ``x`` (``(d,)`` or ``(N, d)``) under the estimate, ``()`` or
        ``(N,)``."""
        if self.seen < self.min_samples:
            return xp.zeros_like(x[..., 0])
        mean, cov = self.moments(xp, x)
        diff = x - mean
        # a linear solve, more accurate than an explicit inverse
        quad = (diff * xp.linalg.solve(cov, diff[..., None])[..., 0]).sum(-1)
        return -0.5 * (self.dim * LOG_2PI + xp.linalg.slogdet(cov)[1] + quad)


class KernelDensity(_Online):
    """A Gaussian kernel density over a ring buffer of the latest samples.

    The kernel is diagonal, with Scott's rule per feature (the feature's SD times
    ``n ** (-1 / (d + 4))``), so in one dimension it equals
    :class:`scipy.stats.gaussian_kde`. It follows a multimodal distribution.

    Args:
        dim: Feature dimension ``d``.
        capacity: Number of latest samples kept. One update with more samples (more
            envs) than this keeps an evenly spaced part of them, shifted by one sample
            per update so that every env is learnt, and the buffer then holds one step.
        bandwidth_scale: Factor on Scott's bandwidth.
        min_bandwidth: Lower bound of each feature's bandwidth.
        min_samples: :meth:`log_prob` is 0 (no correction) until this many samples were
            seen; at least 2, which the bandwidth needs.
        frozen: Start frozen.
    """

    _STATE = ("count", "next", "buffer")

    def __init__(
        self,
        dim: int,
        capacity: int = 4096,
        bandwidth_scale: float = 1.0,
        min_bandwidth: float = 1e-6,
        min_samples: int = 2,
        frozen: bool = False,
    ) -> None:
        if capacity < 2:
            raise ValueError(f"capacity must be at least 2, got {capacity}")
        if min_samples < 2:
            raise ValueError(f"min_samples must be at least 2, got {min_samples}")
        self.dim, self.capacity, self.frozen = dim, capacity, frozen
        self.bandwidth_scale, self.min_bandwidth, self.min_samples = (
            bandwidth_scale,
            min_bandwidth,
            min_samples,
        )
        self.count = 0
        self.next = 0
        self.buffer: Any = None
        # shift of the evenly spaced part of a batch above the capacity
        self._offset = 0

    def update(self, xp: Any, x: Any) -> None:
        """Add the samples ``x`` (``(d,)`` or ``(N, d)``), replacing the oldest; a no-op
        while frozen or without a sample."""
        rows = xp.reshape(x, (-1, self.dim))
        n = int(rows.shape[0])
        if self.frozen or n == 0:
            return
        if n > self.capacity:
            # the rows of one step are envs: an evenly spaced part of them, shifted by
            # one env per update so that every env is learnt
            picks = (_arange(xp, self.capacity, rows) * n) // self.capacity
            rows = rows[(picks + self._offset) % n]
            self._offset = (self._offset + 1) % n
            n = self.capacity
        if self.buffer is None:
            self.buffer = _filled(xp, (self.capacity, self.dim), rows)
        buffer = _on_device(xp, self.buffer, rows)
        if (
            _is_torch(xp)
            and buffer.is_inference()
            and not xp.is_inference_mode_enabled()
        ):
            # made under inference_mode (RSL-RL's rollouts): not writable in place
            # outside it
            buffer = buffer.clone()
        # in the dtype of the first update
        slots = (self.next + _arange(xp, n, rows)) % self.capacity
        buffer[slots] = xp_asarray(xp, rows, dtype=buffer.dtype)
        self.buffer = buffer
        self.next = (self.next + n) % self.capacity
        self.count = min(self.count + n, self.capacity)

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the samples from :meth:`state_dict`; they must come from an estimator
        of this capacity and dim."""
        buffer = state.get("buffer")
        if buffer is not None and tuple(buffer.shape) != (self.capacity, self.dim):
            raise ValueError(
                f"the state's buffer is {tuple(buffer.shape)}, "
                f"this estimator's {(self.capacity, self.dim)}"
            )
        super().load_state_dict(state)

    def _samples(self, xp: Any, like: Any) -> tuple[Any, Any, Any]:
        """The samples ``(n, d)``, their mean and each feature's kernel SD (both
        ``(d,)``), in the backend and dtype of *like*."""
        # kept on the device, in its own dtype, so that a loaded state moves once
        self.buffer = _on_device(xp, self.buffer, like)
        data = xp_asarray(
            xp,
            self.buffer[: self.count],
            dtype=like.dtype,
            device=getattr(like, "device", None),
        )
        centre = data.mean(0)
        var = ((data - centre) ** 2).sum(0) / (self.count - 1)
        h = var**0.5 * self.bandwidth_scale * self.count ** (-1.0 / (self.dim + 4))
        h = (
            xp.clamp(h, min=self.min_bandwidth)
            if _is_torch(xp)
            else xp.clip(h, self.min_bandwidth, None)
        )
        return data, centre, h

    def bandwidth(self, xp: Any, like: Any) -> Any:
        """Each feature's kernel SD, ``(d,)``."""
        return self._samples(xp, like)[2]

    def log_prob(self, xp: Any, x: Any) -> Any:
        """Log density of ``x`` (``(d,)`` or ``(N, d)``) under the estimate, ``()`` or
        ``(N,)``."""
        if self.count < self.min_samples:
            return xp.zeros_like(x[..., 0])
        data, centre, h = self._samples(xp, x)
        # relative to the samples' mean: |z|^2 + |zd|^2 - 2 z.zd cancels in float32 once
        # x / h is large
        z, zd = (x - centre) / h, (data - centre) / h
        exponent = z @ zd.T  # (..., n), the one array of that size: -|z - zd|^2 / 2
        exponent -= 0.5 * (zd * zd).sum(-1)
        exponent -= 0.5 * (z * z).sum(-1)[..., None]
        log_norm = xp.log(h).sum() + 0.5 * self.dim * LOG_2PI + math.log(self.count)
        return logsumexp(xp, exponent) - log_norm


def _arange(xp: Any, n: int, like: Any) -> Any:
    """``0 .. n - 1`` as integers, on the device of *like* for torch."""
    return xp.arange(n, device=like.device) if _is_torch(xp) else np.arange(n)


def _is_torch(xp: Any) -> bool:
    """Whether the array module *xp* is torch."""
    return getattr(xp, "__name__", "") == "torch"
