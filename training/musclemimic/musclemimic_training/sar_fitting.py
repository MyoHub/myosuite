# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Fit muscle synergies (SAR) on MuscleMimic activation arrays.

Implements the fitting step of the Synergistic Action Representation
(Berg, Caggiano and Kumar, RSS 2023, https://arxiv.org/abs/2307.03716):

  1. PCA whitening to decorrelate activations.
  2. FastICA to extract statistically independent components.
  3. MinMaxScaler to normalise synergy activations to [0, 1].

The fitted :class:`~myosuite.integrations.musclemimic.sar_extraction.SynergyModel`,
its encoding / decoding and its file format stay in MyoSuite
(:mod:`myosuite.integrations.musclemimic.sar_extraction`), where the SAR tasks
read it. Typical usage::

    from myosuite.integrations.musclemimic.sar_extraction import save_synergy_model
    from musclemimic_training.activation_collector import collect_activations
    from musclemimic_training.sar_fitting import (
        compute_vaf_curve,
        extract_synergies,
        select_n_synergies,
    )

    acts = collect_activations(policy, model, clips)          # (T, n_muscles)
    vaf = compute_vaf_curve(acts, max_synergies=40)           # {n: vaf_pct}
    n_syn = select_n_synergies(vaf, threshold=0.95)           # e.g. 20
    syn_model = extract_synergies(acts, n_syn)                # SynergyModel
    save_synergy_model(syn_model, save_dir)
"""

from __future__ import annotations

import logging

import numpy as np

from myosuite.integrations.musclemimic.sar_extraction import SynergyModel

logger = logging.getLogger(__name__)

_SAR_SKLEARN_INSTALL = (
    "scikit-learn is required for SAR (PCA / ICA). Install it in the same "
    "environment as your Jupyter kernel, e.g. `pip install scikit-learn` or "
    "`uv pip install scikit-learn` (MyoSuite also lists it in core "
    "dependencies "
    "— run `pip install -e .` from the repo if packages are missing)."
)


def _import_sklearn_pca():
    """Lazy import of ``sklearn.decomposition.PCA`` with a clear error."""
    try:
        from sklearn.decomposition import PCA

        return PCA
    except ImportError as err:
        raise ImportError(_SAR_SKLEARN_INSTALL) from err


def _import_sklearn_sar():
    """Lazy import of PCA, FastICA, and MinMaxScaler with a clear error."""
    try:
        from sklearn.decomposition import FastICA, PCA
        from sklearn.preprocessing import MinMaxScaler

        return PCA, FastICA, MinMaxScaler
    except ImportError as err:
        raise ImportError(_SAR_SKLEARN_INSTALL) from err


def compute_vaf_curve(
    acts: np.ndarray,
    max_synergies: int = 40,
) -> dict[int, float]:
    """Compute the Variance Accounted For (VAF) for increasing synergy counts.

    Fits PCA incrementally for ``n = 1 … max_synergies`` and records the
    cumulative explained variance ratio, which equals VAF under PCA.

    Args:
        acts: Activation array, shape ``(T, n_muscles)``, dtype ``float32``.
        max_synergies: Upper bound on the number of synergies to evaluate.
            Clamped to ``min(max_synergies, n_muscles, T)``.

    Returns:
        ``dict[n_synergies, vaf]`` where VAF is in ``[0, 1]``.

    Raises:
        ValueError: If ``acts`` is empty or not 2-D.
    """
    _validate_acts(acts)
    PCA = _import_sklearn_pca()

    n_max = min(max_synergies, acts.shape[1], acts.shape[0])
    pca = PCA(n_components=n_max, whiten=True)
    pca.fit(acts.astype(np.float64))

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    vaf_curve = {int(n + 1): float(cumvar[n]) for n in range(n_max)}
    logger.debug(
        "VAF curve computed: n_muscles=%d, max_synergies=%d, vaf@%d=%.3f",
        acts.shape[1],
        n_max,
        n_max,
        cumvar[-1],
    )
    return vaf_curve


def select_n_synergies(
    vaf_curve: dict[int, float],
    threshold: float = 0.95,
) -> int:
    """Return the smallest ``n`` such that ``vaf_curve[n] >= threshold``.

    Args:
        vaf_curve: Output of :func:`compute_vaf_curve`.
        threshold: Minimum VAF fraction to accept (default 0.95).

    Returns:
        Minimum synergy count reaching the threshold.  Returns the maximum
        available count when no entry reaches the threshold.
    """
    for n_syn in sorted(vaf_curve):
        if vaf_curve[n_syn] >= threshold:
            return n_syn
    max_n = max(vaf_curve)
    logger.warning(
        "VAF threshold %.2f not reached; using max synergies=%d (vaf=%.3f)",
        threshold,
        max_n,
        vaf_curve[max_n],
    )
    return max_n


def extract_synergies(
    acts: np.ndarray,
    n_synergies: int,
    *,
    random_state: int = 0,
    scaler_margin: float = 0.10,
    source_clips: list[str] | None = None,
) -> SynergyModel:
    """Fit SAR on activations and return a :class:`SynergyModel`.

    Pipeline:
      1. PCA with ``n_synergies`` components and whitening.
      2. FastICA (``max_iter=1000``) on the whitened components.
      3. MinMaxScaler fitted on the ICA-transformed activations.

    Args:
        acts: Activation array, shape ``(T, n_muscles)``, dtype ``float32``.
        n_synergies: Number of synergy components to extract.
        random_state: Seed for FastICA reproducibility.
        scaler_margin: Fractional headroom added to each ICA dimension's
            observed min/max before fitting MinMax scaling.  This reduces
            saturation when encoding rollout states that are near but slightly
            outside the extraction set support.
        source_clips: Optional list of clip paths for provenance tracking.

    Returns:
        Fitted :class:`SynergyModel`.

    Raises:
        ValueError: If ``acts`` is empty, not 2-D, or ``n_synergies`` is
            larger than ``min(T, n_muscles)``.
        ImportError: If ``scikit-learn`` is not installed.
    """
    _validate_acts(acts)
    n_max = min(acts.shape[0], acts.shape[1])
    if n_synergies > n_max:
        raise ValueError(f"n_synergies={n_synergies} exceeds min(T, n_muscles)={n_max}")

    PCA, FastICA, MinMaxScaler = _import_sklearn_sar()

    X = acts.astype(np.float64)

    pca = PCA(n_components=n_synergies, whiten=True, random_state=random_state)
    X_pca = pca.fit_transform(X)

    ica = FastICA(
        n_components=n_synergies,
        max_iter=1000,
        random_state=random_state,
    )
    X_ica = ica.fit_transform(X_pca)

    scaler = MinMaxScaler()
    scaler.fit(X_ica)
    if scaler_margin > 0.0:
        span = scaler.data_max_ - scaler.data_min_
        pad = np.maximum(1e-8, span * float(scaler_margin))
        data_min = scaler.data_min_ - pad
        data_max = scaler.data_max_ + pad
        span_padded = np.maximum(1e-8, data_max - data_min)
        fr_min, fr_max = scaler.feature_range
        scaler.data_min_ = data_min
        scaler.data_max_ = data_max
        scaler.data_range_ = span_padded
        scaler.scale_ = (fr_max - fr_min) / span_padded
        scaler.min_ = fr_min - data_min * scaler.scale_

    vaf = float(np.sum(pca.explained_variance_ratio_))
    logger.info(
        "Extracted %d synergies from (%d, %d) activations; VAF=%.3f",
        n_synergies,
        *acts.shape,
        vaf,
    )

    return SynergyModel(
        pca=pca,
        ica=ica,
        scaler=scaler,
        n_synergies=n_synergies,
        n_muscles=acts.shape[1],
        vaf=vaf,
        source_clips=list(source_clips or []),
    )


def _validate_acts(acts: np.ndarray) -> None:
    if acts.ndim != 2:
        raise ValueError(f"acts must be 2-D, got shape {acts.shape}")
    if acts.shape[0] == 0:
        raise ValueError("acts must contain at least one sample")


__all__ = [
    "compute_vaf_curve",
    "extract_synergies",
    "select_n_synergies",
]
