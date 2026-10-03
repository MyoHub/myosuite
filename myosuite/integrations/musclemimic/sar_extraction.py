# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Extract muscle synergies (SAR) from MuscleMimic activation arrays.

Implements the Synergy Action Representation pipeline described in
"Synergistic Action Representation for Dexterous Locomotion and Manipulation"
(Caggiano et al., 2023, https://arxiv.org/abs/2307.03716):

  1. PCA whitening to decorrelate activations.
  2. FastICA to extract statistically independent components.
  3. MinMaxScaler to normalise synergy activations to [0, 1].

Typical usage::

    from myosuite.integrations.musclemimic.activation_collector import (
        collect_activations,
    )
    from myosuite.integrations.musclemimic.sar_extraction import (
        compute_vaf_curve,
        select_n_synergies,
        extract_synergies,
        save_synergy_model,
        load_synergy_model,
    )

    acts = collect_activations(policy, model, clips)          # (T, n_muscles)
    vaf = compute_vaf_curve(acts, max_synergies=40)           # {n: vaf_pct}
    n_syn = select_n_synergies(vaf, threshold=0.95)           # e.g. 20
    syn_model = extract_synergies(acts, n_syn)                # SynergyModel
    save_synergy_model(syn_model, save_dir)
    # later…
    syn_model = load_synergy_model(save_dir)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

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


# ---------------------------------------------------------------------------
# SynergyModel dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SynergyModel:
    """Trained SAR synergy model.

    Contains the fitted sklearn transformers and descriptive metadata.
    Instances are produced by :func:`extract_synergies` and are
    serialisable via :func:`save_synergy_model` / :func:`load_synergy_model`.

    Args:
        pca: Fitted ``sklearn.decomposition.PCA`` (whitening).
        ica: Fitted ``sklearn.decomposition.FastICA``.
        scaler: Fitted ``sklearn.preprocessing.MinMaxScaler``.
        n_synergies: Number of synergy components extracted.
        n_muscles: Muscle dimensionality of the input activations.
        vaf: Variance accounted for by this synergy count (0–1 scale).
        source_clips: Optional list of clip paths used for training.
    """

    pca: Any  # sklearn.decomposition.PCA
    ica: Any  # sklearn.decomposition.FastICA
    scaler: Any  # sklearn.preprocessing.MinMaxScaler
    n_synergies: int
    n_muscles: int
    vaf: float
    source_clips: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# VAF curve
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Transform helper
# ---------------------------------------------------------------------------


def encode_activations(model: SynergyModel, acts: np.ndarray) -> np.ndarray:
    """Project raw activations into normalised synergy space.

    Args:
        model: Fitted :class:`SynergyModel`.
        acts: Raw activations, shape ``(T, n_muscles)`` or ``(n_muscles,)``.

    Returns:
        Synergy activations, shape ``(T, n_synergies)`` or ``(n_synergies,)``
        with values in ``[0, 1]`` (clipped), dtype ``float32``.

    Raises:
        ValueError: If ``acts`` last dimension does not match ``n_muscles``.
    """
    single = acts.ndim == 1
    X = acts.reshape(1, -1) if single else acts
    if X.shape[1] != model.n_muscles:
        raise ValueError(f"Expected {model.n_muscles} muscles, got {X.shape[1]}")
    X = X.astype(np.float64)
    X_pca = model.pca.transform(X)
    X_ica = model.ica.transform(X_pca)
    X_norm = np.clip(
        model.scaler.transform(X_ica),
        0.0,
        1.0,
    ).astype(np.float32)
    return X_norm[0] if single else X_norm


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def save_synergy_model(model: SynergyModel, save_dir: Path | str) -> None:
    """Save a :class:`SynergyModel` to *save_dir*.

    Writes three ``joblib`` files (``pca.pkl``, ``ica.pkl``, ``scaler.pkl``)
    and a ``metadata.npz`` file containing scalars and clip paths.

    Args:
        model: Fitted model to save.
        save_dir: Destination directory; created if it does not exist.

    Raises:
        ImportError: If ``joblib`` is not installed.
    """
    try:
        import joblib
    except ImportError as err:
        raise ImportError("joblib is required: pip install joblib") from err

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model.pca, save_dir / "pca.pkl")
    joblib.dump(model.ica, save_dir / "ica.pkl")
    joblib.dump(model.scaler, save_dir / "scaler.pkl")

    np.savez(
        save_dir / "metadata.npz",
        n_synergies=np.array(model.n_synergies, dtype=np.int32),
        n_muscles=np.array(model.n_muscles, dtype=np.int32),
        vaf=np.array(model.vaf, dtype=np.float32),
        source_clips=np.array(model.source_clips, dtype=object),
    )
    logger.info(
        "Saved SynergyModel to %s (n_syn=%d, vaf=%.3f)",
        save_dir,
        model.n_synergies,
        model.vaf,
    )


def load_synergy_model(save_dir: Path | str) -> SynergyModel:
    """Load a :class:`SynergyModel` saved by :func:`save_synergy_model`.

    Args:
        save_dir: Directory written by :func:`save_synergy_model`.

    Returns:
        Reconstructed :class:`SynergyModel`.

    Raises:
        FileNotFoundError: If any expected file is missing.
        ImportError: If ``joblib`` is not installed.
    """
    try:
        import joblib
    except ImportError as err:
        raise ImportError("joblib is required: pip install joblib") from err

    save_dir = Path(save_dir)
    for name in ("pca.pkl", "ica.pkl", "scaler.pkl", "metadata.npz"):
        if not (save_dir / name).exists():
            raise FileNotFoundError(
                f"Missing file in synergy model dir: {save_dir / name}"
            )

    pca = joblib.load(save_dir / "pca.pkl")
    ica = joblib.load(save_dir / "ica.pkl")
    scaler = joblib.load(save_dir / "scaler.pkl")

    meta = np.load(save_dir / "metadata.npz", allow_pickle=True)
    source_clips = list(meta["source_clips"])
    model = SynergyModel(
        pca=pca,
        ica=ica,
        scaler=scaler,
        n_synergies=int(meta["n_synergies"]),
        n_muscles=int(meta["n_muscles"]),
        vaf=float(meta["vaf"]),
        source_clips=source_clips,
    )
    logger.info(
        "Loaded SynergyModel from %s (n_syn=%d, vaf=%.3f)",
        save_dir,
        model.n_synergies,
        model.vaf,
    )
    return model


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_acts(acts: np.ndarray) -> None:
    if acts.ndim != 2:
        raise ValueError(f"acts must be 2-D, got shape {acts.shape}")
    if acts.shape[0] == 0:
        raise ValueError("acts must contain at least one sample")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "SynergyModel",
    "compute_vaf_curve",
    "encode_activations",
    "extract_synergies",
    "load_synergy_model",
    "save_synergy_model",
    "select_n_synergies",
]
