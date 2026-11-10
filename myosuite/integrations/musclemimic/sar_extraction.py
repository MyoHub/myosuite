# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""SAR muscle-synergy models: encode, decode, save and load.

A :class:`SynergyModel` holds the fitted transform of the Synergistic Action
Representation (Berg, Caggiano and Kumar, RSS 2023,
https://arxiv.org/abs/2307.03716): PCA whitening, FastICA and a MinMaxScaler.
The SAR tasks load it with :func:`load_synergy_model` (the mjlab action term
applies it through :mod:`~myosuite.integrations.musclemimic.sar_torch_transform`).

Collecting activations and fitting a model are training steps; they live in the
MuscleMimic training project (``training/musclemimic``, import name
``musclemimic_training``: ``activation_collector`` and ``sar_fitting``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from myosuite.core.trajectory_io import read_npz_names

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SynergyModel dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SynergyModel:
    """Trained SAR synergy model.

    Contains the fitted sklearn transformers and descriptive metadata.
    Instances are fitted by ``musclemimic_training.sar_fitting.extract_synergies``
    (``training/musclemimic``) and are serialisable via
    :func:`save_synergy_model` / :func:`load_synergy_model`.

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


def decode_synergies(model: SynergyModel, syn: np.ndarray) -> np.ndarray:
    """Map synergy activations back to muscle activations (inverse of :func:`encode_activations`).

    Pipeline: ``MinMaxScaler``⁻¹ → ``FastICA``⁻¹ → ``PCA``⁻¹ → clamp ``[0, 1]``.

    Args:
        model: Fitted :class:`SynergyModel`.
        syn: Synergy activations (array-like), shape ``(T, n_synergies)`` or ``(n_synergies,)``.

    Returns:
        Muscle activations, shape ``(T, n_muscles)`` or ``(n_muscles,)``, in
        ``[0, 1]``, dtype ``float32``.

    Raises:
        ValueError: If ``syn`` last dimension does not match ``n_synergies``.
    """
    syn = np.asarray(syn)
    single = syn.ndim == 1
    S = syn.reshape(1, -1) if single else syn
    if S.shape[1] != model.n_synergies:
        raise ValueError(f"Expected {model.n_synergies} synergies, got {S.shape[1]}")
    X = model.scaler.inverse_transform(S.astype(np.float64))
    X = model.pca.inverse_transform(model.ica.inverse_transform(X))
    acts = np.clip(X, 0.0, 1.0).astype(np.float32)
    return acts[0] if single else acts


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
        source_clips=np.asarray([str(c) for c in model.source_clips], dtype=str),
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

    meta = np.load(save_dir / "metadata.npz", allow_pickle=False)
    try:
        source_clips = read_npz_names(meta, "source_clips")
    except ValueError:
        # Written as an object array before 3.0; the folder's joblib files above
        # are pickles already, so its metadata may be read the same way.
        legacy = np.load(save_dir / "metadata.npz", allow_pickle=True)
        source_clips = [str(c) for c in legacy["source_clips"]]
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
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "SynergyModel",
    "decode_synergies",
    "encode_activations",
    "load_synergy_model",
    "save_synergy_model",
]
