# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""A small SAR synergy model for tests (the fitting tools are in training/musclemimic)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from myosuite.integrations.musclemimic.sar_extraction import SynergyModel


def fit_synergy_model(
    acts: np.ndarray, n_synergies: int, source_clips: Sequence[str] = ()
) -> SynergyModel:
    """Fit whitened PCA, FastICA and a MinMaxScaler on ``acts`` (T, n_muscles).

    Args:
        acts: Activations, shape ``(T, n_muscles)``.
        n_synergies: Number of components.
        source_clips: Provenance stored in the model.

    Returns:
        A :class:`SynergyModel` laid out as the SAR fitting tools write it.
    """
    from sklearn.decomposition import PCA, FastICA
    from sklearn.preprocessing import MinMaxScaler

    x = np.asarray(acts, dtype=np.float64)
    pca = PCA(n_components=n_synergies, whiten=True, random_state=0)
    ica = FastICA(n_components=n_synergies, max_iter=1000, random_state=0)
    x_ica = ica.fit_transform(pca.fit_transform(x))
    return SynergyModel(
        pca=pca,
        ica=ica,
        scaler=MinMaxScaler().fit(x_ica),
        n_synergies=n_synergies,
        n_muscles=x.shape[1],
        vaf=float(np.sum(pca.explained_variance_ratio_)),
        source_clips=list(source_clips),
    )
