# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""GPU-resident SAR (Synergy-based Action Representation) inverse transform.

Converts sklearn-fitted SAR components (MinMaxScaler → FastICA → PCA) into
fixed ``torch.Tensor`` buffers so that the synergy→muscle mapping runs
entirely on the GPU during mjlab / RSL-RL training — no CPU round-trips.

The inverse transform pipeline mirrors ``synergy_inverse_transform()`` in
``run_benchmark.py`` but uses torch matrix operations instead of sklearn::

    muscle = pca.inverse_transform(
        ica.inverse_transform(
            normalizer.inverse_transform(syn_act)
        )
    )

Which expands to:

    x = (syn_act - norm.min_) / norm.scale_       # MinMaxScaler⁻¹
    x = x @ ica.mixing_.T + ica.mean_             # FastICA⁻¹
    x = x @ pca.components_ + pca.mean_           # PCA⁻¹
    x = clamp(x, 0.0, 1.0)                        # valid activation range

Usage::

    from benchmarks.sar_backends.sar_torch_transform import SARTorchTransform

    sar = SARTorchTransform(ica, pca, normalizer, device="cuda:0")
    # syn_act: torch.Tensor shape (..., n_syn), values in [-1, 1]
    muscle_act = sar(syn_act)  # shape (..., n_muscles), values in [0, 1]
"""

from __future__ import annotations


try:
    import torch
    import torch.nn as nn

    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


class SARTorchTransform(nn.Module if _TORCH_OK else object):
    """Fixed-weight torch module that applies the SAR inverse transform on GPU.

    All sklearn parameters are stored as non-trainable ``register_buffer``
    tensors and moved to the target device at construction time.

    Args:
        ica: Fitted ``sklearn.decomposition.FastICA``.
        pca: Fitted ``sklearn.decomposition.PCA``.
        normalizer: Fitted ``sklearn.preprocessing.MinMaxScaler``.
        device: Target torch device string (e.g. ``"cuda:0"``, ``"cpu"``).

    Raises:
        ImportError: If torch is not installed.
    """

    def __init__(self, ica, pca, normalizer, device: str = "cpu") -> None:
        if not _TORCH_OK:
            raise ImportError("torch is required for SARTorchTransform")
        super().__init__()

        # ── MinMaxScaler inverse ─────────────────────────────────────────────
        # transform:   x_out = x_in * scale_ + min_
        # inverse:     x_in  = (x_out - min_) / scale_
        self.register_buffer(
            "norm_min",
            torch.tensor(normalizer.min_, dtype=torch.float32),
        )
        self.register_buffer(
            "norm_scale",
            torch.tensor(normalizer.scale_, dtype=torch.float32),
        )

        # ── FastICA inverse ──────────────────────────────────────────────────
        # sklearn:  S = (X - mean_) @ components_.T
        # inverse:  X = S @ mixing_.T + mean_
        #   mixing_   shape: (n_features, n_components) = (n_pca_comps, n_syn)
        #   mixing_.T shape: (n_syn, n_pca_comps)
        self.register_buffer(
            "ica_mixing_T",
            torch.tensor(ica.mixing_.T.copy(), dtype=torch.float32),
            # shape: (n_syn, n_pca_comps)
        )
        self.register_buffer(
            "ica_mean",
            torch.tensor(ica.mean_.copy(), dtype=torch.float32),
            # shape: (n_pca_comps,)
        )

        # ── PCA inverse ──────────────────────────────────────────────────────
        # sklearn:  X_reduced = (X - mean_) @ components_.T
        # inverse:  X = X_reduced @ components_ + mean_
        #   components_ shape: (n_components, n_features) = (n_pca_comps, n_muscles)
        self.register_buffer(
            "pca_components",
            torch.tensor(pca.components_.copy(), dtype=torch.float32),
            # shape: (n_pca_comps, n_muscles)
        )
        self.register_buffer(
            "pca_mean",
            torch.tensor(pca.mean_.copy(), dtype=torch.float32),
            # shape: (n_muscles,)
        )

        self.to(device)

    # ------------------------------------------------------------------
    # Shape properties
    # ------------------------------------------------------------------

    @property
    def n_syn(self) -> int:
        """Number of synergy inputs (policy action dimension)."""
        return int(self.norm_min.shape[0])

    @property
    def n_muscles(self) -> int:
        """Number of muscle outputs (env action dimension)."""
        return int(self.pca_mean.shape[0])

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, syn_act: torch.Tensor) -> torch.Tensor:
        """Map synergy actions to muscle activations (GPU-resident).

        Args:
            syn_act: Shape ``(..., n_syn)``, values typically in ``[-1, 1]``.

        Returns:
            Muscle activations, shape ``(..., n_muscles)``, values in
            ``[0, 1]`` after clamping.
        """
        # 1. MinMaxScaler inverse
        x = (syn_act - self.norm_min) / self.norm_scale

        # 2. FastICA inverse:  S @ mixing_.T + mean_
        #    x:          (..., n_syn)
        #    ica_mixing_T: (n_syn, n_pca_comps)
        #    result:     (..., n_pca_comps)
        x = x @ self.ica_mixing_T + self.ica_mean

        # 3. PCA inverse:  x_reduced @ components_ + mean_
        #    x:             (..., n_pca_comps)
        #    pca_components: (n_pca_comps, n_muscles)
        #    result:        (..., n_muscles)
        x = x @ self.pca_components + self.pca_mean

        # Clamp to valid muscle activation range
        return torch.clamp(x, 0.0, 1.0)
