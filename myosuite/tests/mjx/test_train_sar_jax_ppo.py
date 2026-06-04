# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the full SAR + JAX PPO pipeline on MJX (train_sar_jax_ppo.py).

Unit tests (run by default with pytest):
  test_sar_helpers: SAR computation (find_synergies, compute_sar, synergy_to_muscle)
  with synthetic data — no MJX/Brax, runs in seconds.

Slow test (run with pytest -m slow):
  test_sar_jax_ppo_pipeline_smoke: Full pipeline (play → activation rollout → SAR
  → SAR-RL) with minimal steps. Requires MJX/brax/mujoco_playground; takes several
  minutes. Example: JAX_PLATFORMS=cpu pytest -m slow -k smoke -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Repo root (test file is myosuite/tests/mjx/test_*.py)
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Fast tests in this module are tier1 (unit). Slow pipeline smoke is marked @pytest.mark.slow.
pytestmark = [pytest.mark.tier1]

sklearn = pytest.importorskip(
    "sklearn", reason="scikit-learn not installed (pip install myosuite[rl])"
)


def test_sar_helpers():
    """Test SAR computation and synergy_to_muscle with synthetic data (no MJX)."""
    from sklearn.decomposition import FastICA, PCA
    from sklearn.preprocessing import MinMaxScaler

    # Synthetic activations (T=200, n_muscles=20)
    rng = np.random.default_rng(42)
    acts = np.clip(rng.uniform(0, 1, (200, 20)).astype(np.float32), 1e-6, 1 - 1e-6)

    # find_synergies logic
    syn_dict = {}
    for i in range(acts.shape[1]):
        pca = PCA(n_components=i + 1)
        pca.fit_transform(acts)
        syn_dict[i + 1] = round(float(np.sum(pca.explained_variance_ratio_)), 4)
    assert len(syn_dict) == 20
    assert syn_dict[20] <= 1.01

    # compute_sar logic
    n_syn = 4
    pca = PCA(n_components=n_syn, random_state=0)
    pca_act = pca.fit_transform(acts)
    ica = FastICA(n_components=n_syn, random_state=0, max_iter=500)
    pcaica_act = ica.fit_transform(pca_act)
    normalizer = MinMaxScaler(feature_range=(-1.0, 1.0))
    normalizer.fit(pcaica_act)

    # synergy_to_muscle round-trip
    syn = np.array([0.5, -0.5, 0.0, 0.2])
    ica_inv = ica.inverse_transform(normalizer.inverse_transform(syn.reshape(1, -1)))
    muscle = pca.inverse_transform(ica_inv)
    muscle = np.clip(muscle, 0.0, 1.0)
    assert muscle.shape == (1, 20)
    assert np.all(muscle >= 0) and np.all(muscle <= 1)


# ---------------------------------------------------------------------------
# Full pipeline smoke test (requires MJX + Brax; slow)
# ---------------------------------------------------------------------------

try:
    from mujoco import mjx as _mjx_mod  # noqa: F401
    import jax  # noqa: F401
    import mujoco_playground as _mp  # noqa: F401

    _MJX_AVAILABLE = True
except (ImportError, AttributeError):
    _MJX_AVAILABLE = False


@pytest.mark.slow
@pytest.mark.tier3
@pytest.mark.skipif(not _MJX_AVAILABLE, reason="MJX/mujoco_playground not installed")
def test_sar_jax_ppo_pipeline_smoke(tmp_path, monkeypatch):
    """Run full SAR pipeline with minimal steps; assert outputs exist."""
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("SAR_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("MJX_ENV", "MjxLegWalk-v0")
    monkeypatch.setenv("PLAY_STEPS", "50")
    monkeypatch.setenv("SAR_RL_STEPS", "50")
    monkeypatch.setenv("ACTIVATION_EPISODES", "2")
    monkeypatch.setenv("N_SYNERGIES", "2")
    monkeypatch.setenv("JAX_PPO_NUM_ENVS", "4")
    monkeypatch.setenv("JAX_PPO_NUM_EVAL_ENVS", "2")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("RUN_PLAY_PHASE", "true")
    monkeypatch.setenv("SKIP_SAR_RL", "false")

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "train_sar_jax_ppo",
        _REPO_ROOT / "scripts" / "train_sar_jax_ppo.py",
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    script.main()

    output_dir = tmp_path / "MjxLegWalk-v0"
    assert output_dir.is_dir(), f"Expected output dir {output_dir}"
    assert (
        output_dir / "play_phase_params.pkl"
    ).exists(), "Play phase params should be saved"
    assert (
        output_dir / "muscle_activations.npy"
    ).exists(), "Muscle activations should be saved"
    assert (output_dir / "vaf_plot.png").exists(), "VAF plot should be saved"
    assert (output_dir / "locomotion_ica.pkl").exists(), "SAR ica should be saved"
    assert (output_dir / "locomotion_pca.pkl").exists(), "SAR pca should be saved"
    assert (output_dir / "locomotion_scaler.pkl").exists(), "SAR scaler should be saved"
    assert (output_dir / "sar_rl_params.pkl").exists(), "SAR-RL params should be saved"
