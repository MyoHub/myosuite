# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for musclemimic_training.sar_fitting (SAR fitting).

Requires scikit-learn and joblib; tests are skipped when not installed.
"""

from __future__ import annotations

import tempfile
import unittest

import numpy as np


def _sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401
        import joblib  # noqa: F401

        return True
    except ImportError:
        return False


def _make_acts(
    n_samples: int = 200,
    n_muscles: int = 20,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n_samples, n_muscles)).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests: _validate_acts
# ---------------------------------------------------------------------------


class TestValidateActs(unittest.TestCase):
    def test_raises_on_1d(self) -> None:
        from musclemimic_training.sar_fitting import _validate_acts

        with self.assertRaises(ValueError):
            _validate_acts(np.zeros(10))

    def test_raises_on_empty(self) -> None:
        from musclemimic_training.sar_fitting import _validate_acts

        with self.assertRaises(ValueError):
            _validate_acts(np.zeros((0, 10)))

    def test_passes_valid(self) -> None:
        from musclemimic_training.sar_fitting import _validate_acts

        _validate_acts(np.zeros((5, 10)))  # should not raise


# ---------------------------------------------------------------------------
# Tests: compute_vaf_curve
# ---------------------------------------------------------------------------


@unittest.skipUnless(_sklearn_available(), "scikit-learn not installed")
class TestComputeVafCurve(unittest.TestCase):
    def test_returns_dict_with_correct_keys(self) -> None:
        from musclemimic_training.sar_fitting import compute_vaf_curve

        acts = _make_acts(n_samples=100, n_muscles=10)
        curve = compute_vaf_curve(acts, max_synergies=5)
        self.assertEqual(set(curve.keys()), {1, 2, 3, 4, 5})

    def test_vaf_monotonically_increasing(self) -> None:
        from musclemimic_training.sar_fitting import compute_vaf_curve

        acts = _make_acts(n_samples=200, n_muscles=15)
        curve = compute_vaf_curve(acts, max_synergies=10)
        values = [curve[k] for k in sorted(curve)]
        self.assertTrue(all(v2 >= v1 for v1, v2 in zip(values, values[1:])))

    def test_vaf_in_0_1(self) -> None:
        from musclemimic_training.sar_fitting import compute_vaf_curve

        acts = _make_acts(n_samples=100, n_muscles=10)
        curve = compute_vaf_curve(acts, max_synergies=10)
        for v in curve.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0 + 1e-6)

    def test_clamped_by_n_muscles(self) -> None:
        from musclemimic_training.sar_fitting import compute_vaf_curve

        acts = _make_acts(n_samples=50, n_muscles=5)
        curve = compute_vaf_curve(acts, max_synergies=100)
        self.assertLessEqual(max(curve.keys()), 5)

    def test_raises_on_invalid_acts(self) -> None:
        from musclemimic_training.sar_fitting import compute_vaf_curve

        with self.assertRaises(ValueError):
            compute_vaf_curve(np.zeros(10), max_synergies=5)


# ---------------------------------------------------------------------------
# Tests: select_n_synergies
# ---------------------------------------------------------------------------


class TestSelectNSynergies(unittest.TestCase):
    def test_returns_first_above_threshold(self) -> None:
        from musclemimic_training.sar_fitting import select_n_synergies

        curve = {1: 0.5, 2: 0.8, 3: 0.92, 4: 0.96, 5: 0.99}
        n = select_n_synergies(curve, threshold=0.95)
        self.assertEqual(n, 4)

    def test_returns_max_when_threshold_not_reached(self) -> None:
        from musclemimic_training.sar_fitting import select_n_synergies

        curve = {1: 0.5, 2: 0.7, 3: 0.85}
        n = select_n_synergies(curve, threshold=0.99)
        self.assertEqual(n, 3)

    def test_exact_threshold_match(self) -> None:
        from musclemimic_training.sar_fitting import select_n_synergies

        curve = {1: 0.5, 2: 0.95, 3: 0.99}
        n = select_n_synergies(curve, threshold=0.95)
        self.assertEqual(n, 2)


# ---------------------------------------------------------------------------
# Tests: extract_synergies
# ---------------------------------------------------------------------------


@unittest.skipUnless(_sklearn_available(), "scikit-learn not installed")
class TestExtractSynergies(unittest.TestCase):
    def setUp(self) -> None:
        from musclemimic_training.sar_fitting import extract_synergies

        self.acts = _make_acts(n_samples=300, n_muscles=20)
        self.n_syn = 5
        self.model = extract_synergies(self.acts, self.n_syn)

    def test_n_synergies_stored(self) -> None:
        self.assertEqual(self.model.n_synergies, self.n_syn)

    def test_n_muscles_stored(self) -> None:
        self.assertEqual(self.model.n_muscles, 20)

    def test_vaf_in_0_1(self) -> None:
        self.assertGreaterEqual(self.model.vaf, 0.0)
        self.assertLessEqual(self.model.vaf, 1.0 + 1e-6)

    def test_raises_when_n_syn_too_large(self) -> None:
        from musclemimic_training.sar_fitting import extract_synergies

        with self.assertRaises(ValueError):
            extract_synergies(self.acts, n_synergies=9999)

    def test_raises_on_invalid_acts(self) -> None:
        from musclemimic_training.sar_fitting import extract_synergies

        with self.assertRaises(ValueError):
            extract_synergies(np.zeros(10), n_synergies=3)

    def test_source_clips_stored(self) -> None:
        from musclemimic_training.sar_fitting import extract_synergies

        model = extract_synergies(
            self.acts, self.n_syn, source_clips=["clip_a.npz", "clip_b.npz"]
        )
        self.assertEqual(model.source_clips, ["clip_a.npz", "clip_b.npz"])

    def test_default_empty_source_clips(self) -> None:
        self.assertEqual(self.model.source_clips, [])


# ---------------------------------------------------------------------------
# Tests: a fitted model through MyoSuite's encode / decode / save / load
# ---------------------------------------------------------------------------


@unittest.skipUnless(_sklearn_available(), "scikit-learn not installed")
class TestFittedModelInMyoSuite(unittest.TestCase):
    def test_save_load_encode_decode(self) -> None:
        from musclemimic_training.sar_fitting import extract_synergies

        from myosuite.integrations.musclemimic.sar_extraction import (
            decode_synergies,
            encode_activations,
            load_synergy_model,
            save_synergy_model,
        )

        acts = _make_acts(n_samples=300, n_muscles=20)
        model = extract_synergies(acts, n_synergies=5, source_clips=["clip.npz"])
        with tempfile.TemporaryDirectory() as tmp:
            save_synergy_model(model, tmp)
            loaded = load_synergy_model(tmp)
        self.assertEqual(loaded.source_clips, ["clip.npz"])
        np.testing.assert_allclose(
            encode_activations(loaded, acts[:10]),
            encode_activations(model, acts[:10]),
            atol=1e-6,
        )
        # The scaler margin keeps training data inside (0, 1): no clipping, so
        # decode inverts encode up to the PCA reconstruction.
        syn = encode_activations(model, acts)
        self.assertTrue(((syn > 0.0) & (syn < 1.0)).all())
        pca = model.pca
        expected = np.clip(pca.inverse_transform(pca.transform(acts)), 0.0, 1.0)
        np.testing.assert_allclose(decode_synergies(model, syn), expected, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
