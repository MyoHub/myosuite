# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for sar_extraction.py (encode, decode, save, load).

The fitting tests are in training/musclemimic. Requires scikit-learn and joblib;
tests are skipped when not installed.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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
# Tests: encode_activations
# ---------------------------------------------------------------------------


@unittest.skipUnless(_sklearn_available(), "scikit-learn not installed")
class TestEncodeActivations(unittest.TestCase):
    def setUp(self) -> None:
        from myosuite.tests.support.synergy_models import fit_synergy_model

        self.acts = _make_acts(n_samples=300, n_muscles=20)
        self.model = fit_synergy_model(self.acts, n_synergies=5)

    def test_output_shape_batch(self) -> None:
        from myosuite.integrations.musclemimic.sar_extraction import encode_activations

        encoded = encode_activations(self.model, self.acts[:10])
        self.assertEqual(encoded.shape, (10, 5))

    def test_output_shape_single(self) -> None:
        from myosuite.integrations.musclemimic.sar_extraction import encode_activations

        encoded = encode_activations(self.model, self.acts[0])
        self.assertEqual(encoded.shape, (5,))

    def test_output_dtype_float32(self) -> None:
        from myosuite.integrations.musclemimic.sar_extraction import encode_activations

        encoded = encode_activations(self.model, self.acts[:5])
        self.assertEqual(encoded.dtype, np.float32)

    def test_output_in_0_1_on_training_data(self) -> None:
        from myosuite.integrations.musclemimic.sar_extraction import encode_activations

        encoded = encode_activations(self.model, self.acts)
        # Training data should fall mostly in [0, 1] after clip
        self.assertTrue((encoded >= 0.0).all())
        self.assertTrue((encoded <= 1.0).all())

    def test_raises_on_wrong_muscle_dim(self) -> None:
        from myosuite.integrations.musclemimic.sar_extraction import encode_activations

        wrong = np.zeros((5, 99), dtype=np.float32)
        with self.assertRaises(ValueError):
            encode_activations(self.model, wrong)


@unittest.skipUnless(_sklearn_available(), "scikit-learn not installed")
class TestDecodeSynergies(unittest.TestCase):
    def setUp(self) -> None:
        from myosuite.tests.support.synergy_models import fit_synergy_model

        self.acts = _make_acts(n_samples=300, n_muscles=20)
        self.model = fit_synergy_model(self.acts, n_synergies=5)
        self.syn = np.random.default_rng(0).random((10, 5)).astype(np.float32)

    def test_shapes_dtype_and_range(self) -> None:
        from myosuite.integrations.musclemimic.sar_extraction import decode_synergies

        batch = decode_synergies(self.model, self.syn)
        single = decode_synergies(self.model, self.syn[0])
        self.assertEqual(batch.shape, (10, 20))
        self.assertEqual(single.shape, (20,))
        self.assertEqual(batch.dtype, np.float32)
        self.assertTrue(((batch >= 0.0) & (batch <= 1.0)).all())
        np.testing.assert_array_equal(single, batch[0])

    def test_matches_the_inverse_sklearn_pipeline(self) -> None:
        from myosuite.integrations.musclemimic.sar_extraction import decode_synergies

        m = self.model
        x = m.pca.inverse_transform(
            m.ica.inverse_transform(m.scaler.inverse_transform(self.syn))
        )
        expected = np.clip(x, 0.0, 1.0).astype(np.float32)
        # decode_synergies works in float64 (as encode_activations); float32 rounding only.
        np.testing.assert_allclose(decode_synergies(m, self.syn), expected, atol=1e-6)

    def test_roundtrip_reconstructs_activations(self) -> None:
        from myosuite.integrations.musclemimic.sar_extraction import (
            decode_synergies,
            encode_activations,
        )

        # On training data (no scaler clipping) decode inverts encode exactly, so the
        # round trip equals the PCA reconstruction.
        decoded = decode_synergies(
            self.model, encode_activations(self.model, self.acts)
        )
        pca = self.model.pca
        expected = np.clip(pca.inverse_transform(pca.transform(self.acts)), 0.0, 1.0)
        np.testing.assert_allclose(decoded, expected, atol=1e-5)

    def test_raises_on_wrong_synergy_dim(self) -> None:
        from myosuite.integrations.musclemimic.sar_extraction import decode_synergies

        with self.assertRaises(ValueError):
            decode_synergies(self.model, np.zeros((5, 7), dtype=np.float32))

    def test_accepts_a_list(self) -> None:
        from myosuite.integrations.musclemimic.sar_extraction import decode_synergies

        np.testing.assert_array_equal(
            decode_synergies(self.model, self.syn[0].tolist()),
            decode_synergies(self.model, self.syn[0]),
        )


# ---------------------------------------------------------------------------
# Tests: save / load roundtrip
# ---------------------------------------------------------------------------


@unittest.skipUnless(_sklearn_available(), "scikit-learn not installed")
class TestSaveLoadRoundtrip(unittest.TestCase):
    def setUp(self) -> None:
        from myosuite.tests.support.synergy_models import fit_synergy_model

        acts = _make_acts(n_samples=300, n_muscles=20)
        self.model = fit_synergy_model(acts, n_synergies=5, source_clips=["clip.npz"])

    def test_roundtrip_metadata(self) -> None:
        from myosuite.integrations.musclemimic.sar_extraction import (
            load_synergy_model,
            save_synergy_model,
        )

        with tempfile.TemporaryDirectory() as tmp:
            save_synergy_model(self.model, tmp)
            loaded = load_synergy_model(tmp)

        self.assertEqual(loaded.n_synergies, self.model.n_synergies)
        self.assertEqual(loaded.n_muscles, self.model.n_muscles)
        self.assertAlmostEqual(loaded.vaf, self.model.vaf, places=4)
        self.assertEqual(loaded.source_clips, self.model.source_clips)

    def test_roundtrip_encode_parity(self) -> None:
        """Loaded model must produce identical encodings."""
        from myosuite.integrations.musclemimic.sar_extraction import (
            encode_activations,
            load_synergy_model,
            save_synergy_model,
        )

        acts = _make_acts(n_samples=10, n_muscles=20, seed=7)
        original_enc = encode_activations(self.model, acts)

        with tempfile.TemporaryDirectory() as tmp:
            save_synergy_model(self.model, tmp)
            loaded = load_synergy_model(tmp)

        loaded_enc = encode_activations(loaded, acts)
        np.testing.assert_allclose(original_enc, loaded_enc, rtol=1e-5, atol=1e-6)

    def test_metadata_is_written_without_pickle(self) -> None:
        from myosuite.integrations.musclemimic.sar_extraction import save_synergy_model

        with tempfile.TemporaryDirectory() as tmp:
            save_synergy_model(self.model, tmp)
            with np.load(Path(tmp) / "metadata.npz", allow_pickle=False) as meta:
                clips = [str(c) for c in meta["source_clips"]]
        self.assertEqual(clips, ["clip.npz"])

    def test_loads_metadata_written_with_an_object_array(self) -> None:
        """Synergy folders saved before 3.0 (source_clips as objects) still load."""
        from myosuite.integrations.musclemimic.sar_extraction import (
            load_synergy_model,
            save_synergy_model,
        )

        with tempfile.TemporaryDirectory() as tmp:
            save_synergy_model(self.model, tmp)
            np.savez(
                Path(tmp) / "metadata.npz",
                n_synergies=np.array(self.model.n_synergies, dtype=np.int32),
                n_muscles=np.array(self.model.n_muscles, dtype=np.int32),
                vaf=np.array(self.model.vaf, dtype=np.float32),
                source_clips=np.array(self.model.source_clips, dtype=object),
            )
            loaded = load_synergy_model(tmp)
        self.assertEqual(loaded.source_clips, ["clip.npz"])

    def test_load_raises_on_missing_file(self) -> None:
        from myosuite.integrations.musclemimic.sar_extraction import load_synergy_model

        with self.assertRaises(FileNotFoundError):
            load_synergy_model("/nonexistent/path")

    def test_save_creates_expected_files(self) -> None:
        from myosuite.integrations.musclemimic.sar_extraction import save_synergy_model

        with tempfile.TemporaryDirectory() as tmp:
            save_synergy_model(self.model, Path(tmp))
            files = {p.name for p in Path(tmp).iterdir()}
        self.assertIn("pca.pkl", files)
        self.assertIn("ica.pkl", files)
        self.assertIn("scaler.pkl", files)
        self.assertIn("metadata.npz", files)


if __name__ == "__main__":
    unittest.main()
