# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke tests for the SAR backend benchmark utilities.

These tests run the CPU-only paths of `benchmarks/sar_backends/run_benchmark.py`
with very small workloads and assert that no `env.mj_data` deprecation warnings
are emitted. This extends coverage so that regressions in the SAR benchmark
helpers are caught in CI.
"""

from __future__ import annotations

import pathlib
import warnings

import pytest

from benchmarks.sar_backends.run_benchmark import (
    _SAR_DIR,
    run_deterministic_comparison,
    run_training_comparison,
)
from myosuite.tests.support.optional_deps import require_stable_baselines3


pytestmark = pytest.mark.tier1

try:
    import sklearn  # noqa: F401

    _SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    _SKLEARN_AVAILABLE = False


def _skip_if_sar_pkls_missing() -> None:
    """Skip the test suite if the pretrained SAR pkl files are not available."""
    required = ("ica.pkl", "pca.pkl", "normalizer.pkl")
    missing = [name for name in required if not (_SAR_DIR / name).exists()]
    if missing:
        pytest.skip(
            "SAR benchmark pretrained files missing: "
            + ", ".join(str(_SAR_DIR / m) for m in missing)
        )


def _assert_no_mj_data_deprecation(
    recorded_warnings: list[warnings.WarningMessage],
) -> None:
    """Assert that no warning mentions the deprecated env.mj_data property."""
    for w in recorded_warnings:
        msg = str(w.message)
        assert "env.mj_data is deprecated" not in msg, (
            "SAR benchmark emitted deprecated env.mj_data warning: " + msg
        )


@pytest.mark.skipif(
    not _SKLEARN_AVAILABLE,
    reason="sklearn not installed (required for SAR benchmark)",
)
def test_sar_benchmark_phase_a_cpu_no_mj_data_deprecation(
    tmp_path: pathlib.Path,
) -> None:
    """Phase A deterministic rollout (CPU-only) should not touch env.mj_data."""
    _skip_if_sar_pkls_missing()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rewards = run_deterministic_comparison(
            output_dir=tmp_path,
            sar_dir=_SAR_DIR,
            n_steps=20,
            backends=["cpu"],
        )

    assert "cpu" in rewards
    assert len(rewards["cpu"]) <= 20
    _assert_no_mj_data_deprecation(caught)


@pytest.mark.skipif(
    not _SKLEARN_AVAILABLE,
    reason="sklearn not installed (required for SAR benchmark)",
)
def test_sar_benchmark_phase_b_cpu_no_mj_data_deprecation(
    tmp_path: pathlib.Path,
) -> None:
    """Phase B short SAC training (CPU-only) should not touch env.mj_data."""
    require_stable_baselines3()
    _skip_if_sar_pkls_missing()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            run_training_comparison(
                output_dir=tmp_path,
                sar_dir=_SAR_DIR,
                n_steps=200,
                backends=["cpu"],
                seed=0,
                wandb_cfg=None,
                video_interval=0,  # disable rendering/video to avoid GLFW/display issues
            )
        except ImportError as exc:
            # Broken mixed conda/pip stacks often fail when SB3/matplotlib or
            # sklearn C extensions load against an incompatible NumPy build.
            msg = str(exc).lower()
            if "multiarray" in msg or "numpy" in msg:
                pytest.skip(
                    f"SAR Phase B skipped (NumPy / ML stack import issue): {exc}"
                )
            raise

    _assert_no_mj_data_deprecation(caught)
