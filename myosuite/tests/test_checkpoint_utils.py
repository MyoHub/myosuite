# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for checkpoint discovery (``find_checkpoint``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from myosuite.utils.checkpoint_utils import find_checkpoint

pytestmark = pytest.mark.tier1


def test_find_checkpoint_returns_the_explicit_one_unchanged() -> None:
    """An explicit checkpoint short-circuits any search."""
    assert find_checkpoint(
        "myoElbowPose1D6MRandom-v0", checkpoint="some/model.pt"
    ) == Path("some/model.pt")


def test_find_checkpoint_prefers_a_local_logs_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A local ``logs/rsl_rl/<experiment>/<run>/model_*.pt`` wins over everything else."""
    # Pin the experiment name: its mjlab lookup is unavailable where mjlab is not installed (py3.14).
    monkeypatch.setattr(
        "myosuite.utils.checkpoint_utils.mjlab_experiment",
        lambda _env_id: "myo_elbow_pose",
    )
    run = tmp_path / "logs" / "rsl_rl" / "myo_elbow_pose" / "2026-01-01_00-00-00"
    run.mkdir(parents=True)
    (run / "model_0.pt").write_bytes(b"")
    (run / "model_200.pt").write_bytes(b"")

    found = find_checkpoint("myoElbowPose1D6MRandom-v0", roots=(tmp_path,))

    assert found == run / "model_200.pt"


def test_find_checkpoint_falls_back_to_a_local_baseline(tmp_path: Path) -> None:
    """With no local run, the repository's default baseline checkpoint is used."""
    baseline = tmp_path / "baselines" / "checkpoints" / "myoElbowPose1D6MRandom-v0"
    baseline.mkdir(parents=True)
    (baseline / "model_200.pt").write_bytes(b"")

    found = find_checkpoint("myoElbowPose1D6MRandom-v0", roots=(tmp_path,))

    assert found == baseline / "model_200.pt"


def test_find_checkpoint_falls_back_to_the_hf_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No local baseline (e.g. a fresh clone) downloads it from Hugging Face instead."""
    hf_dir = tmp_path / "hf_cache" / "myoElbowPose1D6MRandom-v0"
    hf_dir.mkdir(parents=True)
    (hf_dir / "model_200.pt").write_bytes(b"")

    monkeypatch.setattr(
        "myosuite.core.hf_io.download_baseline_checkpoint",
        lambda env_id: hf_dir if env_id == "myoElbowPose1D6MRandom-v0" else None,
    )

    found = find_checkpoint("myoElbowPose1D6MRandom-v0", roots=(tmp_path,))

    assert found == hf_dir / "model_200.pt"


def test_find_checkpoint_falls_back_to_sb3_zip_when_no_hf_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An env with no baseline anywhere (local or Hugging Face) still tries the SB3 zip."""
    (tmp_path / "policy.zip").write_bytes(b"")
    monkeypatch.setattr(
        "myosuite.core.hf_io.download_baseline_checkpoint", lambda env_id: None
    )

    found = find_checkpoint(
        "myoElbowPose1D6MRandom-v0", roots=(tmp_path,), sb3_zip="policy.zip"
    )

    assert found == tmp_path / "policy.zip"


def test_find_checkpoint_returns_none_when_nothing_is_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No local run, no local baseline, no Hugging Face baseline, no SB3 zip: None."""
    monkeypatch.setattr(
        "myosuite.core.hf_io.download_baseline_checkpoint", lambda env_id: None
    )

    assert find_checkpoint("myoElbowPose1D6MRandom-v0", roots=(tmp_path,)) is None
