"""DynSyn grouping: similarity, k-medoids, groups of a registered task, command line."""

from __future__ import annotations

import numpy as np
import pytest

from myosuite.utils.action_map import DynSyn, DynSynGroups
from myosuite_dynsyn.groups import (
    fit_dynsyn,
    k_medoids,
    length_similarity,
    main,
    synergy_groups,
)

pytestmark = pytest.mark.tier2


def test_k_medoids_recovers_blocks() -> None:
    blocks = np.repeat(np.arange(3), 4)
    d = np.where(blocks[:, None] == blocks[None, :], 0.1, 1.0)
    np.fill_diagonal(d, 0.0)
    _, labels, _ = k_medoids(d, 3, seed=1)
    for b in range(3):
        assert len(set(labels[blocks == b])) == 1
    assert len(set(labels)) == 3


def test_similarity_measures_length_changes_not_slow_drift() -> None:
    """Muscles that only share a slow drift within segments must not look alike."""
    rng = np.random.default_rng(0)
    steps, segment = 2000, 100
    drift = np.linspace(0.0, 0.01, segment)
    a = np.concatenate(
        [
            0.3 + rng.uniform() * 0.1 + drift + 1e-3 * rng.standard_normal(segment)
            for _ in range(20)
        ]
    )
    b = np.concatenate(
        [
            0.3 + rng.uniform() * 0.1 - drift + 1e-3 * rng.standard_normal(segment)
            for _ in range(20)
        ]
    )
    together = a + 0.5
    noise_free = np.stack([a, b, together], axis=1)[:steps]
    similarity = length_similarity(noise_free, segment=segment)
    assert similarity[0, 2] == pytest.approx(1.0)
    assert similarity[0, 1] < 0.05
    # A muscle that doesn't move has no correlation with anything, and no NaN.
    still = np.stack([a, np.full(len(a), 0.3)], axis=1)
    similarity = length_similarity(still, segment=segment)
    assert np.isfinite(similarity).all()
    assert similarity[1, 1] == 1.0


def test_synergy_groups_partition_the_muscles() -> None:
    from myosuite import make_env

    env = make_env("myoLegWalk-v0")
    model = env.unwrapped.model
    groups, meta = synergy_groups(model, np.arange(model.nu), 40, steps=3000)
    env.close()
    assert len(groups) == 40 and len(meta["sizes"]) == 40
    assert sorted(i for g in groups for i in g) == list(range(80))
    m = fit_dynsyn(model, np.arange(model.nu), 40, steps=3000, max_amplitude=0.0)
    assert isinstance(m, DynSyn) and m.action_dim == 40


def test_command_line_writes_the_groups_in_action_order(tmp_path) -> None:
    pytest.importorskip("mjlab")
    out = tmp_path / "elbow.json"
    main(
        [
            "myoElbowPose1D6MRandom-v0",
            "--groups",
            "3",
            "--steps",
            "2000",
            "--out",
            str(out),
        ]
    )
    groups = DynSynGroups.load(out)
    assert groups.meta["task"] == "myoElbowPose1D6MRandom-v0"
    assert len(groups.groups) == 3 and len(groups.muscle_names) == 6
