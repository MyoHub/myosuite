# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""``MuscleMimicClipEnvV0`` returns raw observations; tutorial 5.2 saves its normaliser."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from myosuite.envs.myo.tasks.mimic.clip_env import MuscleMimicClipEnvV0
from myosuite.tests.test_mimic_clip_episode import _write_standing_clip

pytestmark = pytest.mark.tier1

_TRAIN = (
    Path(__file__).resolve().parents[2]
    / "tutorials"
    / "files"
    / "5.2"
    / "train_mimic.py"
)


def test_default_observation_is_raw_and_reproducible(tmp_path: Path) -> None:
    clip = _write_standing_clip(tmp_path / "c.npz", 50, 0.0)
    fresh = MuscleMimicClipEnvV0(clip_path=clip, lookahead_k=1, lookahead_stride=1)
    used = MuscleMimicClipEnvV0(clip_path=clip, lookahead_k=1, lookahead_stride=1)
    first, _ = fresh.reset(seed=3)
    assert np.abs(first).max() > 0.0
    used.reset(seed=0)
    for _ in range(20):
        used.step(used.action_space.sample())
    again, _ = used.reset(seed=3)
    np.testing.assert_array_equal(first, again)
    fresh.close()
    used.close()


def test_tutorial_normaliser_matches_the_batch_statistics_and_is_saved() -> None:
    torch = pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("train_mimic", _TRAIN)
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    rng = np.random.default_rng(0)
    batches = [rng.normal(3.0, 2.0, (n, 5)) for n in (7, 40, 13)]
    policy = train.ActorCritic(5, 2, hidden_dim=8, n_layers=2)
    for batch in batches:
        policy.obs_norm.update(torch.as_tensor(batch, dtype=torch.float32))
    data = np.concatenate(batches)
    np.testing.assert_allclose(policy.obs_norm.mean, data.mean(0), rtol=1e-5)
    np.testing.assert_allclose(policy.obs_norm.var, data.var(0), rtol=1e-4)
    restored = train.ActorCritic(5, 2, hidden_dim=8, n_layers=2)
    restored.load_state_dict(policy.state_dict())
    np.testing.assert_array_equal(restored.obs_norm.mean, policy.obs_norm.mean)
