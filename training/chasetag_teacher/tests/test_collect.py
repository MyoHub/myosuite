# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Contract tests for the ChaseTag teacher collector (no teacher download)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import pytest

from chasetag_teacher.collect import iter_pairs, write_pairs

pytestmark = pytest.mark.tier1


class _FakeEnv:
    def __init__(self) -> None:
        self.unwrapped = SimpleNamespace(model=None, data=None, _meta=None)
        self._t = 0

    def reset(self, seed: int = 0) -> tuple[dict, dict]:
        self._t = 0
        return {"agent_0": np.zeros(546, np.float32)}, {"task": "CHASE"}

    def step(self, _acts: dict) -> tuple:
        self._t += 1
        done = self._t >= 3
        return (
            {"agent_0": np.ones(546, np.float32) * self._t},
            {"agent_0": 0.0},
            {"agent_0": done},
            {"agent_0": False},
            {"task": "CHASE"},
        )


class _FakePair:
    def reset(self, *_a: object) -> None:
        return None

    def actions(self, *_a: object) -> dict[str, np.ndarray]:
        return {"agent_0": np.full(354, 0.2, np.float32)}


def test_write_pairs_roundtrip(tmp_path) -> None:
    path = write_pairs(
        tmp_path / "pairs.npz",
        np.zeros((2, 546)),
        np.ones((2, 354)),
        np.array(["CHASE", "EVADE"]),
    )
    data = np.load(path, allow_pickle=True)
    assert data["obs"].shape == (2, 546)
    assert data["act"].shape == (2, 354)
    assert list(data["task"]) == ["CHASE", "EVADE"]


def test_iter_pairs_stops_on_term() -> None:
    rows = list(iter_pairs(_FakeEnv(), _FakePair(), episodes=1, steps=10))
    assert len(rows) == 3
    assert rows[0][0].shape == (546,)
    assert rows[0][1].shape == (354,)
    assert rows[0][2] == "CHASE"
