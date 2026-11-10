# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The SuperMimic player in the scored game. Needs retargeted KIT clips and a tracker checkpoint (skipped otherwise)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chasetag_supermimic.distill import collect_bout
from chasetag_supermimic.match import IdlePlayer, SmPlayer, make, play

pytestmark = [pytest.mark.tier2, pytest.mark.data]


@pytest.fixture(scope="module", name="lib")
def lib_fixture(kit_root: str) -> object:
    from chasetag_supermimic.library import load_library

    return load_library(kit_root, limit=12)


def test_library_holds_locomotion_clips(lib: object) -> None:
    assert len(lib.names) == 12
    assert all(
        "walk" in n.lower() or "turn" in n.lower() or "circle" in n.lower()
        for n in lib.names
    )
    assert int(lib.valid.sum()) > 0


def test_player_chases_an_idle_player(lib: object, tracker_checkpoint: str) -> None:
    env = make()
    acts = []
    players = {"agent_0": SmPlayer(tracker_checkpoint, lib), "agent_1": IdlePlayer()}
    bout = play(
        env,
        players,
        seed=0,
        task="CHASE",
        max_steps=60,
        record=lambda e: acts.append(e.unwrapped.data.ctrl.copy()),
    )
    env.close()
    assert bout.end in ("tag", "time_up", "chaser_fall")
    assert bout.fell["agent_0"] is None  # the tracker keeps its player up for 0.6 s
    a = np.array(acts)
    assert np.isfinite(a).all() and (a >= 0).all() and (a <= 1).all()


def test_dagger_bout_stores_546_354_pairs(
    lib: object, tracker_checkpoint: str, tmp_path: Path
) -> None:
    env = make()
    r = collect_bout(env, SmPlayer(tracker_checkpoint, lib), None, IdlePlayer(), None, 1.0, seed=0, task="CHASE",
                     noise=0.03, max_steps=20)  # fmt: skip
    env.close()
    assert r["obs"].shape == (20, 546) and r["act"].shape == (20, 354)
