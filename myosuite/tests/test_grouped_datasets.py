# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Trace group iteration (myosuite/logger/grouped_datasets.py)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from myosuite.logger.grouped_datasets import Trace

pytestmark = pytest.mark.tier1


def _trace(n_groups: int = 2) -> Trace:
    trace = Trace("t")
    for i in range(n_groups):
        trace.create_group(f"Trial{i}")
        trace.append_datum(f"Trial{i}", "x", np.full(2, float(i)))
    return trace


def test_loops_after_items_see_every_group() -> None:
    """items() used to leave a shared cursor at the end: the next loop yielded nothing."""
    trace = _trace()
    assert [key for key, _ in trace.items()] == ["Trial0", "Trial1"]
    assert [group["x"][0][0] for group in trace] == [0.0, 1.0]
    assert len(list(trace.items())) == 2


def test_nested_and_abandoned_loops_are_independent() -> None:
    trace = _trace(3)
    # a shared cursor made the inner loop rewind the outer one forever: cap it
    pairs = list(itertools.islice(((a, b) for a in trace for b in trace), 20))
    assert len(pairs) == 9
    for _ in trace:
        break
    assert len(list(trace)) == 3
