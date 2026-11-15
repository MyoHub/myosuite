# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Trace group iteration (myosuite/logger/grouped_datasets.py)."""

from __future__ import annotations

import itertools
from pathlib import Path

import h5py
import numpy as np
import pytest

from myosuite.logger import grouped_datasets
from myosuite.logger.grouped_datasets import Trace, TraceType

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


def test_get_selects_group_dataset_or_item() -> None:
    trace = _trace(1)
    trace.append_datum("Trial0", "x", np.array([2.0, 3.0]))
    assert trace.get("Trial0") is trace.trace["Trial0"]
    assert trace.get("Trial0", "x") is trace.trace["Trial0"]["x"]
    np.testing.assert_array_equal(trace.get("Trial0", "x", 0), [0.0, 0.0])
    np.testing.assert_array_equal(trace.get("Trial0", "x", -1), [2.0, 3.0])
    np.testing.assert_array_equal(trace.get("Trial0", "x", slice(1, 2)), [[2.0, 3.0]])
    with pytest.raises(KeyError):
        trace.get("missing")
    with pytest.raises(KeyError):
        trace.get("Trial0", "missing")
    with pytest.raises(IndexError):
        trace.get("Trial0", "x", 2)


@pytest.mark.parametrize("trace_type", list(TraceType))
def test_trace_type_enum_identity_is_preserved(trace_type: TraceType) -> None:
    assert TraceType.get_type(trace_type) is trace_type


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("robohive", TraceType.ROBOHIVE),
        ("RoBoHiVe", TraceType.ROBOHIVE),
        ("ROBOSET", TraceType.ROBOSET),
    ],
)
def test_trace_type_string_aliases(name: str, expected: TraceType) -> None:
    assert TraceType.get_type(name) is expected


@pytest.mark.parametrize("unknown", ["unknown", "", None, 42])
def test_unknown_trace_type_falls_back_with_a_warning(
    unknown: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    warnings: list[object] = []
    monkeypatch.setattr(
        grouped_datasets, "prompt", lambda message, **kwargs: warnings.append(message)
    )
    assert TraceType.get_type(unknown) is TraceType.UNSET
    assert len(warnings) == 1
    assert "TraceType.UNSET" in str(warnings[0])


@pytest.mark.parametrize("suffix", [".h5", ".pickle"])
@pytest.mark.parametrize("trace_type", [None, TraceType.ROBOSET])
def test_saved_trace_dataset_access_and_enum_loading(
    tmp_path: Path, suffix: str, trace_type: TraceType | None
) -> None:
    trace = _trace(1)
    path = tmp_path / f"trace{suffix}"
    trace.save(str(path))
    loaded = (
        Trace.load(str(path))
        if trace_type is None
        else Trace.load(str(path), trace_type=trace_type)
    )
    try:
        assert loaded.trace_type is (
            TraceType.UNSET if trace_type is None else trace_type
        )
        group = loaded.get("Trial0")
        dataset = loaded.get("Trial0", "x")
        if suffix == ".h5":
            assert isinstance(group, h5py.Group)
            assert isinstance(dataset, h5py.Dataset)
        else:
            assert dataset is group["x"]
        np.testing.assert_array_equal(dataset[:], [[0.0, 0.0]])
        np.testing.assert_array_equal(loaded.get("Trial0", "x", 0), [0.0, 0.0])
    finally:
        if isinstance(loaded.trace, h5py.File):
            loaded.trace.close()
