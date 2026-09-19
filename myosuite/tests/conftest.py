# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Pytest configuration for MyoSuite tests."""

from __future__ import annotations

import sys
import warnings
from typing import Any

import pytest

# Tier markers (see pyproject.toml [tool.pytest.ini_options].markers).
_TIER_MARKERS = frozenset({"tier1", "tier2", "tier3"})
# Test modules without an explicit tier marker are treated as this tier, so
# `pytest -m tierN` never silently drops a whole file. tier2 = "extended";
# a warning below lists offenders so they can be given an explicit tier.
_DEFAULT_TIER = "tier2"


def _assign_default_tiers(items: list[Any]) -> None:
    """Give every test an explicit tier so `-m tierN` selection is complete.

    Any item whose module declares no tier marker gets ``_DEFAULT_TIER`` and its
    module is collected into a single warning, so untagged files surface without
    being silently excluded from tier-filtered runs.
    """
    untagged_modules: set[str] = set()
    default_mark = getattr(pytest.mark, _DEFAULT_TIER)
    for item in items:
        if any(m.name in _TIER_MARKERS for m in item.iter_markers()):
            continue
        item.add_marker(default_mark)
        module = getattr(item, "module", None)
        untagged_modules.add(getattr(module, "__name__", item.nodeid.split("::")[0]))
    if untagged_modules:
        listing = ", ".join(sorted(untagged_modules))
        warnings.warn(
            f"{len(untagged_modules)} test module(s) have no tier marker and were "
            f"defaulted to {_DEFAULT_TIER}; add `pytestmark = pytest.mark.tierN` "
            f"to tier them explicitly: {listing}",
            stacklevel=2,
        )


# MuJoCo Warp + full biped mjlab leg walk has crashed the interpreter on macOS
# (segfault in native forward). Linux CI is the supported environment for these.
_DARWIN_MJLAB_LEG_WALK_SKIP_REASON = (
    "mjlab myoLegWalk-v0 / myoSarcLegWalk-v0 with MuJoCo Warp is unstable on macOS "
    "(native segfault); run on Linux (e.g. GitHub Actions ubuntu job)."
)

# Tests that always construct mjlab biped leg walk on CPU/GPU (non-parametrized).
_DARWIN_SKIP_ORIGINAL_NAMES: frozenset[str] = frozenset(
    {
        "test_mjlab_parallel_qacc_consistency_zero_state",
        "test_walk_tier_a_dense_reward_gate_cpu_vs_mjlab",
        "test_myo_leg_walk_reward_parity_cpu_vs_mjlab",
        "test_myo_leg_walk_reward_manager_matches_term_functions",
        "test_myo_leg_walk_state_parity_cpu_vs_mjlab",
        "test_myo_leg_walk_initial_state_difference_cpu_vs_mjlab",
        "test_myo_leg_walk_ctrl_mapping_cpu_vs_mjlab",
        "test_myo_leg_walk_forced_initial_parity_one_step_stats",
    }
)


def pytest_collection_modifyitems(config: pytest.Config, items: list[Any]) -> None:
    """Assign default tiers, then skip mjlab biped leg-walk on macOS (Warp instability)."""
    _ = config  # hook name is part of the pytest API contract
    _assign_default_tiers(items)
    if sys.platform != "darwin":
        return
    skip = pytest.mark.skip(reason=_DARWIN_MJLAB_LEG_WALK_SKIP_REASON)
    for item in items:
        orig = getattr(item, "originalname", None) or item.name
        if orig == "test_make_reset_step_supported_tasks":
            nid = item.nodeid
            if "[myoLegWalk-v0]" in nid or "[myoSarcLegWalk-v0]" in nid:
                item.add_marker(skip)
            continue
        if orig in _DARWIN_SKIP_ORIGINAL_NAMES:
            item.add_marker(skip)
