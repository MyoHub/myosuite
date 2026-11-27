# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Key parsing for MuscleMimic actor parameter trees (NumPy only; no torch or JAX)."""

from __future__ import annotations

from typing import Any


def sorted_prefix_indices(tree: dict[str, Any], prefix: str) -> list[int]:
    """Return the sorted integer suffixes of keys like ``f"{prefix}{i}"``.

    Args:
        tree: Parameter tree (a dict keyed by layer name).
        prefix: Key prefix, for example ``"Dense_"``.

    Returns:
        The integers ``i`` of the matching keys, ascending.
    """
    indices: list[int] = []
    for key in tree:
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix) :]
        if suffix.isdigit():
            indices.append(int(suffix))
    return sorted(indices)


def residual_block_indices(actor: dict[str, Any]) -> list[int]:
    """Return the sorted residual block indices present in an actor tree.

    Args:
        actor: Actor parameter tree with ``block{i}_layer0_dense`` keys.

    Returns:
        The block indices ``i``, ascending.
    """
    indices: list[int] = []
    suffix = "_layer0_dense"
    for key in actor:
        if not key.startswith("block") or not key.endswith(suffix):
            continue
        idx = key[len("block") : -len(suffix)]
        if idx.isdigit():
            indices.append(int(idx))
    return sorted(indices)
