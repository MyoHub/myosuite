# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Array helpers shared by the backend-agnostic term modules."""

from __future__ import annotations

from typing import Any


def xp_asarray(
    xp: Any,
    value: Any,
    *,
    dtype: Any | None = None,
    device: Any | None = None,
) -> Any:
    """Convert *value* to the backend array type used by *xp*.

    Args:
        xp: Array module from ``accessor.array_module()`` (NumPy, JAX or torch).
        value: Array-like input.
        dtype: Optional dtype of the result.
        device: Optional torch device; ignored by the other backends.

    Returns:
        *value* as an array of *xp*.
    """
    if getattr(xp, "__name__", "") == "torch":
        tensor = (
            xp.as_tensor(value, device=device)
            if device is not None
            else xp.as_tensor(value)
        )
        return tensor.to(dtype=dtype) if dtype is not None else tensor
    return xp.asarray(value, dtype=dtype) if dtype is not None else xp.asarray(value)
