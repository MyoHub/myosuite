# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Unified quaternion math entry point — backend-dispatched.

Usage (in term functions or any backend-aware code)::

    from myosuite.physics.quat import get_quat_ops

    xp = accessor.array_module()   # numpy / jax.numpy / torch
    qm = get_quat_ops(xp)
    q  = qm.mul_quat(qa, qb)      # same call, right backend

Usage (when the array module is already known at import time)::

    from myosuite.physics import quat_math        # numpy
    from myosuite.physics import quat_math_jax    # JAX
    from myosuite.physics import quat_math_torch  # Torch

All three backends expose identical function signatures with [w, x, y, z]
quaternion convention matching MuJoCo.
"""

from __future__ import annotations

import types
from typing import Any


def get_quat_ops(xp: Any) -> types.ModuleType:
    """Return the quaternion math module that matches the array backend *xp*.

    Args:
        xp: An array module — ``numpy``, ``jax.numpy``, or ``torch``.

    Returns:
        The corresponding quat_math module with the same public API.

    Raises:
        TypeError: If *xp* is not a recognised array module.
    """
    name = getattr(xp, "__name__", "") or ""

    if "torch" in name:
        from myosuite.physics import quat_math_torch

        return quat_math_torch

    if "jax" in name:
        from myosuite.physics import quat_math_jax

        return quat_math_jax

    # numpy and numpy-compatible (e.g. cupy)
    try:
        import numpy as np

        if xp is np or getattr(xp, "__name__", "").startswith("numpy"):
            from myosuite.physics import quat_math

            return quat_math
    except ImportError:
        pass

    raise TypeError(
        f"get_quat_ops: unrecognised array module {xp!r}. "
        "Expected numpy, jax.numpy, or torch."
    )


def quat_yaw(xp: Any, quat: Any) -> Any:
    """Yaw (rotation about world ``z``) of ``wxyz`` quaternions.

    Args:
        xp: Array module (numpy, jax.numpy or torch).
        quat: Quaternions ``(..., 4)`` in MuJoCo's ``(w, x, y, z)`` order.

    Returns:
        Yaw angles ``(...)`` in radians.
    """
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return xp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
