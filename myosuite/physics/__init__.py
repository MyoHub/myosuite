# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Biomechanics and physics utilities for MyoSuite environments.

Available classes
-----------------
:class:`CumulativeFatigue`
    NumPy implementation of the 3CC-r cumulative muscle fatigue model
    (Xia & Frey Law, 2008).  Backend: CPU (numpy).

:class:`CumulativeFatigueJax`
    JAX-compatible version of the same fatigue model for use in MJX
    environments.  Backend: MJX (jax.numpy).

Quaternion math
---------------
All three array backends share the same function signatures.  Use the
dispatcher so call sites stay backend-agnostic::

    from myosuite.physics.quat import get_quat_ops
    qm = get_quat_ops(accessor.array_module())
    q  = qm.mul_quat(qa, qb)

Direct backend imports (when the backend is statically known):

- ``myosuite.physics.quat_math``        — NumPy
- ``myosuite.physics.quat_math_jax``    — JAX / MJX
- ``myosuite.physics.quat_math_torch``  — PyTorch / mjlab
"""

from __future__ import annotations

from typing import Any

from myosuite.physics.fatigue import CumulativeFatigue

__all__ = ["CumulativeFatigue", "CumulativeFatigueJax"]


def __getattr__(name: str) -> Any:
    """Lazy-export JAX fatigue model (requires ``myosuite[mjx]`` / JAX)."""
    if name == "CumulativeFatigueJax":
        from myosuite.physics.fatigue_jax import CumulativeFatigue

        return CumulativeFatigue
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
