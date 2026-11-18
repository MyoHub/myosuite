# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Experimental NVIDIA Newton compatibility probe helpers for MyoSuite models.

.. warning::
   **Experimental validation probe only.** MyoSuite's only two supported backends
   are CPU ``MyoGymnasiumEnv`` and GPU ``mjlab`` (see ``docs/source/newton_validation.rst``).
   Like the experimental MJX route, this integration exists solely for external
   solver compatibility diagnostics and may change or be removed without notice;
   do not build new MyoSuite tasks or training pipelines on it.

Submodules and symbols load lazily via PEP 562 so importing this package never
loads ``newton``, ``warp``, or ``torch``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from myosuite.integrations.newton.accessor import (
        IndexMapping,
        NewtonAccessor,
        resolve_env_xml,
    )

_EXPORTS_BY_MODULE: dict[str, tuple[str, ...]] = {
    "myosuite.integrations.newton.accessor": (
        "IndexMapping",
        "NewtonAccessor",
        "resolve_env_xml",
    ),
}

_NAME_TO_MODULE: dict[str, str] = {
    name: mod for mod, names in _EXPORTS_BY_MODULE.items() for name in names
}

__all__ = [
    "IndexMapping",
    "NewtonAccessor",
    "resolve_env_xml",
]


def __getattr__(name: str) -> Any:
    if name in _NAME_TO_MODULE:
        module = importlib.import_module(_NAME_TO_MODULE[name])
        return getattr(module, name)
    if name == "accessor":
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__, "accessor"])
