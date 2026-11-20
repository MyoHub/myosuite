# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Experimental batched NVIDIA Newton compatibility probe for MyoSuite models.

.. warning::
   **Experimental validation probe only.** MyoSuite's only two supported backends
   are CPU ``MyoGymnasiumEnv`` and GPU ``mjlab`` (see ``docs/source/newton_validation.rst``).
   Like the experimental MJX route, :class:`NewtonBatchSim` and :class:`NewtonAccessor`
   exist solely for external solver compatibility diagnostics without registering a
   third ``make_env`` backend, and may change or be removed without notice; do not
   build new MyoSuite tasks or training pipelines on this module.

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
    from myosuite.integrations.newton.batch_sim import NewtonBatchSim
    from myosuite.integrations.newton.playback import (
        ClosedLoopParityReport,
        rollout_closed_loop_parity,
    )
    from myosuite.integrations.newton.task_env import (
        NewtonRslRlVecEnv,
        NewtonTaskEnv,
    )
    from myosuite.integrations.newton.torch_pipeline import (
        NewtonActionCfg,
        NewtonActionPipeline,
        TorchNewtonAccessor,
    )

_EXPORTS_BY_MODULE: dict[str, tuple[str, ...]] = {
    "myosuite.integrations.newton.accessor": (
        "IndexMapping",
        "NewtonAccessor",
        "resolve_env_xml",
    ),
    "myosuite.integrations.newton.batch_sim": ("NewtonBatchSim",),
    "myosuite.integrations.newton.playback": (
        "ClosedLoopParityReport",
        "rollout_closed_loop_parity",
    ),
    "myosuite.integrations.newton.task_env": (
        "NewtonRslRlVecEnv",
        "NewtonTaskEnv",
    ),
    "myosuite.integrations.newton.torch_pipeline": (
        "NewtonActionCfg",
        "NewtonActionPipeline",
        "TorchNewtonAccessor",
    ),
}

_NAME_TO_MODULE: dict[str, str] = {
    name: mod for mod, names in _EXPORTS_BY_MODULE.items() for name in names
}

_SUBMODULES = ("accessor", "batch_sim", "playback", "task_env", "torch_pipeline")

__all__ = [
    "ClosedLoopParityReport",
    "IndexMapping",
    "NewtonAccessor",
    "NewtonActionCfg",
    "NewtonActionPipeline",
    "NewtonBatchSim",
    "NewtonRslRlVecEnv",
    "NewtonTaskEnv",
    "TorchNewtonAccessor",
    "resolve_env_xml",
    "rollout_closed_loop_parity",
]


def __getattr__(name: str) -> Any:
    if name in _NAME_TO_MODULE:
        module = importlib.import_module(_NAME_TO_MODULE[name])
        return getattr(module, name)
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__, *_SUBMODULES])
