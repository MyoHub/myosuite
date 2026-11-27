# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Optional ``amathislab/mm-evalkit`` integration glue for MyoSuite."""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, str] = {
    "CheckoutCapabilityError": "myosuite.integrations.mm_evalkit.workflow",
    "MODEL_SPEC_CAPABILITIES": "myosuite.integrations.mm_evalkit.workflow",
    "compare_emg_reports_mm_evalkit": "myosuite.integrations.mm_evalkit.stats_and_config",
    "compute_rsm_rsa": "myosuite.integrations.mm_evalkit.stats_and_config",
    "compute_wilcoxon_comparison": "myosuite.integrations.mm_evalkit.stats_and_config",
    "dump_policies_yaml": "myosuite.integrations.mm_evalkit.stats_and_config",
    "export_mm_evalkit_rollout": "myosuite.integrations.mm_evalkit.rollout_export",
    "run_emg_validation": "myosuite.integrations.mm_evalkit.workflow",
    "write_mm_evalkit_policy_config": "myosuite.integrations.mm_evalkit.rollout_export",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        mod = importlib.import_module(_EXPORTS[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
