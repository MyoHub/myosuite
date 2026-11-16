# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Thin CLI entry point for musculoskeletal EMG validation and PR A/B comparisons."""

from __future__ import annotations

from collections.abc import Sequence

from myosuite.integrations.mm_evalkit.workflow import (
    build_parser,
    build_source,
    load_target_model,
    registered_dataset_choices,
    run_emg_validation,
    trial_optimization_config,
)

DATASET_CHOICES: tuple[str, ...] = registered_dataset_choices()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the shared EMG validation workflow from the command line."""
    return run_emg_validation(argv)


__all__ = [
    "DATASET_CHOICES",
    "build_parser",
    "build_source",
    "load_target_model",
    "main",
    "trial_optimization_config",
]


if __name__ == "__main__":
    raise SystemExit(main())
