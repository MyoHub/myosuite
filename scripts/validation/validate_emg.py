# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Thin CLI entry point for musculoskeletal EMG validation and PR A/B comparisons."""

from __future__ import annotations

from collections.abc import Sequence

from myosuite.integrations.mm_evalkit.workflow import (
    registered_dataset_choices,
    run_emg_validation,
)

DATASET_CHOICES: tuple[str, ...] = registered_dataset_choices()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the shared EMG validation workflow from the command line."""
    return run_emg_validation(argv)


if __name__ == "__main__":
    raise SystemExit(main())
