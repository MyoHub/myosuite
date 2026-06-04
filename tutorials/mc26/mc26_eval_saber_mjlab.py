#!/usr/bin/env python3
# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Play the mjlab Saber task with ONNX checkpoints by default."""

from __future__ import annotations

import sys

import myosuite

from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
    bootstrap_myosuite_mjlab_registry,
)
from myosuite.viz.mjlab_play import main as play_main

DEFAULT_TASK = "myoChallengeSaberP0-v0"


def _inject_default_task(argv: list[str]) -> list[str]:
    if not argv or argv[0].startswith("-"):
        return [DEFAULT_TASK, *argv]
    return argv


def main(argv: list[str] | None = None) -> int:
    """Register Saber and delegate to ``mjlab_play`` with the default task."""
    cli_args = list(sys.argv[1:] if argv is None else argv)
    myosuite.register_all_envs()
    bootstrap_myosuite_mjlab_registry()
    return play_main(_inject_default_task(cli_args))


if __name__ == "__main__":
    sys.argv[0] = sys.argv[0].removesuffix(".exe")
    raise SystemExit(main())
