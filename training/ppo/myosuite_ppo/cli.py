# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Unified ``myosuite-ppo`` CLI dispatching to CPU SB3 or GPU ``mjlab`` trainers."""

from __future__ import annotations

import sys

_USAGE = """\
usage: myosuite-ppo <backend> [args...]

Backends:
  sb3     CPU Stable-Baselines3 trainer (PPO, SAC, TD3)
  mjlab   GPU mjlab + RSL-RL PPO trainer
"""


def main(argv: list[str] | None = None) -> int:
    """Dispatch ``myosuite-ppo`` subcommands."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(_USAGE.rstrip())
        return 0

    backend, rest = args[0], args[1:]
    if backend == "sb3":
        from myosuite_ppo.sb3_train import main as sb3_main

        sb3_main(rest)
        return 0
    if backend == "mjlab":
        from myosuite_ppo.mjlab_train import main as mjlab_main

        mjlab_main(rest)
        return 0

    print(
        f"error: unknown backend {backend!r} (expected 'sb3' or 'mjlab')",
        file=sys.stderr,
    )
    print(_USAGE.rstrip(), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
