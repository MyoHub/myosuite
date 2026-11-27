#!/usr/bin/env python3
# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Standalone wheel smoke verification without requiring ``myosuite.tests``.

Kept outside ``myosuite/`` so pre- and post-release CI jobs can verify an
isolated wheel install from an empty temporary directory without a repository
checkout on ``sys.path`` (where ``myosuite.tests`` is intentionally absent),
reusing :func:`myosuite.utils.env_check.check_single_env`.
"""

from __future__ import annotations

import argparse
import importlib.util

import gymnasium as gym

import myosuite
from myosuite.core.hf_io import is_gated_or_transient_hf_error
from myosuite.envs.myo.tasks.mimic.tracking import MissingMotionData
from myosuite.utils.env_check import check_single_env, is_multi_agent_env_spec


def verify_single_env(env_id: str, *, seed: int = 1234, steps: int = 3) -> None:
    """Run :func:`check_single_env` (including dict/proprio/extero/seed checks) plus extra steps."""
    check_single_env(
        env_id,
        seed,
        extra_steps=max(0, steps - 1),
        respect_lite_env=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-version", default=None, help="Assert exact myosuite.__version__."
    )
    parser.add_argument(
        "--allow-source-checkout",
        action="store_true",
        help="Skip the wheel-only assertion that myosuite.tests is absent.",
    )
    parser.add_argument(
        "--steps", type=int, default=3, help="Steps per verified environment."
    )
    parser.add_argument(
        "--limit-per-suite",
        type=int,
        default=0,
        help="Optional cap per suite for local dry runs (0 = verify every env).",
    )
    args = parser.parse_args(argv)

    if args.expected_version is not None:
        assert (
            myosuite.__version__ == args.expected_version
        ), f"Expected myosuite=={args.expected_version}, got {myosuite.__version__}"
    if not args.allow_source_checkout:
        assert (
            importlib.util.find_spec("myosuite.tests") is None
        ), "myosuite.tests leaked into installed wheel"

    suites = myosuite.register_all_envs()
    required_suites = ("myobase", "myochallenge", "myomimic", "myoedits")
    for s in required_suites:
        assert suites.get(
            s
        ), f"Required suite {s!r} is empty: { {k: len(v) for k, v in suites.items()} }"

    verified_counts: dict[str, int] = {}
    skipped_counts: dict[str, int] = {}
    for suite_name in required_suites:
        env_ids = suites.get(suite_name, ())
        verified = 0
        skipped = 0
        for env_id in sorted(env_ids):
            if is_multi_agent_env_spec(gym.registry[env_id]):
                skipped += 1
                continue
            try:
                verify_single_env(env_id, steps=args.steps)
            except Exception as exc:
                if isinstance(exc, MissingMotionData) or is_gated_or_transient_hf_error(
                    exc
                ):
                    print(f"Skipping gated/transient HF env {env_id}: {exc}")
                    skipped += 1
                    continue
                raise
            verified += 1
            if args.limit_per_suite and verified >= args.limit_per_suite:
                break
        assert (
            verified > 0
        ), f"Suite {suite_name!r} verified 0 environments (skipped={skipped})"
        verified_counts[suite_name] = verified
        skipped_counts[suite_name] = skipped

    print("Verified wheel suites:", verified_counts, "skipped:", skipped_counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
