# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Summarize data-only run JSON files without importing a training framework."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from myosuite.utils.performance_card import RunRecord, summarize_runs


def main(argv: Sequence[str] | None = None) -> int:
    """Read one record or a list per input, and write a JSON performance card."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        runs = []
        for path in args.inputs:
            payload = json.loads(path.read_text(encoding="utf-8"))
            runs.extend(
                RunRecord.from_dict(row)
                for row in (payload if isinstance(payload, list) else [payload])
            )
        card = summarize_runs(
            runs,
            confidence=args.confidence,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        args.output.write_text(
            json.dumps(card, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    except (OSError, TypeError, ValueError, KeyError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
