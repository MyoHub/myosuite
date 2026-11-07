# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Train a MuscleMimic full-body policy with SuperMimic (world model + BPTT actor-critic)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import tyro

from myosuite.learning.supermimic.trainer import SuperMimicCfg, SuperMimicTrainer


def main(cfg: SuperMimicCfg) -> None:
    Path(cfg.out).mkdir(parents=True, exist_ok=True)
    (Path(cfg.out) / "cfg.json").write_text(json.dumps(asdict(cfg), indent=1))
    SuperMimicTrainer(cfg).train()


if __name__ == "__main__":
    main(tyro.cli(SuperMimicCfg))
