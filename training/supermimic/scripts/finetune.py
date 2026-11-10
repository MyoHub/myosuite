# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Fine-tune a trained ``myoMimic-v0`` baseline on a few clips (see
``supermimic.finetune``).

    python training/supermimic/scripts/finetune.py --checkpoint <baseline run>/model_008000.pt \
        --motion-file my_clips.txt --motion-root <dataset root> --iterations 2000 --out logs/ft_walk
"""

import json

import tyro

from supermimic.finetune import FinetuneCfg, finetune

if __name__ == "__main__":
    res = finetune(tyro.cli(FinetuneCfg))
    keys = ("norm_return", "n_failed", "mpjpe_mm", "joint_rmse_deg")
    print(
        json.dumps(
            {k: {m: res[k][m] for m in keys} for k in ("before", "after") if k in res},
            indent=1,
        )
    )
