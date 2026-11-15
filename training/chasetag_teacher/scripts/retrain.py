# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Fit an FBVs student offline on DAgger data.

    python scripts/retrain.py --out s.pt --data <dagger dir>/data/r0 <dagger dir>/data/r1 ... [--canon] [--augment]

Writes the ActorCritic state dict, ``<out>_full.pt`` and ``<out>.onnx`` (``export_fbvs_onnx``; with ``--canon`` the
ONNX canonicalises the raw observation first, ``export_canon_onnx``).
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from chasetag_teacher.augment import half_turn
from chasetag_teacher.distill import (
    canonicalise_inplace,
    export_canon_onnx,
    load_data,
    train,
)
from myosuite.integrations.musclemimic.fbvs_player import export_fbvs_onnx

ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
ap.add_argument("--out", required=True)
ap.add_argument(
    "--data", nargs="+", required=True, help="folders of bout files (b*.npz)"
)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument(
    "--init",
    default="",
    help="a student_r<k>_full.pt to start from (same observation kind)",
)
ap.add_argument(
    "--augment", action="store_true", help="add every sample's half-turn copy"
)
ap.add_argument(
    "--yaw-aug",
    action="store_true",
    help="rotate every sample about z by a random angle",
)
ap.add_argument(
    "--canon",
    action="store_true",
    help="train on yaw-canonicalised observations; the ONNX canonicalises",
)
a = ap.parse_args()
X, Y = load_data([Path(d) for d in a.data])
if a.augment:
    X, Y = np.concatenate((X, half_turn(X))), np.concatenate((Y, Y))
print(len(X), "samples", flush=True)
if a.canon:
    canonicalise_inplace(X)
init = torch.load(a.init, weights_only=True) if a.init else None
res = train(X, Y, epochs=a.epochs, batch=a.batch, lr=a.lr, device=a.device, init=init, yaw_aug=a.yaw_aug,
            log=lambda s: print(s, flush=True))  # fmt: skip
res["canon"] = a.canon
out = Path(a.out)
torch.save(res["state"], out)
torch.save(res, out.with_name(out.stem + "_full.pt"))
if a.canon:
    export_canon_onnx(res["state"], out.with_suffix(".onnx"))
else:
    export_fbvs_onnx(out, out.with_suffix(".onnx"))
print("val", res["val_mse"], "wrote", out, out.with_suffix(".onnx"))
