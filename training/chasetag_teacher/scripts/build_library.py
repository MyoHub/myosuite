# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Build (and cache) the steering clip library: python scripts/build_library.py [--root DIR] [--limit N]."""

import argparse

from chasetag_teacher.library import load_library

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument(
    "--root",
    default=None,
    help="folder of retargeted KIT clips (default: $MYOSUITE_KIT_ROOT)",
)
ap.add_argument(
    "--limit", type=int, default=None, help="use only the first N locomotion clips"
)
a = ap.parse_args()
lib = load_library(a.root, a.limit)
print(
    len(lib.names),
    "clips",
    lib.m.num_frames,
    "frames",
    int(lib.valid.sum()),
    "valid transition frames",
)
