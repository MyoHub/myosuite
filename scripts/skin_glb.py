#!/usr/bin/env python3
# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Round-trip a body skin through a rigged .glb, to edit its blend weights in Blender.

Example::

    python scripts/skin_glb.py export skin.glb                  # bundled skin -> .glb
    # Blender: File > Import > glTF, Weight Paint, File > Export > glTF Binary
    #          (Data > Armature > "Include All Bone Influences" keeps more than 4 per vertex)
    python scripts/skin_glb.py import skin_edited.glb skin.skn
"""

import argparse
from pathlib import Path

from myosuite import make_env
from myosuite.viz.skin import load_skn, save_skn
from myosuite.viz.skin_glb import read_skin_glb, write_skin_glb

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument(
    "--env", default="myoMimicFullbody-v0", help="Model the skin binds to."
)
sub = parser.add_subparsers(dest="command", required=True)
export = sub.add_parser("export", help=".skn -> rigged .glb in the model's rest pose")
export.add_argument("output", type=Path)
export.add_argument(
    "--skin", default="fullbody", help='.skn file, or "fullbody" (bundled).'
)
load = sub.add_parser("import", help="rigged .glb -> .skn")
load.add_argument("glb", type=Path)
load.add_argument("output", type=Path)
args = parser.parse_args()

env = make_env(args.env)
model = env.unwrapped.model
if args.command == "export":
    write_skin_glb(load_skn(args.skin), model, args.output)
else:
    save_skn(read_skin_glb(args.glb, model), args.output)
env.close()
print(f"wrote {args.output}")
