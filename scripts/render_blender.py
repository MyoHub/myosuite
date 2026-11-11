#!/usr/bin/env python3
# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Render a CPU rollout in Blender (see :mod:`myosuite.viz.blender_render`).

Example::

    python scripts/render_blender.py --env myoHandReorient8-v0 --random \
        --seconds 0.2 --output hand-render --preview
"""

from myosuite.viz.blender_render import main

if __name__ == "__main__":
    main()
