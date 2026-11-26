# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Backward-compatible package facade multiplexing resource lookups to ``myosuite/envs/assets``."""

import importlib.machinery as _importlib_machinery
from pathlib import Path as _Path

_canonical_init = str(_Path(__file__).resolve().parents[2] / "assets" / "__init__.py")
if __spec__ is not None:
    __spec__.origin = _canonical_init
    __spec__.loader = _importlib_machinery.SourceFileLoader(__name__, _canonical_init)
