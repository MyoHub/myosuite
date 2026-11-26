#!/usr/bin/env python3
# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Backward-compatible CLI forwarder to ``myosuite_ppo.mjlab_train`` (``training/ppo``).

Note: loading this file directly via ``importlib.util.spec_from_file_location`` returns this
wrapper module; import ``myosuite_ppo.mjlab_train`` directly when monkeypatching internals.
"""

from __future__ import annotations

import importlib as _importlib
import os as _os
from pathlib import Path as _Path
import sys as _sys
from typing import TYPE_CHECKING as _TYPE_CHECKING

_PPO_ROOT = _Path(__file__).resolve().parents[1] / "training" / "ppo"
if _PPO_ROOT.is_dir():
    _ppo_str = str(_PPO_ROOT)
    while _ppo_str in _sys.path:
        _sys.path.remove(_ppo_str)
    _sys.path.insert(0, _ppo_str)
    _py_parts = [
        p
        for p in _os.environ.get("PYTHONPATH", "").split(_os.pathsep)
        if p and p != _ppo_str
    ]
    _os.environ["PYTHONPATH"] = _os.pathsep.join([_ppo_str, *_py_parts])

try:
    _target = _importlib.import_module("myosuite_ppo.mjlab_train")
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("myosuite_ppo"):
        raise ImportError(
            "`scripts/train_mjlab.py` delegates to the standalone `myosuite-ppo` package. "
            "Install it from a MyoSuite checkout with: pip install -e '.[mjlab]' && pip install --no-deps -e training/ppo"
        ) from exc
    raise

if _TYPE_CHECKING:
    from myosuite_ppo.mjlab_train import *  # noqa: F403

main = _target.main

if __name__ == "__main__":
    main()
else:
    _sys.modules[__name__] = _target
