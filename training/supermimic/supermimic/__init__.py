# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""myoSuperMimic: SuperMimic on MyoSuite's multi-clip full-body tracking task (``myoMimic-v0``).

A port of muscle-supermimic (github.com/kaibohex/muscle-supermimic, Apache-2.0, commit 8a6244b), which ports its
SuperMimic runner from supermimic_vibed (MIT) and its fast recipe from BlinkMimic and InstantTrack (MIT): see
``NOTICE-muscle-supermimic.txt`` and the ``LICENSE-*.txt`` files of this project (``training/supermimic/``).
SuperMimic is Liao et al., arXiv:2610.28480. Every ported file names its source.

The task (observation, reward, failure rule, reference motion) is MyoSuite's; this method builds it with ``make_env``
and evaluates the task's pure term functions (``myosuite.terms.mimic_track``) on imagined states, through its own
differentiable kinematics, muscle geometry and activation dynamics. The task never imports this method.

Usage, install and results: ``training/supermimic/README.md``. Command lines: ``training/supermimic/scripts/``.
"""

from importlib.util import find_spec

_missing = [m for m in ("torch", "mjlab", "mujoco_warp") if find_spec(m) is None]
if _missing:
    raise ImportError(
        f"supermimic needs {', '.join(_missing)}: install this project "
        "(pip install training/supermimic), which pulls in myosuite[mjlab]"
    )
