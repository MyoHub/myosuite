# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""The steering clip library: the KIT locomotion clips of the tracker's training split, cached on disk.

Loading and evaluating the kinematics of ~800 clips takes minutes, so their motion library is stored in
``$CHASETAG_TEACHER_CACHE`` (default ``~/.cache/myosuite/chasetag_teacher``) by the task's motion cache: NumPy
arrays without pickles, keyed by the bytes of every clip, the compiled body, the tracked bodies, the control step
and the preprocessing code. The library's features are recomputed from it on every load.
"""

from __future__ import annotations

import os
from pathlib import Path

from myosuite.biomechanics.retarget import check_refit_meta
from myosuite.envs.myo.tasks.mimic.tracking import (
    KIT_ROOT_ENV,
    MissingMotionData,
    kit_split,
)
from myosuite.envs.myo.tasks.mimic.tracking.models import (
    compiled_scene,
    model_cfg,
    model_version,
    root_offset,
)
from myosuite.envs.myo.tasks.mimic.tracking.motion import build_motion_lib

from .reference import ClipLibrary, locomotion_files

MODEL = "myofullbody"
"""The body of the FBVs players and of the tracker."""
CACHE_ENV = "CHASETAG_TEACHER_CACHE"
"""Environment variable naming the library cache folder."""


def cache_dir() -> Path:
    """The library cache folder."""
    return Path(
        os.environ.get(
            CACHE_ENV, Path.home() / ".cache" / "myosuite" / "chasetag_teacher"
        )
    )


def load_library(
    root: str | None = None, limit: int | None = None, ctrl_dt: float = 0.02
) -> ClipLibrary:
    """The KIT locomotion clips of MuscleMimic's training split as a :class:`ClipLibrary` (motion data cached).

    Args:
        root: The folder of retargeted clips (default ``$MYOSUITE_KIT_ROOT``), as ``scripts/mimic/retarget_amass.py``
            writes it.
        limit: Use only the first ``limit`` locomotion clips (tests, quick runs).
        ctrl_dt: Control step of the tracker (s).

    Returns:
        The library.

    Raises:
        MissingMotionData: No clip folder is given or the clips are missing.
    """
    root = root or os.environ.get(KIT_ROOT_ENV, "")
    if not root or not Path(root).is_dir():
        raise MissingMotionData(
            f"no retargeted KIT clips: set ${KIT_ROOT_ENV} or pass root= (got {root!r})"
        )
    version = model_version(MODEL)
    files = locomotion_files(str(kit_split("train")), root, limit)
    missing = [f for f in files if not Path(f).is_file()]
    if missing:
        raise MissingMotionData(
            f"{len(missing)} of {len(files)} clips missing under {root} (e.g. {missing[0]})"
        )
    for f in files:
        check_refit_meta(f, MODEL, version)
    m = compiled_scene(MODEL)
    motion = build_motion_lib(
        files, m, model_cfg(MODEL).tracked_bodies, ctrl_dt, cache_dir=cache_dir()
    )
    return ClipLibrary(motion, root_offset(m))
