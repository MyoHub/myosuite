# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""mjlab backend task discovery module.

.. warning::
    **Experimental** — the mjlab integration is a work in progress.  The API
    (task configs, ``REGISTERED_TASKS`` keys, observation layouts) may change
    without notice until this warning is removed.  Do not depend on this module
    in production code.

This module is the entry-point for mjlab's auto-discovery mechanism, registered
in ``pyproject.toml`` as::

    [project.entry-points."mjlab.tasks"]
    myosuite = "myosuite.envs.myo.backends.mjlab"

When mjlab is installed it imports this module and consults ``REGISTERED_TASKS``
to discover which task configs MyoSuite exposes.  Each value in the dict is a
config dataclass that mjlab's ``envs.make()`` can instantiate.

Public API::

    REGISTERED_TASKS  — mapping env_id → task config class
    MjlabEnvAccessor  — EnvAccessor implementation returning torch.Tensor
"""

from __future__ import annotations

from myosuite.envs.myo.backends.mjlab.configs.baoding_cfg import BaodingCfg
from myosuite.envs.myo.backends.mjlab.configs.elbow_pose_cfg import ElbowPoseCfg
from myosuite.envs.myo.backends.mjlab.configs.finger_pose_cfg import FingerPoseCfg
from myosuite.envs.myo.backends.mjlab.configs.finger_reach_cfg import FingerReachCfg
from myosuite.envs.myo.backends.mjlab.configs.hand_pose_cfg import HandPoseCfg
from myosuite.envs.myo.backends.mjlab.configs.hand_reach_cfg import HandReachCfg
from myosuite.envs.myo.backends.mjlab.configs.musclemimic_bimanual_cfg import (
    MuscleMimicBimanualCfg,
)
from myosuite.envs.myo.backends.mjlab.configs.musclemimic_fullbody_cfg import (
    MuscleMimicFullbodyCfg,
)
from myosuite.envs.myo.backends.mjlab.configs.table_tennis_cfg import TableTennisCfg
from myosuite.envs.myo.backends.mjlab.configs.walk_cfg import WalkCfg
from myosuite.envs.myo.backends.mjlab.mjlab_env_base import MjlabEnvAccessor

# ---------------------------------------------------------------------------
# Task registry — mjlab discovers tasks by reading this dict.
# ---------------------------------------------------------------------------

REGISTERED_TASKS: dict[str, type] = {
    "myoElbowPose1D6MRandom-v0": ElbowPoseCfg,
    "myoElbowPose1D6MFixed-v0": ElbowPoseCfg,
    "myoFingerPoseFixed-v0": FingerPoseCfg,
    "myoFingerPoseRandom-v0": FingerPoseCfg,
    "myoFingerReachRandom-v0": FingerReachCfg,
    "myoHandPoseRandom-v0": HandPoseCfg,
    "myoHandReachRandom-v0": HandReachCfg,
    "myoLegWalk-v0": WalkCfg,
    "myoSarcLegWalk-v0": WalkCfg,
    "myoChallengeBaodingP2-v1": BaodingCfg,
    "myoChallengeTableTennisP0-v0": TableTennisCfg,
    "myoChallengeTableTennisP1-v0": TableTennisCfg,
    "myoChallengeTableTennisP2-v0": TableTennisCfg,
    "myoMimicBimanual-v0": MuscleMimicBimanualCfg,
    "myoMimicFullbody-v0": MuscleMimicFullbodyCfg,
    "myoMuscleMimicBimanual-v0": MuscleMimicBimanualCfg,
    "myoMuscleMimicFullbody-v0": MuscleMimicFullbodyCfg,
}

# Register with mjlab.tasks.registry when mjlab loads this package (entry point).
try:
    from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
        bootstrap_myosuite_mjlab_registry,
    )

    bootstrap_myosuite_mjlab_registry()
except (ImportError, RuntimeError, ValueError):
    # Keep import side effects best-effort: task registration should not break
    # package import on systems without a full mjlab installation.
    pass
except Exception:
    import logging

    logging.getLogger(__name__).warning(
        "myosuite mjlab backend: unexpected error during task registration",
        exc_info=True,
    )

__all__ = [
    "REGISTERED_TASKS",
    "MjlabEnvAccessor",
    "ElbowPoseCfg",
    "FingerPoseCfg",
    "FingerReachCfg",
    "HandPoseCfg",
    "HandReachCfg",
    "WalkCfg",
    "BaodingCfg",
    "TableTennisCfg",
    "MuscleMimicBimanualCfg",
    "MuscleMimicFullbodyCfg",
]
