# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""mjlab backend task discovery module.

.. note::
    This is the **supported GPU training path**. MJX (JAX) is experimental.
    Task IDs share the CPU ``env_id`` (see ``REGISTERED_TASKS``).

Task registration with ``mjlab.tasks`` runs via the dedicated entry-point module
registered in ``pyproject.toml`` as::

    [project.entry-points."mjlab.tasks"]
    myosuite = "myosuite.envs.backends.mjlab.bootstrap"

This package exposes ``REGISTERED_TASKS`` (a pure config-class dictionary) so
callers on CPU-only installs can inspect which task configs MyoSuite exposes
without ``torch`` or ``mjlab`` installed; when ``mjlab`` has triggered deferred
entry-point discovery, registration completes once package initialization finishes.

Public API::

    REGISTERED_TASKS  — mapping env_id → task config class
    MjlabEnvAccessor  — EnvAccessor implementation returning torch.Tensor
"""

from __future__ import annotations

import warnings

from myosuite.envs.backends.mjlab.mjlab_env_base import MjlabEnvAccessor
from myosuite.envs.tasks.hand.baoding.mjlab_cfg import BaodingCfg
from myosuite.envs.backends.mjlab.configs.elbow_pose_cfg import ElbowPoseCfg
from myosuite.envs.backends.mjlab.configs.finger_pose_cfg import FingerPoseCfg
from myosuite.envs.backends.mjlab.configs.finger_reach_cfg import FingerReachCfg
from myosuite.envs.backends.mjlab.configs.hand_pose_cfg import HandPoseCfg
from myosuite.envs.backends.mjlab.configs.hand_reach_cfg import HandReachCfg
from myosuite.envs.tasks.fullbody.bimanual.mjlab_cfg import (
    MuscleMimicBimanualCfg,
)
from myosuite.envs.tasks.mimic.fullbody_mjlab_cfg import (
    MuscleMimicFullbodyCfg,
)
from myosuite.envs.tasks.games.tabletennis.mjlab_cfg import TableTennisCfg
from myosuite.envs.backends.mjlab.configs.walk_cfg import WalkCfg

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


# mjlab attaches each entity spec to its scene: the entity's own <option> is dropped (the
# physics options come from MujocoCfg) and attach reports differing compiler/visual settings.
# Masses, inertias and timestep are unaffected; hide exactly these two messages.
for _message in (
    r"Entity '.*' has non-default <option> fields",
    r"Attach conflict when attaching",
):
    warnings.filterwarnings("ignore", message=_message, category=UserWarning)

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

from myosuite.envs.backends.mjlab.bootstrap import (  # noqa: E402
    finish_deferred_mjlab_bootstrap as _finish_deferred_mjlab_bootstrap,
)

_finish_deferred_mjlab_bootstrap(__name__)
del _finish_deferred_mjlab_bootstrap
