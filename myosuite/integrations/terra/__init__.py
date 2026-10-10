# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Run the TERRA-4B policy (https://github.com/amathislab/terra) natively in MyoSuite.

TERRA is a terrain-aware MyoFullBody motion tracker. This package rebuilds its actor,
observation and inference without the TERRA, MuscleMimic or JAX runtimes (JAX is only
needed to read the Orbax checkpoint), and composes a reference walk along waypoints.
Cite TERRA when you use it; the checkpoint has its own model card and license.
"""

from myosuite.integrations.terra.actor import (
    TERRA_MODELS_VERSION,
    TERRAIN_GROUP,
    apply_terra_model_fixes,
    terra_fullbody_spec,
)
from myosuite.integrations.terra.observation import (
    MIMIC_SITES,
    TerraObsCfg,
    TerraObservation,
    TerrainHeights,
)
from myosuite.integrations.terra.policy import (
    TERRA_REPO,
    TERRA_REVISION,
    TerraController,
    TerraPolicy,
    download_terra_checkpoint,
)
from myosuite.integrations.terra.reference import (
    ReferenceMotion,
    compose_waypoint_reference,
)

__all__ = [
    "MIMIC_SITES",
    "TERRAIN_GROUP",
    "TERRA_MODELS_VERSION",
    "TERRA_REPO",
    "TERRA_REVISION",
    "ReferenceMotion",
    "TerraController",
    "TerraObsCfg",
    "TerraObservation",
    "TerraPolicy",
    "TerrainHeights",
    "apply_terra_model_fixes",
    "compose_waypoint_reference",
    "download_terra_checkpoint",
    "terra_fullbody_spec",
]
