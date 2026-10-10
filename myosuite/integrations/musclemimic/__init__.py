# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""MuscleMimic compatibility helpers.

Run examples: ``myosuite/integrations/musclemimic/README.md``.
Upstream: https://github.com/amathislab/musclemimic

* **Eval / playback** — ``myosuite-musclemimic-fullbody-eval``: with ``--path``
  runs MyoSuite-native trajectory replay in MuJoCo viewer from motion cache
  (checkpoint path is resolved and validated). Without ``--path``, supports
  preview mode (``--use_mujoco --mujoco_viewer``) or MJX smoke on
  ``MjxMimicFullbody-v0`` (alias: ``MjxMuscleMimicFullbody-v0``).
* **Demo cache** — ``myosuite-musclemimic-setup-demo-cache`` downloads HF demo
  motions via ``huggingface_hub`` (``MyoSuite[musclemimic]``); no upstream
  ``musclemimic`` package required.

The names below and the submodules are imported on first access (PEP 562), so
importing one submodule (e.g. ``bc_directional_collector`` for the ChaseTag
observation) does not import the others (``fullbody_native_playback`` pulls in
``scipy.spatial``).
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from myosuite.integrations.musclemimic.citation import (
        MUSCLEMIMIC_ARXIV_URL,
        MUSCLEMIMIC_CITATION,
        MUSCLEMIMIC_CITATION_BIBTEX,
        MUSCLEMIMIC_PROJECT_URL,
    )
    from myosuite.integrations.musclemimic.bimanual_model import (
        BODY2SITES_FOR_MIMIC,
        FINGER_JOINT_TOKENS,
        FINGER_MUSCLE_TOKENS,
        apply_mimic_bimanual_spec_edits,
        build_mimic_bimanual_spec,
        compile_mimic_bimanual_mjmodel,
        compile_musclemimic_bimanual_mjmodel,
        default_mimic_config,
        default_musclemimic_config,
        resolve_mimic_bimanual_xml,
        resolve_musclemimic_bimanual_xml,
    )
    from myosuite.integrations.musclemimic.myotorso_bimanual_model import (
        build_myotorso_bimanual_mimic_spec,
        compile_myotorso_bimanual_mimic_mjmodel,
        default_myotorso_bimanual_mimic_config,
        save_myotorso_bimanual_mimic_xml,
    )
    from myosuite.integrations.musclemimic.fullbody_model import (
        FULLBODY_BODY2SITES_FOR_MIMIC,
        build_mimic_fullbody_spec,
        build_native_mimic_fullbody_spec,
        compile_mimic_fullbody_mjmodel,
        compile_musclemimic_fullbody_mjmodel,
        default_mimic_fullbody_config,
        default_musclemimic_fullbody_config,
        resolve_mimic_fullbody_xml,
        resolve_musclemimic_fullbody_xml,
    )
    from myosuite.integrations.musclemimic.fullbody_native_playback import (
        FullbodyNativePlaybackRunner,
        NativePlaybackArgs,
        parse_native_playback_argv,
        run_native_playback,
    )
    from myosuite.core.playback_contract import (
        PlaybackArtifacts,
        PlaybackRequest,
        PlaybackResult,
        PlaybackRunner,
    )

# Defining module of every public name.
_EXPORTS_BY_MODULE: dict[str, tuple[str, ...]] = {
    "myosuite.integrations.musclemimic.citation": (
        "MUSCLEMIMIC_ARXIV_URL",
        "MUSCLEMIMIC_CITATION",
        "MUSCLEMIMIC_CITATION_BIBTEX",
        "MUSCLEMIMIC_PROJECT_URL",
    ),
    "myosuite.integrations.musclemimic.bimanual_model": (
        "BODY2SITES_FOR_MIMIC",
        "FINGER_JOINT_TOKENS",
        "FINGER_MUSCLE_TOKENS",
        "apply_mimic_bimanual_spec_edits",
        "build_mimic_bimanual_spec",
        "compile_mimic_bimanual_mjmodel",
        "compile_musclemimic_bimanual_mjmodel",
        "default_mimic_config",
        "default_musclemimic_config",
        "resolve_mimic_bimanual_xml",
        "resolve_musclemimic_bimanual_xml",
    ),
    "myosuite.integrations.musclemimic.myotorso_bimanual_model": (
        "build_myotorso_bimanual_mimic_spec",
        "compile_myotorso_bimanual_mimic_mjmodel",
        "default_myotorso_bimanual_mimic_config",
        "save_myotorso_bimanual_mimic_xml",
    ),
    "myosuite.integrations.musclemimic.fullbody_model": (
        "build_terrain_fullbody_spec",
        "FULLBODY_BODY2SITES_FOR_MIMIC",
        "build_mimic_fullbody_spec",
        "build_native_mimic_fullbody_spec",
        "compile_mimic_fullbody_mjmodel",
        "compile_musclemimic_fullbody_mjmodel",
        "default_mimic_fullbody_config",
        "default_musclemimic_fullbody_config",
        "resolve_mimic_fullbody_xml",
        "resolve_musclemimic_fullbody_xml",
    ),
    "myosuite.integrations.musclemimic.fullbody_native_playback": (
        "FullbodyNativePlaybackRunner",
        "NativePlaybackArgs",
        "parse_native_playback_argv",
        "run_native_playback",
    ),
    "myosuite.integrations.musclemimic.terrain_observation": (
        "MIMIC_SITES",
        "TerrainObsCfg",
        "TerrainObservation",
        "TerrainHeights",
    ),
    "myosuite.integrations.musclemimic.terrain_policy": (
        "load_terrain_policy",
        "TerrainController",
        "download_terrain_checkpoint",
    ),
    "myosuite.integrations.musclemimic.reference_motion": (
        "compose_waypoint_reference",
        "plan_waypoint_reference",
        "GaitParameters",
    ),
    "myosuite.core.playback_contract": (
        "PlaybackArtifacts",
        "PlaybackRequest",
        "PlaybackResult",
        "PlaybackRunner",
    ),
}
_MODULE_OF = {
    name: module for module, names in _EXPORTS_BY_MODULE.items() for name in names
}

__all__ = [
    "BODY2SITES_FOR_MIMIC",
    "FINGER_JOINT_TOKENS",
    "FINGER_MUSCLE_TOKENS",
    "FULLBODY_BODY2SITES_FOR_MIMIC",
    "MUSCLEMIMIC_ARXIV_URL",
    "MUSCLEMIMIC_CITATION",
    "MUSCLEMIMIC_CITATION_BIBTEX",
    "MUSCLEMIMIC_PROJECT_URL",
    "apply_mimic_bimanual_spec_edits",
    "build_mimic_bimanual_spec",
    "build_myotorso_bimanual_mimic_spec",
    "build_mimic_fullbody_spec",
    "build_native_mimic_fullbody_spec",
    "compile_mimic_bimanual_mjmodel",
    "compile_myotorso_bimanual_mimic_mjmodel",
    "compile_mimic_fullbody_mjmodel",
    "default_mimic_config",
    "default_myotorso_bimanual_mimic_config",
    "save_myotorso_bimanual_mimic_xml",
    "default_mimic_fullbody_config",
    "resolve_mimic_bimanual_xml",
    "resolve_mimic_fullbody_xml",
    "compile_musclemimic_bimanual_mjmodel",
    "compile_musclemimic_fullbody_mjmodel",
    "NativePlaybackArgs",
    "FullbodyNativePlaybackRunner",
    "parse_native_playback_argv",
    "run_native_playback",
    "PlaybackRequest",
    "PlaybackArtifacts",
    "PlaybackResult",
    "PlaybackRunner",
    "default_musclemimic_config",
    "default_musclemimic_fullbody_config",
    "resolve_musclemimic_bimanual_xml",
    "resolve_musclemimic_fullbody_xml",
]

__all__ += [
    "build_terrain_fullbody_spec",
    "MIMIC_SITES",
    "TerrainObsCfg",
    "TerrainObservation",
    "TerrainHeights",
    "load_terrain_policy",
    "TerrainController",
    "download_terrain_checkpoint",
    "compose_waypoint_reference",
    "plan_waypoint_reference",
    "GaitParameters",
]


def __getattr__(name: str) -> Any:
    """Import the submodule defining *name*, or named *name*, on first access."""
    module = _MODULE_OF.get(name)
    if module is not None:
        value = getattr(importlib.import_module(module), name)
    elif not name.startswith("_") and importlib.util.find_spec(f"{__name__}.{name}"):
        # Submodules stay attributes of the package, as with eager imports.
        value = importlib.import_module(f"{__name__}.{name}")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Module attributes plus the not yet imported public names."""
    return sorted({*globals(), *__all__})
