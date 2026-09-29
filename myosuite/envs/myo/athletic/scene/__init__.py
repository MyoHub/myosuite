# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Optional athletic scene plugins (lanes, hurdles, blocks, curve, vertical)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScenePlugin:
    """Declarative scene attachment for an athletic TaskSpec."""

    key: str
    description: str = ""

    def apply_spec(self, mj_spec: Any) -> Any:
        """Hook for future MjSpec mutation; default is identity."""
        return mj_spec


SCENE_PLUGINS: dict[str, ScenePlugin] = {
    "lanes": ScenePlugin(key="lanes", description="Lane markers / mid-lane lateral constraint"),
    "hurdles": ScenePlugin(key="hurdles", description="Men's 110m hurdles colliders"),
    "blocks": ScenePlugin(key="blocks", description="Starting-block colliders"),
    "curve": ScenePlugin(key="curve", description="Olympic curve track geometry"),
    "vertical": ScenePlugin(key="vertical", description="Vertical jump / rim target"),
    "uphill": ScenePlugin(key="uphill", description="Tilted ground plane"),
}


def get_scene_plugin(key: str | None) -> ScenePlugin | None:
    if key is None:
        return None
    return SCENE_PLUGINS.get(key)
