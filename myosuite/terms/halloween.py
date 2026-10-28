"""Backend-independent Halloween scoring (clearance is shaping, never a hit)."""

from typing import Any


def halloween_scores(xp: Any, root: Any, head: Any, clearance: Any) -> dict[str, Any]:
    """Compute posture and distance shaping for scalar or batched states."""
    upright = xp.clip((head[..., 2] - 1.0) / 0.5, 0.0, 1.0)
    direction = head - root
    posture = xp.clip(
        direction[..., 2] / xp.clip(xp.linalg.norm(direction, axis=-1), 1e-6, None),
        0.0,
        1.0,
    )
    dodge = xp.clip((clearance - 0.20) / 0.3, 0.0, 1.0)
    return {"upright": upright, "posture": posture, "dodge": dodge}
