# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Backend-agnostic spatial enclosure proximity and contact-clearance MDP terms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from myosuite.core.protocols import EnvAccessor


def _is_torch(xp: Any) -> bool:
    return getattr(xp, "__name__", "") == "torch"


def _as_xp_array(xp: Any, value: Any, ref: Any) -> Any:
    if _is_torch(xp):
        return xp.as_tensor(value, dtype=ref.dtype, device=ref.device)
    return xp.asarray(value, dtype=ref.dtype)


def _expand_axis(xp: Any, arr: Any, axis: int) -> Any:
    if _is_torch(xp):
        return arr.unsqueeze(axis)
    return xp.expand_dims(arr, axis=axis)


def _relu(xp: Any, arr: Any) -> Any:
    if _is_torch(xp):
        return xp.clamp_min(arr, 0.0)
    return xp.maximum(arr, 0.0)


def _min_last2(xp: Any, arr: Any) -> Any:
    if _is_torch(xp):
        return xp.amin(arr, dim=(-2, -1))
    return xp.amin(arr, axis=(-2, -1))


def _norm_last(xp: Any, arr: Any, *, keepdim: bool = False) -> Any:
    if _is_torch(xp):
        return xp.linalg.norm(arr, dim=-1, keepdim=keepdim)
    return xp.linalg.norm(arr, axis=-1, keepdims=keepdim)


def _cast_float32(xp: Any, arr: Any) -> Any:
    if _is_torch(xp):
        return arr.to(dtype=xp.float32)
    return arr.astype(xp.float32)


def _cast_bool(xp: Any, arr: Any) -> Any:
    if _is_torch(xp):
        return arr.to(dtype=xp.bool)
    return arr.astype(bool)


def _site_positions_3d(accessor: EnvAccessor, site_ids: Any) -> tuple[Any, bool]:
    xp = accessor.array_module()
    batched = accessor.joint_pos().ndim > 1
    raw = accessor.site_xpos(site_ids)
    if batched:
        if raw.ndim == 2:
            return _expand_axis(xp, raw, -2), True
        return raw, True
    if raw.ndim == 1:
        return _expand_axis(xp, raw, 0), False
    return raw, False


def enclosure_margins(
    xp: Any,
    site_pos: Any,
    bounds_low: Any,
    bounds_high: Any,
    *,
    plane_normals: Any | None = None,
    plane_offsets: Any | None = None,
) -> Any:
    """Compute signed inward wall clearances ``(..., K, F)`` for ``K`` tracked sites.

    Positive values indicate the site lies strictly inside the enclosure; negative
    values indicate boundary penetration.

    Args:
        xp: Backend array module from ``accessor.array_module()``.
        site_pos: Site positions of shape ``(K, 3)`` or ``(N, K, 3)``.
        bounds_low: Lower AABB corner ``(3,)`` or ``(N, 3)`` [m].
        bounds_high: Upper AABB corner ``(3,)`` or ``(N, 3)`` [m].
        plane_normals: Optional inward-pointing plane normals ``(M, 3)``.
        plane_offsets: Optional plane offsets ``(M,)`` for ``n . p - d >= 0``.

    Returns:
        Signed clearances of shape ``(..., K, 6)`` or ``(..., K, 6 + M)``.
    """
    lo = _as_xp_array(xp, bounds_low, site_pos)
    hi = _as_xp_array(xp, bounds_high, site_pos)
    if lo.ndim == site_pos.ndim - 1 and lo.shape[-1] == 3 and site_pos.ndim == 3:
        lo = _expand_axis(xp, lo, -2)
    if hi.ndim == site_pos.ndim - 1 and hi.shape[-1] == 3 and site_pos.ndim == 3:
        hi = _expand_axis(xp, hi, -2)

    d_low = site_pos - lo
    d_high = hi - site_pos
    parts = [d_low, d_high]

    if (plane_normals is None) != (plane_offsets is None):
        raise ValueError(
            "plane_normals and plane_offsets must both be provided or both be None"
        )
    if plane_normals is not None and plane_offsets is not None:
        normals = _as_xp_array(xp, plane_normals, site_pos)
        offsets = _as_xp_array(xp, plane_offsets, site_pos)
        norm_len = _norm_last(xp, normals, keepdim=True)
        norm_safe = (
            xp.clamp_min(norm_len, 1e-12)
            if _is_torch(xp)
            else xp.maximum(norm_len, 1e-12)
        )
        unit_n = normals / norm_safe
        norm_flat = (
            norm_safe.squeeze(-1) if _is_torch(xp) else xp.squeeze(norm_safe, axis=-1)
        )
        unit_d = offsets / norm_flat
        expanded = _expand_axis(xp, site_pos, -2) * unit_n
        proj = xp.sum(expanded, dim=-1) if _is_torch(xp) else xp.sum(expanded, axis=-1)
        parts.append(proj - unit_d)

    if _is_torch(xp):
        return xp.cat(parts, dim=-1)
    return xp.concatenate(parts, axis=-1)


def boundary_proximity_obs(
    accessor: EnvAccessor,
    *,
    site_ids: Any,
    bounds_low: Any,
    bounds_high: Any,
    soft_margin: float = 0.15,
    signed: bool = False,
    plane_normals: Any | None = None,
    plane_offsets: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Return float32 enclosure wall proximity or signed clearance observation.

    When ``signed=False`` (default), returns ``max(0, 1 - clearance / soft_margin)``
    with no upper clip so boundary penetrations remain smoothly differentiable and
    unbounded.

    Args:
        accessor: Environment state accessor.
        site_ids: Tracked site id or sequence of ``K`` site ids.
        bounds_low: Lower enclosure corner ``(3,)`` or ``(N, 3)`` [m].
        bounds_high: Upper enclosure corner ``(3,)`` or ``(N, 3)`` [m].
        soft_margin: Sensing horizon [m] inside the boundary (positive).
        signed: Return raw signed inward clearances [m] when ``True``.
        plane_normals: Optional inward plane normals ``(M, 3)``.
        plane_offsets: Optional plane offsets ``(M,)``.
        **kwargs: Unused extra keyword arguments.

    Returns:
        Float32 array of shape ``(K * F,)`` or ``(N, K * F)``.
    """
    del kwargs
    if soft_margin <= 0.0:
        raise ValueError(f"soft_margin must be positive, got {soft_margin}")
    xp = accessor.array_module()
    pos, batched = _site_positions_3d(accessor, site_ids)
    margins = enclosure_margins(
        xp,
        pos,
        bounds_low,
        bounds_high,
        plane_normals=plane_normals,
        plane_offsets=plane_offsets,
    )
    features = margins if signed else _relu(xp, 1.0 - margins / float(soft_margin))
    flat = features.reshape(features.shape[0], -1) if batched else features.reshape(-1)
    return _cast_float32(xp, flat)


def enclosure_clearance_reward(
    accessor: EnvAccessor,
    task_state: dict[str, Any],
    *,
    margin: float = 0.05,
    violation_weight: float = 5.0,
    contact_weight: float = 1.0,
    contact_threshold: float = 1.0,
    terminate_on_violation: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Penalize spatial boundary intrusion and unintended enclosure contact forces.

    Args:
        accessor: Environment state accessor.
        task_state: Dict containing ``"enclosure_site_ids"`` (or ``"tip_site_ids"``),
            ``"bounds_low"``, ``"bounds_high"``, and optional ``"plane_normals"``,
            ``"plane_offsets"``, and ``"enclosure_contact_force"`` (scalar or 3-vector
            contact force in Newtons).
        margin: Safe inward clearance buffer [m] before intrusion penalty begins.
        violation_weight: Scale factor for boundary clearance intrusion penalty.
        contact_weight: Scale factor for excess physical contact force penalty.
        contact_threshold: Allowed contact force magnitude [N] before penalty.
        terminate_on_violation: Set ``done=True`` when outside the enclosure or
            exceeding ``contact_threshold``.
        **kwargs: Unused extra keyword arguments.

    Returns:
        Reward dict with ``min_clearance``, ``clearance_penalty``,
        ``contact_penalty``, ``dense``, ``solved``, and ``done``.
    """
    del kwargs
    xp = accessor.array_module()
    site_ids = (
        task_state["enclosure_site_ids"]
        if "enclosure_site_ids" in task_state
        else task_state["tip_site_ids"]
    )
    pos, batched = _site_positions_3d(accessor, site_ids)
    margins = enclosure_margins(
        xp,
        pos,
        task_state["bounds_low"],
        task_state["bounds_high"],
        plane_normals=task_state.get("plane_normals"),
        plane_offsets=task_state.get("plane_offsets"),
    )
    min_clearance = _min_last2(xp, margins)
    intrusion = _relu(xp, float(margin) - min_clearance)
    denom = max(float(margin), 1e-6)
    clearance_penalty = -float(violation_weight) * (
        intrusion + (intrusion * intrusion) / denom
    )

    raw_force = task_state.get("enclosure_contact_force")
    if raw_force is not None:
        f_arr = _as_xp_array(xp, raw_force, min_clearance)
        expected_scalar_ndim = 1 if batched else 0
        force_mag = (
            _norm_last(xp, f_arr)
            if f_arr.ndim > expected_scalar_ndim
            else xp.abs(f_arr)
        )
        excess_force = _relu(xp, force_mag - float(contact_threshold))
        contact_penalty = -float(contact_weight) * excess_force
    else:
        excess_force = xp.zeros_like(min_clearance)
        contact_penalty = xp.zeros_like(min_clearance)

    violated = (min_clearance < 0.0) | (excess_force > 0.0)
    dense = clearance_penalty + contact_penalty
    solved = ~violated
    done = (
        _cast_bool(xp, violated)
        if terminate_on_violation
        else _cast_bool(xp, xp.zeros_like(min_clearance))
    )
    return {
        "min_clearance": min_clearance,
        "clearance_penalty": clearance_penalty,
        "contact_penalty": contact_penalty,
        "dense": dense,
        "solved": _cast_bool(xp, solved),
        "done": done,
    }


enclosure_contact_penalty = enclosure_clearance_reward

__all__ = [
    "boundary_proximity_obs",
    "enclosure_clearance_reward",
    "enclosure_contact_penalty",
    "enclosure_margins",
]
