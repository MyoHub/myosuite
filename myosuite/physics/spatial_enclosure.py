# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Backend-agnostic 3D AABB and half-space enclosure clearance geometry."""

from __future__ import annotations

from typing import Any


def _is_torch(xp: Any) -> bool:
    return getattr(xp, "__name__", "") == "torch"


def _as_like(xp: Any, value: Any, ref: Any) -> Any:
    if _is_torch(xp):
        return xp.as_tensor(value, dtype=ref.dtype, device=ref.device)
    return xp.asarray(value, dtype=ref.dtype)


def _expand_axis(xp: Any, arr: Any, axis: int) -> Any:
    if _is_torch(xp):
        return arr.unsqueeze(axis)
    return xp.expand_dims(arr, axis=axis)


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
        xp: Backend array module (``numpy``, ``jax.numpy``, or ``torch``).
        site_pos: Site positions of shape ``(K, 3)`` or ``(N, K, 3)``.
        bounds_low: Lower AABB corner ``(3,)`` or ``(N, 3)`` [m].
        bounds_high: Upper AABB corner ``(3,)`` or ``(N, 3)`` [m].
        plane_normals: Optional inward-pointing plane normals ``(M, 3)``.
        plane_offsets: Optional plane offsets ``(M,)`` for ``n . p - d >= 0``.

    Returns:
        Signed clearances of shape ``(..., K, 6)`` or ``(..., K, 6 + M)``.
    """
    lo = _as_like(xp, bounds_low, site_pos)
    hi = _as_like(xp, bounds_high, site_pos)
    if lo.ndim == site_pos.ndim - 1 and lo.shape[-1] == 3 and site_pos.ndim == 3:
        lo = _expand_axis(xp, lo, -2)
    if hi.ndim == site_pos.ndim - 1 and hi.shape[-1] == 3 and site_pos.ndim == 3:
        hi = _expand_axis(xp, hi, -2)

    parts = [site_pos - lo, hi - site_pos]

    if (plane_normals is None) != (plane_offsets is None):
        raise ValueError(
            "plane_normals and plane_offsets must both be provided or both be None"
        )
    if plane_normals is not None and plane_offsets is not None:
        normals = _as_like(xp, plane_normals, site_pos)
        offsets = _as_like(xp, plane_offsets, site_pos)
        if _is_torch(xp):
            norm_len = xp.linalg.norm(normals, dim=-1, keepdim=True)
            norm_safe = xp.clamp_min(norm_len, 1e-12)
            unit_n = normals / norm_safe
            unit_d = offsets / norm_safe.squeeze(-1)
            proj = xp.sum(site_pos.unsqueeze(-2) * unit_n, dim=-1)
        else:
            norm_len = xp.linalg.norm(normals, axis=-1, keepdims=True)
            norm_safe = xp.maximum(norm_len, 1e-12)
            unit_n = normals / norm_safe
            unit_d = offsets / xp.squeeze(norm_safe, axis=-1)
            proj = xp.sum(xp.expand_dims(site_pos, axis=-2) * unit_n, axis=-1)
        parts.append(proj - unit_d)

    if _is_torch(xp):
        return xp.cat(parts, dim=-1)
    return xp.concatenate(parts, axis=-1)


def min_enclosure_clearance(
    xp: Any,
    site_pos: Any,
    bounds_low: Any,
    bounds_high: Any,
    *,
    plane_normals: Any | None = None,
    plane_offsets: Any | None = None,
) -> Any:
    """Return minimum signed inward clearance ``(...,)`` across sites and faces."""
    margins = enclosure_margins(
        xp,
        site_pos,
        bounds_low,
        bounds_high,
        plane_normals=plane_normals,
        plane_offsets=plane_offsets,
    )
    if _is_torch(xp):
        return xp.amin(margins, dim=(-2, -1))
    return xp.amin(margins, axis=(-2, -1))


def excess_contact_force(
    xp: Any,
    contact_force: Any | None,
    *,
    threshold: float = 1.0,
    reference: Any,
    batched: bool = False,
) -> Any:
    """Return non-negative contact force magnitude above *threshold* [N]."""
    if contact_force is None:
        return xp.zeros_like(reference)
    f_arr = _as_like(xp, contact_force, reference)
    expected_scalar_ndim = 1 if batched else 0
    if f_arr.ndim > expected_scalar_ndim:
        mag = (
            xp.linalg.norm(f_arr, dim=-1)
            if _is_torch(xp)
            else xp.linalg.norm(f_arr, axis=-1)
        )
    else:
        mag = xp.abs(f_arr)
    diff = mag - float(threshold)
    return xp.clamp_min(diff, 0.0) if _is_torch(xp) else xp.maximum(diff, 0.0)


__all__ = [
    "enclosure_margins",
    "excess_contact_force",
    "min_enclosure_clearance",
]
