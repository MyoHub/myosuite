# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Volumetric muscle tubes along MuJoCo tendon paths.

Each muscle becomes a fusiform belly with thin tendons, after the volumetric
visualiser of MuSkeMo (van Bijlert et al.). The peak cross-section is
``F_max / CROSS_SECTION_STRESS``, a stress fitted to measured cross-sections of
forearm and leg muscles: MyoSuite's ``F_max`` are not anatomical forces, so the
``F_max / specific tension * L0`` volume would make forearm muscles 5-15x too
large. MyoSuite's forces are not proportional to anatomical size either, so the
large hip, thigh, calf and shoulder muscles take measured volumes instead
(:data:`REFERENCE_VOLUMES`). The belly is as long as the optimal fibre length,
or longer when its cross-section needs it (pennate leg muscles), and sits
towards the origin, so long tendons run distally.
"""

from __future__ import annotations

import re

import mujoco
import numpy as np

CROSS_SECTION_STRESS = 1.6e6  # N/m^2
BELLY_FRACTION = (0.4, 0.85)  # belly length as a fraction of the path
SLENDERNESS = 0.16  # peak radius at most this fraction of the belly length
TENDON_RADIUS_RATIO = 0.2
PATH_POINTS = 32
# Adult volumes (m^3) of whole muscles, shared among a muscle's MyoSuite parts
# (e.g. glmax1-3) in proportion to their forces. Leg: Handsfield et al. 2014,
# J Biomech 47:631; shoulder: approximate, from Holzbaur et al. 2007, J Biomech
# 40:742.
REFERENCE_VOLUMES = {
    "glmax": 849e-6,
    "glmed": 323e-6,
    "addmag": 559e-6,
    "vaslat": 514e-6,
    "vasmed": 424e-6,
    "vasint": 375e-6,
    "recfem": 237e-6,
    "semimem": 245e-6,
    "semiten": 199e-6,
    "bflh": 192e-6,
    "tfl": 69e-6,
    "soleus": 476e-6,
    "gasmed": 258e-6,
    "gaslat": 141e-6,
    "DELT": 275e-6,
    "PECM": 210e-6,
    "LAT": 190e-6,
}
RING_POINTS = 12


def muscle_actuators(model: mujoco.MjModel) -> np.ndarray:
    """Muscle actuators that act through a spatial tendon.

    Args:
        model: Compiled MuJoCo model.

    Returns:
        Actuator indices, in model order.
    """
    return np.array(
        [
            a
            for a in range(model.nu)
            if model.actuator_gaintype[a] == mujoco.mjtGain.mjGAIN_MUSCLE
            and model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_TENDON
            and model.tendon_num[model.actuator_trnid[a, 0]] > 1
            and model.wrap_type[model.tendon_adr[model.actuator_trnid[a, 0]]]
            == mujoco.mjtWrap.mjWRAP_SITE
        ],
        dtype=int,
    )


def muscle_shape(
    model: mujoco.MjModel, path_lengths: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Peak belly radius and belly fraction of each muscle.

    Args:
        model: Compiled MuJoCo model.
        path_lengths: Tendon length of each :func:`muscle_actuators` entry (m).

    Returns:
        ``(peak_radius, belly_fraction)``: radii in metres and the fraction of the
        path the belly covers.
    """
    act = muscle_actuators(model)
    gain = model.actuator_gainprm[act]
    # A negative force means MuJoCo derives F_max as scale / acc0.
    force = np.where(
        gain[:, 2] > 0,
        gain[:, 2],
        gain[:, 3] / np.maximum(model.actuator_acc0[act], 1e-10),
    )
    lrange = model.actuator_lengthrange[act]
    l0 = (lrange[:, 1] - lrange[:, 0]) / np.maximum(gain[:, 1] - gain[:, 0], 1e-6)
    radius = np.sqrt(force / (np.pi * CROSS_SECTION_STRESS))
    # Whole-muscle groups by name, per side: "addmagDist_r" -> ("addmag", "_r").
    groups = [_reference_group(model.actuator(a).name) for a in act]
    reference = np.array([REFERENCE_VOLUMES.get(g[0], 0.0) for g in groups])
    share = np.array([force[[h == g for h in groups]].sum() for g in groups])
    target = reference * force / np.maximum(share, 1e-9)
    # Fibre-length belly with the measured volume: r = sqrt(V / (pi * L0)).
    radius = np.where(
        target > 0, np.sqrt(target / (np.pi * np.maximum(l0, 1e-3))), radius
    )
    belly = np.maximum(l0, radius / SLENDERNESS)
    fraction = np.clip(belly / np.maximum(path_lengths, 1e-6), *BELLY_FRACTION)
    peak = np.minimum(radius, SLENDERNESS * fraction * path_lengths)
    profile = radius_profile(fraction)
    # Referenced muscles keep their volume whatever belly length they end up with.
    fitted = belly_radius(target, path_lengths, profile)
    peak = np.where(
        target > 0, np.minimum(fitted, SLENDERNESS * fraction * path_lengths), peak
    )
    return peak, fraction


def _reference_group(name: str) -> tuple[str, str]:
    side = name[-2:] if name[-2:] in ("_r", "_l") else ""
    stem = name[: len(name) - len(side)]
    for key in REFERENCE_VOLUMES:
        if re.fullmatch(rf"{key}(\d*|Dist|Isch|Mid|Prox)", stem):
            return key, side
    return stem, side


def resample_path(
    points: np.ndarray, count: int = PATH_POINTS
) -> tuple[np.ndarray, float]:
    """Resample a tendon polyline to points evenly spaced by arc length.

    Args:
        points: ``(P, 3)`` wrap points of one tendon path.
        count: Number of output points.

    Returns:
        ``(centreline, length)``: ``(count, 3)`` points and the path length.
    """
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    target = np.linspace(0.0, arc[-1], count)
    centreline = np.stack([np.interp(target, arc, points[:, i]) for i in range(3)], 1)
    return centreline, float(arc[-1])


def radius_profile(
    fraction: np.ndarray | float, count: int = PATH_POINTS
) -> np.ndarray:
    """Relative radius along the path: tendon, fusiform belly, tendon.

    A quarter of the tendon length lies at the origin, the rest towards the
    insertion.

    Args:
        fraction: Belly fraction of the path, scalar or one per muscle.
        count: Number of samples along the path.

    Returns:
        ``(..., count)`` radii relative to the peak (1 at the belly's widest point).
    """
    fraction = np.asarray(fraction, dtype=float)[..., None]
    s = np.linspace(0.0, 1.0, count)
    u = np.clip((s - (1 - fraction) / 4) / fraction, 0.0, 1.0)
    return TENDON_RADIUS_RATIO + (1 - TENDON_RADIUS_RATIO) * np.sin(np.pi * u)


def _mean_square(profile: np.ndarray) -> np.ndarray:
    squared = profile**2
    return ((squared[..., :-1] + squared[..., 1:]) / 2).mean(-1)


def belly_radius(
    volume: np.ndarray | float, length: np.ndarray | float, profile: np.ndarray
) -> np.ndarray:
    """Peak radius of a tube that encloses *volume* along *length*.

    Args:
        volume: Tube volume (m^3).
        length: Path length (m).
        profile: Relative radius profile from :func:`radius_profile`.

    Returns:
        Peak radius (m), broadcast over the inputs.
    """
    return np.sqrt(volume / (np.pi * np.maximum(length, 1e-6) * _mean_square(profile)))


def belly_volume(
    radius: np.ndarray | float, length: np.ndarray | float, profile: np.ndarray
) -> np.ndarray:
    """Volume of a tube with peak *radius* along *length* (inverse of :func:`belly_radius`).

    Args:
        radius: Peak radius (m).
        length: Path length (m).
        profile: Relative radius profile from :func:`radius_profile`.

    Returns:
        Volume (m^3), broadcast over the inputs.
    """
    return np.pi * radius**2 * length * _mean_square(profile)


def tube_mesh(
    centres: np.ndarray, radii: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Capped tubes around centrelines, with a twist-free frame.

    Args:
        centres: ``(..., N, 3)`` centreline points.
        radii: ``(..., N)`` radii.

    Returns:
        ``(points, face_counts, face_indices)``. Points are ``(..., N*K + 2, 3)``;
        the topology is shared by every leading index.
    """
    n, k = centres.shape[-2], RING_POINTS
    tangent = np.gradient(centres, axis=-2)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=-1, keepdims=True), 1e-12)
    # Parallel transport of a normal along the path avoids twisting the rings.
    ref = np.where(
        np.abs(tangent[..., :1, 2:3]) < 0.9, [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]
    )
    normals = [
        np.cross(np.cross(tangent[..., 0, :], ref[..., 0, :]), tangent[..., 0, :])
    ]
    for i in range(1, n):
        previous, t = normals[-1], tangent[..., i, :]
        normals.append(previous - (previous * t).sum(-1, keepdims=True) * t)
    normal = np.stack(normals, -2)
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-12)
    binormal = np.cross(tangent, normal)
    angle = np.linspace(0, 2 * np.pi, k, endpoint=False)
    ring = (
        np.cos(angle)[:, None] * normal[..., None, :]
        + np.sin(angle)[:, None] * binormal[..., None, :]
    )
    rings = centres[..., None, :] + radii[..., None, None] * ring  # (..., N, K, 3)
    points = np.concatenate(
        [
            rings.reshape(*rings.shape[:-3], n * k, 3),
            centres[..., :1, :],
            centres[..., -1:, :],
        ],
        -2,
    )
    faces = [
        [i * k + j, i * k + (j + 1) % k, (i + 1) * k + (j + 1) % k, (i + 1) * k + j]
        for i in range(n - 1)
        for j in range(k)
    ]
    first, last = n * k, n * k + 1
    faces += [[first, (j + 1) % k, j] for j in range(k)]
    faces += [[last, (n - 1) * k + j, (n - 1) * k + (j + 1) % k] for j in range(k)]
    return points, np.array([len(f) for f in faces]), np.concatenate(faces)
