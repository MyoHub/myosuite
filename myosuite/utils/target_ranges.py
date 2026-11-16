# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Resolve per-joint and per-site target ranges by name against a compiled model.

Target ranges arrive as ``{name: (lo, hi)}`` mappings whose iteration order is
not reliable (``ml_collections.ConfigDict`` iterates keys alphabetically), so
each range is matched to the model by name and laid out in model order:
qpos-address order for joints, site-id order for sites. Unknown names raise
through MuJoCo's named accessors (``model.joint(name)``, ``model.site(name)``),
as in the CPU envs. An unchecked ``mj_name2id`` would return -1, and JAX
indexing silently wraps -1 to the last element.

Pure numpy + MuJoCo (no JAX), so these helpers run on CPU-only installs.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import mujoco
import numpy as np

_SCALAR_JOINT_TYPES = (
    int(mujoco.mjtJoint.mjJNT_HINGE),
    int(mujoco.mjtJoint.mjJNT_SLIDE),
)


class SiteTargets(NamedTuple):
    """Fingertip target ranges in model (site id) order.

    Attributes:
        names: Tip site names.
        tip_ids: Tip site ids, shape ``(n,)``.
        target_ids: Ids of the ``<name>_target`` visualisation sites, shape ``(n,)``.
        lo: Lower corner of each target box, shape ``(n, 3)``.
        hi: Upper corner of each target box, shape ``(n, 3)``.
    """

    names: tuple[str, ...]
    tip_ids: np.ndarray
    target_ids: np.ndarray
    lo: np.ndarray
    hi: np.ndarray


def resolve_joint_target_ranges(
    model: mujoco.MjModel, target_jnt_range: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-joint target bounds laid out like ``qpos[:n]``.

    The shared pose terms (``pose_error_obs``, ``pose_reward``) compare the
    target with the leading ``qpos`` entries, so the named joints must be
    scalar joints that occupy exactly ``qpos[:n]``.

    Args:
        model: Compiled MuJoCo model.
        target_jnt_range: ``{joint_name: (lo, hi)}``, in any order.

    Returns:
        ``(lo, hi)``, each of shape ``(n,)``. Entry ``i`` bounds ``qpos[i]``.

    Raises:
        KeyError: If a joint name is not in the model.
        ValueError: If the mapping is empty, a range is not a ``(lo, hi)``
            pair, a joint is not a hinge/slide joint, or the joints do not
            occupy the leading ``qpos`` entries.
    """
    if not target_jnt_range:
        raise ValueError("target_jnt_range is empty.")
    rows = []
    for name, span in target_jnt_range.items():
        jid = model.joint(name).id
        if int(model.jnt_type[jid]) not in _SCALAR_JOINT_TYPES:
            raise ValueError(f"Joint {name!r} is not a hinge/slide joint.")
        bounds = np.asarray(span, dtype=np.float64)
        if bounds.size != 2:
            raise ValueError(
                f"Range for joint {name!r} must be a (lo, hi) pair, got shape {bounds.shape}."
            )
        rows.append((int(model.jnt_qposadr[jid]), name, bounds.reshape(2)))
    rows.sort(key=lambda row: row[0])
    adrs = [adr for adr, _, _ in rows]
    if adrs != list(range(len(rows))):
        raise ValueError(
            f"Target joints {[name for _, name, _ in rows]} occupy qpos addresses {adrs}; "
            f"the pose terms compare the target with qpos[:{len(rows)}]."
        )
    bounds = np.stack([b for _, _, b in rows])
    return bounds[:, 0], bounds[:, 1]


class TargetCoupling(NamedTuple):
    """A limit ``lo <= coef * t[index] + other_coef * t[other] <= hi`` on a target.

    Taken from a limited fixed tendon over two target joints (e.g. the
    flexion-dependent MCP abduction limit of the revised MyoHand). ``index``
    is the joint with the larger coefficient, the one a sampler adjusts.
    """

    index: int
    other: int
    coef: float
    other_coef: float
    lo: float
    hi: float


class JointTargetLayout(NamedTuple):
    """Where a named joint target lives in ``qpos``.

    Attributes:
        qpos_ids: ``qpos`` address of each target entry, or ``None`` when the
            target covers the leading ``qpos`` entries in order (the layout the
            pose terms assume by default).
        couplings: Limits that couple two target entries.
    """

    qpos_ids: tuple[int, ...] | None
    couplings: tuple[TargetCoupling, ...]


def select_target_jnt_range(
    joint_names: tuple[str, ...],
    target_jnt_range: Mapping[str, Any] | None,
    model_target_jnt_ranges: Mapping[str, Mapping[str, Any]] | None,
) -> Mapping[str, Any] | None:
    """Pick the target range for the loaded model.

    Args:
        joint_names: Joint names of the loaded model.
        target_jnt_range: Default ``{joint_name: (lo, hi)}``.
        model_target_jnt_ranges: ``{marker_joint: target_jnt_range}``; the first
            entry whose marker joint the model has replaces the default.

    Returns:
        The selected range (the default when no marker matches).
    """
    for marker, ranges in (model_target_jnt_ranges or {}).items():
        if marker in joint_names:
            return ranges
    return target_jnt_range


def joint_target_layout(
    model: mujoco.MjModel, joint_names: tuple[str, ...]
) -> JointTargetLayout:
    """Lay out a target over *joint_names* (one entry per joint) on *model*.

    Args:
        model: Compiled MuJoCo model.
        joint_names: Hinge/slide joint of each target entry.

    Returns:
        The :class:`JointTargetLayout`.

    Raises:
        KeyError: If a joint name is not in the model.
        ValueError: If a joint is not a hinge/slide joint.
    """
    adrs = []
    for name in joint_names:
        jid = model.joint(name).id
        if int(model.jnt_type[jid]) not in _SCALAR_JOINT_TYPES:
            raise ValueError(f"Joint {name!r} is not a hinge/slide joint.")
        adrs.append(int(model.jnt_qposadr[jid]))
    entry = {adr: i for i, adr in enumerate(adrs)}
    couplings = []
    for t in range(model.ntendon):
        start, num = int(model.tendon_adr[t]), int(model.tendon_num[t])
        wraps = range(start, start + num)
        if not model.tendon_limited[t] or num != 2:
            continue
        if any(model.wrap_type[w] != mujoco.mjtWrap.mjWRAP_JOINT for w in wraps):
            continue
        terms = [
            (entry.get(int(model.jnt_qposadr[model.wrap_objid[w]])), model.wrap_prm[w])
            for w in wraps
        ]
        if any(i is None for i, _ in terms):
            continue
        (i, ci), (j, cj) = sorted(terms, key=lambda term: -abs(term[1]))
        lo, hi = model.tendon_range[t]
        couplings.append(
            TargetCoupling(i, j, float(ci), float(cj), float(lo), float(hi))
        )
    leading = adrs == list(range(len(adrs)))
    return JointTargetLayout(None if leading else tuple(adrs), tuple(couplings))


def resolve_site_target_ranges(
    model: mujoco.MjModel, target_reach_range: Mapping[str, Any]
) -> SiteTargets:
    """Resolve ``{site_name: (lo_xyz, hi_xyz)}`` to site ids and target boxes.

    Each tip stays paired with its own box. Rows are sorted by tip site id,
    which for the hand model is the CPU registration order
    (TH, IF, MF, RF, LF).

    Args:
        model: Compiled MuJoCo model.
        target_reach_range: ``{site_name: (lo_xyz, hi_xyz)}``, in any order.
            The model must also contain a ``<site_name>_target`` site.

    Returns:
        :class:`SiteTargets` in model order.

    Raises:
        KeyError: If a tip site or its ``_target`` site is not in the model.
        ValueError: If the mapping is empty or a range is not ``(2, 3)``.
    """
    if not target_reach_range:
        raise ValueError("target_reach_range is empty.")
    rows = []
    for name, span in target_reach_range.items():
        tip_id = model.site(name).id
        target_id = model.site(f"{name}_target").id
        bounds = np.asarray(span, dtype=np.float64)
        if bounds.shape != (2, 3):
            raise ValueError(
                f"Range for site {name!r} must be (lo_xyz, hi_xyz), got shape {bounds.shape}."
            )
        rows.append((tip_id, target_id, name, bounds))
    rows.sort(key=lambda row: row[0])
    bounds = np.stack([b for *_, b in rows])
    return SiteTargets(
        names=tuple(name for _, _, name, _ in rows),
        tip_ids=np.array([tip for tip, _, _, _ in rows], dtype=np.int32),
        target_ids=np.array([tgt for _, tgt, _, _ in rows], dtype=np.int32),
        lo=bounds[:, 0],
        hi=bounds[:, 1],
    )
