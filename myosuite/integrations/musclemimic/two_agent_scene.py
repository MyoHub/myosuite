# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Shared utilities for building two-agent MuJoCo competitive scenes.

Two-agent competitive model files use frame attachment to compose two
independent agent specs into a single combined scene.  This module extracts
the common scaffolding so each model file only needs to implement its own
agent-specific geometry (limbs, scoring zones, sensors).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

import mujoco
import numpy as np

from myosuite.physics import quat_math

logger = logging.getLogger(__name__)

# 180° rotation around Z: places agent 1 facing agent 0 (MJCF: w first).
_QUAT_180Z: list[float] = [0.0, 0.0, 0.0, 1.0]


def build_combined_spec(
    agent0_spec: mujoco.MjSpec,
    agent1_spec: mujoco.MjSpec,
    *,
    separation_m: float,
    sim_dt: float,
    model_name: str = "combined",
    prefixes: tuple[str, str] = ("a0_", "a1_"),
    floor_rgba: list[float] | None = None,
    floor_collision: bool = True,
) -> mujoco.MjSpec:
    """Attach two agent specs into a combined scene and return the combined MjSpec.

    Agent 0 is placed at ``[0, +separation_m/2, 0]`` in its default orientation.
    Agent 1 is placed at ``[0, -separation_m/2, 0]`` and rotated 180° around Z
    so both agents face each other along the Y axis.

    A floor plane and a fill light are added to the combined spec.  Each agent
    spec is attached under a named frame so MuJoCo scopes all names automatically
    via the supplied prefixes.

    Args:
        agent0_spec: MjSpec for agent 0 (not yet attached to any parent).
        agent1_spec: MjSpec for agent 1 (not yet attached to any parent).
        separation_m: Frame-to-frame Y-axis distance (metres).
        sim_dt: Simulation timestep forwarded to ``spec.option.timestep``.
        model_name: Value assigned to ``spec.modelname``.
        prefixes: Name prefixes for agent 0 and agent 1 respectively.
        floor_rgba: RGBA colour of the floor plane.  Defaults to light grey.
        floor_collision: Whether the floor participates in collisions.

    Returns:
        Combined ``mujoco.MjSpec`` with both agents attached.
    """
    if floor_rgba is None:
        floor_rgba = [0.75, 0.75, 0.75, 1.0]

    spec = mujoco.MjSpec()
    spec.modelname = model_name
    spec.option.timestep = sim_dt

    floor = spec.worldbody.add_geom(name="floor")
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [8.0, 8.0, 0.1]
    floor.rgba = floor_rgba
    floor.contype = 1 if floor_collision else 0
    floor.conaffinity = 1 if floor_collision else 0

    fill = spec.worldbody.add_light(name="scene_fill")
    fill.pos = [0.0, 0.0, 4.0]
    fill.diffuse = [0.5, 0.5, 0.55]
    fill.castshadow = False

    frame0 = spec.worldbody.add_frame()
    frame0.pos = [0.0, separation_m / 2.0, 0.0]
    frame0.quat = [1.0, 0.0, 0.0, 0.0]
    spec.attach(agent0_spec, prefix=prefixes[0], suffix="", frame=frame0)

    frame1 = spec.worldbody.add_frame()
    frame1.pos = [0.0, -separation_m / 2.0, 0.0]
    frame1.quat = _QUAT_180Z
    spec.attach(agent1_spec, prefix=prefixes[1], suffix="", frame=frame1)

    return spec


def mj_name2id_strict(model: mujoco.MjModel, obj_type: int, name: str) -> int:
    """Return the MuJoCo ID for *name*, raising ``RuntimeError`` if not found.

    Args:
        model: Compiled MjModel.
        obj_type: MuJoCo object type constant (e.g. ``mujoco.mjtObj.mjOBJ_SITE``).
        name: Fully-qualified object name (including agent prefix).

    Returns:
        Non-negative integer ID.

    Raises:
        RuntimeError: If the object is not found in the model.
    """
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx == -1:
        raise RuntimeError(f"MuJoCo object not found: type={obj_type} name={name!r}")
    return idx


def extract_prefix_indices(
    model: mujoco.MjModel,
    prefix: str,
    *,
    arm_joint_tokens: tuple[str, ...] = (),
) -> tuple[list[int], list[int], list[int]]:
    """Extract actuator indices, joint ids, and arm-joint subset for one agent prefix.

    Args:
        model: Compiled combined MjModel.
        prefix: Agent name prefix (e.g. ``"a0_"``).
        arm_joint_tokens: Short joint names (without prefix) that constitute the
            arm-only subset used for opponent observations.  If empty, the arm
            subset is also empty.

    Returns:
        Tuple of ``(act_indices, jnt_ids, arm_jnt_ids)``:
        - ``act_indices``: All actuator indices whose name starts with *prefix*.
        - ``jnt_ids``: All joint ids whose name starts with *prefix*.
        - ``arm_jnt_ids``: Subset of *jnt_ids* whose full name is in
          ``{prefix + tok for tok in arm_joint_tokens}``.
    """
    act_indices = [
        i
        for i in range(model.nu)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "").startswith(
            prefix
        )
    ]

    jnt_ids = [
        j
        for j in range(model.njnt)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith(
            prefix
        )
    ]

    arm_names = {f"{prefix}{tok}" for tok in arm_joint_tokens}
    arm_jnt_ids = [
        j
        for j in jnt_ids
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "") in arm_names
    ]

    return act_indices, jnt_ids, arm_jnt_ids


def extract_site_ids(
    model: mujoco.MjModel,
    prefix: str,
    suffixes: tuple[str, ...],
) -> dict[str, int]:
    """Return ``{suffix: site_id}`` for each suffix in *suffixes*.

    Args:
        model: Compiled combined MjModel.
        prefix: Agent name prefix (e.g. ``"a0_"``).
        suffixes: Short site name suffixes (without prefix) to look up.

    Returns:
        Dict mapping each suffix to its site id.

    Raises:
        RuntimeError: If any site is not found.
    """
    return {
        suffix: mj_name2id_strict(model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}{suffix}")
        for suffix in suffixes
    }


def extract_sensor_addrs(
    model: mujoco.MjModel,
    prefix: str,
    suffixes: tuple[str, ...],
) -> dict[str, int]:
    """Return ``{suffix_vel: sensor_adr}`` for each suffix, looking up ``{prefix}{suffix}_vel``.

    Args:
        model: Compiled combined MjModel.
        prefix: Agent name prefix (e.g. ``"a0_"``).
        suffixes: Short site/sensor name suffixes (without prefix or ``_vel``).
            The sensor name looked up is ``{prefix}{suffix}_vel``.

    Returns:
        Dict mapping ``"{suffix}_vel"`` to the first sensordata index.

    Raises:
        RuntimeError: If any sensor is not found.
    """
    result: dict[str, int] = {}
    for suffix in suffixes:
        sensor_name = f"{prefix}{suffix}_vel"
        sid = mj_name2id_strict(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        result[f"{suffix}_vel"] = int(model.sensor_adr[sid])
    return result


_JOINT_QPOS_WIDTH: dict[int, int] = {
    int(mujoco.mjtJoint.mjJNT_FREE): 7,
    int(mujoco.mjtJoint.mjJNT_BALL): 4,
    int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
    int(mujoco.mjtJoint.mjJNT_HINGE): 1,
}

_JOINT_QVEL_WIDTH: dict[int, int] = {
    int(mujoco.mjtJoint.mjJNT_FREE): 6,
    int(mujoco.mjtJoint.mjJNT_BALL): 3,
    int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
    int(mujoco.mjtJoint.mjJNT_HINGE): 1,
}


@dataclass(frozen=True)
class AgentShadowMap:
    """Address map from one prefixed FBVs agent onto an unprefixed shadow model."""

    qpos_pairs: tuple[tuple[int, int, int], ...]
    qvel_pairs: tuple[tuple[int, int, int], ...]
    act_idx: np.ndarray
    prefix: str


def build_agent_shadow_map(
    fbvs_model: mujoco.MjModel,
    shadow_model: mujoco.MjModel,
    prefix: str,
    act_indices: np.ndarray | list[int],
) -> AgentShadowMap:
    """Map every shadow joint onto the same joint name under *prefix* in *fbvs_model*.

    Args:
        fbvs_model: Combined two-agent model (names like ``a0_root``).
        shadow_model: Single-agent MuscleMimic model (names like ``root``).
        prefix: Agent prefix (``a0_`` / ``a1_``).
        act_indices: FBVs actuator indices for this agent.

    Returns:
        Address pairs for qpos, qvel, and actuators.
    """
    qpos_pairs: list[tuple[int, int, int]] = []
    qvel_pairs: list[tuple[int, int, int]] = []
    for sj in range(shadow_model.njnt):
        sname = mujoco.mj_id2name(shadow_model, mujoco.mjtObj.mjOBJ_JOINT, sj)
        if sname is None:
            raise RuntimeError(f"shadow joint {sj} has no name")
        fj = mujoco.mj_name2id(fbvs_model, mujoco.mjtObj.mjOBJ_JOINT, prefix + sname)
        if fj < 0:
            raise RuntimeError(f"missing FBVs joint {prefix}{sname}")
        jt = int(shadow_model.jnt_type[sj])
        qpos_pairs.append(
            (
                int(fbvs_model.jnt_qposadr[fj]),
                int(shadow_model.jnt_qposadr[sj]),
                _JOINT_QPOS_WIDTH[jt],
            )
        )
        qvel_pairs.append(
            (
                int(fbvs_model.jnt_dofadr[fj]),
                int(shadow_model.jnt_dofadr[sj]),
                _JOINT_QVEL_WIDTH[jt],
            )
        )
    return AgentShadowMap(
        qpos_pairs=tuple(qpos_pairs),
        qvel_pairs=tuple(qvel_pairs),
        act_idx=np.asarray(act_indices, dtype=int),
        prefix=prefix,
    )


def copy_agent_to_shadow(
    fbvs_data: mujoco.MjData,
    shadow_data: mujoco.MjData,
    amap: AgentShadowMap,
) -> None:
    """Copy one FBVs agent's qpos/qvel/act onto the shadow ``MjData``."""
    for fa, sa, w in amap.qpos_pairs:
        shadow_data.qpos[sa : sa + w] = fbvs_data.qpos[fa : fa + w]
    for fa, sa, w in amap.qvel_pairs:
        shadow_data.qvel[sa : sa + w] = fbvs_data.qvel[fa : fa + w]
    if (
        shadow_data.act is not None
        and fbvs_data.act is not None
        and shadow_data.act.size
        and fbvs_data.act.size
    ):
        shadow_data.act[:] = fbvs_data.act[amap.act_idx]


def _nlerp_quat(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Normalized lerp; flips *q1* so the blend takes the short arc."""
    a = np.asarray(q0, dtype=np.float64)
    b = np.asarray(q1, dtype=np.float64)
    if float(np.dot(a, b)) < 0.0:
        b = -b
    out = (1.0 - alpha) * a + alpha * b
    n = float(np.linalg.norm(out))
    return out / n if n > 1e-12 else a


def _blend_qpos(dst: np.ndarray, src: np.ndarray, alpha: float) -> None:
    """Lerp *dst* toward *src*. Width 4/7 is a quaternion (ball / free)."""
    w = int(dst.shape[0])
    if w == 7:
        dst[:3] = (1.0 - alpha) * dst[:3] + alpha * src[:3]
        dst[3:7] = _nlerp_quat(dst[3:7], src[3:7], alpha)
    elif w == 4:
        dst[:] = _nlerp_quat(dst, src, alpha)
    else:
        dst[:] = (1.0 - alpha) * dst + alpha * src


def write_clip_pose(
    fbvs_data: mujoco.MjData,
    amap: AgentShadowMap,
    clip_qpos: np.ndarray,
    clip_qvel: np.ndarray,
    *,
    keep_xy: bool = True,
    keep_yaw: bool = False,
    act: float | np.ndarray | None = 0.05,
    alpha: float = 1.0,
) -> None:
    """Write a single-agent clip frame onto one FBVs agent.

    Args:
        fbvs_data: Combined two-agent ``MjData``.
        amap: Address map from :func:`build_agent_shadow_map`.
        clip_qpos: Shadow-model qpos frame.
        clip_qvel: Shadow-model qvel frame.
        keep_xy: Keep the agent's current root XY (spawn / mid-episode).
        keep_yaw: Keep the agent's current root yaw and rotate clip planar
            velocity into that frame (RSI without a heading change).
        act: Muscle activation written on this agent (``0.05`` matches a
            standing overlay). ``None`` leaves ``act`` unchanged.
        alpha: Blend toward the clip (``1`` snaps, ``0`` is a no-op). A
            per-step value in ``(0, 1)`` tracks the gait without a pop.
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0.0:
        return
    fa, sa, _ = amap.qpos_pairs[0]
    xy = fbvs_data.qpos[fa : fa + 2].copy() if keep_xy else None
    yaw = (
        float(quat_math.quat2yaw(fbvs_data.qpos[fa + 3 : fa + 7])) if keep_yaw else None
    )
    clip_yaw = float(quat_math.quat2yaw(clip_qpos[sa + 3 : sa + 7]))
    if alpha >= 1.0:
        for fadr, sadr, w in amap.qpos_pairs:
            fbvs_data.qpos[fadr : fadr + w] = clip_qpos[sadr : sadr + w]
        for fadr, sadr, w in amap.qvel_pairs:
            fbvs_data.qvel[fadr : fadr + w] = clip_qvel[sadr : sadr + w]
    else:
        for fadr, sadr, w in amap.qpos_pairs:
            _blend_qpos(
                fbvs_data.qpos[fadr : fadr + w],
                clip_qpos[sadr : sadr + w],
                alpha,
            )
        for fadr, sadr, w in amap.qvel_pairs:
            fbvs_data.qvel[fadr : fadr + w] = (1.0 - alpha) * fbvs_data.qvel[
                fadr : fadr + w
            ] + alpha * clip_qvel[sadr : sadr + w]
    if xy is not None:
        fbvs_data.qpos[fa : fa + 2] = xy
    if yaw is not None:
        dpsi = yaw - clip_yaw
        qz = quat_math.axis_angle2quat(np.array([0.0, 0.0, 1.0]), dpsi)
        fbvs_data.qpos[fa + 3 : fa + 7] = quat_math.mul_quat(
            qz, clip_qpos[sa + 3 : sa + 7]
        )
        fva, sva, _ = amap.qvel_pairs[0]
        lin = np.array([float(clip_qvel[sva]), float(clip_qvel[sva + 1]), 0.0])
        lin = quat_math.rot_vec_quat(lin, qz)
        fbvs_data.qvel[fva] = lin[0]
        fbvs_data.qvel[fva + 1] = lin[1]
    if (
        act is not None
        and fbvs_data.act is not None
        and fbvs_data.act.size
        and amap.act_idx.size
    ):
        fbvs_data.act[amap.act_idx] = act


def default_root_offsets(
    separation_m: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return the standard ``agent_0``/``agent_1`` root offsets for a 1v1 scene.

    Mirrors the placement convention already used by ``build_combined_spec``:
    agent 0 sits at ``[0, +separation_m/2, 0]`` in its default orientation,
    agent 1 sits at ``[0, -separation_m/2, 0]`` rotated 180 degrees about Z
    so the two agents face each other along the Y axis.

    Args:
        separation_m: Total Y-axis distance between the two agents (metres).

    Returns:
        ``{"agent_0": (offset_xyz, offset_quat_wxyz), "agent_1": (...)}``.
    """
    identity_quat = np.array([1.0, 0.0, 0.0, 0.0])
    quat_180z = np.array(_QUAT_180Z)
    return {
        "agent_0": (np.array([0.0, separation_m / 2.0, 0.0]), identity_quat),
        "agent_1": (np.array([0.0, -separation_m / 2.0, 0.0]), quat_180z),
    }


def _apply_root_offset(
    root_qpos: np.ndarray, offset_pos: np.ndarray, offset_quat: np.ndarray
) -> np.ndarray:
    """Rigidly transform a free-joint's ``[xyz, wxyz]`` qpos by an offset pose.

    Free-joint qpos is an *absolute* world pose (unlike hinge/slide qpos,
    which is relative to the body's default frame), so any per-agent
    placement offset must be composed into it explicitly rather than relying
    on MJCF-level ``<frame>`` translation/rotation (those are folded into the
    body's default pose at compile time and have no effect once qpos
    overrides a free joint's position/orientation).

    Args:
        root_qpos: Length-7 ``[x, y, z, qw, qx, qy, qz]`` free-joint qpos.
        offset_pos: Length-3 world-frame translation to apply.
        offset_quat: Length-4 ``[qw, qx, qy, qz]`` world-frame rotation to
            apply (rotates the source pose, then translates).

    Returns:
        New length-7 ``[xyz, wxyz]`` array with the offset applied.
    """
    out = np.array(root_qpos, dtype=float, copy=True)
    src_pos = out[:3]
    src_quat = out[3:7]

    rotated_pos = np.zeros(3)
    mujoco.mju_rotVecQuat(rotated_pos, src_pos, offset_quat)
    out[:3] = rotated_pos + offset_pos

    composed_quat = np.zeros(4)
    mujoco.mju_mulQuat(composed_quat, offset_quat, src_quat)
    out[3:7] = composed_quat
    return out


def two_agent_standing_qpos(
    model: mujoco.MjModel,
    jnt_ids_by_agent: Mapping[str, list[int]],
    source_key_qpos_by_agent: Mapping[str, np.ndarray | None],
    root_offset_by_agent: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> np.ndarray | None:
    """Reconstruct a combined-model standing ``qpos`` from each agent's own keyframe.

    ``MjSpec.attach`` does not carry a source model's ``<keyframe>`` into the
    combined two-agent scene (each agent's keyframe would collide on name
    after prefixing), so any standing pose the source model ships is
    otherwise silently lost -- the combined model resets to all-zero joint
    angles, i.e. a rigid "pedestal" pose that collapses immediately under
    gravity (see ``ModularMultiAgentTaskEnv.reset()`` / the CPU
    ``ModularTaskEnv.reset()`` fix this mirrors).

    Attach preserves each agent's joint declaration order and address
    contiguity, so one agent's qpos block in the combined model is a
    straight positional copy of that agent's own (pre-attach) qpos layout.
    This reconstructs the full combined ``qpos`` by placing each agent's
    captured keyframe-0 ``qpos`` (captured from the standalone single-agent
    model *before* attachment) at that agent's contiguous address range in
    the combined model.

    Because both agents' keyframes come from the same single-agent source
    model, they carry the *same* root position (and orientation) -- without
    an explicit per-agent offset both agents would land on top of each other
    in the combined model, since free-joint qpos is an absolute world pose
    and the ``<frame>`` translation ``build_combined_spec`` applies at the
    MJCF level has no effect once qpos overrides it (see
    ``_apply_root_offset``). ``root_offset_by_agent`` supplies that missing
    per-agent translation/rotation; use ``default_root_offsets`` to match the
    same placement convention already used by ``build_combined_spec``.

    Args:
        model: Compiled combined ``MjModel``.
        jnt_ids_by_agent: ``{agent_id: [joint ids in the combined model]}``,
            as returned by ``extract_prefix_indices``.
        source_key_qpos_by_agent: ``{agent_id: standalone source model's
            key_qpos[0]}`` (or ``None`` if that source model had no
            keyframe).
        root_offset_by_agent: Optional ``{agent_id: (offset_xyz,
            offset_quat_wxyz)}`` rigid transform applied to that agent's
            root (first 7 qpos values, i.e. a free joint) before it is
            written into the combined qpos. Agents without a free-joint root,
            or without an entry here, are left untransformed. If ``None``,
            no offset is applied (legacy behaviour -- callers that build
            their own placement must supply this to avoid overlapping
            agents).

    Returns:
        Full-length ``qpos`` array for the combined model, or ``None`` if
        any agent's source model had no keyframe to reconstruct from.
    """
    if any(v is None for v in source_key_qpos_by_agent.values()):
        return None
    root_offset_by_agent = root_offset_by_agent or {}
    standing = np.zeros(model.nq)
    for agent, jnt_ids in jnt_ids_by_agent.items():
        src = np.asarray(source_key_qpos_by_agent[agent], dtype=float).copy()
        addrs = [
            (
                int(model.jnt_qposadr[jid]),
                int(model.jnt_qposadr[jid])
                + _JOINT_QPOS_WIDTH[int(model.jnt_type[jid])],
                int(model.jnt_type[jid]),
            )
            for jid in jnt_ids
        ]
        lo = min(a for a, _, _ in addrs)
        hi = max(b for _, b, _ in addrs)
        if hi - lo != src.shape[0]:
            raise ValueError(
                f"Agent {agent!r} qpos block width {hi - lo} does not match "
                f"source keyframe width {src.shape[0]} -- attach did not "
                "preserve contiguous per-agent joint ordering as expected."
            )
        offset = root_offset_by_agent.get(agent)
        if offset is not None:
            free_addrs = [
                (a, b) for a, b, t in addrs if t == int(mujoco.mjtJoint.mjJNT_FREE)
            ]
            if free_addrs:
                free_lo, free_hi = free_addrs[0]
                assert free_hi - free_lo == 7
                offset_pos, offset_quat = offset
                src[free_lo - lo : free_hi - lo] = _apply_root_offset(
                    src[free_lo - lo : free_hi - lo], offset_pos, offset_quat
                )
        standing[lo:hi] = src
    return standing
