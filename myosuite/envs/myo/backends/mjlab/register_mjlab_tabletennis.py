# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Register MyoChallenge TableTennis with mjlab (ManagerBasedRlEnvCfg + task registry)."""

from __future__ import annotations

import logging
import weakref
from collections.abc import Callable
from typing import Any

import mujoco
import numpy as np
from mjlab.actuator import XmlActuatorCfg as _XmlActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp import terminations as mdp_terminations
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.registry import register_mjlab_task
from scipy.spatial.transform import Rotation as R

from myosuite.core.model_builder import build_from_recipe
from myosuite.core.model_recipes import (
    _add_tabletennis_furniture,
    _tabletennis_body_spec,
)
from myosuite.envs.myo.backends.mjlab.configs.table_tennis_cfg import TableTennisCfg
from myosuite.envs.myo.tasks.challenge.tabletennis import (
    ContactTrajIssue,
    PingpongContactLabels,
    evaluate_pingpong_trajectory,
)
from myosuite.terms.base_action import sigmoid_muscle_activation

logger = logging.getLogger(__name__)

_TT_ENTITY_NAME = "table_tennis_robot"
_MAX_TIME = 3.0
_TT_RWD_WEIGHTS: dict[str, float] = {
    "reach_dist": 1.0,
    "palm_dist": 1.0,
    "paddle_quat": 2.0,
    "act_reg": 0.5,
    "torso_up": 2.0,
    "sparse": 100.0,
    "solved": 1000.0,
    "done": -10.0,
}

_REF_MODEL: mujoco.MjModel | None = None
_REF_QPOS_KEY0: np.ndarray | None = None
_REF_QVEL_INIT: np.ndarray | None = None
_TT_INDEX: dict[str, Any] | None = None

# Weak-keyed caches: entries are evicted when the env is garbage collected,
# preventing id(env) stale-key bugs and memory leaks in long-running processes.
_TT_RUNTIME: weakref.WeakKeyDictionary[Any, dict[str, Any]] = (
    weakref.WeakKeyDictionary()
)
_TT_SCENE: weakref.WeakKeyDictionary[Any, dict[str, dict]] = weakref.WeakKeyDictionary()
# GPU tensor cache: label_lookup and geom_bodyid_t, built once per env on first step.
_TT_GPU: weakref.WeakKeyDictionary[Any, dict[str, Any]] = weakref.WeakKeyDictionary()
_TT_TENDON_NAMES: tuple[str, ...] | None = None
_TT_POSITION_ACTUATOR_NAMES: tuple[str, ...] | None = None


def _reference_model() -> mujoco.MjModel:
    global _REF_MODEL, _REF_QPOS_KEY0, _REF_QVEL_INIT, _TT_INDEX
    if _REF_MODEL is not None:
        return _REF_MODEL
    model, _ = build_from_recipe("challenge_tabletennis")
    _REF_MODEL = model
    d = mujoco.MjData(model)
    _REF_QPOS_KEY0 = model.key_qpos[0].copy()
    _REF_QVEL_INIT = d.qvel.copy()
    start_vel = np.array([5.6, 1.6, 0.1], dtype=np.float64)
    ball_dofadr = int(model.body_dofadr[model.body("pingpong").id])
    _REF_QVEL_INIT[ball_dofadr : ball_dofadr + 3] = start_vel

    myo_joint_range = np.concatenate(
        [
            model.joint(i).qposadr
            for i in range(model.njnt)
            if not model.joint(i).name.startswith("ping")
            and model.joint(i).name not in ("pingpong_freejoint", "paddle_freejoint")
        ]
    )
    myo_dof_range = np.concatenate(
        [
            model.joint(i).dofadr
            for i in range(model.njnt)
            if not model.joint(i).name.startswith("ping")
            and model.joint(i).name != "paddle_freejoint"
        ]
    )
    init_paddle_quat = R.from_euler(
        "xyz", np.array([-0.3, 1.57, 0]), degrees=False
    ).as_quat()[[3, 0, 1, 2]]
    flex_adr = int(model.jnt_qposadr[model.joint("flex_extension").id])
    ball_bid = int(model.body("pingpong").id)
    ball_gid = int(model.geom("pingpong").id)
    _TT_INDEX = {
        "pelvis_site_id": int(model.site("pelvis").id),
        "ball_site_id": int(model.site("pingpong").id),
        "paddle_site_id": int(model.site("paddle").id),
        "grasp_site_id": int(model.site("S_grasp").id),
        "paddle_body_id": int(model.body("paddle").id),
        "ball_body_id": ball_bid,
        "ball_geom_id": ball_gid,
        "geom_bodyid": np.asarray(model.geom_bodyid, dtype=np.int32),
        "gids": {
            "pad": int(model.geom("pad").id),
            "own": int(model.geom("coll_own_half").id),
            "opp": int(model.geom("coll_opponent_half").id),
            "net": int(model.geom("coll_net").id),
            "ground": int(model.geom("ground").id),
        },
        "myo_joint_idx": np.asarray(myo_joint_range, dtype=np.int64),
        "myo_dof_idx": np.asarray(myo_dof_range, dtype=np.int64),
        "sensor_pingpong_vel_adr": int(
            model.sensor_adr[model.sensor("pingpong_vel_sensor").id]
        ),
        "sensor_pingpong_vel_dim": int(
            model.sensor_dim[model.sensor("pingpong_vel_sensor").id]
        ),
        "sensor_paddle_vel_adr": int(
            model.sensor_adr[model.sensor("paddle_vel_sensor").id]
        ),
        "sensor_paddle_vel_dim": int(
            model.sensor_dim[model.sensor("paddle_vel_sensor").id]
        ),
        "flex_qpos_adr": flex_adr,
        "ball_qpos_adr": int(model.joint("pingpong_freejoint").qposadr[0]),
        "ball_dof_adr": ball_dofadr,
        "init_paddle_quat": init_paddle_quat.astype(np.float32),
        "muscle_act_mask": np.asarray(
            model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE, dtype=bool
        ),
        "actuator_ctrlrange": np.asarray(model.actuator_ctrlrange, dtype=np.float64),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "na": int(model.na),
        "jnt_range": np.asarray(model.jnt_range, dtype=np.float64),
        "jnt_qposadr": np.asarray(model.jnt_qposadr, dtype=np.int32),
        "njnt": int(model.njnt),
    }
    return model


def _tt_actuator_xml_groups() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return (muscle tendon names, position actuator names) for mjlab Xml wrapping."""
    global _TT_TENDON_NAMES, _TT_POSITION_ACTUATOR_NAMES
    if _TT_TENDON_NAMES is not None and _TT_POSITION_ACTUATOR_NAMES is not None:
        return _TT_TENDON_NAMES, _TT_POSITION_ACTUATOR_NAMES
    m = _reference_model()
    tendons: list[str] = []
    positions: list[str] = []
    for i in range(m.nu):
        aname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if m.actuator_dyntype[i] == mujoco.mjtDyn.mjDYN_MUSCLE:
            tid = int(m.actuator_trnid[i, 0])
            tname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TENDON, tid)
            if tname is not None:
                tendons.append(tname)
        else:
            if aname is not None:
                positions.append(aname)
    _TT_TENDON_NAMES = tuple(tendons)
    _TT_POSITION_ACTUATOR_NAMES = tuple(positions)
    return _TT_TENDON_NAMES, _TT_POSITION_ACTUATOR_NAMES


_TT_BALL_ENTITY_NAME = "pingpong"


def _resolve_tt_scene_ids(env: Any, entity_name: str) -> dict[str, Any]:
    """Resolve all scene-level indices from the compiled mjlab model.

    Called once per env instance and cached.  All IDs are scene-level (i.e.
    suitable for indexing into ``entity.data.data.*`` Warp buffers).  The
    reference model (``_TT_INDEX``) is only used for quantities that do not
    depend on scene-level addressing: actuator masks, ctrlrange, muscle flags.
    """
    env_cache = _TT_SCENE.setdefault(env, {})
    if entity_name in env_cache:
        return env_cache[entity_name]

    ent = env.scene[entity_name]
    ball_ent = env.scene[_TT_BALL_ENTITY_NAME]
    mj_m = env.sim.mj_model

    arm_prefix = f"{entity_name}/"
    ball_prefix = f"{_TT_BALL_ENTITY_NAME}/"
    paddle_fj = f"{arm_prefix}paddle_freejoint"

    # --- site IDs (scene-level via entity.indexing.site_ids) ---
    def _scene_site(e: Any, short_name: str) -> int:
        local, _ = e.find_sites([short_name])
        return int(e.indexing.site_ids[local[0]])

    def _scene_body(e: Any, short_name: str) -> int:
        local, _ = e.find_bodies([short_name])
        return int(e.indexing.body_ids[local[0]])

    # --- arm joint qpos / dof addresses in the scene model ---
    myo_qpos: list[int] = []
    myo_dof: list[int] = []
    jnt_qposadr_arm: list[int] = []
    jnt_range_arm: list[tuple[float, float]] = []
    for i in range(mj_m.njnt):
        jname = mujoco.mj_id2name(mj_m, mujoco.mjtObj.mjOBJ_JOINT, i)
        if not jname or not jname.startswith(arm_prefix):
            continue
        jnt = mj_m.joint(jname)
        if jname == paddle_fj:
            continue
        jnt_qposadr_arm.append(int(jnt.qposadr[0]))
        jnt_range_arm.append((float(mj_m.jnt_range[i, 0]), float(mj_m.jnt_range[i, 1])))
        myo_qpos.append(int(jnt.qposadr[0]))
        myo_dof.append(int(jnt.dofadr[0]))

    flex_jnt = mj_m.joint(f"{arm_prefix}flex_extension")
    flex_qpos_adr = int(flex_jnt.qposadr[0])

    # --- ball body / contact geom IDs in the scene model ---
    ball_bid = mj_m.body(f"{ball_prefix}pingpong").id
    geom_bodyid = np.asarray(mj_m.geom_bodyid, dtype=np.int32)

    def _gid(name: str) -> int:
        return int(mj_m.geom(name).id)

    gids = {
        "pad": _gid(f"{arm_prefix}pad"),
        "own": _gid(f"{arm_prefix}coll_own_half"),
        "opp": _gid(f"{arm_prefix}coll_opponent_half"),
        "net": _gid(f"{arm_prefix}coll_net"),
        # ground is inside the arm entity spec (attached via table tennis XML)
        "ground": _gid(f"{arm_prefix}ground"),
    }

    # --- sensor addresses (added via SceneCfg.spec_fn, not inside any entity) ---
    sid_pv = mj_m.sensor("pingpong_vel_sensor").id
    sid_pad = mj_m.sensor("paddle_vel_sensor").id

    # --- ball qpos / dof addresses in the scene model ---
    ball_fj = mj_m.joint(f"{ball_prefix}pingpong_freejoint")
    ball_qpos_adr = int(ball_fj.qposadr[0])
    ball_dof_adr = int(ball_fj.dofadr[0])

    env_cache[entity_name] = {
        "pelvis_site": _scene_site(ent, "pelvis"),
        "ball_site": _scene_site(ball_ent, "pingpong"),
        "paddle_site": _scene_site(ent, "paddle"),
        "grasp_site": _scene_site(ent, "S_grasp"),
        "paddle_body": _scene_body(ent, "paddle"),
        "ball_body_id": int(ball_bid),
        "geom_bodyid": geom_bodyid,
        "gids": gids,
        "flex_qpos_adr": flex_qpos_adr,
        "myo_joint_idx": np.asarray(myo_qpos, dtype=np.int64),
        "myo_dof_idx": np.asarray(myo_dof, dtype=np.int64),
        "jnt_qposadr": np.asarray(jnt_qposadr_arm, dtype=np.int32),
        "jnt_range": np.asarray(jnt_range_arm, dtype=np.float64),
        "njnt_arm": len(jnt_qposadr_arm),
        "ball_qpos_adr": ball_qpos_adr,
        "ball_dof_adr": ball_dof_adr,
        # Sensors live in the scene spec (via SceneCfg.spec_fn).
        "sensor_pingpong_vel_adr": int(mj_m.sensor_adr[sid_pv]),
        "sensor_pingpong_vel_dim": int(mj_m.sensor_dim[sid_pv]),
        "sensor_paddle_vel_adr": int(mj_m.sensor_adr[sid_pad]),
        "sensor_paddle_vel_dim": int(mj_m.sensor_dim[sid_pad]),
    }
    return env_cache[entity_name]


def _table_tennis_full_spec() -> mujoco.MjSpec:
    """Build the same torso+arms+legs+furniture spec as the CPU recipe.

    Reuses ``model_recipes._tabletennis_body_spec``/``_add_tabletennis_furniture``
    directly (not :func:`build_from_recipe`) so mjlab and CPU always compose
    the identical un-keyframed spec — the mjlab entity/scene split below then
    strips/re-adds pieces from this one shared spec instead of maintaining a
    second, independently-written composition.
    """
    return _add_tabletennis_furniture(_tabletennis_body_spec())


def _table_tennis_spec_fn() -> mujoco.MjSpec:
    spec = _table_tennis_full_spec()
    # Remove the ball body — it becomes its own entity so it can be reset via
    # Entity.write_root_state_to_sim.  The two velocimeter sensors reference
    # sites from both trees (paddle + pingpong) and are re-added post-attachment
    # by _tt_scene_spec_fn via SceneCfg.spec_fn.
    for s in list(spec.sensors):
        if s.name in ("pingpong_vel_sensor", "paddle_vel_sensor"):
            spec.delete(s)
    for b in list(spec.bodies):
        if b.name == "pingpong":
            spec.delete(b)
            break
    return spec


def _pingpong_spec_fn() -> mujoco.MjSpec:
    """Minimal spec containing only the pingpong ball (freejoint + site + geom)."""
    full = _table_tennis_full_spec()
    # ``full.body("pingpong")`` returns None for this composed/attached spec
    # (name lookup isn't populated pre-compile for attached subtrees) —
    # iterate instead.
    ball_body = next(b for b in full.bodies if b.name == "pingpong")
    # Build a clean spec with just the ball subtree.
    spec = mujoco.MjSpec()
    frame = spec.worldbody.add_frame()
    frame.attach_body(ball_body, "", "")
    # attach_body may carry sensors that reference the ball site; strip them
    # here — they will be re-added at scene level via SceneCfg.spec_fn.
    for s in list(spec.sensors):
        spec.delete(s)
    return spec


def _tt_scene_spec_fn(
    arm_entity_name: str, ball_entity_name: str
) -> Callable[[mujoco.MjSpec], None]:
    """Return a SceneCfg.spec_fn that re-adds the two cross-tree velocimeters.

    After mjlab attaches entities the site names are prefixed as
    ``{entity_name}/{site_name}``.  The two velocimeters reference sites in
    different entity trees so they cannot live inside either entity spec and
    must be added here, at the combined-spec level.
    """
    paddle_site = f"{arm_entity_name}/paddle"
    ball_site = f"{ball_entity_name}/pingpong"

    def _fn(spec: mujoco.MjSpec) -> None:
        s1 = spec.add_sensor()
        s1.name = "pingpong_vel_sensor"
        s1.type = mujoco.mjtSensor.mjSENS_VELOCIMETER
        s1.objtype = mujoco.mjtObj.mjOBJ_SITE
        s1.objname = ball_site

        s2 = spec.add_sensor()
        s2.name = "paddle_vel_sensor"
        s2.type = mujoco.mjtSensor.mjSENS_VELOCIMETER
        s2.objtype = mujoco.mjtObj.mjOBJ_SITE
        s2.objname = paddle_site

    return _fn


def _get_tt_runtime(env: Any) -> dict[str, Any]:
    if env not in _TT_RUNTIME:
        n = int(env.num_envs)
        _TT_RUNTIME[env] = {
            "trajectories": [[] for _ in range(n)],
            "cur_rally": [0] * n,
            "last_done": [False] * n,
        }
    return _TT_RUNTIME[env]


def _geom_to_label(gid: int, gids: dict[str, int]) -> PingpongContactLabels | None:
    if gid == gids["pad"]:
        return PingpongContactLabels.PADDLE
    if gid == gids["own"]:
        return PingpongContactLabels.OWN
    if gid == gids["opp"]:
        return PingpongContactLabels.OPPONENT
    if gid == gids["net"]:
        return PingpongContactLabels.NET
    if gid == gids["ground"]:
        return PingpongContactLabels.GROUND
    return PingpongContactLabels.ENV


def _contacts_for_env(
    data: Any,
    env_i: int,
    ball_bid: int,
    geom_bodyid: np.ndarray,
    gids: dict[str, int],
) -> frozenset[PingpongContactLabels]:
    labels: set[PingpongContactLabels] = set()
    try:
        nacon = int(data.nacon.detach().cpu().item())
    except (AttributeError, TypeError):
        # mjlab API shape varies: try indexed form (nacon is a per-env tensor)
        try:
            nacon = int(data.nacon[0].item())
        except (AttributeError, IndexError, TypeError):
            return frozenset()
    if nacon <= 0:
        return frozenset()
    try:
        geom = data.contact.geom
        wid = data.contact.worldid
    except AttributeError:
        return frozenset()
    nmax = min(nacon, int(geom.shape[0]))
    for k in range(nmax):
        try:
            if int(wid[k].detach().cpu().item()) != env_i:
                continue
        except (IndexError, TypeError):
            continue
        try:
            g1 = int(geom[k, 0].detach().cpu().item())
            g2 = int(geom[k, 1].detach().cpu().item())
        except (IndexError, TypeError):
            try:
                g1 = int(geom[k][0].detach().cpu().item())
                g2 = int(geom[k][1].detach().cpu().item())
            except (IndexError, TypeError):
                continue
        b1 = int(geom_bodyid[g1])
        b2 = int(geom_bodyid[g2])
        if b1 == ball_bid:
            lab = _geom_to_label(g2, gids)
        elif b2 == ball_bid:
            lab = _geom_to_label(g1, gids)
        else:
            continue
        if lab is not None:
            labels.add(lab)
    return frozenset(labels)


def _ball_label_vec(labels: frozenset[PingpongContactLabels], ref_vec: Any) -> Any:
    import torch

    v = torch.zeros(6, dtype=torch.float32, device=ref_vec.device)
    for lab in labels:
        if lab == PingpongContactLabels.PADDLE:
            v[0] += 1.0
        elif lab == PingpongContactLabels.OWN:
            v[1] += 1.0
        elif lab == PingpongContactLabels.OPPONENT:
            v[2] += 1.0
        elif lab == PingpongContactLabels.NET:
            v[3] += 1.0
        elif lab == PingpongContactLabels.GROUND:
            v[4] += 1.0
        else:
            v[5] += 1.0
    return v


_LABEL_ENUM_MAP: list[PingpongContactLabels] = [
    PingpongContactLabels.PADDLE,
    PingpongContactLabels.OWN,
    PingpongContactLabels.OPPONENT,
    PingpongContactLabels.NET,
    PingpongContactLabels.GROUND,
    PingpongContactLabels.ENV,
]


def _build_label_lookup(n_geoms: int, gids: dict[str, int], device: Any) -> Any:
    """Build a GPU tensor mapping geom_id → label index (0=PADDLE … 5=ENV)."""
    import torch

    t = torch.full((n_geoms,), 5, dtype=torch.long, device=device)
    t[gids["pad"]] = 0
    t[gids["own"]] = 1
    t[gids["opp"]] = 2
    t[gids["net"]] = 3
    t[gids["ground"]] = 4
    return t


def _get_gpu_tensors(env: Any, sc: dict[str, Any], device: Any) -> dict[str, Any]:
    """Return (and cache) the GPU tensors needed for vectorized contact detection."""
    import torch

    if env not in _TT_GPU:
        geom_bodyid = sc["geom_bodyid"]
        n_geoms = len(geom_bodyid)
        _TT_GPU[env] = {
            "geom_bodyid_t": torch.as_tensor(
                geom_bodyid, device=device, dtype=torch.long
            ),
            "label_lookup": _build_label_lookup(n_geoms, sc["gids"], device),
            "n_geoms": n_geoms,
        }
    return _TT_GPU[env]


def _contacts_all_envs_vectorized(
    data: Any,
    ball_bid: int,
    gpu_tensors: dict[str, Any],
    n: int,
    device: Any,
) -> tuple[list[frozenset[PingpongContactLabels]], Any]:
    """Vectorized contact detection for all envs in a single GPU pass.

    Returns:
        contact_sets: list of frozensets (one per env) for trajectory tracking.
        touching_info: float32 tensor of shape (n, 6) on ``device``.
    """
    import torch

    empty_sets: list[frozenset[PingpongContactLabels]] = [frozenset() for _ in range(n)]
    touching_info = torch.zeros(n, 6, dtype=torch.float32, device=device)

    try:
        geom = data.contact.geom  # (nconmax, 2)
        wid = data.contact.worldid  # (nconmax,)
    except AttributeError:
        return empty_sets, touching_info

    if geom.numel() == 0:
        return empty_sets, touching_info

    n_geoms = gpu_tensors["n_geoms"]
    geom_bodyid_t = gpu_tensors["geom_bodyid_t"]
    label_lookup = gpu_tensors["label_lookup"]

    wid_l = wid.long()
    g1 = torch.clamp(geom[:, 0].long(), 0, n_geoms - 1)
    g2 = torch.clamp(geom[:, 1].long(), 0, n_geoms - 1)

    b1 = geom_bodyid_t[g1]
    b2 = geom_bodyid_t[g2]
    is_ball1 = b1 == ball_bid
    is_ball2 = b2 == ball_bid

    valid = (wid_l >= 0) & (wid_l < n) & (is_ball1 | is_ball2)

    if not valid.any():
        return empty_sets, touching_info

    other_geom = torch.where(is_ball1, g2, g1)
    label_idx = label_lookup[other_geom]

    valid_idx = valid.nonzero(as_tuple=True)[0]
    env_idx = wid_l[valid_idx]
    lab_idx = label_idx[valid_idx]

    # Deduplicate (env_id, label_id) pairs — matches CPU frozenset semantics.
    pairs = torch.stack([env_idx, lab_idx], dim=1)
    unique_pairs = pairs.unique(dim=0)
    flat_idx_dedup = unique_pairs[:, 0] * 6 + unique_pairs[:, 1]
    touching_info.view(-1).scatter_add_(
        0,
        flat_idx_dedup,
        torch.ones(flat_idx_dedup.shape[0], dtype=torch.float32, device=device),
    )

    # One batched CPU transfer for trajectory tracking (frozensets).
    env_cpu = env_idx.cpu().numpy()
    lab_cpu = lab_idx.cpu().numpy()
    per_env: list[set[PingpongContactLabels]] = [set() for _ in range(n)]
    for c, e in zip(lab_cpu, env_cpu):
        per_env[int(e)].add(_LABEL_ENUM_MAP[int(c)])
    return [frozenset(s) for s in per_env], touching_info


def _cal_ball_qvel_torch(ball_qpos: Any) -> tuple[Any, Any]:
    """Return (v_low[3], v_high[3]) tensors for sampling initial ball linear velocity."""
    import torch

    table_upper = torch.tensor(
        [1.35, 0.70, 0.785], dtype=ball_qpos.dtype, device=ball_qpos.device
    )
    table_lower = torch.tensor(
        [0.5, -0.60, 0.785], dtype=ball_qpos.dtype, device=ball_qpos.device
    )
    gravity = 9.81
    v_z = (
        torch.rand(1, device=ball_qpos.device, dtype=ball_qpos.dtype) * 0.2 - 0.1
    ).squeeze(0)
    a = -0.5 * gravity
    b = v_z
    c = ball_qpos[2] - table_upper[2]
    disc = b * b - 4 * a * c
    t = (-b - torch.sqrt(torch.clamp(disc, min=0.0))) / (2 * a + 1e-8)
    v_upper = torch.stack([(table_upper[i] - ball_qpos[i]) / t for i in range(2)])
    v_lower = torch.stack([(table_lower[i] - ball_qpos[i]) / t for i in range(2)])
    v_high = torch.cat([v_upper, v_z.unsqueeze(0)], dim=0)
    v_low = torch.cat([v_lower, v_z.unsqueeze(0)], dim=0)
    return v_low, v_high


class TableTennisMixedCtrlActionCfg:
    """Config for mixed muscle + position actuator control (TableTennis)."""

    def __init__(self, *, entity_name: str) -> None:
        self.entity_name = entity_name

    def build(self, env: ManagerBasedRlEnv) -> TableTennisMixedCtrlAction:
        return TableTennisMixedCtrlAction(self, env)


class TableTennisMixedCtrlAction:
    """Map policy actions in [-1, 1] to MuJoCo ctrl (muscles + pelvis position)."""

    def __init__(
        self, cfg: TableTennisMixedCtrlActionCfg, env: ManagerBasedRlEnv
    ) -> None:
        import torch

        self.cfg = cfg
        self._env = env
        self.num_envs = env.num_envs
        self.device = env.device
        self._entity = env.scene[cfg.entity_name]
        _reference_model()
        idx = _TT_INDEX
        assert idx is not None
        self._nu = int(idx["nu"])
        self._muscle_mask = torch.as_tensor(idx["muscle_act_mask"], device=self.device)
        lo = torch.as_tensor(
            idx["actuator_ctrlrange"][:, 0], dtype=torch.float32, device=self.device
        )
        hi = torch.as_tensor(
            idx["actuator_ctrlrange"][:, 1], dtype=torch.float32, device=self.device
        )
        self._mid = 0.5 * (lo + hi)
        self._half = 0.5 * (hi - lo)
        self._raw_actions = torch.zeros((self.num_envs, self._nu), device=self.device)
        self._processed = torch.zeros_like(self._raw_actions)

    @property
    def action_dim(self) -> int:
        return self._nu

    @property
    def raw_action(self) -> Any:
        return self._raw_actions

    def process_actions(self, actions: Any) -> None:
        import torch

        a = torch.clamp(actions.to(self.device), -1.0, 1.0)
        self._raw_actions[:] = a
        ctrl = torch.zeros_like(a)
        m = self._muscle_mask[None, :].expand_as(a)
        ctrl = torch.where(
            m, sigmoid_muscle_activation(a, torch), self._mid + a * self._half
        )
        self._processed[:] = ctrl

    def apply_actions(self) -> None:
        self._entity.write_ctrl_to_sim(self._processed)

    def reset(self, env_ids: Any | None = None) -> None:
        if env_ids is None:
            self._raw_actions.zero_()
        else:
            self._raw_actions[env_ids] = 0.0


def _make_tt_obs_closure(
    entity_name: str, _tt_cfg: TableTennisCfg
) -> Callable[[Any], Any]:
    def _obs(env: Any) -> Any:
        import torch

        _reference_model()
        idx = _TT_INDEX
        assert idx is not None
        rt = _get_tt_runtime(env)
        sc = _resolve_tt_scene_ids(env, entity_name)
        data = env.scene[entity_name].data.data
        n = int(data.qpos.shape[0])
        device = data.qpos.device
        pelvis_sid = sc["pelvis_site"]
        ball_sid = sc["ball_site"]
        paddle_sid = sc["paddle_site"]
        paddle_bid = sc["paddle_body"]

        # All addressing uses scene-level IDs from sc (resolved from the
        # actual compiled mjlab model, not from the reference model _TT_INDEX).
        jix = torch.as_tensor(sc["myo_joint_idx"], device=device, dtype=torch.long)
        dix = torch.as_tensor(sc["myo_dof_idx"], device=device, dtype=torch.long)
        pelvis_pos = data.site_xpos[:, pelvis_sid, :]
        body_qpos = data.qpos[:, jix]
        body_qvel = data.qvel[:, dix]
        ball_pos = data.site_xpos[:, ball_sid, :]
        adr_p = sc["sensor_pingpong_vel_adr"]
        dim_p = sc["sensor_pingpong_vel_dim"]
        ball_vel = data.sensordata[:, adr_p : adr_p + dim_p]
        paddle_pos = data.site_xpos[:, paddle_sid, :]
        adr_pad = sc["sensor_paddle_vel_adr"]
        dim_pad = sc["sensor_paddle_vel_dim"]
        paddle_vel = data.sensordata[:, adr_pad : adr_pad + dim_pad]
        paddle_ori = data.xquat[:, paddle_bid, :]
        reach_err = paddle_pos - ball_pos

        # Vectorized contact detection: one GPU pass for all envs.
        gpu_t = _get_gpu_tensors(env, sc, device)
        contact_sets, touching_info = _contacts_all_envs_vectorized(
            data, sc["ball_body_id"], gpu_t, n, device
        )
        for i in range(n):
            rt["trajectories"][i].append(contact_sets[i])

        parts = [
            pelvis_pos,
            body_qpos,
            body_qvel,
            ball_pos,
            ball_vel,
            paddle_pos,
            paddle_vel,
            paddle_ori,
            reach_err,
            touching_info,
        ]
        if int(idx["na"]) > 0:
            parts.append(data.act)

        # Keep tensors on-device: avoid the GPU→CPU→GPU round-trip of _unwrap.
        return torch.cat([p.to(dtype=torch.float32) for p in parts], dim=-1)

    return _obs


def _make_tt_reward_closure(
    entity_name: str, _tt_cfg: TableTennisCfg
) -> Callable[[Any], Any]:
    def _rew(env: Any) -> Any:
        import torch

        _reference_model()
        idx = _TT_INDEX
        assert idx is not None
        rt = _get_tt_runtime(env)
        sc = _resolve_tt_scene_ids(env, entity_name)
        data = env.scene[entity_name].data.data
        n = int(data.qpos.shape[0])
        device = data.qpos.device

        ball_sid = sc["ball_site"]
        paddle_sid = sc["paddle_site"]
        grasp_sid = sc["grasp_site"]
        paddle_bid = sc["paddle_body"]

        paddle_pos = data.site_xpos[:, paddle_sid, :]
        ball_pos = data.site_xpos[:, ball_sid, :]
        reach_err = paddle_pos - ball_pos
        palm_pos = data.site_xpos[:, grasp_sid, :]
        palm_err = palm_pos - paddle_pos
        paddle_ori = data.xquat[:, paddle_bid, :]
        init_q = paddle_ori.new_tensor(
            idx["init_paddle_quat"], dtype=torch.float32
        ).expand_as(paddle_ori)
        padde_ori_err = paddle_ori - init_q
        flex_adr = int(sc["flex_qpos_adr"])
        torso_err = torch.abs(data.qpos[:, flex_adr])

        reach_dist = torch.linalg.norm(reach_err, dim=-1)
        palm_dist = torch.linalg.norm(palm_err, dim=-1)
        paddle_quat_err = torch.linalg.norm(padde_ori_err, dim=-1)
        act_mag = (
            torch.linalg.norm(data.act, dim=-1) / float(idx["na"])
            if int(idx["na"]) > 0
            else torch.zeros(n, device=device)
        )

        # Single batched CPU transfer instead of N×5 individual .item() CUDA syncs.
        cpu_block = (
            torch.stack(
                [
                    torch.exp(-1.0 * reach_dist),  # [0] reach_dist reward
                    torch.exp(-5.0 * palm_dist),  # [1] palm_dist reward
                    torch.exp(-5.0 * paddle_quat_err),  # [2] paddle_quat reward
                    torch.exp(-5.0 * torso_err),  # [3] torso_up reward
                    -act_mag,  # [4] act_reg reward
                    ball_pos[:, 2].float(),  # [5] ball z-height
                    data.time.float(),  # [6] sim time
                ],
                dim=0,
            )
            .cpu()
            .numpy()
        )  # shape (7, n) — one CUDA sync for all envs

        rw_rd, rw_pd, rw_pq, rw_tu, rw_ar = (
            cpu_block[0],
            cpu_block[1],
            cpu_block[2],
            cpu_block[3],
            cpu_block[4],
        )
        ball_z_arr = cpu_block[5]
        time_arr = cpu_block[6]

        wt_rd = _TT_RWD_WEIGHTS["reach_dist"]
        wt_pd = _TT_RWD_WEIGHTS["palm_dist"]
        wt_pq = _TT_RWD_WEIGHTS["paddle_quat"]
        wt_tu = _TT_RWD_WEIGHTS["torso_up"]
        wt_ar = _TT_RWD_WEIGHTS["act_reg"]
        wt_sp = _TT_RWD_WEIGHTS["sparse"]
        wt_sv = _TT_RWD_WEIGHTS["solved"]
        wt_dn = _TT_RWD_WEIGHTS["done"]

        dense_np = np.zeros(n, dtype=np.float32)
        for i in range(n):
            traj = rt["trajectories"][i]
            traj_py = [set(s) for s in traj]
            ev = evaluate_pingpong_trajectory(traj_py)
            solved = ev is None
            last = traj[-1] if traj else frozenset()
            sparse = PingpongContactLabels.PADDLE in last
            ball_z = float(ball_z_arr[i])
            t = float(time_arr[i])
            done_for_dense = _dense_channel_done(t, ball_z, solved, traj_py)
            dense_np[i] = (
                wt_rd * float(rw_rd[i])
                + wt_pd * float(rw_pd[i])
                + wt_pq * float(rw_pq[i])
                + wt_tu * float(rw_tu[i])
                + wt_ar * float(rw_ar[i])
                + wt_sp * (1.0 if sparse else 0.0)
                + wt_sv * (1.0 if solved else 0.0)
                + wt_dn * float(done_for_dense)
            )

            if solved:
                rt["cur_rally"][i] += 1
                cr = rt["cur_rally"][i]
                if cr < int(_tt_cfg.rally_count):
                    rt["trajectories"][i] = []
                    data.time[i] = 0.0
                    _relaunch_ball_torch(
                        env,
                        entity_name,
                        torch.tensor([i], device=device, dtype=torch.long),
                        _tt_cfg,
                    )
            else:
                cr = rt["cur_rally"][i]
            rt["last_done"][i] = _episode_terminal(
                t, ball_z, solved, traj_py, int(_tt_cfg.rally_count), cr
            )

        return torch.from_numpy(dense_np).to(device=device, dtype=torch.float32)

    return _rew


def _dense_channel_done(
    t: float,
    ball_z: float,
    solved: bool,
    traj_py: list[set[PingpongContactLabels]],
) -> float:
    """Match CPU ``_get_done`` contribution to ``rwd_dict['done']`` (before rally override)."""
    if t > _MAX_TIME:
        return 1.0
    if ball_z < 0.3:
        return 1.0
    if solved:
        return 1.0
    ev2 = evaluate_pingpong_trajectory(traj_py)
    if ev2 in (
        ContactTrajIssue.OWN_HALF,
        ContactTrajIssue.NO_PADDLE,
        ContactTrajIssue.DOUBLE_TOUCH,
    ):
        return 1.0
    return 0.0


def _episode_terminal(
    t: float,
    ball_z: float,
    solved: bool,
    traj_py: list[set[PingpongContactLabels]],
    rally_count: int,
    cur_rally_after_increment_if_solved_else_current: int,
) -> bool:
    """Episode termination flag after CPU-style rally bookkeeping for the current step."""
    if t > _MAX_TIME:
        return True
    if ball_z < 0.3:
        return True
    if solved:
        return cur_rally_after_increment_if_solved_else_current >= rally_count
    ev2 = evaluate_pingpong_trajectory(traj_py)
    return ev2 in (
        ContactTrajIssue.OWN_HALF,
        ContactTrajIssue.NO_PADDLE,
        ContactTrajIssue.DOUBLE_TOUCH,
    )


def _relaunch_ball_torch(
    env: Any,
    entity_name: str,
    env_ids: Any,
    tt_cfg: TableTennisCfg,
) -> None:
    """Reset ball state for env_ids (subset) after an intermediate rally."""
    import torch

    _reference_model()
    idx = _TT_INDEX
    assert idx is not None
    assert _REF_QPOS_KEY0 is not None and _REF_QVEL_INIT is not None
    ball_ent = env.scene[_TT_BALL_ENTITY_NAME]
    device = env.device
    n = len(env_ids) if hasattr(env_ids, "__len__") else int(env_ids.shape[0])

    qadr = idx["ball_qpos_adr"]
    vadr = idx["ball_dof_adr"]
    q0 = torch.as_tensor(_REF_QPOS_KEY0, dtype=torch.float32, device=device)
    v0 = torch.as_tensor(_REF_QVEL_INIT, dtype=torch.float32, device=device)

    # root_state: pos(3) + quat(4) + linvel(3) + angvel(3)
    pos_init = q0[qadr : qadr + 3].expand(n, -1).clone()
    quat_init = q0[qadr + 3 : qadr + 7].expand(n, -1).clone()
    linvel_init = v0[vadr : vadr + 3].expand(n, -1).clone()
    angvel_init = v0[vadr + 3 : vadr + 6].expand(n, -1).clone()

    if tt_cfg.ball_xyz_range is not None:
        low = torch.tensor(
            tt_cfg.ball_xyz_range["low"], device=device, dtype=torch.float32
        )
        high = torch.tensor(
            tt_cfg.ball_xyz_range["high"], device=device, dtype=torch.float32
        )
        pos_init = torch.rand(n, 3, device=device) * (high - low) + low
    if tt_cfg.ball_qvel:
        for row in range(n):
            v_lo, v_hi = _cal_ball_qvel_torch(pos_init[row])
            linvel_init[row] = torch.rand(3, device=device) * (v_hi - v_lo) + v_lo

    root_state = torch.cat([pos_init, quat_init, linvel_init, angvel_init], dim=-1)
    ball_ent.write_root_state_to_sim(root_state, env_ids=env_ids)


def _make_tt_term_closure(
    entity_name: str, _tt_cfg: TableTennisCfg
) -> Callable[[Any], Any]:
    def _term(env: Any) -> Any:
        import torch

        rt = _get_tt_runtime(env)
        device = env.scene[entity_name].data.data.qpos.device
        return torch.tensor(rt["last_done"], dtype=torch.bool, device=device)

    return _term


def _make_tt_reset_event(
    entity_name: str, tt_cfg: TableTennisCfg
) -> Callable[[Any, Any], None]:
    def _fn(env: Any, env_ids: Any) -> None:
        import torch

        _reference_model()
        idx = _TT_INDEX
        assert idx is not None
        assert _REF_QPOS_KEY0 is not None and _REF_QVEL_INIT is not None
        # sc is resolved lazily on first call (needs compiled mjlab model).
        sc = _resolve_tt_scene_ids(env, entity_name)
        rt = _get_tt_runtime(env)
        arm_ent = env.scene[entity_name]
        ball_ent = env.scene[_TT_BALL_ENTITY_NAME]
        device = env.device
        env_ids_long = torch.as_tensor(env_ids, device=device, dtype=torch.long)
        n_reset = int(env_ids_long.shape[0])

        for e in env_ids_long.tolist():
            rt["trajectories"][int(e)] = []
            rt["cur_rally"][int(e)] = 0
            rt["last_done"][int(e)] = False

        # --- paddle mass DR (AP-8 known exception; TODO: dr.body_mass once supported) ---
        if tt_cfg.paddle_mass_range is not None:
            lo, hi = tt_cfg.paddle_mass_range
            mass = lo + torch.rand(n_reset, device=device) * (hi - lo)
            bid = sc["paddle_body"]
            try:
                arm_ent.data.model.body_mass[env_ids_long, bid] = mass
            except (AttributeError, TypeError) as e:
                logger.debug("paddle mass DR skipped (AP-8): %s", e)

        # --- ball friction DR (AP-8 known exception; TODO: dr.geom_friction once supported) ---
        if tt_cfg.ball_friction_range is not None:
            low = torch.tensor(
                tt_cfg.ball_friction_range["low"], device=device, dtype=torch.float32
            )
            high = torch.tensor(
                tt_cfg.ball_friction_range["high"], device=device, dtype=torch.float32
            )
            fr = torch.rand(n_reset, 3, device=device) * (high - low) + low
            try:
                ball_ent.data.model.geom_friction[env_ids_long, 0, :3] = fr
            except (AttributeError, TypeError) as e:
                logger.debug("ball friction DR skipped (AP-8): %s", e)

        # --- arm joint reset via entity.write_joint_state_to_sim ---
        # joint_q_adr has one entry per qpos slot (not per joint) for non-free
        # joints.  write_joint_state_to_sim takes indices into joint_q_adr.
        # Build the initial positions from the reference keyframe by matching
        # each joint's scene-level qpos address to its reference-model value.
        ref_model = _reference_model()
        n_qpos_slots = int(arm_ent.indexing.joint_q_adr.shape[0])
        kf_pos = torch.zeros(n_reset, n_qpos_slots, device=device)
        for jnt_spec in arm_ent.spec.joints:
            jname = jnt_spec.name
            ref_jname = jname.split("/")[-1]
            try:
                ref_jnt = ref_model.joint(ref_jname)
                scene_jnt = env.sim.mj_model.joint(jname)
                scene_qadr = int(scene_jnt.qposadr[0])
                ref_kf_val = float(_REF_QPOS_KEY0[int(ref_jnt.qposadr[0])])
                matches = (arm_ent.indexing.joint_q_adr == scene_qadr).nonzero(
                    as_tuple=True
                )[0]
                if matches.numel() > 0:
                    kf_pos[:, int(matches[0])] = ref_kf_val
            except (KeyError, ValueError) as e:
                logger.debug("joint %s not in reference model: %s", ref_jname, e)
        kf_vel = torch.zeros_like(kf_pos)

        if tt_cfg.qpos_noise_range is not None:
            jnt_range = torch.as_tensor(
                sc["jnt_range"], device=device, dtype=kf_pos.dtype
            )
            span = jnt_range[:, 1] - jnt_range[:, 0]
            nj_loop = int(sc["njnt_arm"])
            qr = tt_cfg.qpos_noise_range
            lo_np = np.asarray(qr.get("low", 0.0), dtype=np.float64).ravel()
            hi_np = np.asarray(qr.get("high", 1.0), dtype=np.float64).ravel()
            lo_np = lo_np[0] * np.ones(nj_loop) if lo_np.size == 1 else lo_np[:nj_loop]
            hi_np = hi_np[0] * np.ones(nj_loop) if hi_np.size == 1 else hi_np[:nj_loop]
            lo_t = torch.as_tensor(lo_np, device=device, dtype=kf_pos.dtype).view(1, -1)
            hi_t = torch.as_tensor(hi_np, device=device, dtype=kf_pos.dtype).view(1, -1)
            u = torch.rand(n_reset, nj_loop, device=device, dtype=kf_pos.dtype)
            noise = lo_t + u * (hi_t - lo_t)
            for j in range(nj_loop):
                if j < kf_pos.shape[1]:
                    kf_pos[:, j] = torch.clamp(
                        kf_pos[:, j] + noise[:, j] * span[j],
                        min=float(jnt_range[j, 0]),
                        max=float(jnt_range[j, 1]),
                    )

        all_slot_ids = torch.arange(n_qpos_slots, device=device, dtype=torch.int)
        arm_ent.write_joint_state_to_sim(
            kf_pos, kf_vel, joint_ids=all_slot_ids, env_ids=env_ids_long
        )

        # --- ball reset via Entity API (no raw Warp writes) ---
        ref_qadr = int(idx["ball_qpos_adr"])
        ref_vadr = int(idx["ball_dof_adr"])
        pos_init = (
            torch.as_tensor(
                _REF_QPOS_KEY0[ref_qadr : ref_qadr + 3],
                device=device,
                dtype=torch.float32,
            )
            .expand(n_reset, -1)
            .clone()
        )
        quat_init = (
            torch.as_tensor(
                _REF_QPOS_KEY0[ref_qadr + 3 : ref_qadr + 7],
                device=device,
                dtype=torch.float32,
            )
            .expand(n_reset, -1)
            .clone()
        )
        linvel_init = (
            torch.as_tensor(
                _REF_QVEL_INIT[ref_vadr : ref_vadr + 3],
                device=device,
                dtype=torch.float32,
            )
            .expand(n_reset, -1)
            .clone()
        )
        angvel_init = (
            torch.as_tensor(
                _REF_QVEL_INIT[ref_vadr + 3 : ref_vadr + 6],
                device=device,
                dtype=torch.float32,
            )
            .expand(n_reset, -1)
            .clone()
        )

        if tt_cfg.ball_xyz_range is not None:
            low = torch.tensor(
                tt_cfg.ball_xyz_range["low"], device=device, dtype=torch.float32
            )
            high = torch.tensor(
                tt_cfg.ball_xyz_range["high"], device=device, dtype=torch.float32
            )
            pos_init = torch.rand(n_reset, 3, device=device) * (high - low) + low

        if tt_cfg.ball_qvel:
            for row in range(n_reset):
                v_lo, v_hi = _cal_ball_qvel_torch(pos_init[row])
                linvel_init[row] = torch.rand(3, device=device) * (v_hi - v_lo) + v_lo

        ball_root_state = torch.cat(
            [pos_init, quat_init, linvel_init, angvel_init], dim=-1
        )
        ball_ent.write_root_state_to_sim(ball_root_state, env_ids=env_ids_long)

    return _fn


def _table_tennis_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    # Match register_mjlab_tasks: env observation group is ``policy`` (flat 417-d).
    _policy_obs_groups: dict[str, tuple[str, ...]] = {
        "actor": ("policy",),
        "critic": ("policy",),
    }
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(256, 128, 64),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(256, 128, 64),
            activation="elu",
            obs_normalization=True,
            distribution_cfg=None,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=4,
            num_mini_batches=4,
            learning_rate=3e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="myo_table_tennis",
        save_interval=100,
        num_steps_per_env=48,
        max_iterations=500,
        obs_groups=dict(_policy_obs_groups),
    )


def make_table_tennis_mjlab_env_cfg(tt_cfg: TableTennisCfg) -> ManagerBasedRlEnvCfg:
    """Build mjlab ``ManagerBasedRlEnvCfg`` for TableTennis (vectorised)."""
    _reference_model()
    tendon_names, pos_names = _tt_actuator_xml_groups()
    articulation = EntityArticulationInfoCfg(
        actuators=(
            _XmlActuatorCfg(
                target_names_expr=tendon_names,
                transmission_type=TransmissionType.TENDON,
            ),
            _XmlActuatorCfg(
                target_names_expr=pos_names,
                transmission_type=TransmissionType.JOINT,
            ),
        ),
    )

    entity_cfg = EntityCfg(
        spec_fn=_table_tennis_spec_fn,
        articulation=articulation,
    )
    ball_entity_cfg = EntityCfg(spec_fn=_pingpong_spec_fn)
    scene_cfg = SceneCfg(
        num_envs=int(tt_cfg.num_envs),
        entities={
            _TT_ENTITY_NAME: entity_cfg,
            _TT_BALL_ENTITY_NAME: ball_entity_cfg,
        },
        # Re-add the two velocimeters that cross entity trees (paddle site lives
        # in table_tennis_robot; pingpong site lives in pingpong).  They are
        # stripped from both entity specs and re-added here after attachment so
        # MuJoCo can resolve the prefixed site names in the combined spec.
        spec_fn=_tt_scene_spec_fn(_TT_ENTITY_NAME, _TT_BALL_ENTITY_NAME),
    )

    decimation = max(1, int(round(tt_cfg.ctrl_dt / tt_cfg.sim_dt)))
    episode_length_s = float(tt_cfg.max_episode_steps) * float(tt_cfg.ctrl_dt)

    observations = {
        "policy": ObservationGroupCfg(
            terms={
                "table_tennis_vec": ObservationTermCfg(
                    func=_make_tt_obs_closure(_TT_ENTITY_NAME, tt_cfg),
                ),
            },
        ),
    }
    actions = {"ctrl": TableTennisMixedCtrlActionCfg(entity_name=_TT_ENTITY_NAME)}
    terminations = {
        "time_out": TerminationTermCfg(
            func=mdp_terminations.time_out,
            time_out=True,
        ),
        "task_done": TerminationTermCfg(
            func=_make_tt_term_closure(_TT_ENTITY_NAME, tt_cfg),
            time_out=False,
        ),
    }
    rewards = {
        "dense": RewardTermCfg(
            func=_make_tt_reward_closure(_TT_ENTITY_NAME, tt_cfg),
            weight=1.0,
        ),
    }
    events = {
        "tt_reset": EventTermCfg(
            func=_make_tt_reset_event(_TT_ENTITY_NAME, tt_cfg),
            mode="reset",
        ),
    }

    return ManagerBasedRlEnvCfg(
        scene=scene_cfg,
        decimation=decimation,
        episode_length_s=episode_length_s,
        observations=observations,
        actions=actions,
        terminations=terminations,
        rewards=rewards,
        events=events,
        sim=SimulationCfg(
            mujoco=MujocoCfg(timestep=float(tt_cfg.sim_dt), ccd_iterations=500),
            # The myo_sim-native torso+both-arms+legs body has more
            # equality-constraint rows (nefc) than the legacy single-arm
            # chain — 512 overflowed at nefc=1071; use 1536 for headroom.
            njmax=1536,
            nconmax=1024,
        ),
    )


def register_table_tennis_mjlab_tasks() -> None:
    """Register ``myoChallengeTableTennisP{0,1,2}-v0`` with mjlab (idempotent)."""
    try:
        _reference_model()
    except Exception as exc:
        logger = logging.getLogger(__name__)
        if isinstance(exc, FileNotFoundError) or "Error opening file" in str(exc):
            logger.info("mjlab: skipping optional table tennis registration: %s", exc)
        else:
            logger.warning(
                "mjlab: skipping optional table tennis registration: %s", exc
            )
        return

    rl_cfg = _table_tennis_ppo_runner_cfg()
    tasks: tuple[tuple[str, TableTennisCfg], ...] = (
        ("myoChallengeTableTennisP0-v0", TableTennisCfg.p0()),
        ("myoChallengeTableTennisP1-v0", TableTennisCfg.p1()),
        ("myoChallengeTableTennisP2-v0", TableTennisCfg.p2()),
    )
    for task_id, cfg in tasks:
        try:
            env_cfg = make_table_tennis_mjlab_env_cfg(cfg)
            play_cfg = make_table_tennis_mjlab_env_cfg(cfg)
            register_mjlab_task(
                task_id=task_id,
                env_cfg=env_cfg,
                play_env_cfg=play_cfg,
                rl_cfg=rl_cfg,
                runner_cls=None,
            )
        except ValueError:
            pass
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Table tennis mjlab: skip %s: %s", task_id, exc
            )
