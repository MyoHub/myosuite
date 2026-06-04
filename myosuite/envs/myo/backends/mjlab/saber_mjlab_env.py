from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any
from collections.abc import Callable

import numpy as np
import torch
from mjlab.envs.mdp import Entity
from mjlab.managers import ManagerTermBase
from mjlab.managers.event_manager import requires_model_fields

from myosuite.envs.myo.backends.mjlab.mjlab_env_base import MjlabEnvAccessor
from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
    _resolve_mimic_mjlab_ids,
)
from myosuite.envs.modular_env import (
    _LEFT_SABER_GRASP_OFFSET,
    _RIGHT_SABER_GRASP_OFFSET,
    _TABLETENNIS_GRASP_SEED,
)
from myosuite.envs.myo.tasks.challenge.saber.saber_target_pool import (
    SABER_POOL_HEALTH_BAD_CUT_DELTA,
    SABER_POOL_HEALTH_HIT_DELTA,
    SABER_POOL_HEALTH_MISS_DELTA,
    SABER_POOL_HEALTH_OBSTACLE_DELTA,
    SABER_POOL_HIT_AREA_NONE,
    SaberTargetPoolConfig,
    face_surface_distance as compute_face_surface_distance,
    graveyard_offset_xy,
    plan_target_spawn,
    pool_reset_torch,
    slicing_accuracy as compute_slicing_accuracy,
    timing_accuracy as compute_timing_accuracy,
)
from myosuite.terms.base_obs import (
    health_status_obs as shared_health_status_obs,
    joint_pos_obs as shared_joint_pos_obs,
    joint_vel_obs as shared_joint_vel_obs,
    muscle_act_obs as shared_muscle_act_obs,
    saber_target_pool_obs as shared_saber_target_pool_obs,
    time_obs as shared_time_obs,
)

if TYPE_CHECKING:
    from myosuite.core.trajectory_io import MotionClip

_SABER_ENTITY_NAME = "saber_p0_robot"
_LEFT_SABER_ENTITY_NAME = "left_saber"
_RIGHT_SABER_ENTITY_NAME = "right_saber"
_TARGET_ENTITY_PREFIX = "saber_target_entity_"


def _wp_tensor(array: Any, device: str) -> torch.Tensor:
    return torch.as_tensor(array, device=device)


def _get_saber_entity(env: Any) -> Entity:
    return env.scene[_SABER_ENTITY_NAME]


def _target_entity_name(index: int) -> str:
    return f"{_TARGET_ENTITY_PREFIX}{int(index)}"


def _saber_keyframe_reset_event(entity_name: str, mj_model: Any) -> Any:
    # Store (joint_name, seed_value) for hinge joints — looked up via Entity API at reset.
    grasp_seed_joints: list[tuple[str, float]] = []
    for stem, value in _TABLETENNIS_GRASP_SEED.items():
        for suffix in ("_r", "_l"):
            jname = f"{stem}{suffix}"
            try:
                mj_model.joint(jname)
            except KeyError:
                continue
            grasp_seed_joints.append((jname, float(value)))
    # Store body/site NAMES (not IDs) so that the runtime lookup uses the full
    # combined-scene model — the standalone robot_mj_model body IDs can differ
    # from the global scene IDs once all entities are assembled together.
    split_saber_mappings: list[tuple[str, str, str, Any]] = []
    for saber_entity_name, site_name, hand_body_name, site_local_offset in (
        (
            _LEFT_SABER_ENTITY_NAME,
            "S_grasp_left",
            "lunate_l",
            _LEFT_SABER_GRASP_OFFSET,
        ),
        (
            _RIGHT_SABER_ENTITY_NAME,
            "S_grasp",
            "lunate_r",
            _RIGHT_SABER_GRASP_OFFSET,
        ),
    ):
        try:
            mj_model.site(site_name)
            mj_model.body(hand_body_name)
        except KeyError:
            continue
        split_saber_mappings.append(
            (
                saber_entity_name,
                site_name,
                hand_body_name,
                np.asarray(site_local_offset, dtype=np.float64),
            )
        )
    del mj_model
    mount_quat = np.array(
        [0.7071067811865476, 0.0, 0.7071067811865476, 0.0], dtype=np.float64
    )
    # Cache (body_id, site_id) for each entry, resolved lazily on first call
    # using the global scene model so that entity-prefixed names map correctly.
    _global_ids_cache: list[tuple[int, int]] = []

    def _fn(env: Any, env_ids: Any) -> None:
        env_ids_t = _normalize_env_ids(env, env_ids)
        if len(env_ids_t) == 0:
            return
        entity = env.scene[entity_name]
        data = entity.data.data

        # --- grasp seed: hinge joints → Entity API (no Warp direct write needed) ---
        if grasp_seed_joints:
            joint_slot_lookup: dict[str, int] = {}
            for slot_index, scene_qadr in enumerate(
                entity.data.indexing.joint_q_adr.tolist()
            ):
                for joint_spec in entity.spec.joints:
                    try:
                        spec_joint = env.sim.mj_model.joint(joint_spec.name)
                    except KeyError:
                        continue
                    if int(spec_joint.qposadr[0]) != int(scene_qadr):
                        continue
                    joint_slot_lookup[joint_spec.name.split("/")[-1]] = int(slot_index)
                    break
            ordered_slots: list[int] = []
            ordered_values: list[float] = []
            for joint_name, value in grasp_seed_joints:
                slot_index = joint_slot_lookup.get(joint_name)
                if slot_index is None:
                    continue
                ordered_slots.append(slot_index)
                ordered_values.append(float(value))
            if ordered_slots:
                jnt_ids_t = torch.tensor(
                    ordered_slots, device=env.device, dtype=torch.long
                )
                pos_t = (
                    torch.tensor(ordered_values, device=env.device, dtype=torch.float32)
                    .unsqueeze(0)
                    .expand(len(env_ids_t), -1)
                )
                entity.write_joint_position_to_sim(
                    pos_t, joint_ids=jnt_ids_t, env_ids=env_ids_t
                )
        env.sim.forward()

        # Resolve global body/site IDs once, using the full combined-scene model.
        if not _global_ids_cache:
            scene_model = env.sim.mj_model
            prefix = entity_name + "/"
            for _, site_name, body_name, _ in split_saber_mappings:
                try:
                    bid = int(scene_model.body(prefix + body_name).id)
                    sid = int(scene_model.site(prefix + site_name).id)
                except KeyError:
                    bid = int(scene_model.body(body_name).id)
                    sid = int(scene_model.site(site_name).id)
                _global_ids_cache.append((bid, sid))

        mount_quat_t = torch.as_tensor(
            mount_quat, device=env.device, dtype=torch.float32
        )
        # Split saber entities own their freejoints, so write the computed grasp
        # poses through the root-state API instead of robot qpos addresses.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        env_origins = _env_origins_tensor(env, torch.float32)
        zero_vel = torch.zeros(
            (len(env_ids_t), 6), device=env.device, dtype=torch.float32
        )
        for (
            saber_entity_name,
            _site_name,
            _body_name,
            site_local_offset,
        ), (body_id, site_id) in zip(split_saber_mappings, _global_ids_cache):
            hand_quat = data.xquat[env_ids_t, body_id].to(torch.float32)
            quat = torch.stack(
                (
                    hand_quat[:, 0] * mount_quat_t[0]
                    - hand_quat[:, 1] * mount_quat_t[1]
                    - hand_quat[:, 2] * mount_quat_t[2]
                    - hand_quat[:, 3] * mount_quat_t[3],
                    hand_quat[:, 0] * mount_quat_t[1]
                    + hand_quat[:, 1] * mount_quat_t[0]
                    + hand_quat[:, 2] * mount_quat_t[3]
                    - hand_quat[:, 3] * mount_quat_t[2],
                    hand_quat[:, 0] * mount_quat_t[2]
                    - hand_quat[:, 1] * mount_quat_t[3]
                    + hand_quat[:, 2] * mount_quat_t[0]
                    + hand_quat[:, 3] * mount_quat_t[1],
                    hand_quat[:, 0] * mount_quat_t[3]
                    + hand_quat[:, 1] * mount_quat_t[2]
                    - hand_quat[:, 2] * mount_quat_t[1]
                    + hand_quat[:, 3] * mount_quat_t[0],
                ),
                dim=1,
            )
            quat = quat / torch.clamp(
                torch.linalg.norm(quat, dim=-1, keepdim=True), min=1e-12
            )
            site_rot = (
                data.site_xmat[env_ids_t, site_id]
                .to(torch.float32)
                .reshape(len(env_ids_t), 3, 3)
            )
            site_pos = data.site_xpos[env_ids_t, site_id].to(torch.float32)
            offset_t = torch.as_tensor(
                site_local_offset, device=env.device, dtype=torch.float32
            )
            pos = site_pos + torch.matmul(site_rot, offset_t.reshape(1, 3, 1)).squeeze(
                -1
            )
            state = torch.cat((pos + env_origins[env_ids_t], quat, zero_vel), dim=1)
            env.scene[saber_entity_name].write_root_state_to_sim(
                state, env_ids=env_ids_t
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        env.sim.forward()

    return _fn


try:
    from mjlab.envs.mdp.events import resolve_env_ids as _normalize_env_ids
except ImportError:
    # mjlab<1.4: resolve_env_ids not yet exported; remove fallback on upgrade
    def _normalize_env_ids(env: Any, env_ids: Any) -> torch.Tensor:  # type: ignore[misc]
        if env_ids is None:
            return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        if isinstance(env_ids, slice):
            return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[
                env_ids
            ]
        return torch.as_tensor(env_ids, device=env.device, dtype=torch.long).reshape(-1)


def _sync_saber_visuals_to_mj_model(env: Any) -> None:
    geom_rgba = env.sim.model.geom_rgba
    if len(geom_rgba.shape) == 3:
        env_idx = int(getattr(env.cfg.viewer, "env_idx", 0))
        env_idx = max(0, min(env_idx, env.num_envs - 1))
        env.sim.mj_model.geom_rgba[:] = geom_rgba[env_idx].detach().cpu().numpy()
    else:
        env.sim.mj_model.geom_rgba[:] = geom_rgba.detach().cpu().numpy()


def _update_saber_target_pool_visuals(env: Any, env_ids: Any = None) -> None:
    logic = getattr(env, "_saber_logic", None)
    if logic is None:
        if (
            not hasattr(env, "_target_pool_geom_ids")
            or not hasattr(env, "_target_pool_marker_geom_ids")
            or not hasattr(env, "_pool_active")
        ):
            return
        target_geom_ids = env._target_pool_geom_ids
        marker_geom_ids = env._target_pool_marker_geom_ids
        pool_active = env._pool_active
        pool_preferred_hand = env._pool_preferred_hand
        pool_preferred_hit_area = env._pool_preferred_hit_area
        pool_is_obstacle = env._pool_is_obstacle
    else:
        target_geom_ids = logic.target_pool_geom_ids
        marker_geom_ids = logic.target_pool_marker_geom_ids
        pool_active = logic.pool_active
        pool_preferred_hand = logic.pool_preferred_hand
        pool_preferred_hit_area = logic.pool_preferred_hit_area
        pool_is_obstacle = logic.pool_is_obstacle

    env_ids_t = _normalize_env_ids(env, env_ids)
    if len(env_ids_t) == 0:
        return

    geom_rgba = env.sim.model.geom_rgba
    cube_inactive = torch.tensor(
        [0.55, 0.55, 0.55, 0.85], device=env.device, dtype=geom_rgba.dtype
    )
    cube_active = torch.tensor(
        [0.80, 0.80, 0.80, 1.0], device=env.device, dtype=geom_rgba.dtype
    )
    obstacle_active = torch.tensor(
        [0.0, 0.0, 0.0, 1.0], device=env.device, dtype=geom_rgba.dtype
    )
    left_marker = torch.tensor(
        [0.05, 0.25, 1.0, 1.0], device=env.device, dtype=geom_rgba.dtype
    )
    right_marker = torch.tensor(
        [1.0, 0.10, 0.10, 1.0], device=env.device, dtype=geom_rgba.dtype
    )
    clear = torch.tensor([0.0, 0.0, 0.0, 0.0], device=env.device, dtype=geom_rgba.dtype)

    if len(geom_rgba.shape) == 2:
        active = pool_active[0]
        preferred = pool_preferred_hand[0]
        hit_face = pool_preferred_hit_area[0]
        is_obstacle = pool_is_obstacle[0]
        geom_rgba[target_geom_ids] = torch.where(
            active.unsqueeze(-1),
            torch.where(is_obstacle.unsqueeze(-1), obstacle_active, cube_active),
            cube_inactive,
        )
        geom_rgba[marker_geom_ids.reshape(-1)] = clear
        active_targets = torch.nonzero(active & (~is_obstacle), as_tuple=False).squeeze(
            -1
        )
        for target_idx in active_targets.tolist():
            face_idx = int(hit_face[target_idx])
            if face_idx < 0 or face_idx >= int(marker_geom_ids.shape[1]):
                continue
            marker_color = (
                left_marker if int(preferred[target_idx]) == 0 else right_marker
            )
            geom_rgba[int(marker_geom_ids[target_idx, face_idx])] = marker_color
        _sync_saber_visuals_to_mj_model(env)
        return

    active = pool_active[env_ids_t]
    preferred = pool_preferred_hand[env_ids_t]
    hit_face = pool_preferred_hit_area[env_ids_t]
    is_obstacle = pool_is_obstacle[env_ids_t]
    geom_rgba[env_ids_t[:, None], target_geom_ids[None, :]] = torch.where(
        active.unsqueeze(-1),
        torch.where(is_obstacle.unsqueeze(-1), obstacle_active, cube_active),
        cube_inactive,
    )
    flat_marker_ids = marker_geom_ids.reshape(-1)
    geom_rgba[env_ids_t[:, None], flat_marker_ids[None, :]] = clear
    for row_idx, env_idx in enumerate(env_ids_t.tolist()):
        active_targets = torch.nonzero(
            active[row_idx] & (~is_obstacle[row_idx]), as_tuple=False
        ).squeeze(-1)
        for target_idx in active_targets.tolist():
            face_idx = int(hit_face[row_idx, target_idx])
            if face_idx < 0 or face_idx >= int(marker_geom_ids.shape[1]):
                continue
            marker_color = (
                left_marker
                if int(preferred[row_idx, target_idx]) == 0
                else right_marker
            )
            geom_rgba[env_idx, int(marker_geom_ids[target_idx, face_idx])] = (
                marker_color
            )
    _sync_saber_visuals_to_mj_model(env)


class SaberTaskLogicState(ManagerTermBase):
    def __init__(self, cfg: Any, env: Any) -> None:
        super().__init__(env)
        self.cfg = cfg
        self.env = env
        self.env._saber_logic = self  # noqa: SLF001
        self.robot_entity = _get_saber_entity(env)
        self.left_saber_entity = env.scene[_LEFT_SABER_ENTITY_NAME]
        self.right_saber_entity = env.scene[_RIGHT_SABER_ENTITY_NAME]
        self.task_cfg = cfg.params["task_cfg"]
        self.mj_model = cfg.params["mj_model"]
        self._configure_entity_joint_defaults()
        self.target_pool_cfg = self._build_target_pool_cfg()
        self.target_entities = tuple(
            env.scene[_target_entity_name(i)]
            for i in range(self.target_pool_cfg.pool_size)
        )
        self.target_pool_entity_names = tuple(
            _target_entity_name(i) for i in range(self.target_pool_cfg.pool_size)
        )
        target_pool_mocap_ids: list[int] = []
        target_pool_site_ids: list[int] = []
        target_pool_geom_ids: list[int] = []
        target_pool_marker_geom_ids: list[list[int]] = []
        for index, target_entity in enumerate(self.target_entities):
            body_name = f"{self.target_pool_cfg.body_name_prefix}{index}"
            site_name = f"{body_name}{self.target_pool_cfg.site_suffix}"
            geom_name = f"{self.target_pool_cfg.geom_name_prefix}{index}"
            marker_names = (
                f"{geom_name}_marker_front",
                f"{geom_name}_marker_top",
                f"{geom_name}_marker_bottom",
                f"{geom_name}_marker_right",
                f"{geom_name}_marker_left",
            )
            site_local = target_entity.find_sites(site_name)[0][0]
            geom_local = target_entity.find_geoms(geom_name)[0][0]
            marker_local, _ = target_entity.find_geoms(
                marker_names, preserve_order=True
            )
            if target_entity.data.indexing.mocap_id is None:
                raise ValueError(
                    f"Target entity {_target_entity_name(index)!r} is not a mocap entity."
                )
            target_pool_mocap_ids.append(int(target_entity.data.indexing.mocap_id))
            target_pool_site_ids.append(
                int(target_entity.data.indexing.site_ids[site_local])
            )
            target_pool_geom_ids.append(
                int(target_entity.data.indexing.geom_ids[geom_local])
            )
            target_pool_marker_geom_ids.append(
                [
                    int(target_entity.data.indexing.geom_ids[local])
                    for local in marker_local
                ]
            )
        self.target_pool_mocap_ids_np = np.asarray(
            target_pool_mocap_ids, dtype=np.int32
        )
        self.target_pool_site_ids_np = np.asarray(target_pool_site_ids, dtype=np.int32)
        self.target_pool_geom_ids_np = np.asarray(target_pool_geom_ids, dtype=np.int32)
        self.target_pool_marker_geom_ids_np = np.asarray(
            target_pool_marker_geom_ids, dtype=np.int32
        )
        self.target_pool_mocap_ids = torch.as_tensor(
            self.target_pool_mocap_ids_np, device=env.device, dtype=torch.long
        )
        self.target_pool_site_ids = torch.as_tensor(
            self.target_pool_site_ids_np, device=env.device, dtype=torch.long
        )
        self.target_pool_geom_ids = torch.as_tensor(
            self.target_pool_geom_ids_np, device=env.device, dtype=torch.long
        )
        self.target_pool_marker_geom_ids = torch.as_tensor(
            self.target_pool_marker_geom_ids_np, device=env.device, dtype=torch.long
        )
        left_tip_local = self.left_saber_entity.find_sites("left_saber_tip")[0][0]
        right_tip_local = self.right_saber_entity.find_sites("right_saber_tip")[0][0]
        self.left_tip_site_id = int(
            self.left_saber_entity.data.indexing.site_ids[left_tip_local]
        )
        self.right_tip_site_id = int(
            self.right_saber_entity.data.indexing.site_ids[right_tip_local]
        )
        self.left_saber_body_id = int(self.left_saber_entity.data.indexing.root_body_id)
        self.right_saber_body_id = int(
            self.right_saber_entity.data.indexing.root_body_id
        )
        upright_body_name = str(self.task_cfg.saber_cfg.upright_body_name)
        upright_body_local = self.robot_entity.find_bodies(upright_body_name)[0][0]
        self.upright_body_id = int(
            self.robot_entity.data.indexing.body_ids[upright_body_local]
        )
        self.lane_axis = (
            2
            if abs(float(self.target_pool_cfg.advance_direction[1]))
            >= max(
                abs(float(self.target_pool_cfg.advance_direction[0])),
                abs(float(self.target_pool_cfg.advance_direction[2])),
            )
            else 1
        )
        self.lane_values = torch.linspace(
            float(self.target_pool_cfg.spawn_portal_z[0])
            if self.lane_axis == 2
            else float(self.target_pool_cfg.spawn_portal_y[0]),
            float(self.target_pool_cfg.spawn_portal_z[1])
            if self.lane_axis == 2
            else float(self.target_pool_cfg.spawn_portal_y[1]),
            steps=max(1, int(self.target_pool_cfg.along_advance_lane_count)),
            device=env.device,
        )
        self._allocate_target_pool_state()
        self.target_identity_quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=env.device, dtype=torch.float32
        )
        self._seed: int | None = None
        self._spawn_random = tuple(
            np.random.default_rng() for _ in range(int(self.env.num_envs))
        )
        self.last_synced_common_step = -1
        self.publish_target_pool_task_state()
        self.env._saber_task_state = self.task_state  # noqa: SLF001
        _install_saber_seed_hooks(env, self)

    def _configure_entity_joint_defaults(self) -> None:
        entity = self.robot_entity
        joint_names = tuple(entity.joint_names)
        joint_ids, _ = entity.find_joints(joint_names, preserve_order=True)
        joint_qpos_t = torch.zeros(
            (len(joint_ids),),
            device=self.env.device,
            dtype=entity.data.default_joint_pos.dtype,
        )
        joint_qvel_t = torch.zeros(
            (len(joint_ids),),
            device=self.env.device,
            dtype=entity.data.default_joint_vel.dtype,
        )
        for local_id, name in zip(joint_ids, joint_names, strict=True):
            jid = int(self.mj_model.joint(name).id)
            value = float(self.mj_model.qpos0[int(self.mj_model.jnt_qposadr[jid])])
            for suffix in ("_r", "_l"):
                stem = name.removesuffix(suffix)
                if stem != name and stem in _TABLETENNIS_GRASP_SEED:
                    value = float(_TABLETENNIS_GRASP_SEED[stem])
                    break
            joint_qpos_t[int(local_id)] = float(value)
        entity.data.default_joint_pos[:] = joint_qpos_t.unsqueeze(0)
        entity.data.default_joint_vel[:] = joint_qvel_t.unsqueeze(0)

    def _build_target_pool_cfg(self) -> SaberTargetPoolConfig:
        c = self.task_cfg.saber_cfg
        return SaberTargetPoolConfig(
            pool_size=int(c.pool_size),
            graveyard_z=float(c.graveyard_z),
            spawn_interval_steps=int(c.spawn_interval_steps),
            sphere_radius=float(c.sphere_radius),
            kill_plane_y=float(c.kill_plane_y),
            miss_zone_y=float(c.miss_zone_y),
            ideal_hit_y_range=tuple(float(v) for v in c.ideal_hit_y_range),
            max_target_misses=c.max_target_misses,
            spawn_portal_x=tuple(float(v) for v in c.spawn_portal_x),
            spawn_portal_y=tuple(float(v) for v in c.spawn_portal_y),
            spawn_portal_z=tuple(float(v) for v in c.spawn_portal_z),
            advance_direction=tuple(float(v) for v in c.advance_direction),
            speed_mps_range=tuple(float(v) for v in c.saber_target_speed_mps_range),
            recycle_on_body_or_helmet_contact=bool(c.recycle_on_body_or_helmet_contact),
            recycle_contact_geom_substrings=tuple(c.recycle_contact_geom_substrings),
            enforce_preferred_hand=bool(c.enforce_preferred_hand),
            enforce_preferred_hit_area=bool(c.enforce_preferred_hit_area),
            obstacle_fraction=float(c.obstacle_fraction),
            slicing_accuracy_epsilon=float(c.slicing_accuracy_epsilon),
        )

    def _build_recycle_contact_geom_ids(self) -> torch.Tensor:
        """Return scene-global geom IDs for body/helmet recycling, from env.sim.mj_model."""
        import mujoco

        mj_model = self.env.sim.mj_model
        target_ids = {int(g) for g in self.target_pool_geom_ids.tolist()}
        keys = tuple(
            k.lower() for k in self.target_pool_cfg.recycle_contact_geom_substrings
        )
        selected: list[int] = []
        for gid in range(int(mj_model.ngeom)):
            if gid in target_ids:
                continue
            gname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if any(key in gname.lower() for key in keys):
                selected.append(gid)
        if not selected:
            return torch.zeros(0, device=self.env.device, dtype=torch.long)
        return torch.tensor(selected, device=self.env.device, dtype=torch.long)

    def _find_body_contact_slots(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (miss_mask, obstacle_mask) of shape (num_envs, pool_size).

        Both are True where a target slot's geom is in contact with a body/helmet
        geom (from `recycle_contact_geom_substrings`) this step.  Targets flagged
        in miss_mask are NOT obstacles; targets in obstacle_mask ARE obstacles.
        Only active slots are meaningful — callers must AND with `pool_active`.
        """
        n, p = self.env.num_envs, self.target_pool_cfg.pool_size
        miss_mask = torch.zeros((n, p), device=self.env.device, dtype=torch.bool)
        obstacle_mask = torch.zeros((n, p), device=self.env.device, dtype=torch.bool)

        nacon = int(
            _wp_tensor(self.env.sim.wp_data.nacon, self.env.device).sum().item()
        )
        if nacon <= 0:
            return miss_mask, obstacle_mask

        # contact.geom is vec2i → (naconmax, 2), contact.worldid → (naconmax,)
        contact_geom = _wp_tensor(self.env.sim.wp_data.contact.geom, self.env.device)[
            :nacon
        ]  # (nacon, 2)
        contact_world = _wp_tensor(
            self.env.sim.wp_data.contact.worldid, self.env.device
        )[:nacon]  # (nacon,)

        g1 = contact_geom[:, 0]  # (nacon,)
        g2 = contact_geom[:, 1]  # (nacon,)
        recycle_ids = self._recycle_contact_geom_ids  # (R,)
        target_ids = self.target_pool_geom_ids  # (p,)

        # (nacon, p): whether geom1/geom2 matches each target slot
        g1_is_target = g1.unsqueeze(1) == target_ids.unsqueeze(0)  # (nacon, p)
        g2_is_target = g2.unsqueeze(1) == target_ids.unsqueeze(0)  # (nacon, p)
        # (nacon,): whether the other geom is a body/helmet geom
        g2_is_recycle = (g2.unsqueeze(1) == recycle_ids.unsqueeze(0)).any(dim=1)
        g1_is_recycle = (g1.unsqueeze(1) == recycle_ids.unsqueeze(0)).any(dim=1)

        # (nacon, p): this contact involves target slot p and a recycle geom
        match = (g1_is_target & g2_is_recycle.unsqueeze(1)) | (
            g2_is_target & g1_is_recycle.unsqueeze(1)
        )

        if not match.any():
            return miss_mask, obstacle_mask

        # Accumulate per (world, pool_slot): scatter match into (num_envs, p)
        world_clamp = contact_world.clamp(0, n - 1).unsqueeze(1).expand(-1, p)
        accumulated = torch.zeros((n, p), device=self.env.device, dtype=torch.float32)
        accumulated.scatter_add_(0, world_clamp, match.to(torch.float32))
        body_contact = accumulated > 0  # (n, p)

        # Split by obstacle status for currently-active slots
        miss_mask = body_contact & (~self.pool_is_obstacle)
        obstacle_mask = body_contact & self.pool_is_obstacle
        return miss_mask, obstacle_mask

    def _allocate_target_pool_state(self) -> None:
        n, p = self.env.num_envs, self.target_pool_cfg.pool_size
        self.pool_active = torch.zeros((n, p), device=self.env.device, dtype=torch.bool)
        self.pool_vel = torch.zeros(
            (n, p, 3), device=self.env.device, dtype=torch.float32
        )
        self.pool_preferred_hand = torch.zeros(
            (n, p), device=self.env.device, dtype=torch.int64
        )
        self.pool_preferred_hit_area = torch.full(
            (n, p),
            SABER_POOL_HIT_AREA_NONE,
            device=self.env.device,
            dtype=torch.int64,
        )
        self.pool_is_obstacle = torch.zeros(
            (n, p), device=self.env.device, dtype=torch.bool
        )
        self.pool_step_index = torch.zeros(n, device=self.env.device, dtype=torch.long)
        self.pool_total_spawned = torch.zeros(
            n, device=self.env.device, dtype=torch.long
        )
        self.pool_hits = torch.zeros(n, device=self.env.device, dtype=torch.long)
        self.pool_misses = torch.zeros(n, device=self.env.device, dtype=torch.long)
        self.pool_correct_hand_hits = torch.zeros(
            n, device=self.env.device, dtype=torch.long
        )
        self.pool_correct_face_hits = torch.zeros(
            n, device=self.env.device, dtype=torch.long
        )
        self.pool_wrong_hand_hits = torch.zeros(
            n, device=self.env.device, dtype=torch.long
        )
        self.pool_wrong_face_hits = torch.zeros(
            n, device=self.env.device, dtype=torch.long
        )
        self.pool_obstacle_contacts = torch.zeros(
            n, device=self.env.device, dtype=torch.long
        )
        self.pool_spawn_lane_cursor = torch.zeros(
            n, device=self.env.device, dtype=torch.long
        )
        self.pool_spawn_hand_cursor = torch.zeros(
            n, device=self.env.device, dtype=torch.long
        )
        self.pool_spawn_hit_area_cursor = torch.zeros(
            n, device=self.env.device, dtype=torch.long
        )
        self.pool_last_spawn_step = torch.full(
            (n,), -10_000, device=self.env.device, dtype=torch.long
        )
        self.health_status = torch.full(
            (n,), 0.5, device=self.env.device, dtype=torch.float32
        )
        self.keyframe_ref_pos = torch.zeros(
            (n, 2, 3),
            device=self.env.device,
            dtype=torch.float32,
        )
        self.target_pos = torch.zeros(
            (n, p, 3), device=self.env.device, dtype=torch.float32
        )
        self.task_state: dict[str, torch.Tensor] = {}
        self._recycle_contact_geom_ids: torch.Tensor | None = None

    def set_seed(self, seed: int | None) -> None:
        """Reset per-env target-pool RNG streams from a base seed."""

        if seed is None:
            return
        self._seed = int(seed)
        self._spawn_random = tuple(
            np.random.default_rng(self._seed + env_idx)
            for env_idx in range(int(self.env.num_envs))
        )

    def _current_saber_positions(self) -> torch.Tensor:
        left_pos = _entity_local_qpos(self.left_saber_entity)[:, :3]
        right_pos = _entity_local_qpos(self.right_saber_entity)[:, :3]
        return torch.stack((left_pos, right_pos), dim=1)

    def _write_target_slot_poses(
        self, slot_index: int, env_ids: torch.Tensor | None = None
    ) -> None:
        env_ids_t = (
            _normalize_env_ids(self.env, env_ids)
            if env_ids is not None
            else torch.arange(
                self.env.num_envs, device=self.env.device, dtype=torch.long
            )
        )
        if len(env_ids_t) == 0:
            return
        env_origins = _env_origins_tensor(self.env, self.target_pos.dtype)
        pose = torch.cat(
            (
                self.target_pos[env_ids_t, int(slot_index)] + env_origins[env_ids_t],
                self.target_identity_quat.unsqueeze(0).expand(len(env_ids_t), -1),
            ),
            dim=1,
        )
        self.target_entities[int(slot_index)].write_mocap_pose_to_sim(
            pose, env_ids=env_ids_t
        )

    def _capture_keyframe_pose_reference(self, env_ids: torch.Tensor) -> None:
        self.keyframe_ref_pos[env_ids] = self._current_saber_positions()[env_ids]

    def _set_graveyard_positions(self, env_ids: torch.Tensor) -> None:
        for i in range(self.target_pool_cfg.pool_size):
            ox, oy = graveyard_offset_xy(i, self.target_pool_cfg.pool_size)
            self.target_pos[env_ids, i] = torch.tensor(
                [ox, oy, float(self.target_pool_cfg.graveyard_z)],
                device=self.env.device,
                dtype=torch.float32,
            )
            self._write_target_slot_poses(i, env_ids)

    def reset(self, env_ids: Any) -> None:
        env_ids_t = _normalize_env_ids(self.env, env_ids)
        pool_reset_torch(self, env_ids_t)
        self._set_graveyard_positions(env_ids_t)
        self.last_synced_common_step = -1

    def finalize_reset(self, env_ids: Any) -> None:
        env_ids_t = _normalize_env_ids(self.env, env_ids)
        if len(env_ids_t) == 0:
            return
        # Capture the saber reference only after reset events have written the
        # new robot and saber state into the split scene.
        self._capture_keyframe_pose_reference(env_ids_t)
        self.last_synced_common_step = int(self.env.common_step_counter)
        self.publish_target_pool_task_state()
        self.sync_extras()

    def publish_target_pool_task_state(
        self,
        *,
        hit_this_step: torch.Tensor | None = None,
        miss_zone_this_step: torch.Tensor | None = None,
        d_min_closest: torch.Tensor | None = None,
        offset_at_hit: torch.Tensor | None = None,
        hit_y_at_hit: torch.Tensor | None = None,
        hit_y_deviation_at_hit: torch.Tensor | None = None,
        face_surface_distance_at_hit: torch.Tensor | None = None,
        slicing_accuracy_this_step: torch.Tensor | None = None,
        timing_accuracy_this_step: torch.Tensor | None = None,
        correct_hand_this_step: torch.Tensor | None = None,
        correct_face_this_step: torch.Tensor | None = None,
        wrong_hand_this_step: torch.Tensor | None = None,
        wrong_face_this_step: torch.Tensor | None = None,
        obstacle_contact_this_step: torch.Tensor | None = None,
        d_min_obstacle: torch.Tensor | None = None,
        health_depleted_this_step: torch.Tensor | None = None,
    ) -> None:
        n = self.env.num_envs
        zeros = torch.zeros(n, device=self.env.device, dtype=torch.float32)
        falses = torch.zeros(n, device=self.env.device, dtype=torch.bool)
        body_xquat = self.env.scene[_SABER_ENTITY_NAME].data.data.xquat
        upright_posture = _saber_upright_posture_from_xquat(
            body_xquat[:, self.upright_body_id, :]
        )
        current_keyframe_pos = self._current_saber_positions()
        keyframe_pose_error = torch.mean(
            torch.linalg.norm(current_keyframe_pos - self.keyframe_ref_pos, dim=-1),
            dim=1,
        )
        self.task_state = {
            "target_pool_active": self.pool_active.clone(),
            "target_pool_vel": self.pool_vel.clone(),
            "target_pool_is_obstacle": self.pool_is_obstacle.clone(),
            "target_pool_hit_area": self.pool_preferred_hit_area.clone(),
            "target_pool_preferred_hand": self.pool_preferred_hand.clone(),
            "target_pool_hit_this_step": (
                hit_this_step if hit_this_step is not None else falses
            ).clone(),
            "target_pool_offset_at_hit": (
                offset_at_hit if offset_at_hit is not None else zeros
            ).clone(),
            "target_pool_hit_y_at_hit": (
                hit_y_at_hit if hit_y_at_hit is not None else zeros
            ).clone(),
            "target_pool_hit_y_deviation_at_hit": (
                hit_y_deviation_at_hit if hit_y_deviation_at_hit is not None else zeros
            ).clone(),
            "target_pool_face_surface_distance_at_hit": (
                face_surface_distance_at_hit
                if face_surface_distance_at_hit is not None
                else zeros
            ).clone(),
            "target_pool_slicing_accuracy_this_step": (
                slicing_accuracy_this_step
                if slicing_accuracy_this_step is not None
                else zeros
            ).clone(),
            "target_pool_timing_accuracy_this_step": (
                timing_accuracy_this_step
                if timing_accuracy_this_step is not None
                else zeros
            ).clone(),
            "target_pool_d_min": (
                d_min_closest
                if d_min_closest is not None
                else torch.full((n,), 1e6, device=self.env.device)
            ).clone(),
            "target_pool_obstacle_d_min": (
                d_min_obstacle
                if d_min_obstacle is not None
                else torch.full((n,), 1e6, device=self.env.device)
            ).clone(),
            "target_pool_obstacle_contact_this_step": (
                obstacle_contact_this_step
                if obstacle_contact_this_step is not None
                else falses
            ).clone(),
            "target_pool_obstacle_contacts": self.pool_obstacle_contacts.clone(),
            "target_pool_miss_zone": (
                miss_zone_this_step if miss_zone_this_step is not None else falses
            ).clone(),
            "target_pool_hits": self.pool_hits.clone(),
            "target_pool_misses": self.pool_misses.clone(),
            "target_pool_correct_hand_this_step": (
                correct_hand_this_step if correct_hand_this_step is not None else falses
            ).clone(),
            "target_pool_correct_face_this_step": (
                correct_face_this_step if correct_face_this_step is not None else falses
            ).clone(),
            "target_pool_wrong_hand_this_step": (
                wrong_hand_this_step if wrong_hand_this_step is not None else falses
            ).clone(),
            "target_pool_wrong_face_this_step": (
                wrong_face_this_step if wrong_face_this_step is not None else falses
            ).clone(),
            "target_pool_correct_hand_hits": self.pool_correct_hand_hits.clone(),
            "target_pool_correct_face_hits": self.pool_correct_face_hits.clone(),
            "target_pool_wrong_hand_hits": self.pool_wrong_hand_hits.clone(),
            "target_pool_wrong_face_hits": self.pool_wrong_face_hits.clone(),
            "target_pool_total_spawned": self.pool_total_spawned.clone(),
            "health_status": self.health_status.clone(),
            "upright_posture": upright_posture.clone(),
            "saber_keyframe_pose_error": keyframe_pose_error.clone(),
            "target_pool_health_depleted_this_step": (
                health_depleted_this_step
                if health_depleted_this_step is not None
                else falses
            ).clone(),
        }
        self.env._saber_task_state = self.task_state  # noqa: SLF001

    def advance_target_pool(self) -> None:
        self.target_pos += (
            self.pool_vel
            * self.pool_active.unsqueeze(-1).to(torch.float32)
            * float(self.env.step_dt)
        )
        _any_moving_targets = self.pool_active.any(dim=1)
        env_ids = _any_moving_targets.nonzero(as_tuple=False).squeeze(-1)
        for i in range(self.target_pool_cfg.pool_size):
            self._write_target_slot_poses(i, env_ids)
        self.pool_step_index += 1
        min_step_gap = max(
            1,
            int(
                np.ceil(
                    float(self.target_pool_cfg.min_spawn_along_advance_separation)
                    / (
                        max(float(self.target_pool_cfg.speed_mps_range[0]), 1e-6)
                        * float(self.env.step_dt)
                    )
                )
            ),
        )
        inactive = ~self.pool_active
        can_spawn = (
            inactive.any(dim=1)
            & (
                self.pool_step_index % int(self.target_pool_cfg.spawn_interval_steps)
                == 0
            )
            & ((self.pool_step_index - self.pool_last_spawn_step) >= min_step_gap)
        )
        if not torch.any(can_spawn):
            return
        env_ids = can_spawn.nonzero(as_tuple=False).squeeze(-1)

        for env_idx_t in env_ids:
            env_idx = int(env_idx_t.item())
            rng = self._spawn_random[env_idx]
            inactive_ids = torch.nonzero(
                ~self.pool_active[env_idx], as_tuple=False
            ).squeeze(-1)
            if int(inactive_ids.numel()) == 0:
                continue

            active_ids = torch.nonzero(
                self.pool_active[env_idx], as_tuple=False
            ).squeeze(-1)
            active_axis_vals = [
                float(
                    self.target_pos[
                        env_idx, int(active_idx.item()), self.lane_axis
                    ].item()
                )
                for active_idx in active_ids
            ]
            spawn_plan = plan_target_spawn(
                self.target_pool_cfg,
                available_slot_ids=inactive_ids.detach().cpu().numpy(),
                active_axis_vals=active_axis_vals,
                np_random=rng,
                spawn_lane_cursor=int(self.pool_spawn_lane_cursor[env_idx].item()),
                spawn_hand_cursor=int(self.pool_spawn_hand_cursor[env_idx].item()),
                spawn_hit_area_cursor=int(
                    self.pool_spawn_hit_area_cursor[env_idx].item()
                ),
            )
            if spawn_plan is None:
                continue
            slot = int(spawn_plan.slot_index)
            self.pool_spawn_lane_cursor[env_idx] = int(
                spawn_plan.next_spawn_lane_cursor
            )
            self.pool_spawn_hand_cursor[env_idx] = int(
                spawn_plan.next_spawn_hand_cursor
            )
            self.pool_spawn_hit_area_cursor[env_idx] = int(
                spawn_plan.next_spawn_hit_area_cursor
            )

            self.pool_active[env_idx, slot] = True
            self.pool_vel[env_idx, slot] = torch.as_tensor(
                spawn_plan.velocity,
                device=self.env.device,
                dtype=torch.float32,
            )
            self.pool_is_obstacle[env_idx, slot] = bool(spawn_plan.is_obstacle)
            self.pool_preferred_hand[env_idx, slot] = int(spawn_plan.preferred_hand)
            self.pool_preferred_hit_area[env_idx, slot] = int(
                spawn_plan.preferred_hit_area
            )
            self.pool_total_spawned[env_idx] += 1
            self.pool_last_spawn_step[env_idx] = self.pool_step_index[env_idx]
            self.target_pos[env_idx, slot] = torch.as_tensor(
                spawn_plan.spawn_pos,
                device=self.env.device,
                dtype=torch.float32,
            )

        active_slots = torch.nonzero(
            self.pool_active.any(dim=0), as_tuple=False
        ).squeeze(-1)
        for slot in active_slots.tolist():
            active_env_ids = torch.nonzero(
                self.pool_active[:, slot], as_tuple=False
            ).squeeze(-1)
            self._write_target_slot_poses(slot, active_env_ids)

    def ensure_post_step(self) -> None:
        current_step = int(self.env.common_step_counter)
        if self.last_synced_common_step == current_step:
            return
        site_xpos = _wp_tensor(self.env.sim.wp_data.site_xpos, self.env.device)
        env_origins = _env_origins_tensor(self.env, site_xpos.dtype).unsqueeze(1)
        target_pos = self.target_pos
        left_tip = site_xpos[:, self.left_tip_site_id, :].unsqueeze(1) - env_origins
        right_tip = site_xpos[:, self.right_tip_site_id, :].unsqueeze(1) - env_origins
        body_xpos = _wp_tensor(self.env.sim.wp_data.xpos, self.env.device)
        left_base = body_xpos[:, self.left_saber_body_id, :].unsqueeze(1) - env_origins
        right_base = (
            body_xpos[:, self.right_saber_body_id, :].unsqueeze(1) - env_origins
        )
        d_l, closest_l = _segment_distance_and_closest(target_pos, left_base, left_tip)
        d_r, closest_r = _segment_distance_and_closest(
            target_pos, right_base, right_tip
        )
        use_left = d_l <= d_r
        d_min = torch.minimum(d_l, d_r)
        closest = torch.where(use_left.unsqueeze(-1), closest_l, closest_r)
        hit_thresh = float(self.target_pool_cfg.resolved_hit_threshold())
        forward_positive_y = float(self.target_pool_cfg.advance_direction[1]) >= 0.0

        active = self.pool_active
        is_obstacle = self.pool_is_obstacle
        hit_candidate = active & (d_min < hit_thresh)
        target_hit = hit_candidate & (~is_obstacle)
        required_left = self.pool_preferred_hand == 0
        if self.target_pool_cfg.enforce_preferred_hand:
            correct_hand = target_hit & (required_left == use_left)
        else:
            correct_hand = target_hit.clone()
        wrong_hand = target_hit & (~correct_hand)
        face = _impact_face_index(closest - target_pos)
        if self.target_pool_cfg.enforce_preferred_hit_area:
            correct_face = target_hit & (face == self.pool_preferred_hit_area)
        else:
            correct_face = target_hit.clone()
        wrong_face = target_hit & (~correct_face)
        obstacle_contact = hit_candidate & is_obstacle

        target_y = self.target_pos[:, :, 1]
        miss_zone = (
            active
            & (~is_obstacle)
            & (
                (target_y > float(self.target_pool_cfg.miss_zone_y))
                if forward_positive_y
                else (target_y < float(self.target_pool_cfg.miss_zone_y))
            )
        )
        crossed_kill = active & (
            (target_y > float(self.target_pool_cfg.kill_plane_y))
            if forward_positive_y
            else (target_y < float(self.target_pool_cfg.kill_plane_y))
        )
        misses = crossed_kill & (~is_obstacle) & (~target_hit)
        to_recycle = (
            target_hit | obstacle_contact | misses | (crossed_kill & is_obstacle)
        )

        hit_this_step = torch.any(target_hit, dim=1)
        miss_zone_this_step = torch.any(miss_zone, dim=1)
        correct_hand_this_step = torch.any(correct_hand, dim=1)
        correct_face_this_step = torch.any(correct_face, dim=1)
        wrong_hand_this_step = torch.any(wrong_hand, dim=1)
        wrong_face_this_step = torch.any(wrong_face, dim=1)
        obstacle_contact_this_step = torch.any(obstacle_contact, dim=1)
        d_min_closest = torch.where(
            torch.any(active & (~is_obstacle), dim=1),
            torch.min(
                torch.where(
                    active & (~is_obstacle), d_min, torch.full_like(d_min, 1e6)
                ),
                dim=1,
            ).values,
            torch.full((self.env.num_envs,), 1e6, device=self.env.device),
        )
        d_min_obstacle = torch.where(
            torch.any(active & is_obstacle, dim=1),
            torch.min(
                torch.where(active & is_obstacle, d_min, torch.full_like(d_min, 1e6)),
                dim=1,
            ).values,
            torch.full((self.env.num_envs,), 1e6, device=self.env.device),
        )
        hit_index = (
            _first_true_index(target_hit)
            if torch.any(target_hit)
            else torch.zeros(
                self.env.num_envs, device=self.env.device, dtype=torch.long
            )
        )
        batch_ids = torch.arange(self.env.num_envs, device=self.env.device)
        offset_at_hit = torch.where(
            hit_this_step,
            d_min[batch_ids, hit_index],
            torch.zeros(self.env.num_envs, device=self.env.device),
        )
        hit_y_at_hit = torch.where(
            hit_this_step,
            target_pos[batch_ids, hit_index, 1],
            torch.zeros(self.env.num_envs, device=self.env.device),
        )
        ideal_lo = float(self.target_pool_cfg.ideal_hit_y_range[0])
        ideal_hi = float(self.target_pool_cfg.ideal_hit_y_range[1])
        hit_y_deviation = torch.where(
            hit_y_at_hit < ideal_lo,
            torch.full_like(hit_y_at_hit, ideal_lo) - hit_y_at_hit,
            torch.where(
                hit_y_at_hit > ideal_hi,
                hit_y_at_hit - torch.full_like(hit_y_at_hit, ideal_hi),
                torch.zeros_like(hit_y_at_hit),
            ),
        )
        face_surface_distance = torch.zeros(
            self.env.num_envs, device=self.env.device, dtype=torch.float32
        )
        slicing_accuracy = torch.zeros_like(face_surface_distance)
        timing_accuracy = torch.zeros_like(face_surface_distance)
        radius = float(self.target_pool_cfg.sphere_radius)
        epsilon = float(self.target_pool_cfg.slicing_accuracy_epsilon)
        for env_idx in (
            torch.nonzero(hit_this_step, as_tuple=False).squeeze(-1).tolist()
        ):
            pool_idx = int(hit_index[env_idx])
            timing_accuracy[env_idx] = float(
                compute_timing_accuracy(
                    float(hit_y_at_hit[env_idx].item()),
                    self.target_pool_cfg.miss_zone_y,
                    radius,
                    forward_positive_y=forward_positive_y,
                )
            )
            if bool(correct_face[env_idx, pool_idx]):
                distance = float(
                    compute_face_surface_distance(
                        int(face[env_idx, pool_idx].item()),
                        target_pos[env_idx, pool_idx].detach().cpu().numpy(),
                        closest[env_idx, pool_idx].detach().cpu().numpy(),
                        radius,
                    )
                )
                face_surface_distance[env_idx] = distance
                slicing_accuracy[env_idx] = float(
                    compute_slicing_accuracy(distance, radius, epsilon)
                )

        # Body/helmet contact recycling (recycle_on_body_or_helmet_contact).
        # Targets that touch body/helmet geoms are recycled as misses (CPU semantics:
        # saber hits take priority; a target already recycled by saber is not also
        # counted as a body-contact miss).
        if self.target_pool_cfg.recycle_on_body_or_helmet_contact:
            if self._recycle_contact_geom_ids is None:
                self._recycle_contact_geom_ids = self._build_recycle_contact_geom_ids()
            if self._recycle_contact_geom_ids.numel() > 0:
                raw_body_miss, raw_body_obstacle = self._find_body_contact_slots()
                # Exclude slots already processed as saber hits/obstacle contacts.
                body_contact_miss = raw_body_miss & (~target_hit) & (~misses)
                body_contact_obstacle_new = raw_body_obstacle & (~obstacle_contact)
                to_recycle = (
                    to_recycle
                    | (body_contact_miss & active)
                    | (body_contact_obstacle_new & active)
                )
                obstacle_contact = obstacle_contact | body_contact_obstacle_new
                misses = misses | body_contact_miss

        self.pool_hits += target_hit.sum(dim=1).to(torch.long)
        self.pool_misses += misses.sum(dim=1).to(torch.long)
        self.pool_correct_hand_hits += correct_hand.sum(dim=1).to(torch.long)
        self.pool_correct_face_hits += correct_face.sum(dim=1).to(torch.long)
        self.pool_wrong_hand_hits += wrong_hand.sum(dim=1).to(torch.long)
        self.pool_wrong_face_hits += wrong_face.sum(dim=1).to(torch.long)
        self.pool_obstacle_contacts += obstacle_contact.sum(dim=1).to(torch.long)

        prev_health = self.health_status.clone()
        self.health_status = torch.clamp(
            self.health_status
            + target_hit.sum(dim=1).to(torch.float32) * SABER_POOL_HEALTH_HIT_DELTA
            + misses.sum(dim=1).to(torch.float32) * SABER_POOL_HEALTH_MISS_DELTA
            + obstacle_contact.sum(dim=1).to(torch.float32)
            * SABER_POOL_HEALTH_OBSTACLE_DELTA
            + (wrong_hand.sum(dim=1) + wrong_face.sum(dim=1)).to(torch.float32)
            * SABER_POOL_HEALTH_BAD_CUT_DELTA,
            0.0,
            1.0,
        )
        health_depleted_this_step = (prev_health > 0.0) & (self.health_status <= 0.0)

        if torch.any(to_recycle):
            env_idx, pool_idx = torch.nonzero(to_recycle, as_tuple=True)
            self.pool_active[env_idx, pool_idx] = False
            self.pool_vel[env_idx, pool_idx] = 0.0
            self.pool_is_obstacle[env_idx, pool_idx] = False
            self.pool_preferred_hand[env_idx, pool_idx] = 0
            self.pool_preferred_hit_area[env_idx, pool_idx] = SABER_POOL_HIT_AREA_NONE
            for e, p in zip(env_idx.tolist(), pool_idx.tolist(), strict=False):
                ox, oy = graveyard_offset_xy(int(p), self.target_pool_cfg.pool_size)
                self.target_pos[e, p, 0] = ox
                self.target_pos[e, p, 1] = oy
                self.target_pos[e, p, 2] = float(self.target_pool_cfg.graveyard_z)
                self._write_target_slot_poses(
                    p,
                    torch.tensor([e], device=self.env.device, dtype=torch.long),
                )

        self.publish_target_pool_task_state(
            hit_this_step=hit_this_step,
            miss_zone_this_step=miss_zone_this_step,
            d_min_closest=d_min_closest,
            offset_at_hit=offset_at_hit,
            hit_y_at_hit=hit_y_at_hit,
            hit_y_deviation_at_hit=hit_y_deviation,
            face_surface_distance_at_hit=face_surface_distance,
            slicing_accuracy_this_step=slicing_accuracy,
            timing_accuracy_this_step=timing_accuracy,
            correct_hand_this_step=correct_hand_this_step,
            correct_face_this_step=correct_face_this_step,
            wrong_hand_this_step=wrong_hand_this_step,
            wrong_face_this_step=wrong_face_this_step,
            obstacle_contact_this_step=obstacle_contact_this_step,
            d_min_obstacle=d_min_obstacle,
            health_depleted_this_step=health_depleted_this_step,
        )
        self.last_synced_common_step = current_step

    def sync_extras(self) -> None:
        if hasattr(self.env, "extras") and isinstance(self.env.extras, dict):
            self.env.extras["task_state"] = self.task_state

    def __call__(self, env: Any, env_ids: Any, **_: Any) -> None:
        del env_ids, _
        self.ensure_post_step()
        self.sync_extras()
        _update_saber_target_pool_visuals(env, None)


def _install_saber_seed_hooks(env: Any, logic: SaberTaskLogicState) -> None:
    if getattr(env, "_saber_seed_hooks_installed", False):
        return

    original_seed = getattr(env, "seed", None)
    if callable(original_seed):

        @wraps(original_seed)
        def _seed_wrapper(seed: int | None = None, *args: Any, **kwargs: Any) -> Any:
            result = original_seed(seed, *args, **kwargs)
            logic.set_seed(seed)
            return result

        env.seed = _seed_wrapper  # type: ignore[method-assign]

    original_reset = getattr(env, "reset", None)
    if callable(original_reset):

        @wraps(original_reset)
        def _reset_wrapper(*args: Any, **kwargs: Any) -> Any:
            seed = kwargs.get("seed")
            logic.set_seed(seed)
            result = original_reset(*args, **kwargs)
            logic.set_seed(seed)
            return result

        env.reset = _reset_wrapper  # type: ignore[method-assign]

    env._saber_seed_hooks_installed = True  # noqa: SLF001


def _get_saber_logic(env: Any) -> SaberTaskLogicState:
    logic = getattr(env, "_saber_logic", None)
    if logic is None:
        raise RuntimeError(
            "SaberTaskLogicState was not initialized by the event manager."
        )
    return logic


def _ensure_saber_task_state(env: Any) -> dict[str, torch.Tensor]:
    logic = _get_saber_logic(env)
    logic.ensure_post_step()
    return logic.task_state


def _saber_startup_event(
    env: Any,
    env_ids: Any,
    keyframe_reset_fn: Any | None = None,
    **_: Any,
) -> None:
    if keyframe_reset_fn is not None:
        keyframe_reset_fn(env, env_ids)
    _get_saber_logic(env).sync_extras()


@requires_model_fields("geom_rgba")
def _saber_reset_event(env: Any, env_ids: Any, **_: Any) -> None:
    keyframe_reset_fn = _.get("keyframe_reset_fn")
    if keyframe_reset_fn is not None:
        keyframe_reset_fn(env, env_ids)
    _get_saber_logic(env).finalize_reset(env_ids)
    _update_saber_target_pool_visuals(env, env_ids)


def _saber_mimic_rsi_event(
    clip: MotionClip | tuple[MotionClip, ...],
    ctrl_dt: float,
) -> Callable[[Any, Any], None]:
    """Reference-state initialization for the split saber scene."""

    def _fn(env: Any, env_ids: Any) -> None:
        cache = _resolve_mimic_mjlab_ids(
            env, _SABER_ENTITY_NAME, "saber", clip, ctrl_dt
        )
        clip_source = cache.get("clip_source")
        if clip_source is None:
            return
        has_qpos = getattr(clip_source, "_qpos_tensor", None) is not None or (
            getattr(clip_source, "_qpos_tensors", None) is not None
        )
        if not has_qpos:
            return

        logic = _get_saber_logic(env)
        robot_data = logic.robot_entity.data.data
        device = robot_data.qpos.device
        env_ids_long = torch.as_tensor(env_ids, device=device, dtype=torch.long)
        if env_ids_long.numel() == 0:
            return
        n_envs = int(robot_data.qpos.shape[0])
        clip_source._ensure_device(device, n_envs)

        n_reset = int(env_ids_long.shape[0])
        if hasattr(clip_source, "_clip_indices") and hasattr(clip_source, "clips"):
            new_clip_indices = torch.randint(
                0,
                len(clip_source.clips),
                (n_reset,),
                device=device,
                dtype=torch.long,
            )
            clip_lengths = clip_source._clip_lengths.index_select(0, new_clip_indices)
            new_offsets = torch.floor(
                torch.rand(n_reset, device=device, dtype=torch.float32)
                * clip_lengths.float()
            ).to(dtype=torch.long)
            clip_source._clip_indices[env_ids_long] = new_clip_indices
        else:
            new_offsets = torch.randint(
                0, clip_source.n_frames, (n_reset,), device=device, dtype=torch.long
            )
        clip_source._start_offsets[env_ids_long] = new_offsets
        if clip_source._last_t is not None:
            clip_source._last_t[env_ids_long] = 0.0

        ref_qpos = clip_source.ref_qpos_at_frames(new_offsets)
        if ref_qpos is None:
            return
        ref_qvel = clip_source.ref_qvel_at_frames(new_offsets)

        robot_nq = int(_entity_local_qpos(logic.robot_entity).shape[1])
        robot_nv = int(_entity_local_qvel(logic.robot_entity).shape[1])
        env_origins = env.scene.env_origins[env_ids_long].to(device=device)

        robot_qpos = ref_qpos[:, :robot_nq].float()
        left_qpos = ref_qpos[:, robot_nq : robot_nq + 7].float()
        right_qpos = ref_qpos[:, robot_nq + 7 : robot_nq + 14].float()
        if ref_qvel is not None:
            robot_qvel = ref_qvel[:, :robot_nv].float()
            left_qvel = ref_qvel[:, robot_nv : robot_nv + 6].float()
            right_qvel = ref_qvel[:, robot_nv + 6 : robot_nv + 12].float()
        else:
            robot_qvel = torch.zeros(n_reset, robot_nv, device=device)
            left_qvel = torch.zeros(n_reset, 6, device=device)
            right_qvel = torch.zeros(n_reset, 6, device=device)

        logic.robot_entity.write_joint_state_to_sim(
            robot_qpos, robot_qvel, env_ids=env_ids_long
        )

        left_state = torch.cat(
            (left_qpos[:, :3] + env_origins, left_qpos[:, 3:7], left_qvel), dim=1
        )
        right_state = torch.cat(
            (right_qpos[:, :3] + env_origins, right_qpos[:, 3:7], right_qvel), dim=1
        )
        logic.left_saber_entity.write_root_state_to_sim(
            left_state, env_ids=env_ids_long
        )
        logic.right_saber_entity.write_root_state_to_sim(
            right_state, env_ids=env_ids_long
        )
        env.sim.forward()

    return _fn


def _segment_distance_and_closest(
    points: torch.Tensor,
    seg_start: torch.Tensor,
    seg_end: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    seg = seg_end - seg_start
    denom = torch.sum(seg * seg, dim=-1, keepdim=True)
    denom = torch.clamp(denom, min=1e-12)
    t = torch.sum((points - seg_start) * seg, dim=-1, keepdim=True) / denom
    t = torch.clamp(t, 0.0, 1.0)
    closest = seg_start + t * seg
    return torch.linalg.norm(points - closest, dim=-1), closest


def _impact_face_index(delta: torch.Tensor) -> torch.Tensor:
    ax = delta[..., 0].abs()
    ay = delta[..., 1].abs()
    az = delta[..., 2].abs()
    face = torch.full_like(ax, 3, dtype=torch.long)
    face = torch.where(delta[..., 0] < 0.0, torch.full_like(face, 4), face)
    z_face = torch.where(
        delta[..., 2] >= 0.0, torch.ones_like(face), torch.full_like(face, 2)
    )
    face = torch.where((az >= ax) & (az >= ay), z_face, face)
    face = torch.where((ay >= ax) & (ay >= az), torch.zeros_like(face), face)
    return face


def _first_true_index(mask: torch.Tensor) -> torch.Tensor:
    return torch.argmax(mask.to(torch.int64), dim=1)


def _entity_local_qpos(entity: Any) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    if entity.data.indexing.free_joint_q_adr.numel() > 0:
        pieces.append(entity.data.data.qpos[:, entity.data.indexing.free_joint_q_adr])
    if entity.data.indexing.joint_q_adr.numel() > 0:
        pieces.append(entity.data.data.qpos[:, entity.data.indexing.joint_q_adr])
    if not pieces:
        return torch.zeros(
            (entity.data.data.qpos.shape[0], 0),
            device=entity.data.data.qpos.device,
            dtype=entity.data.data.qpos.dtype,
        )
    return torch.cat(pieces, dim=1)


def _entity_local_qvel(entity: Any) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    if entity.data.indexing.free_joint_v_adr.numel() > 0:
        pieces.append(entity.data.data.qvel[:, entity.data.indexing.free_joint_v_adr])
    if entity.data.indexing.joint_v_adr.numel() > 0:
        pieces.append(entity.data.data.qvel[:, entity.data.indexing.joint_v_adr])
    if not pieces:
        return torch.zeros(
            (entity.data.data.qvel.shape[0], 0),
            device=entity.data.data.qvel.device,
            dtype=entity.data.data.qvel.dtype,
        )
    return torch.cat(pieces, dim=1)


def _env_origins_tensor(env: Any, dtype: torch.dtype) -> torch.Tensor:
    return env.scene.env_origins.to(device=env.device, dtype=dtype)


class SaberMjlabEnvAccessor(MjlabEnvAccessor):
    """EnvAccessor exposing CPU-parity saber state from the split-scene mjlab env."""

    def __init__(self, env: Any) -> None:
        super().__init__(
            env.sim.mj_model,
            env.sim.wp_data,
            ctrl_dt=float(env.physics_dt) * float(env.cfg.decimation),
        )
        self._env = env

    def joint_pos(self) -> torch.Tensor:
        return _saber_combined_qpos_state(self._env)

    def joint_vel(self) -> torch.Tensor:
        return _saber_combined_qvel_state(self._env)

    def muscle_act(self) -> torch.Tensor:
        return _wp_tensor(self._env.sim.wp_data.act, self._env.device)

    def site_xpos(self, site_ids: Any) -> torch.Tensor:
        return _wp_tensor(self._env.sim.wp_data.site_xpos, self._env.device)[
            :, site_ids, :
        ]

    def time(self) -> torch.Tensor:
        return _wp_tensor(self._env.sim.wp_data.time, self._env.device)


def _as_saber_accessor(env: Any) -> SaberMjlabEnvAccessor:
    return SaberMjlabEnvAccessor(env)


def _saber_obs_from_shared(
    env: Any,
    shared_fn: Callable[..., Any],
    *,
    include_task_state: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    accessor = _as_saber_accessor(env)
    params = dict(kwargs)
    if include_task_state:
        params.update(_ensure_saber_task_state(env))
    obs = shared_fn(accessor, **params)
    if not isinstance(obs, torch.Tensor):
        obs = torch.as_tensor(obs, device=env.device, dtype=torch.float32)
    if obs.ndim == 1:
        return obs.unsqueeze(-1)
    return obs.to(device=env.device)


def _saber_reward_dense_from_shared(
    env: Any,
    shared_fn: Callable[..., dict[str, Any]],
    **kwargs: Any,
) -> torch.Tensor:
    accessor = _as_saber_accessor(env)
    result = shared_fn(accessor, _ensure_saber_task_state(env), **kwargs)
    dense = result["dense"]
    if isinstance(dense, torch.Tensor):
        return dense.to(device=env.device, dtype=torch.float32)
    return torch.as_tensor(dense, device=env.device, dtype=torch.float32)


def _saber_termination_from_shared(
    env: Any,
    shared_fn: Callable[..., Any],
    **kwargs: Any,
) -> torch.Tensor:
    accessor = _as_saber_accessor(env)
    done = shared_fn(accessor, _ensure_saber_task_state(env), **kwargs)
    if isinstance(done, torch.Tensor):
        return done.to(device=env.device, dtype=torch.bool)
    return torch.as_tensor(done, device=env.device, dtype=torch.bool)


def _saber_combined_qpos_state(env: Any) -> torch.Tensor:
    logic = _get_saber_logic(env)
    env_origins = _env_origins_tensor(env, logic.left_saber_entity.data.data.qpos.dtype)
    left_qpos = _entity_local_qpos(logic.left_saber_entity).clone()
    right_qpos = _entity_local_qpos(logic.right_saber_entity).clone()
    left_qpos[:, :3] -= env_origins
    right_qpos[:, :3] -= env_origins
    return torch.cat(
        (
            _entity_local_qpos(logic.robot_entity),
            left_qpos,
            right_qpos,
        ),
        dim=1,
    )


def _saber_mimic_qpos_obs(env: Any) -> torch.Tensor:
    return _saber_combined_qpos_state(env)


def _saber_combined_qvel_state(env: Any) -> torch.Tensor:
    logic = _get_saber_logic(env)
    return torch.cat(
        (
            _entity_local_qvel(logic.robot_entity),
            _entity_local_qvel(logic.left_saber_entity),
            _entity_local_qvel(logic.right_saber_entity),
        ),
        dim=1,
    )


def _saber_mimic_qvel_obs(env: Any) -> torch.Tensor:
    ctrl_dt = env.physics_dt * env.cfg.decimation
    return _saber_combined_qvel_state(env) * ctrl_dt


def _saber_mimic_qvel_state(env: Any) -> torch.Tensor:
    return _saber_combined_qvel_state(env)


def _saber_joint_pos_obs(env: Any) -> torch.Tensor:
    return _saber_obs_from_shared(env, shared_joint_pos_obs)


def _saber_joint_vel_obs(env: Any) -> torch.Tensor:
    return _saber_obs_from_shared(env, shared_joint_vel_obs)


def _saber_muscle_act_obs(env: Any) -> torch.Tensor:
    return _saber_obs_from_shared(env, shared_muscle_act_obs)


def _saber_time_obs(env: Any) -> torch.Tensor:
    return _saber_obs_from_shared(env, shared_time_obs)


def _saber_upright_posture_from_xquat(quat: torch.Tensor) -> torch.Tensor:
    qw = quat[..., 0]
    qx = quat[..., 1]
    qy = quat[..., 2]
    qz = quat[..., 3]
    return torch.clamp(2.0 * (qy * qz + qw * qx), 0.0, 1.0)


def _saber_health_status_obs(env: Any) -> torch.Tensor:
    return _saber_obs_from_shared(
        env,
        shared_health_status_obs,
        include_task_state=True,
    )


def _saber_target_pool_obs(env: Any) -> torch.Tensor:
    task_state = _ensure_saber_task_state(env)
    logic = _get_saber_logic(env)
    num_visible = int(logic.task_cfg.obs.extra.get("num_visible", 10))
    return _saber_obs_from_shared(
        env,
        shared_saber_target_pool_obs,
        pool_size=logic.target_pos.shape[1],
        num_visible=num_visible,
        target_pool_pos=logic.target_pos,
        target_pool_active=task_state["target_pool_active"],
        target_pool_vel=task_state["target_pool_vel"],
        target_pool_hit_area=task_state["target_pool_hit_area"].to(torch.float32),
        target_pool_preferred_hand=task_state["target_pool_preferred_hand"].to(
            torch.float32
        ),
        left_saber_site_name=f"{_LEFT_SABER_ENTITY_NAME}/left_saber_tip",
        right_saber_site_name=f"{_RIGHT_SABER_ENTITY_NAME}/right_saber_tip",
    )
