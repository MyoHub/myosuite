# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Reset events shared by the MyoSuite mjlab tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
import re
import copy
import mujoco

import torch
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg

from myosuite.envs.myo.backends.mjlab.mjlab_env_base import normalize_mjlab_env_ids
from myosuite.physics.quat_math_torch import quat2mat
from myosuite.physics.geometry import primitive_support_extent

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class reset_joints_uniform_in_range(ManagerTermBase):  # noqa: N801  (mjlab term style)
    """CPU ``reset_type="random"``: every joint uniform in its model ``range``.

    Unlimited joints have ``range = [0, 0]`` in MuJoCo and therefore reset to
    zero, exactly as on the CPU. Joint velocities keep their default (zero).
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self._entity = env.scene[asset_cfg.name]
        ids = self._entity.indexing.joint_ids.cpu().numpy()
        jnt_range = torch.as_tensor(
            env.sim.mj_model.jnt_range[ids], dtype=torch.float32, device=env.device
        )
        self._low, self._high = jnt_range[:, 0], jnt_range[:, 1]

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
    ) -> None:
        env_ids = normalize_mjlab_env_ids(env, env_ids)
        u = torch.rand(len(env_ids), self._low.numel(), device=env.device)
        pos = self._low + u * (self._high - self._low)
        vel = self._entity.data.default_joint_vel[env_ids].clone()
        self._entity.write_joint_state_to_sim(pos, vel, env_ids=env_ids)


@requires_model_fields("body_mass", "geom_size")
def randomize_carry_weight(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    mass_range: tuple[float, float],
) -> None:
    """CPU ``weight_bodyname``/``weight_range`` perturbation of ``PoseEnvV0``.

    Sets the body's mass to ``U(mass_range)`` and its first geom's radius to
    ``0.01 + 2.5 * mass / 100``. Like the CPU env, derived constants (inertia,
    ``set_const`` quantities, broadphase bounds) are left untouched.
    """
    env_ids = normalize_mjlab_env_ids(env, env_ids)
    entity = env.scene[asset_cfg.name]
    body_id = int(entity.indexing.body_ids[asset_cfg.body_ids][0])
    geom_id = int(env.sim.mj_model.body_geomadr[body_id])
    lo, hi = mass_range
    mass = lo + (hi - lo) * torch.rand(len(env_ids), device=env.device)
    env.sim.model.body_mass[env_ids, body_id] = mass
    env.sim.model.geom_size[env_ids, geom_id, 0] = 0.01 + 2.5 * mass / 100


def write_cpu_state(
    env: ManagerBasedRlEnv,
    entity_name: str,
    env_ids: torch.Tensor,
    qpos: torch.Tensor,
    qvel: torch.Tensor,
) -> None:
    """Write CPU-layout ``qpos`` / ``qvel`` (free root first) of one entity.

    Args:
        env: The environment.
        entity_name: Scene entity key.
        env_ids: Environments to write, shape ``(k,)``.
        qpos: CPU-layout positions, shape ``(k, nq)``; the root position is
            relative to the env origin.
        qvel: CPU-layout velocities, shape ``(k, nv)``.
    """
    entity = env.scene[entity_name]
    idx = entity.indexing
    q_adr = torch.sort(torch.cat([idx.free_joint_q_adr, idx.joint_q_adr]).long())[0]
    v_adr = torch.sort(torch.cat([idx.free_joint_v_adr, idx.joint_v_adr]).long())[0]
    if not len(q_adr):
        return
    q_base, v_base = q_adr[0], v_adr[0]
    if len(idx.free_joint_q_adr):
        root = torch.cat((qpos[:, :7], qvel[:, :6]), dim=-1)
        root[:, :3] += env.scene.env_origins[env_ids]
        root[:, 10:13] = torch.einsum(
            "nij,nj->ni", quat2mat(root[:, 3:7]), qvel[:, 3:6]
        )
        entity.write_root_state_to_sim(root, env_ids=env_ids)
    if len(idx.joint_q_adr):
        entity.write_joint_state_to_sim(
            qpos[:, idx.joint_q_adr.long() - q_base],
            qvel[:, idx.joint_v_adr.long() - v_base],
            env_ids=env_ids,
        )


def subject_provenance(
    env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, metadata: dict[str, Any]
) -> None:
    """Expose static construction provenance in every full or partial reset."""
    del env_ids
    env.subject_model_metadata = copy.deepcopy(metadata)
    env.extras["subject_model"] = copy.deepcopy(metadata)


class reset_to_cpu_state(ManagerTermBase):  # noqa: N801  (mjlab term style)
    """CPU keyframe reset, optionally with per-joint noise.

    With ``joint_noise=(lo, hi)`` the first ``qpos`` coordinate of every joint
    gets ``U(lo, hi)`` added and is clipped to the joint ``range`` (``[0, 0]``
    for unlimited joints, as in MuJoCo), exactly as ``LegReachEnvV0``.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        p = cfg.params
        entity = env.scene[p["asset_cfg"].name]
        idx = entity.indexing
        self._qpos = torch.tensor(p["qpos"], dtype=torch.float32, device=env.device)
        self._qvel = torch.tensor(p["qvel"], dtype=torch.float32, device=env.device)
        model = env.sim.mj_model
        entity_q = sorted(torch.cat([idx.free_joint_q_adr, idx.joint_q_adr]).tolist())
        base, members = entity_q[0], set(entity_q)
        joints = [j for j in range(model.njnt) if int(model.jnt_qposadr[j]) in members]
        self._noise_adr = torch.tensor(
            [int(model.jnt_qposadr[j]) - base for j in joints], device=env.device
        )
        rng = torch.as_tensor(
            model.jnt_range[joints], dtype=torch.float32, device=env.device
        )
        self._lo, self._hi = rng[:, 0], rng[:, 1]
        offsets = p.get("joint_offsets", {})
        local_joints, _ = (
            entity.find_joints(
                [f"^{re.escape(name)}$" for name in offsets], preserve_order=True
            )
            if offsets
            else ([], [])
        )
        selected = idx.joint_ids[local_joints].tolist()
        self._offset_adr = torch.tensor(
            [int(model.jnt_qposadr[j]) - base for j in selected],
            dtype=torch.long,
            device=env.device,
        )
        self._offset_lo = torch.tensor(
            [offsets[name][0] for name in offsets], device=env.device
        )
        self._offset_hi = torch.tensor(
            [offsets[name][1] for name in offsets], device=env.device
        )
        self._limited = torch.tensor(
            model.jnt_limited[selected], dtype=torch.bool, device=env.device
        )
        self._ranges = torch.tensor(
            model.jnt_range[selected], dtype=torch.float32, device=env.device
        )
        ground = p.get("ground")
        self._ground = ground
        if ground is not None:
            local, _ = entity.find_geoms(
                [f"^{re.escape(name)}$" for name in ground["sole_geoms"]],
                preserve_order=True,
            )
            self._sole_ids = idx.geom_ids[local].long()
            self._sole_shapes = [
                mujoco.mjtGeom(int(kind)).name.removeprefix("mjGEOM_").lower()
                for kind in model.geom_type[self._sole_ids.cpu().numpy()]
            ]
            self._sole_sizes = torch.tensor(
                model.geom_size[self._sole_ids.cpu().numpy()],
                device=env.device,
                dtype=torch.float32,
            )
            self._sole_vertices = {}
            for index, geom_id in enumerate(self._sole_ids.cpu().tolist()):
                if self._sole_shapes[index] == "mesh":
                    mesh_id = int(model.geom_dataid[geom_id])
                    start = int(model.mesh_vertadr[mesh_id])
                    count = int(model.mesh_vertnum[mesh_id])
                    self._sole_vertices[index] = torch.tensor(
                        model.mesh_vert[start : start + count],
                        device=env.device,
                        dtype=torch.float32,
                    )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        qpos: tuple[float, ...],
        qvel: tuple[float, ...],
        joint_noise: tuple[float, float] | None = None,
        joint_offsets: dict[str, tuple[float, float]] | None = None,
        ground: dict[str, Any] | None = None,
    ) -> None:
        env_ids = normalize_mjlab_env_ids(env, env_ids)
        q = self._qpos.repeat(len(env_ids), 1)
        if joint_offsets:
            noise = self._offset_lo + (self._offset_hi - self._offset_lo) * torch.rand(
                len(env_ids),
                len(self._offset_adr),
                device=env.device,
            )
            values = q[:, self._offset_adr] + noise
            q[:, self._offset_adr] = torch.where(
                self._limited,
                torch.clamp(values, self._ranges[:, 0], self._ranges[:, 1]),
                values,
            )
        if joint_noise is not None and joint_noise[1] > joint_noise[0]:
            lo, hi = joint_noise
            noise = lo + (hi - lo) * torch.rand(
                len(env_ids), len(self._noise_adr), device=env.device
            )
            q[:, self._noise_adr] = torch.clamp(
                q[:, self._noise_adr] + noise, self._lo, self._hi
            )
        write_cpu_state(
            env, asset_cfg.name, env_ids, q, self._qvel.repeat(len(env_ids), 1)
        )
        if joint_offsets and ground is not None:
            env.sim.forward()
            vertical = env.sim.data.geom_xmat[env_ids][:, self._sole_ids].reshape(
                len(env_ids), -1, 3, 3
            )[:, :, 2]
            extents = []
            for index, shape in enumerate(self._sole_shapes):
                if shape == "mesh":
                    vertices = self._sole_vertices[index]
                    extents.append(
                        -torch.amin(
                            torch.sum(
                                vertices[None] * vertical[:, index, None],
                                dim=-1,
                            ),
                            dim=-1,
                        )
                    )
                else:
                    extents.append(
                        primitive_support_extent(
                            shape, self._sole_sizes[index], vertical[:, index], torch
                        )
                    )
            bottoms = env.sim.data.geom_xpos[env_ids][
                :, self._sole_ids, 2
            ] - torch.stack(extents, dim=-1)
            target = (
                ground["ground_height"]
                + ground["clearance"]
                + env.scene.env_origins[env_ids, 2]
            )
            q[:, 2] += target - torch.min(bottoms, dim=-1).values
            write_cpu_state(
                env, asset_cfg.name, env_ids, q, self._qvel.repeat(len(env_ids), 1)
            )
