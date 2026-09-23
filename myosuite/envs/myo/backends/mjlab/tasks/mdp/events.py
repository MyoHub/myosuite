# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Reset events shared by the MyoSuite mjlab tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg

from myosuite.envs.myo.backends.mjlab.mjlab_env_base import normalize_mjlab_env_ids

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
    idx = env.scene[entity_name].indexing
    q_adr = torch.sort(torch.cat([idx.free_joint_q_adr, idx.joint_q_adr]).long())[0]
    v_adr = torch.sort(torch.cat([idx.free_joint_v_adr, idx.joint_v_adr]).long())[0]
    qpos = qpos.clone()
    if len(idx.free_joint_q_adr):
        qpos[:, :3] += env.scene.env_origins[env_ids]
    env.sim.data.qpos[env_ids[:, None], q_adr] = qpos
    env.sim.data.qvel[env_ids[:, None], v_adr] = qvel


class reset_to_cpu_state(ManagerTermBase):  # noqa: N801  (mjlab term style)
    """CPU keyframe reset, optionally with per-joint noise.

    With ``joint_noise=(lo, hi)`` the first ``qpos`` coordinate of every joint
    gets ``U(lo, hi)`` added and is clipped to the joint ``range`` (``[0, 0]``
    for unlimited joints, as in MuJoCo), exactly as ``LegReachEnvV0``.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        p = cfg.params
        idx = env.scene[p["asset_cfg"].name].indexing
        self._qpos = torch.tensor(p["qpos"], dtype=torch.float32, device=env.device)
        self._qvel = torch.tensor(p["qvel"], dtype=torch.float32, device=env.device)
        model = env.sim.mj_model
        entity_q = sorted(torch.cat([idx.free_joint_q_adr, idx.joint_q_adr]).tolist())
        base, members = entity_q[0], set(entity_q)
        joints = [j for j in range(model.njnt) if int(model.jnt_qposadr[j]) in members]
        self._noise_adr = torch.tensor(
            [int(model.jnt_qposadr[j]) - base for j in joints], device=env.device
        )
        rng = torch.as_tensor(model.jnt_range[joints], dtype=torch.float32, device=env.device)
        self._lo, self._hi = rng[:, 0], rng[:, 1]

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        qpos: tuple[float, ...],
        qvel: tuple[float, ...],
        joint_noise: tuple[float, float] | None = None,
    ) -> None:
        env_ids = normalize_mjlab_env_ids(env, env_ids)
        q = self._qpos.repeat(len(env_ids), 1)
        if joint_noise is not None and joint_noise[1] > joint_noise[0]:
            lo, hi = joint_noise
            noise = lo + (hi - lo) * torch.rand(len(env_ids), len(self._noise_adr), device=env.device)
            q[:, self._noise_adr] = torch.clamp(q[:, self._noise_adr] + noise, self._lo, self._hi)
        write_cpu_state(env, asset_cfg.name, env_ids, q, self._qvel.repeat(len(env_ids), 1))
