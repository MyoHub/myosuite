# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Reset events shared by the MyoSuite mjlab tasks."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

from myosuite.envs.myo.backends.mjlab.mjlab_env_base import normalize_mjlab_env_ids
from myosuite.physics.base_excitation import (
    CompositeBaseProfile,
    HarmonicBaseProfile,
    PulseBaseProfile,
    RecordedBaseProfile,
    inertial_body_wrenches,
)
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def torch_joint_couplings(
    couplings: JointCouplings, device: Any, dtype: torch.dtype = torch.float32
) -> JointCouplings:
    """*couplings* as tensors on *device*: addresses ``long``, values in *dtype*.

    Args:
        couplings: NumPy couplings (:meth:`JointCouplings.from_model`).
        device: Torch device of the states they will project.
        dtype: Floating dtype of those states.

    Returns:
        The couplings for :func:`~myosuite.physics.joint_coupling.project_joint_couplings`.
    """
    return couplings.map_arrays(
        lambda a: torch.as_tensor(
            a, device=device, dtype=dtype if a.dtype.kind == "f" else torch.long
        )
    )


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
    if entity.is_fixed_base:
        entity.write_joint_state_to_sim(qpos, qvel, env_ids=env_ids)
        return
    root = torch.cat(
        [
            qpos[:, :3] + env.scene.env_origins[env_ids],
            qpos[:, 3:7],
            qvel[:, :3],
            quat_apply(qpos[:, 3:7], qvel[:, 3:6]),
        ],
        dim=-1,
    )
    entity.write_root_state_to_sim(root, env_ids=env_ids)
    entity.write_joint_state_to_sim(qpos[:, 7:], qvel[:, 6:], env_ids=env_ids)


class reset_to_cpu_state(ManagerTermBase):  # noqa: N801  (mjlab term style)
    """CPU keyframe reset, optionally with per-joint noise, on the joint couplings.

    With ``joint_noise=(lo, hi)`` the first ``qpos`` coordinate of every joint
    gets ``U(lo, hi)`` added and is clipped to the joint ``range`` (``[0, 0]``
    for unlimited joints, as in MuJoCo), exactly as ``LegReachEnvV0``. The
    coupled joints are then put on their polynomials, as the CPU resets do.

    ``couplings`` (required) are the
    :class:`~myosuite.physics.joint_coupling.JointCouplings` of the CPU model,
    whose layout ``qpos`` / ``qvel`` follow, or ``None`` to skip the projection.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        p = cfg.params
        idx = env.scene[p["asset_cfg"].name].indexing
        self._qpos = torch.tensor(p["qpos"], dtype=torch.float32, device=env.device)
        self._qvel = torch.tensor(p["qvel"], dtype=torch.float32, device=env.device)
        self._couplings = (
            None
            if p["couplings"] is None
            else torch_joint_couplings(p["couplings"], env.device)
        )
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

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        qpos: tuple[float, ...],
        qvel: tuple[float, ...],
        couplings: JointCouplings | None,
        joint_noise: tuple[float, float] | None = None,
    ) -> None:
        env_ids = normalize_mjlab_env_ids(env, env_ids)
        q = self._qpos.repeat(len(env_ids), 1)
        qd = self._qvel.repeat(len(env_ids), 1)
        if joint_noise is not None and joint_noise[1] > joint_noise[0]:
            lo, hi = joint_noise
            noise = lo + (hi - lo) * torch.rand(
                len(env_ids), len(self._noise_adr), device=env.device
            )
            q[:, self._noise_adr] = torch.clamp(
                q[:, self._noise_adr] + noise, self._lo, self._hi
            )
        if self._couplings is not None:
            project_joint_couplings(self._couplings, q, qd)
        write_cpu_state(env, asset_cfg.name, env_ids, q, qd)


def _quat_wxyz_to_mat(quat: torch.Tensor) -> torch.Tensor:
    """Convert unit ``(..., 4)`` ``(w, x, y, z)`` quaternions to ``(..., 3, 3)`` matrices."""
    w, x, y, z = quat.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack(
        (
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            1.0 - 2.0 * (xx + yy),
        ),
        dim=-1,
    ).reshape(*quat.shape[:-1], 3, 3)


def _flatten_base_profiles(profile: Any) -> list[Any]:
    if isinstance(profile, CompositeBaseProfile):
        out: list[Any] = []
        for sub in profile.profiles:
            out.extend(_flatten_base_profiles(sub))
        return out
    if isinstance(
        profile, (HarmonicBaseProfile, PulseBaseProfile, RecordedBaseProfile)
    ):
        return [profile]
    raise TypeError(
        "apply_base_excitation supports HarmonicBaseProfile, PulseBaseProfile, "
        f"RecordedBaseProfile, and CompositeBaseProfile; got {type(profile).__name__}"
    )


class apply_base_excitation(ManagerTermBase):  # noqa: N801  (mjlab term style)
    """Step-rate non-inertial D'Alembert body wrenches via ``Entity.write_external_wrench_to_sim``."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(env)
        p = cfg.params
        asset_cfg: SceneEntityCfg = p["asset_cfg"]
        self._entity = env.scene[asset_cfg.name]
        self._step_dt = float(env.step_dt)
        device = env.device
        dtype = torch.float32

        body_ids = self._entity.indexing.body_ids.cpu().numpy()
        mj_model = env.sim.mj_model
        dyn = (mj_model.body_weldid[body_ids] != 0).astype("float32")
        masses = mj_model.body_mass[body_ids].astype("float32") * dyn
        inertias = mj_model.body_inertia[body_ids].astype("float32") * dyn[:, None]

        self._body_mass = torch.tensor(masses, dtype=dtype, device=device)
        self._body_inertia = torch.tensor(inertias, dtype=dtype, device=device)
        self._pivot = torch.tensor(
            p.get("pivot", (0.0, 0.0, 0.0)), dtype=dtype, device=device
        )
        nb = len(body_ids)
        self._zero_wrench = torch.zeros(
            (env.num_envs, nb, 3), dtype=dtype, device=device
        )
        self._zero_vec3 = torch.zeros((env.num_envs, 3), dtype=dtype, device=device)
        self._elapsed = torch.zeros(env.num_envs, dtype=dtype, device=device)

        self._harmonic: list[dict[str, Any]] = []
        self._pulse: list[dict[str, Any]] = []
        self._recorded: list[dict[str, Any]] = []
        for leaf in _flatten_base_profiles(p["profile"]):
            if isinstance(leaf, HarmonicBaseProfile):
                freqs, lin, ang, phase = leaf._normalized_params()
                base_phase = torch.tensor(phase, dtype=dtype, device=device)
                lin_phase = base_phase.unsqueeze(0).repeat(env.num_envs, 1, 1)
                ang_phase = base_phase.unsqueeze(0).repeat(env.num_envs, 1, 1)
                if leaf.random_phase:
                    lin_phase.uniform_(0.0, 2.0 * math.pi)
                    ang_phase.uniform_(0.0, 2.0 * math.pi)
                self._harmonic.append(
                    {
                        "omega": torch.tensor(
                            2.0 * math.pi * freqs, dtype=dtype, device=device
                        ).view(1, -1, 1),
                        "lin_amp": torch.tensor(
                            lin, dtype=dtype, device=device
                        ).unsqueeze(0),
                        "ang_amp": torch.tensor(
                            ang, dtype=dtype, device=device
                        ).unsqueeze(0),
                        "base_phase": base_phase,
                        "lin_phase": lin_phase,
                        "ang_phase": ang_phase,
                        "random_phase": leaf.random_phase,
                    }
                )
            elif isinstance(leaf, PulseBaseProfile):
                self._pulse.append(
                    {
                        "start_s": float(leaf.start_s),
                        "duration_s": float(leaf.duration_s),
                        "peak_lin": torch.tensor(
                            leaf.peak_lin_acc, dtype=dtype, device=device
                        ).view(1, 3),
                        "peak_ang": torch.tensor(
                            leaf.peak_ang_vel, dtype=dtype, device=device
                        ).view(1, 3),
                    }
                )
            elif isinstance(leaf, RecordedBaseProfile):
                self._recorded.append(
                    {
                        "times": torch.tensor(leaf.times, dtype=dtype, device=device),
                        "lin_acc": torch.tensor(
                            leaf.lin_acc, dtype=dtype, device=device
                        ),
                        "ang_vel": torch.tensor(
                            leaf.ang_vel, dtype=dtype, device=device
                        ),
                        "ang_acc": torch.tensor(
                            leaf.ang_acc, dtype=dtype, device=device
                        ),
                        "loop": bool(leaf.loop),
                    }
                )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = normalize_mjlab_env_ids(self._env, env_ids)
        n = len(ids)
        if n == 0:
            return
        self._elapsed[ids] = 0.0
        for h in self._harmonic:
            if h["random_phase"]:
                shape = (n, h["base_phase"].shape[0], 3)
                h["lin_phase"][ids] = torch.rand(
                    shape, dtype=self._elapsed.dtype, device=self._env.device
                ) * (2.0 * math.pi)
                h["ang_phase"][ids] = torch.rand(
                    shape, dtype=self._elapsed.dtype, device=self._env.device
                ) * (2.0 * math.pi)
            else:
                h["lin_phase"][ids] = h["base_phase"]
                h["ang_phase"][ids] = h["base_phase"]
        zeros = self._zero_wrench[:n]
        self._entity.write_external_wrench_to_sim(zeros, zeros, env_ids=ids)

    def _sample_kinematics(
        self, t: torch.Tensor, ids: torch.Tensor | slice
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = t.shape[0]
        lin_acc = self._zero_vec3[:n].clone()
        ang_vel = self._zero_vec3[:n].clone()
        ang_acc = self._zero_vec3[:n].clone()

        t_col = t[:, None, None]
        for h in self._harmonic:
            lp = h["lin_phase"][ids]
            ap = h["ang_phase"][ids]
            arg_lin = h["omega"] * t_col + lp
            arg_ang = h["omega"] * t_col + ap
            lin_acc = lin_acc + torch.sum(h["lin_amp"] * torch.sin(arg_lin), dim=1)
            ang_vel = ang_vel + torch.sum(h["ang_amp"] * torch.sin(arg_ang), dim=1)
            ang_acc = ang_acc + torch.sum(
                h["ang_amp"] * h["omega"] * torch.cos(arg_ang), dim=1
            )

        for pulse in self._pulse:
            tau = (t - pulse["start_s"]) / pulse["duration_s"]
            active = ((tau >= 0.0) & (tau <= 1.0)).to(dtype=t.dtype).unsqueeze(-1)
            tau_c = torch.clamp(tau, 0.0, 1.0).unsqueeze(-1)
            env_val = torch.sin(math.pi * tau_c) * active
            denv_dt = (
                (math.pi / pulse["duration_s"]) * torch.cos(math.pi * tau_c)
            ) * active
            lin_acc = lin_acc + pulse["peak_lin"] * env_val
            ang_vel = ang_vel + pulse["peak_ang"] * env_val
            ang_acc = ang_acc + pulse["peak_ang"] * denv_dt

        for rec in self._recorded:
            times = rec["times"]
            if rec["loop"]:
                span = times[-1] - times[0]
                query = times[0] + torch.remainder(
                    torch.clamp_min(t - times[0], 0.0), span
                )
            else:
                query = torch.clamp(t, min=times[0], max=times[-1])
            idx1 = torch.clamp(torch.searchsorted(times, query), 1, times.shape[0] - 1)
            idx0 = idx1 - 1
            t0 = times[idx0]
            t1 = times[idx1]
            alpha = ((query - t0) / torch.clamp_min(t1 - t0, 1e-12)).unsqueeze(-1)
            lin_acc = (
                lin_acc
                + rec["lin_acc"][idx0]
                + alpha * (rec["lin_acc"][idx1] - rec["lin_acc"][idx0])
            )
            ang_vel = (
                ang_vel
                + rec["ang_vel"][idx0]
                + alpha * (rec["ang_vel"][idx1] - rec["ang_vel"][idx0])
            )
            ang_acc = (
                ang_acc
                + rec["ang_acc"][idx0]
                + alpha * (rec["ang_acc"][idx1] - rec["ang_acc"][idx0])
            )

        return lin_acc, ang_vel, ang_acc

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        profile: Any,
        pivot: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        del asset_cfg, profile, pivot
        if env_ids is None:
            idx: torch.Tensor | slice = slice(None)
            t = self._elapsed
            xipos = self._entity.data.body_com_pos_w - env.scene.env_origins[:, None, :]
            quat = self._entity.data.body_com_quat_w
            v_rel = self._entity.data.body_com_lin_vel_w
            omega_rel = self._entity.data.body_com_ang_vel_w
            self._elapsed = self._elapsed + self._step_dt
        else:
            ids = normalize_mjlab_env_ids(env, env_ids)
            if len(ids) == 0:
                return
            idx = ids
            t = self._elapsed[ids]
            xipos = (
                self._entity.data.body_com_pos_w[ids]
                - env.scene.env_origins[ids, None, :]
            )
            quat = self._entity.data.body_com_quat_w[ids]
            v_rel = self._entity.data.body_com_lin_vel_w[ids]
            omega_rel = self._entity.data.body_com_ang_vel_w[ids]
            self._elapsed[ids] = t + self._step_dt

        lin_acc, ang_vel, ang_acc = self._sample_kinematics(t, idx)
        ximat = _quat_wxyz_to_mat(quat)
        cvel = torch.cat((omega_rel, v_rel), dim=-1)
        wrenches = inertial_body_wrenches(
            torch,
            self._body_mass,
            self._body_inertia,
            xipos,
            ximat,
            cvel,
            xipos,
            lin_acc,
            ang_vel=ang_vel,
            ang_acc=ang_acc,
            pivot=self._pivot,
        )
        self._entity.write_external_wrench_to_sim(
            wrenches[..., :3], wrenches[..., 3:], env_ids=env_ids
        )
