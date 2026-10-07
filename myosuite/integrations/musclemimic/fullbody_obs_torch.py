# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Batched torch twin of the MuscleMimic full-body observation builder.

:class:`TorchFullbodyObsAdapter` builds the
:class:`~myosuite.integrations.musclemimic.fullbody_local_policy.FullbodyObsAdapter`
observation for a whole batch of envs from batched MuJoCo / MuJoCo-Warp data
(e.g. mjlab's post-``forward()`` sim data), on the data's device and without
host synchronisation.  It is float32 throughout, so it matches the CPU builder to
float32 rounding rather than bitwise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from myosuite.integrations.musclemimic.fullbody_local_policy import (
        FullbodyObsAdapter,
    )

# Data fields the builder reads (all ``(N, ...)`` batched, MuJoCo model layout).
FULLBODY_OBS_DATA_FIELDS = (
    "qpos",
    "qvel",
    "ctrl",
    "act",
    "actuator_length",
    "actuator_velocity",
    "actuator_force",
    "sensordata",
    "site_xpos",
    "site_xmat",
    "cvel",
    "subtree_com",
)


def _relative_site_quantities(
    *,
    site_ids: torch.Tensor,
    site_xpos: Any,
    site_xmat: Any,
    cvel_parent: Any,
    subtree_com_root: Any,
    site_bodyid: torch.Tensor,
    body_rootid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched port of ``fullbody_local_policy._relative_site_quantities``.

    Args:
        site_ids: ``(n,)`` site ids; the first is the main (reference) site.
        site_xpos: ``(B, nsite, 3)`` site positions.
        site_xmat: ``(B, nsite, 9)`` or ``(B, nsite, 3, 3)`` site rotations.
        cvel_parent: ``(B, nbody, 6)`` body com velocities ``[ang, lin]``.
        subtree_com_root: ``(B, nbody, 3)`` subtree centres of mass.
        site_bodyid: ``(nsite,)`` parent body of each site.
        body_rootid: ``(nbody,)`` root body of each body.

    Returns:
        ``site_rpos (B, n-1, 3)``, ``site_rangles (B, n-1, 3)`` and
        ``site_rvel (B, n-1, 6)`` of the other sites relative to the main one.
    """
    from mjlab.utils.lab_api.math import axis_angle_from_quat, quat_from_matrix

    parent_body = site_bodyid[site_ids]
    pos = site_xpos[:, site_ids, :]
    mat = site_xmat[:, site_ids].reshape(pos.shape[0], -1, 3, 3)
    body_cvel = cvel_parent[:, parent_body, :]
    rpos_com = pos - subtree_com_root[:, body_rootid[parent_body], :]
    ang = body_cvel[..., :3]
    lin = body_cvel[..., 3:] - torch.cross(rpos_com, ang, dim=-1)

    main_mat = mat[:, 0]
    site_rpos = pos[:, 1:] - pos[:, :1]
    rel_rot = torch.einsum("bji,bnjk->bnik", main_mat, mat[:, 1:])
    site_rangles = axis_angle_from_quat(quat_from_matrix(rel_rot))
    rel_lin = torch.einsum("bij,bnj->bni", main_mat, lin[:, :1] - lin[:, 1:])
    # rel_rot^T @ w, as the CPU adapter and upstream loco-mujoco define it.
    rel_ang = torch.einsum("bnki,bnk->bni", rel_rot, ang[:, 1:]) - ang[:, :1]
    return site_rpos, site_rangles, torch.cat([rel_ang, rel_lin], dim=-1)


class TorchFullbodyObsAdapter:
    """Torch port of :class:`~...fullbody_local_policy.FullbodyObsAdapter`.

    Copies the CPU adapter's index arrays and trajectory buffers to *device* once;
    :meth:`build` then gathers the observation of every env in one batch.

    Args:
        adapter: CPU adapter that defines the layout (goal params, obs flags,
            clip, model indices).
        device: Device of the batched data passed to :meth:`build`.
    """

    def __init__(self, adapter: FullbodyObsAdapter, *, device: torch.device) -> None:
        self.device = torch.device(device)
        self._goal = adapter._goal
        self._obs_flags = dict(adapter._obs_flags)
        self._traj_len = int(adapter._traj_len)
        self.goal_dim = int(adapter.goal_dim)

        def _li(v: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(np.asarray(v, dtype=np.int64), device=self.device)

        def _lf(v: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(np.asarray(v, dtype=np.float32), device=self.device)

        self._root_qpos_idx_full = _li(adapter._root_qpos_idx_full)
        self._root_qvel_idx_full = _li(adapter._root_qvel_idx_full)
        self._root_qpos_idx_xyz = _li(adapter._root_qpos_idx_xyz)
        self._qpos_ind = _li(adapter._qpos_ind)
        self._qvel_ind = _li(adapter._qvel_ind)
        self._qpos_non_root_ind = _li(adapter._qpos_non_root_ind)
        self._qvel_non_root_ind = _li(adapter._qvel_non_root_ind)
        self._actuator_ids = _li(adapter._actuator_ids)
        self._site_ids = _li(adapter._site_ids)
        self._traj_site_ids = _li(adapter._traj_site_ids)
        self._sim_site_bodyid = _li(adapter._sim_site_bodyid)
        self._sim_body_rootid = _li(adapter._sim_body_rootid)
        self._traj_site_bodyid = _li(adapter._traj_site_bodyid)
        self._traj_body_rootid = _li(adapter._traj_body_rootid)
        self._traj_site_xpos = _lf(adapter._traj_site_xpos)
        self._traj_site_xmat = _lf(adapter._traj_site_xmat)
        self._traj_cvel = _lf(adapter._traj_cvel)
        self._traj_subtree_com = _lf(adapter._traj_subtree_com)
        self._clip_qpos = _lf(adapter._clip.qpos)
        # The CPU adapter reads zeros of width nv when the clip has no qvel.
        self._clip_qvel = _lf(
            adapter._clip.qvel
            if adapter._clip.qvel is not None
            else np.zeros((self._traj_len, adapter._model.nv))
        )
        self._lookahead = _li(
            np.arange(self._goal.n_step_lookahead) * int(self._goal.n_step_stride)
        )
        sensor_adr = np.asarray(adapter._model.sensor_adr, dtype=np.int64)
        sensor_dim = np.asarray(adapter._model.sensor_dim, dtype=np.int64)
        self._touch_sensor_slices = tuple(
            (int(sensor_adr[s]), int(sensor_adr[s] + sensor_dim[s]))
            for s in np.asarray(adapter._touch_sensor_ids, dtype=np.int64)
        )
        self._muscle_fields = tuple(
            field
            for flag, field in (
                ("enable_muscle_length_observations", "actuator_length"),
                ("enable_muscle_velocity_observations", "actuator_velocity"),
                ("enable_muscle_force_observations", "actuator_force"),
                ("enable_muscle_excitation_observations", "ctrl"),
                ("enable_muscle_activation_observations", "act"),
            )
            if self._obs_flags[flag]
        )

    def _traj_goal_obs(self, frame_idx: torch.Tensor) -> torch.Tensor:
        goal = self._goal
        batch, steps = int(frame_idx.shape[0]), int(goal.n_step_lookahead)
        future = torch.clamp(
            frame_idx[:, None] + self._lookahead[None, :], max=self._traj_len - 1
        )
        flat = future.reshape(-1)
        qpos = self._clip_qpos[flat].reshape(batch, steps, -1)
        qvel = self._clip_qvel[flat].reshape(batch, steps, -1)
        site_rpos, site_rangles, site_rvel = _relative_site_quantities(
            site_ids=self._traj_site_ids,
            site_xpos=self._traj_site_xpos[flat],
            site_xmat=self._traj_site_xmat[flat],
            cvel_parent=self._traj_cvel[flat],
            subtree_com_root=self._traj_subtree_com[flat],
            site_bodyid=self._traj_site_bodyid,
            body_rootid=self._traj_body_rootid,
        )
        site_rpos = site_rpos.reshape(batch, steps, -1)
        if goal.use_concise_lookahead:
            ref_root_pos = self._clip_qpos[frame_idx][:, self._root_qpos_idx_xyz]
            ref_root_vel = self._clip_qvel[frame_idx][:, self._root_qvel_idx_full]
            parts: list[torch.Tensor] = [site_rpos[:, 0]]
            for s in range(1, steps):
                parts += [
                    qpos[:, s, self._root_qpos_idx_xyz] - ref_root_pos,
                    qvel[:, s, self._root_qvel_idx_full] - ref_root_vel,
                    site_rpos[:, s],
                ]
            return torch.cat(parts, dim=1)
        return torch.cat(
            [
                qpos[:, :, self._qpos_ind].reshape(batch, -1),
                qvel[:, :, self._qvel_ind].reshape(batch, -1),
                site_rpos.reshape(batch, -1),
                site_rangles.reshape(batch, -1),
                site_rvel.reshape(batch, -1),
            ],
            dim=1,
        )

    def build(self, data: Any, frame_idx: torch.Tensor) -> torch.Tensor:
        """Return the ``(B, obs_dim)`` float32 observation of every env in *data*.

        Args:
            data: Batched MuJoCo data with the :data:`FULLBODY_OBS_DATA_FIELDS`
                as ``(B, ...)`` tensors (or mjlab ``TorchArray``) on
                :attr:`device`, read after ``forward``.
            frame_idx: ``(B,)`` int64 clip frame of each env.
        """
        goal = self._goal
        n = int(data.qpos.shape[0])
        obs: list[torch.Tensor] = []
        if self._obs_flags["enable_joint_pos_observations"]:
            obs += [
                data.qpos[:, self._root_qpos_idx_full[2:]],
                data.qpos[:, self._qpos_non_root_ind],
            ]
        if self._obs_flags["enable_joint_vel_observations"]:
            obs += [
                data.qvel[:, self._root_qvel_idx_full],
                data.qvel[:, self._qvel_non_root_ind],
            ]
        if self._muscle_fields:
            # Interleaved per actuator: [length, velocity, force, ctrl, act, ...].
            blocks = [
                getattr(data, f)[:, self._actuator_ids] for f in self._muscle_fields
            ]
            obs.append(torch.stack(blocks, dim=2).reshape(n, -1))
        obs += [
            data.sensordata[:, s:e].sum(dim=1, keepdim=True)
            for s, e in self._touch_sensor_slices
        ]
        site_rpos, site_rangles, site_rvel = _relative_site_quantities(
            site_ids=self._site_ids,
            site_xpos=data.site_xpos,
            site_xmat=data.site_xmat,
            cvel_parent=data.cvel,
            subtree_com_root=data.subtree_com,
            site_bodyid=self._sim_site_bodyid,
            body_rootid=self._sim_body_rootid,
        )
        if goal.enable_mimic_site_rpos_observations:
            obs.append(site_rpos.reshape(n, -1))
        obs += [
            site_rangles.reshape(n, -1),
            site_rvel.reshape(n, -1),
            self._traj_goal_obs(frame_idx),
        ]
        if goal.enable_motion_phase:
            obs.append(frame_idx[:, None].to(torch.float32) / max(self._traj_len, 1))
        return torch.cat(obs, dim=1).to(torch.float32)


__all__ = ["FULLBODY_OBS_DATA_FIELDS", "TorchFullbodyObsAdapter"]
