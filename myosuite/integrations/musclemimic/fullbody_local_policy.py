# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Minimal local policy loader/inference for full-body checkpoints.

This module intentionally avoids importing upstream runtime packages.
It loads Orbax checkpoint artifacts directly and executes a small NumPy
forward pass for the actor network.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# pylint: disable=no-member
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as np_R

from myosuite.integrations.musclemimic.running_stats import (
    numpy_running_mean_std_update as running_mean_std_update,
)
from myosuite.integrations.musclemimic.trajectory_io import MotionClip


def _to_numpy_tree(tree: Any) -> Any:
    if isinstance(tree, dict):
        return {k: _to_numpy_tree(v) for k, v in tree.items()}
    return np.asarray(tree, dtype=np.float32)


def _layer_norm(x: np.ndarray, scale: np.ndarray, bias: np.ndarray) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    eps = 1e-5
    return ((x - mean) / np.sqrt(var + eps)) * scale + bias


def _silu(x: np.ndarray) -> np.ndarray:
    # numpy inference path — torch.nn.functional.silu not available here
    return x / (1.0 + np.exp(-x))


def _dense(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.matmul(x, kernel) + bias


def _sorted_prefix_indices(tree: dict[str, Any], prefix: str) -> list[int]:
    """Return sorted integer suffixes for keys like ``f"{prefix}{i}"``."""
    indices: list[int] = []
    for key in tree:
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix) :]
        if suffix.isdigit():
            indices.append(int(suffix))
    return sorted(indices)


def _residual_block_indices(actor: dict[str, Any]) -> list[int]:
    """Return sorted residual block indices present in an actor tree."""
    indices: list[int] = []
    suffix = "_layer0_dense"
    for key in actor:
        if not key.startswith("block") or not key.endswith(suffix):
            continue
        idx = key[len("block") : -len(suffix)]
        if idx.isdigit():
            indices.append(int(idx))
    return sorted(indices)


@dataclass(frozen=True)
class LocalPolicyArtifacts:
    """Local policy artifacts parsed from a checkpoint."""

    params: dict[str, Any]
    obs_mean: np.ndarray
    obs_var: np.ndarray
    obs_count: np.ndarray
    obs_dim: int
    action_dim: int


@dataclass(frozen=True)
class LocalPolicyActionTrace:
    """Intermediate tensors for action-level parity diagnostics."""

    raw_obs: np.ndarray
    policy_obs: np.ndarray
    norm_obs: np.ndarray
    mean_action: np.ndarray
    action: np.ndarray
    run_mean_before: np.ndarray
    run_var_before: np.ndarray
    run_count_before: np.ndarray
    run_mean_after: np.ndarray
    run_var_after: np.ndarray
    run_count_after: np.ndarray


_FULLBODY_OBS_FLAG_DEFAULTS: dict[str, bool] = {
    # Defaults preserve the historical local fullbody checkpoint path.  The
    # upstream MyoFullBody env can disable any of these via checkpoint metadata.
    "enable_joint_pos_observations": True,
    "enable_joint_vel_observations": True,
    "enable_muscle_length_observations": True,
    "enable_muscle_velocity_observations": True,
    "enable_muscle_force_observations": True,
    "enable_muscle_excitation_observations": True,
    "enable_muscle_activation_observations": True,
    "enable_touch_sensor_observations": True,
}


def fullbody_obs_adapter_params_from_metadata(
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Return ``FullbodyObsAdapter`` kwargs encoded in checkpoint metadata.

    MuscleMimic stores goal settings under ``experiment.env_params.goal_params``
    and the non-goal observation toggles as sibling ``env_params`` keys.  Passing
    only ``goal_params`` happens to work for ``mm-fullbody-base`` because it uses
    every muscle observation, but it is not enough to reproduce variants that
    change the observation specification.
    """
    experiment = metadata.get("experiment", {})
    env_params = experiment.get("env_params", {})
    params = dict(env_params.get("goal_params", {}) or {})
    for key in _FULLBODY_OBS_FLAG_DEFAULTS:
        if key in env_params:
            params[key] = bool(env_params[key])
    return params


def fullbody_history_settings_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return upstream MuJoCo inference history settings from metadata."""
    experiment = metadata.get("experiment", {})
    return {
        "len_obs_history": int(experiment.get("len_obs_history", 1) or 1),
        "split_goal": bool(experiment.get("split_goal", False)),
    }


class ObservationHistoryBuffer:
    """Upstream-compatible observation history for single-env inference.

    Matches ``musclemimic.algorithms.ppo.inference.ObservationHistoryBuffer``.
    When ``split_goal=True``, only state indices are stacked and goal indices are
    taken from the current raw observation.
    """

    def __init__(
        self,
        n_steps: int,
        *,
        split_goal: bool = False,
        state_indices: np.ndarray | None = None,
        goal_indices: np.ndarray | None = None,
    ) -> None:
        self.n_steps = int(n_steps)
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}.")
        self.split_goal = bool(split_goal)
        self.state_indices = (
            None if state_indices is None else np.asarray(state_indices, dtype=int)
        )
        self.goal_indices = (
            None if goal_indices is None else np.asarray(goal_indices, dtype=int)
        )
        self._buffer: np.ndarray | None = None
        if self.split_goal and (
            self.state_indices is None or self.goal_indices is None
        ):
            raise ValueError("split_goal=True requires state_indices and goal_indices.")

    def reset(self, obs: np.ndarray) -> np.ndarray:
        """Initialize history with zeros and place ``obs`` in the latest slot."""
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if self.split_goal:
            assert self.state_indices is not None
            assert self.goal_indices is not None
            state_obs = obs[self.state_indices]
            goal_obs = obs[self.goal_indices]
            self._buffer = np.zeros((self.n_steps, state_obs.shape[0]), dtype=obs.dtype)
            self._buffer[-1] = state_obs
            return np.concatenate([self._buffer.reshape(-1), goal_obs]).astype(
                np.float32
            )

        self._buffer = np.zeros((self.n_steps, obs.shape[0]), dtype=obs.dtype)
        self._buffer[-1] = obs
        return self._buffer.reshape(-1).astype(np.float32)

    def step(self, obs: np.ndarray) -> np.ndarray:
        """Roll history left and append ``obs``."""
        if self._buffer is None:
            return self.reset(obs)
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if self.split_goal:
            assert self.state_indices is not None
            assert self.goal_indices is not None
            state_obs = obs[self.state_indices]
            goal_obs = obs[self.goal_indices]
            self._buffer = np.roll(self._buffer, shift=-1, axis=0)
            self._buffer[-1] = state_obs
            return np.concatenate([self._buffer.reshape(-1), goal_obs]).astype(
                np.float32
            )

        self._buffer = np.roll(self._buffer, shift=-1, axis=0)
        self._buffer[-1] = obs
        return self._buffer.reshape(-1).astype(np.float32)

    def clear(self) -> None:
        """Clear buffered state so the next observation is treated as reset."""
        self._buffer = None


def has_local_policy_artifacts(checkpoint_root: Path) -> bool:
    """Return ``True`` if checkpoint root looks loadable for local policy."""
    p = Path(checkpoint_root)
    return (p / "train_state").is_dir() and (p / "config" / "metadata").exists()


def load_local_policy_artifacts(checkpoint_root: Path) -> LocalPolicyArtifacts:
    """Load policy parameters and normalization stats from Orbax artifacts."""
    p = Path(checkpoint_root)
    train_state_dir = p / "train_state"
    if not train_state_dir.is_dir():
        raise FileNotFoundError(f"train_state directory missing: {train_state_dir}")
    try:
        import orbax.checkpoint as ocp
    except ImportError as err:
        raise ImportError(
            "orbax.checkpoint is required for local policy inference."
        ) from err

    restored = ocp.StandardCheckpointer().restore(str(train_state_dir))
    params_raw = restored["params"]
    run_stats = restored["run_stats"]["RunningMeanStd_0"]
    params = _to_numpy_tree(params_raw)
    obs_mean = np.asarray(run_stats["mean"], dtype=np.float32)
    obs_var = np.asarray(run_stats["var"], dtype=np.float32)
    obs_count = np.asarray(run_stats.get("count", 1e-6), dtype=np.float32)
    actor = params["actor"]
    if "output" in actor:
        action_dim = int(np.asarray(actor["output"]["bias"]).shape[0])
        obs_dim = int(np.asarray(actor["block0_layer0_dense"]["kernel"]).shape[0])
    else:
        dense_indices = _sorted_prefix_indices(actor, "Dense_")
        if not dense_indices:
            raise ValueError("Unsupported actor params: no Dense_* layers found.")
        first_dense = actor[f"Dense_{dense_indices[0]}"]
        last_dense = actor[f"Dense_{dense_indices[-1]}"]
        obs_dim = int(np.asarray(first_dense["kernel"]).shape[0])
        action_dim = int(np.asarray(last_dense["bias"]).shape[0])
    return LocalPolicyArtifacts(
        params=params,
        obs_mean=obs_mean,
        obs_var=obs_var,
        obs_count=obs_count,
        obs_dim=obs_dim,
        action_dim=action_dim,
    )


def _actor_forward(params: dict[str, Any], obs: np.ndarray) -> np.ndarray:
    actor = params["actor"]
    if "output" not in actor:
        dense_indices = _sorted_prefix_indices(actor, "Dense_")
        if not dense_indices:
            raise ValueError("Unsupported actor params: no Dense_* layers found.")
        hidden_indices = dense_indices[:-1]
        out_idx = dense_indices[-1]
        h = obs
        for idx in hidden_indices:
            dense = actor[f"Dense_{idx}"]
            h = _dense(h, dense["kernel"], dense["bias"])
            ln = actor.get(f"LayerNorm_{idx}")
            if ln is not None:
                h = _layer_norm(h, ln["scale"], ln["bias"])
            h = _silu(h)
        out = actor[f"Dense_{out_idx}"]
        return _dense(h, out["kernel"], out["bias"]).astype(np.float32)

    h = obs
    block_indices = _residual_block_indices(actor)
    if not block_indices:
        raise ValueError("Unsupported actor params: no residual blocks found.")
    for block_idx in block_indices:
        dense0 = actor[f"block{block_idx}_layer0_dense"]
        ln0 = actor[f"block{block_idx}_layer0_ln"]
        y = _dense(h, dense0["kernel"], dense0["bias"])
        y = _silu(_layer_norm(y, ln0["scale"], ln0["bias"]))

        dense1 = actor[f"block{block_idx}_layer1_dense"]
        ln1 = actor[f"block{block_idx}_layer1_ln"]
        y = _dense(y, dense1["kernel"], dense1["bias"])
        y = _layer_norm(y, ln1["scale"], ln1["bias"])

        proj = actor.get(f"block{block_idx}_proj")
        shortcut = h if proj is None else _dense(h, proj["kernel"], proj["bias"])
        gate_raw = np.asarray(actor[f"res_gate_{block_idx}"], dtype=np.float32).reshape(
            ()
        )
        gate = 1.0 / (1.0 + np.exp(-float(gate_raw)))
        h = _silu(shortcut + gate * y)

    tail = actor.get("tail_dense")
    tail_ln = actor.get("tail_ln")
    if tail is not None:
        if tail_ln is None:
            raise ValueError("Unsupported actor params: tail_dense without tail_ln.")
        h = _dense(h, tail["kernel"], tail["bias"])
        h = _silu(_layer_norm(h, tail_ln["scale"], tail_ln["bias"]))
    out = actor["output"]
    return _dense(h, out["kernel"], out["bias"]).astype(np.float32)


def _pack_features_for_policy(
    data: mujoco.MjData,
    clip: MotionClip,
    frame_idx: int,
    obs_dim: int,
) -> np.ndarray:
    """Build a deterministic feature vector and fit to policy obs_dim."""
    qpos = np.asarray(data.qpos, dtype=np.float32)
    qvel = np.asarray(data.qvel, dtype=np.float32)
    act = np.asarray(data.act, dtype=np.float32)
    target_qpos = np.asarray(clip.qpos[frame_idx], dtype=np.float32)
    target_qvel = (
        np.asarray(clip.qvel[frame_idx], dtype=np.float32)
        if clip.qvel is not None and frame_idx < clip.qvel.shape[0]
        else np.zeros_like(qvel)
    )
    features = [
        qpos,
        qvel,
        act,
        target_qpos,
        target_qvel,
        target_qpos - qpos,
        target_qvel - qvel,
    ]
    # Add short lookahead target qpos context.
    n_frames = int(clip.qpos.shape[0])
    for k in (1, 2, 3, 4, 5):
        idx = min(n_frames - 1, frame_idx + k)
        features.append(np.asarray(clip.qpos[idx], dtype=np.float32))

    vec = np.concatenate(features, axis=0).astype(np.float32)
    if vec.shape[0] >= obs_dim:
        return vec[:obs_dim]
    out = np.zeros((obs_dim,), dtype=np.float32)
    out[: vec.shape[0]] = vec
    return out


def _joint_qpos_size(jnt_type: int) -> int:
    if jnt_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7
    if jnt_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return 4
    return 1


def _joint_qvel_size(jnt_type: int) -> int:
    if jnt_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return 6
    if jnt_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return 3
    return 1


def _calc_site_velocities_for_data(
    *,
    site_ids: np.ndarray,
    site_xpos: np.ndarray,
    cvel_parent: np.ndarray,
    subtree_com_root: np.ndarray,
    site_bodyid: np.ndarray,
    body_rootid: np.ndarray,
) -> np.ndarray:
    """Approximate upstream calc_site_velocities for a site batch."""
    parent_body_id = site_bodyid[site_ids]
    root_body_id = body_rootid[parent_body_id]
    body_cvel = cvel_parent[parent_body_id]
    root_com = subtree_com_root[root_body_id]
    rpos = site_xpos[site_ids] - root_com
    lin_vel = body_cvel[:, 3:] - np.cross(rpos, body_cvel[:, :3], axis=-1)
    rot_vel = body_cvel[:, :3]
    return np.hstack([rot_vel, lin_vel])


def _relative_site_quantities(
    *,
    site_ids: np.ndarray,
    site_xpos: np.ndarray,
    site_xmat: np.ndarray,
    cvel_parent: np.ndarray,
    subtree_com_root: np.ndarray,
    site_bodyid: np.ndarray,
    body_rootid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Port of upstream relative site quantities (numpy path)."""
    site_vel = _calc_site_velocities_for_data(
        site_ids=site_ids,
        site_xpos=site_xpos,
        cvel_parent=cvel_parent,
        subtree_com_root=subtree_com_root,
        site_bodyid=site_bodyid,
        body_rootid=body_rootid,
    )
    main_site_id = 0
    main_pos = site_xpos[site_ids[main_site_id]]
    main_mat = site_xmat[site_ids[main_site_id]].reshape(3, 3)
    main_vel = site_vel[main_site_id]

    other_site_ids = site_ids[1:]
    other_pos = site_xpos[other_site_ids]
    other_mat = site_xmat[other_site_ids].reshape(-1, 3, 3)
    other_vel = site_vel[1:]

    site_rpos = other_pos - main_pos
    rel_rot = np.einsum("ik,nkj->nij", main_mat.T, other_mat)
    site_rangles = np_R.from_matrix(rel_rot).as_rotvec()
    rel_lin = np.einsum("jk,nk->nj", main_mat, main_vel[3:] - other_vel[:, 3:])
    other_ang_in_main = np.einsum("nkj,nk->nj", rel_rot, other_vel[:, :3])
    rel_ang = other_ang_in_main - main_vel[:3]
    site_rvel = np.hstack([rel_ang, rel_lin])
    return site_rpos, site_rangles, site_rvel


@dataclass(frozen=True)
class _TrajectoryGoalSpec:
    n_step_lookahead: int
    n_step_stride: int
    enable_motion_phase: bool
    use_concise_lookahead: bool
    enable_mimic_site_rpos_observations: bool
    sites_for_mimic: tuple[str, ...]


class FullbodyObsAdapter:
    """Upstream-compatible fullbody observation builder for local inference."""

    def __init__(
        self,
        model: mujoco.MjModel,
        clip: MotionClip,
        goal_params: dict[str, Any] | None = None,
    ) -> None:
        self._model = model
        self._clip = clip
        gp = dict(goal_params or {})
        self._goal = _TrajectoryGoalSpec(
            n_step_lookahead=int(gp.get("n_step_lookahead", 5)),
            n_step_stride=int(gp.get("n_step_stride", 1)),
            enable_motion_phase=bool(gp.get("enable_motion_phase", True)),
            use_concise_lookahead=bool(gp.get("use_concise_lookahead", False)),
            enable_mimic_site_rpos_observations=bool(
                gp.get("enable_mimic_site_rpos_observations", True)
            ),
            sites_for_mimic=tuple(gp.get("sites_for_mimic", ())),
        )
        if not self._goal.sites_for_mimic:
            raise ValueError("goal_params.sites_for_mimic must be provided.")
        self._obs_flags = {
            key: bool(gp.get(key, default))
            for key, default in _FULLBODY_OBS_FLAG_DEFAULTS.items()
        }

        root_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        if root_jid < 0:
            raise ValueError("Root free joint 'root' not found in model.")
        self._root_qpos_adr = int(model.jnt_qposadr[root_jid])
        self._root_qvel_adr = int(model.jnt_dofadr[root_jid])
        self._root_qpos_idx_full = np.arange(
            self._root_qpos_adr, self._root_qpos_adr + 7
        )
        self._root_qvel_idx_full = np.arange(
            self._root_qvel_adr, self._root_qvel_adr + 6
        )
        self._root_qpos_idx_xyz = self._root_qpos_idx_full[:3]

        qpos_indices: list[np.ndarray] = []
        qvel_indices: list[np.ndarray] = []
        qpos_non_root: list[np.ndarray] = []
        qvel_non_root: list[np.ndarray] = []
        for jid in range(model.njnt):
            jtype = int(model.jnt_type[jid])
            qpos_size = _joint_qpos_size(jtype)
            qvel_size = _joint_qvel_size(jtype)
            qpos_adr = int(model.jnt_qposadr[jid])
            qvel_adr = int(model.jnt_dofadr[jid])
            qpos_idx = np.arange(qpos_adr, qpos_adr + qpos_size)
            qvel_idx = np.arange(qvel_adr, qvel_adr + qvel_size)
            if jid == root_jid:
                qpos_idx = qpos_idx[2:]
            else:
                qpos_non_root.append(np.arange(qpos_adr, qpos_adr + qpos_size))
                qvel_non_root.append(np.arange(qvel_adr, qvel_adr + qvel_size))
            qpos_indices.append(qpos_idx)
            qvel_indices.append(qvel_idx)
        self._qpos_ind = np.concatenate(qpos_indices).astype(np.int32)
        self._qvel_ind = np.concatenate(qvel_indices).astype(np.int32)
        self._qpos_non_root_ind = np.concatenate(qpos_non_root).astype(np.int32)
        self._qvel_non_root_ind = np.concatenate(qvel_non_root).astype(np.int32)

        self._actuator_ids = np.arange(model.nu, dtype=np.int32)

        touch_names = ("r_foot", "r_toes", "l_foot", "l_toes")
        touch_ids: list[int] = []
        if self._obs_flags["enable_touch_sensor_observations"]:
            for name in touch_names:
                sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
                if sid >= 0:
                    touch_ids.append(int(sid))
        self._touch_sensor_ids = np.asarray(touch_ids, dtype=np.int32)

        site_ids: list[int] = []
        for name in self._goal.sites_for_mimic:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            if sid < 0:
                raise ValueError(f"Mimic site not found in model: {name}")
            site_ids.append(int(sid))
        self._site_ids = np.asarray(site_ids, dtype=np.int32)

        npz = np.load(self._clip.source_path, allow_pickle=True)
        self._traj_site_xpos = np.asarray(npz["site_xpos"], dtype=np.float32)
        self._traj_site_xmat = np.asarray(npz["site_xmat"], dtype=np.float32)
        self._traj_cvel = np.asarray(npz["cvel"], dtype=np.float32)
        self._traj_subtree_com = np.asarray(npz["subtree_com"], dtype=np.float32)
        self._traj_site_bodyid = np.asarray(npz["site_bodyid"], dtype=np.int32)
        self._traj_body_rootid = np.asarray(npz["body_rootid"], dtype=np.int32)
        traj_site_names = tuple(str(x) for x in np.asarray(npz["site_names"]))
        name_to_traj_idx = {name: i for i, name in enumerate(traj_site_names)}
        try:
            self._traj_site_ids = np.asarray(
                [name_to_traj_idx[n] for n in self._goal.sites_for_mimic],
                dtype=np.int32,
            )
        except KeyError as err:
            raise ValueError(
                f"Trajectory site mapping missing for mimic site: {err.args[0]}"
            ) from err

        self._sim_site_bodyid = np.asarray(model.site_bodyid, dtype=np.int32)
        self._sim_body_rootid = np.asarray(model.body_rootid, dtype=np.int32)
        self._traj_len = int(self._clip.qpos.shape[0])
        self.goal_dim = self._compute_goal_dim()

    def with_clip(self, clip: MotionClip) -> FullbodyObsAdapter:
        """New adapter sharing goal layout and model indices but *clip* trajectory.

        Use when the policy is rolled out on a different :class:`MotionClip` than
        the one this adapter was constructed with (same ``sites_for_mimic`` /
        lookahead settings, new NPZ trajectory buffers).
        """
        g = self._goal
        gp: dict[str, Any] = {
            "n_step_lookahead": g.n_step_lookahead,
            "n_step_stride": g.n_step_stride,
            "enable_motion_phase": g.enable_motion_phase,
            "use_concise_lookahead": g.use_concise_lookahead,
            "enable_mimic_site_rpos_observations": g.enable_mimic_site_rpos_observations,
            "sites_for_mimic": list(g.sites_for_mimic),
        }
        gp.update(self._obs_flags)
        return FullbodyObsAdapter(self._model, clip, gp)

    def _compute_goal_dim(self) -> int:
        """Return the stateful goal observation width for split-goal history."""
        g = self._goal
        n_relative_sites = len(g.sites_for_mimic) - 1
        site_rpos_dim = 3 * n_relative_sites
        site_full_dim = (3 + 3 + 6) * n_relative_sites

        if g.use_concise_lookahead:
            traj_step_dim = 3 + 6 + site_rpos_dim
            traj_goal_dim = site_rpos_dim
            if g.n_step_lookahead > 1:
                traj_goal_dim += traj_step_dim * (g.n_step_lookahead - 1)
        else:
            traj_step_dim = len(self._qpos_ind) + len(self._qvel_ind) + site_full_dim
            traj_goal_dim = traj_step_dim * g.n_step_lookahead

        if g.enable_mimic_site_rpos_observations:
            current_site_dim = site_full_dim
        else:
            current_site_dim = (3 + 6) * n_relative_sites
        phase_dim = 1 if g.enable_motion_phase else 0
        return int(current_site_dim + traj_goal_dim + phase_dim)

    def goal_indices_for_obs_dim(self, obs_dim: int) -> np.ndarray:
        """Return indices occupied by the goal group in this adapter output."""
        if self.goal_dim <= 0 or self.goal_dim > obs_dim:
            raise ValueError(
                f"goal_dim {self.goal_dim} is incompatible with obs_dim {obs_dim}."
            )
        return np.arange(obs_dim - self.goal_dim, obs_dim, dtype=int)

    def state_indices_for_obs_dim(self, obs_dim: int) -> np.ndarray:
        """Return non-goal indices for ``split_goal`` observation history."""
        goal_indices = self.goal_indices_for_obs_dim(obs_dim)
        mask = np.ones(obs_dim, dtype=bool)
        mask[goal_indices] = False
        return np.arange(obs_dim, dtype=int)[mask]

    def _traj_goal_obs(self, frame_idx: int) -> np.ndarray:
        g = self._goal
        if g.use_concise_lookahead:
            ref_idx = int(frame_idx)
            ref_qpos = np.asarray(self._clip.qpos[ref_idx], dtype=np.float32)
            ref_qvel = (
                np.asarray(self._clip.qvel[ref_idx], dtype=np.float32)
                if self._clip.qvel is not None
                else np.zeros((self._model.nv,), dtype=np.float32)
            )
            ref_root_pos = ref_qpos[self._root_qpos_idx_xyz]
            ref_root_vel = ref_qvel[self._root_qvel_idx_full]

            goal_site_rpos_all: list[np.ndarray] = []
            root_pos_deltas: list[np.ndarray] = []
            root_vel_deltas: list[np.ndarray] = []
            for step_offset in range(g.n_step_lookahead):
                future = min(
                    self._traj_len - 1,
                    frame_idx + step_offset * g.n_step_stride,
                )
                site_rpos, _site_rangles, _site_rvel = _relative_site_quantities(
                    site_ids=self._traj_site_ids,
                    site_xpos=self._traj_site_xpos[future],
                    site_xmat=self._traj_site_xmat[future],
                    cvel_parent=self._traj_cvel[future],
                    subtree_com_root=self._traj_subtree_com[future],
                    site_bodyid=self._traj_site_bodyid,
                    body_rootid=self._traj_body_rootid,
                )
                goal_site_rpos_all.append(site_rpos)
                if step_offset > 0:
                    qpos = np.asarray(self._clip.qpos[future], dtype=np.float32)
                    qvel = (
                        np.asarray(self._clip.qvel[future], dtype=np.float32)
                        if self._clip.qvel is not None
                        else np.zeros((self._model.nv,), dtype=np.float32)
                    )
                    root_pos_deltas.append(qpos[self._root_qpos_idx_xyz] - ref_root_pos)
                    root_vel_deltas.append(
                        qvel[self._root_qvel_idx_full] - ref_root_vel
                    )

            comps: list[np.ndarray] = [goal_site_rpos_all[0].reshape(-1)]
            for i in range(len(root_pos_deltas)):
                comps.append(root_pos_deltas[i])
                comps.append(root_vel_deltas[i])
                comps.append(goal_site_rpos_all[i + 1].reshape(-1))
            return np.concatenate(comps).astype(np.float32)

        # Full lookahead path (unused by current fullbody config).
        all_qpos: list[np.ndarray] = []
        all_qvel: list[np.ndarray] = []
        all_site_rpos: list[np.ndarray] = []
        all_site_rangles: list[np.ndarray] = []
        all_site_rvel: list[np.ndarray] = []
        for step_offset in range(g.n_step_lookahead):
            future = min(self._traj_len - 1, frame_idx + step_offset * g.n_step_stride)
            qpos = np.asarray(self._clip.qpos[future], dtype=np.float32)
            qvel = (
                np.asarray(self._clip.qvel[future], dtype=np.float32)
                if self._clip.qvel is not None
                else np.zeros((self._model.nv,), dtype=np.float32)
            )
            site_rpos, site_rangles, site_rvel = _relative_site_quantities(
                site_ids=self._traj_site_ids,
                site_xpos=self._traj_site_xpos[future],
                site_xmat=self._traj_site_xmat[future],
                cvel_parent=self._traj_cvel[future],
                subtree_com_root=self._traj_subtree_com[future],
                site_bodyid=self._traj_site_bodyid,
                body_rootid=self._traj_body_rootid,
            )
            all_qpos.append(qpos[self._qpos_ind])
            all_qvel.append(qvel[self._qvel_ind])
            all_site_rpos.append(site_rpos)
            all_site_rangles.append(site_rangles)
            all_site_rvel.append(site_rvel)
        return np.concatenate(
            [
                np.concatenate(all_qpos),
                np.concatenate(all_qvel),
                np.concatenate(all_site_rpos).reshape(-1),
                np.concatenate(all_site_rangles).reshape(-1),
                np.concatenate(all_site_rvel).reshape(-1),
            ]
        ).astype(np.float32)

    def build(self, data: mujoco.MjData, frame_idx: int) -> np.ndarray:
        g = self._goal
        obs: list[np.ndarray] = []
        root_qpos = np.asarray(data.qpos[self._root_qpos_idx_full], dtype=np.float32)
        root_qvel = np.asarray(data.qvel[self._root_qvel_idx_full], dtype=np.float32)
        if self._obs_flags["enable_joint_pos_observations"]:
            obs.append(root_qpos[2:])
            obs.append(np.asarray(data.qpos[self._qpos_non_root_ind], dtype=np.float32))
        if self._obs_flags["enable_joint_vel_observations"]:
            obs.append(root_qvel)
            obs.append(np.asarray(data.qvel[self._qvel_non_root_ind], dtype=np.float32))

        for act_idx in self._actuator_ids:
            if self._obs_flags["enable_muscle_length_observations"]:
                obs.append(
                    np.asarray([data.actuator_length[act_idx]], dtype=np.float32)
                )
            if self._obs_flags["enable_muscle_velocity_observations"]:
                obs.append(
                    np.asarray([data.actuator_velocity[act_idx]], dtype=np.float32)
                )
            if self._obs_flags["enable_muscle_force_observations"]:
                obs.append(np.asarray([data.actuator_force[act_idx]], dtype=np.float32))
            if self._obs_flags["enable_muscle_excitation_observations"]:
                obs.append(np.asarray([data.ctrl[act_idx]], dtype=np.float32))
            if self._obs_flags["enable_muscle_activation_observations"]:
                obs.append(np.asarray([data.act[act_idx]], dtype=np.float32))

        if self._touch_sensor_ids.size:
            sens = np.asarray(data.sensordata, dtype=np.float32)
            for sid in self._touch_sensor_ids:
                adr = int(self._model.sensor_adr[sid])
                dim = int(self._model.sensor_dim[sid])
                val = sens[adr : adr + dim]
                obs.append(np.asarray([float(np.sum(val))], dtype=np.float32))

        site_rpos, site_rangles, site_rvel = _relative_site_quantities(
            site_ids=self._site_ids,
            site_xpos=np.asarray(data.site_xpos, dtype=np.float32),
            site_xmat=np.asarray(data.site_xmat, dtype=np.float32),
            cvel_parent=np.asarray(data.cvel, dtype=np.float32),
            subtree_com_root=np.asarray(data.subtree_com, dtype=np.float32),
            site_bodyid=self._sim_site_bodyid,
            body_rootid=self._sim_body_rootid,
        )
        traj_goal_obs = self._traj_goal_obs(frame_idx)
        goal_comps: list[np.ndarray] = []
        if g.enable_mimic_site_rpos_observations:
            goal_comps.append(site_rpos.reshape(-1))
        goal_comps.extend(
            [site_rangles.reshape(-1), site_rvel.reshape(-1), traj_goal_obs]
        )
        if g.enable_motion_phase:
            motion_phase = float(frame_idx) / float(max(self._traj_len, 1))
            goal_comps.append(np.asarray([motion_phase], dtype=np.float32))

        obs.append(np.concatenate(goal_comps).astype(np.float32))
        return np.concatenate(obs).astype(np.float32)


@dataclass
class LocalPolicyRunner:
    """Local actor inference + MuJoCo stepping helper."""

    artifacts: LocalPolicyArtifacts
    stochastic: bool
    seed: int
    frame_skip: int = 5
    obs_adapter: FullbodyObsAdapter | None = None
    len_obs_history: int = 1
    split_goal: bool = False
    goal_indices: np.ndarray | None = None
    state_indices: np.ndarray | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(int(self.seed))
        self._history: ObservationHistoryBuffer | None = None
        self._last_frame_idx: int | None = None
        if int(self.len_obs_history) > 1:
            if (
                self.split_goal
                and self.goal_indices is None
                and self.state_indices is None
                and self.obs_adapter is not None
            ):
                goal_dim = int(self.obs_adapter.goal_dim)
                history_steps = int(self.len_obs_history)
                stacked_state_dim = self.artifacts.obs_dim - goal_dim
                if stacked_state_dim % history_steps != 0:
                    raise ValueError(
                        "Checkpoint obs_dim is incompatible with split-goal "
                        f"history: obs_dim={self.artifacts.obs_dim}, "
                        f"goal_dim={goal_dim}, len_obs_history={history_steps}."
                    )
                raw_obs_dim = int(stacked_state_dim // history_steps + goal_dim)
                self.goal_indices = self.obs_adapter.goal_indices_for_obs_dim(
                    raw_obs_dim
                )
                self.state_indices = self.obs_adapter.state_indices_for_obs_dim(
                    raw_obs_dim
                )
            self._history = ObservationHistoryBuffer(
                int(self.len_obs_history),
                split_goal=bool(self.split_goal),
                state_indices=self.state_indices,
                goal_indices=self.goal_indices,
            )
        self.reset()

    def reset(self) -> None:
        """Reset running-normalizer and observation-history inference state."""
        self._run_mean = np.asarray(self.artifacts.obs_mean, dtype=np.float32).copy()
        self._run_var = np.asarray(self.artifacts.obs_var, dtype=np.float32).copy()
        self._run_count = np.asarray(self.artifacts.obs_count, dtype=np.float32).copy()
        if self._history is not None:
            self._history.clear()
        self._last_frame_idx = None

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """RunningMeanStd update + normalization (upstream-compatible)."""
        normalized, self._run_mean, self._run_var, self._run_count = (
            running_mean_std_update(
                obs=obs,
                mean=self._run_mean,
                var=self._run_var,
                count=self._run_count,
            )
        )
        return normalized

    def _apply_history(self, obs: np.ndarray, frame_idx: int) -> np.ndarray:
        if self._history is None:
            return np.asarray(obs, dtype=np.float32)
        if self._last_frame_idx is None or frame_idx <= self._last_frame_idx:
            out = self._history.reset(obs)
        else:
            out = self._history.step(obs)
        self._last_frame_idx = int(frame_idx)
        return out

    def _build_policy_obs(
        self,
        data: mujoco.MjData,
        clip: MotionClip,
        frame_idx: int,
        *,
        obs_adapter: FullbodyObsAdapter | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        adapter = obs_adapter if obs_adapter is not None else self.obs_adapter
        if adapter is not None:
            raw_obs = adapter.build(data, frame_idx)
            policy_obs = self._apply_history(raw_obs, frame_idx)
            if policy_obs.shape[0] != self.artifacts.obs_dim:
                raise ValueError(
                    "Upstream obs adapter dim mismatch: "
                    f"{policy_obs.shape[0]} != checkpoint {self.artifacts.obs_dim}"
                )
        else:
            raw_obs = _pack_features_for_policy(
                data, clip, frame_idx, self.artifacts.obs_dim
            )
            policy_obs = raw_obs
        return (
            np.asarray(raw_obs, dtype=np.float32),
            np.asarray(policy_obs, dtype=np.float32),
        )

    def action_trace_for(
        self,
        data: mujoco.MjData,
        clip: MotionClip,
        frame_idx: int,
        *,
        obs_adapter: FullbodyObsAdapter | None = None,
    ) -> LocalPolicyActionTrace:
        """Return action and intermediate tensors for one inference step.

        This method intentionally mutates the runner exactly like
        :meth:`action_for`: history, running mean/std, and stochastic RNG state
        advance once.
        """
        raw_obs, policy_obs = self._build_policy_obs(
            data,
            clip,
            frame_idx,
            obs_adapter=obs_adapter,
        )
        mean_before = self._run_mean.copy()
        var_before = self._run_var.copy()
        count_before = self._run_count.copy()
        norm_obs = self._normalize_obs(policy_obs)
        mean_action = _actor_forward(self.artifacts.params, norm_obs)
        if self.stochastic and "log_std" in self.artifacts.params:
            log_std = np.asarray(self.artifacts.params["log_std"], dtype=np.float32)
            std = np.exp(log_std)
            action = (
                mean_action
                + self._rng.normal(0.0, 1.0, size=mean_action.shape).astype(np.float32)
                * std
            )
        else:
            action = mean_action
        action = np.asarray(np.clip(action, -1.0, 1.0), dtype=np.float32)
        return LocalPolicyActionTrace(
            raw_obs=raw_obs,
            policy_obs=policy_obs,
            norm_obs=np.asarray(norm_obs, dtype=np.float32),
            mean_action=np.asarray(mean_action, dtype=np.float32),
            action=action,
            run_mean_before=mean_before,
            run_var_before=var_before,
            run_count_before=count_before,
            run_mean_after=self._run_mean.copy(),
            run_var_after=self._run_var.copy(),
            run_count_after=self._run_count.copy(),
        )

    def action_for(
        self,
        data: mujoco.MjData,
        clip: MotionClip,
        frame_idx: int,
        *,
        obs_adapter: FullbodyObsAdapter | None = None,
    ) -> np.ndarray:
        return self.action_trace_for(
            data,
            clip,
            frame_idx,
            obs_adapter=obs_adapter,
        ).action

    def step(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        action: np.ndarray,
    ) -> None:
        ctrl = np.asarray(action, dtype=np.float32).reshape(-1)
        if ctrl.shape[0] != model.nu:
            raise ValueError(
                f"Policy action dim {ctrl.shape[0]} does not match model.nu {model.nu}."
            )
        ctrl_range = np.asarray(model.actuator_ctrlrange, dtype=np.float32)
        ctrl = np.clip(ctrl, ctrl_range[:, 0], ctrl_range[:, 1])
        data.ctrl[:] = ctrl
        n_substeps = max(1, int(self.frame_skip))
        for _ in range(n_substeps):
            mujoco.mj_step(model, data)


def read_checkpoint_config_metadata(checkpoint_root: Path) -> dict[str, Any]:
    """Read raw ``config/metadata`` JSON for diagnostics."""
    p = Path(checkpoint_root) / "config" / "metadata"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


@dataclass
class OnnxPolicyRunner:
    """Inference runner backed by an ONNX model instead of Orbax/numpy weights.

    Obs normalization is baked into the ONNX graph, so raw observations are
    passed directly to the session.  All other behaviour (obs building,
    MuJoCo stepping) matches ``LocalPolicyRunner``.

    Requires: ``pip install onnxruntime``

    Example::

        from myosuite.integrations.musclemimic.fullbody_local_policy import (
            OnnxPolicyRunner,
        )

        runner = OnnxPolicyRunner(
            onnx_path="mm-10m-2.onnx",
            obs_dim=2418,
            action_dim=354,
            obs_adapter=adapter,   # FullbodyObsAdapter instance
        )
        action = runner.action_for(data, clip, frame_idx)
        runner.step(model, data, action)
    """

    onnx_path: str | Path
    obs_dim: int
    action_dim: int
    frame_skip: int = 5
    obs_adapter: FullbodyObsAdapter | None = None
    len_obs_history: int = 1
    split_goal: bool = False
    goal_indices: np.ndarray | None = None
    state_indices: np.ndarray | None = None

    def __post_init__(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as err:
            raise ImportError(
                "onnxruntime is required for OnnxPolicyRunner. "
                "Install with: pip install onnxruntime"
            ) from err
        self._session = ort.InferenceSession(
            str(self.onnx_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_name: str = self._session.get_inputs()[0].name
        self._output_name: str = self._session.get_outputs()[0].name
        graph_obs_dim = self._session.get_inputs()[0].shape[1]
        if isinstance(graph_obs_dim, int) and graph_obs_dim != int(self.obs_dim):
            raise ValueError(
                f"ONNX input dim {graph_obs_dim} does not match obs_dim {self.obs_dim}."
            )

        self._history: ObservationHistoryBuffer | None = None
        self._last_frame_idx: int | None = None
        if int(self.len_obs_history) > 1:
            if self.obs_adapter is None:
                raise ValueError("len_obs_history > 1 requires obs_adapter.")
            if (
                self.split_goal
                and self.goal_indices is None
                and self.state_indices is None
            ):
                goal_dim = int(self.obs_adapter.goal_dim)
                history_steps = int(self.len_obs_history)
                stacked_state_dim = int(self.obs_dim) - goal_dim
                if stacked_state_dim % history_steps != 0:
                    raise ValueError(
                        "ONNX obs_dim is incompatible with split-goal history: "
                        f"obs_dim={self.obs_dim}, goal_dim={goal_dim}, "
                        f"len_obs_history={history_steps}."
                    )
                raw_obs_dim = int(stacked_state_dim // history_steps + goal_dim)
                self.goal_indices = self.obs_adapter.goal_indices_for_obs_dim(
                    raw_obs_dim
                )
                self.state_indices = self.obs_adapter.state_indices_for_obs_dim(
                    raw_obs_dim
                )
            self._history = ObservationHistoryBuffer(
                int(self.len_obs_history),
                split_goal=bool(self.split_goal),
                state_indices=self.state_indices,
                goal_indices=self.goal_indices,
            )
        self.reset()

    def reset(self) -> None:
        """Reset observation-history inference state."""
        if self._history is not None:
            self._history.clear()
        self._last_frame_idx = None

    def _apply_history(self, obs: np.ndarray, frame_idx: int) -> np.ndarray:
        if self._history is None:
            return np.asarray(obs, dtype=np.float32)
        if self._last_frame_idx is None or frame_idx <= self._last_frame_idx:
            out = self._history.reset(obs)
        else:
            out = self._history.step(obs)
        self._last_frame_idx = int(frame_idx)
        return out

    def action_for(
        self,
        data: mujoco.MjData,
        clip: MotionClip,
        frame_idx: int,
        *,
        obs_adapter: FullbodyObsAdapter | None = None,
    ) -> np.ndarray:
        """Build observation and run ONNX inference.

        Args:
            data: Current MuJoCo simulation state.
            clip: Motion clip providing reference trajectory.
            frame_idx: Current frame index into the clip.
            obs_adapter: Optional override; defaults to this runner's adapter.

        Returns:
            Action array of shape ``(action_dim,)`` clipped to ``[-1, 1]``.
        """
        adapter = obs_adapter if obs_adapter is not None else self.obs_adapter
        if adapter is not None:
            raw_obs = adapter.build(data, frame_idx)
            obs = self._apply_history(raw_obs, frame_idx)
        else:
            obs = _pack_features_for_policy(data, clip, frame_idx, self.obs_dim)
        if obs.shape[0] != self.obs_dim:
            raise ValueError(
                f"ONNX observation dim {obs.shape[0]} does not match obs_dim {self.obs_dim}."
            )

        obs_f32 = np.asarray(obs, dtype=np.float32)[None]  # (1, obs_dim)
        action = self._session.run([self._output_name], {self._input_name: obs_f32})[
            0
        ].squeeze(0)
        return np.asarray(action, dtype=np.float32)

    def step(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        action: np.ndarray,
    ) -> None:
        """Apply action to MuJoCo and advance by ``frame_skip`` substeps.

        Args:
            model: MuJoCo model.
            data: MuJoCo data to step in-place.
            action: Control vector of shape ``(nu,)``.
        """
        ctrl = np.asarray(action, dtype=np.float32).reshape(-1)
        if ctrl.shape[0] != model.nu:
            raise ValueError(
                f"Policy action dim {ctrl.shape[0]} does not match model.nu {model.nu}."
            )
        ctrl_range = np.asarray(model.actuator_ctrlrange, dtype=np.float32)
        ctrl = np.clip(ctrl, ctrl_range[:, 0], ctrl_range[:, 1])
        data.ctrl[:] = ctrl
        for _ in range(max(1, int(self.frame_skip))):
            mujoco.mj_step(model, data)


__all__ = [
    "FullbodyObsAdapter",
    "LocalPolicyActionTrace",
    "LocalPolicyArtifacts",
    "LocalPolicyRunner",
    "ObservationHistoryBuffer",
    "OnnxPolicyRunner",
    "fullbody_history_settings_from_metadata",
    "fullbody_obs_adapter_params_from_metadata",
    "has_local_policy_artifacts",
    "load_local_policy_artifacts",
    "read_checkpoint_config_metadata",
    "running_mean_std_update",
    "StandaloneBCPolicy",
]


class StandaloneBCPolicy:
    """Checkpoint-independent BC policy with frozen obs normalisation.

    Serialised to / from a ``.npz`` file that embeds the actor parameters as a
    pickled blob alongside obs normalisation statistics and goal params.  Use
    :meth:`load` to restore a saved policy and :meth:`action` to run inference.

    Observation normalisation matches exported checkpoint stats (same as
    :class:`MimicActorModule`): ``(obs - mean) / sqrt(var + eps)`` with **no**
    per-step running updates.

    Example::

        policy, goal_params = StandaloneBCPolicy.load("tutorials/mc26/baselines/boxing/mannequin_exact_clone.npz")
        action = policy.action(fullbody_obs)   # fullbody_obs: (obs_dim,) float32
    """

    def __init__(self, artifacts: LocalPolicyArtifacts) -> None:
        self.artifacts = artifacts

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        mean = np.asarray(self.artifacts.obs_mean, dtype=np.float32)
        var = np.asarray(self.artifacts.obs_var, dtype=np.float32)
        return (obs - mean) / np.sqrt(var + 1e-8)

    def action(self, fullbody_obs: np.ndarray) -> np.ndarray:
        """Predict clipped action for one fullbody observation.

        Args:
            fullbody_obs: Observation vector of shape ``(obs_dim,)``, dtype float32.

        Returns:
            Clipped action array of shape ``(action_dim,)`` in ``[-1, 1]``.
        """
        norm = self._normalize_obs(fullbody_obs)
        out = _actor_forward(self.artifacts.params, norm)
        return np.asarray(np.clip(out, -1.0, 1.0), dtype=np.float32)

    def save(self, path: str | Path, goal_params: dict) -> None:
        """Write standalone policy to a ``.npz`` file.

        Args:
            path: Destination file path (parent dirs created automatically).
            goal_params: Serialisable dict of goal/adapter params stored alongside
                the actor weights so the file is self-contained.
        """
        import pickle

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        params_blob = np.frombuffer(
            pickle.dumps(self.artifacts.params, protocol=pickle.HIGHEST_PROTOCOL),
            dtype=np.uint8,
        )
        np.savez_compressed(
            str(out),
            obs_mean=np.asarray(self.artifacts.obs_mean, dtype=np.float32),
            obs_var=np.asarray(self.artifacts.obs_var, dtype=np.float32),
            obs_count=np.asarray(self.artifacts.obs_count, dtype=np.float32),
            obs_dim=np.asarray(self.artifacts.obs_dim, dtype=np.int32),
            action_dim=np.asarray(self.artifacts.action_dim, dtype=np.int32),
            actor_params_blob=params_blob,
            goal_params_json=np.asarray(json.dumps(goal_params), dtype=np.str_),
        )

    @classmethod
    def load(cls, path: str | Path) -> tuple[StandaloneBCPolicy, dict]:
        """Load a standalone policy saved by :meth:`save`.

        Args:
            path: Path to the ``.npz`` file.

        Returns:
            Tuple ``(policy, goal_params)`` where *goal_params* is the dict
            stored at save time (e.g. ``FullbodyObsAdapter`` construction kwargs).
        """
        import pickle

        pack = np.load(str(Path(path)), allow_pickle=False)
        params = pickle.loads(pack["actor_params_blob"].tobytes())
        artifacts = LocalPolicyArtifacts(
            params=params,
            obs_mean=np.asarray(pack["obs_mean"], dtype=np.float32),
            obs_var=np.asarray(pack["obs_var"], dtype=np.float32),
            obs_count=np.asarray(pack["obs_count"], dtype=np.float32),
            obs_dim=int(np.asarray(pack["obs_dim"]).item()),
            action_dim=int(np.asarray(pack["action_dim"]).item()),
        )
        goal_params = json.loads(str(pack["goal_params_json"]))
        return cls(artifacts), goal_params
