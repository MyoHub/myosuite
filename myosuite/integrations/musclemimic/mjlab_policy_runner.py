# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Batched mjlab policy runner for MuscleMimic actor inference.

Wraps either a :class:`~...MimicActorModule` (PyTorch, GPU-native) or
an :class:`~...OnnxActorSession` (onnxruntime, CPU or CUDA) behind a unified
interface that accepts ``(N, obs_dim)`` torch tensors and returns
``(N, act_dim)`` torch tensors — matching the mjlab training loop's data
flow.

Usage with torch backend (GPU-native, recommended for mjlab)::

    from myosuite.integrations.musclemimic.actor_torch import MimicActorModule
    from myosuite.integrations.musclemimic.mjlab_policy_runner import (
        MjlabPolicyRunner,
    )

    module = MimicActorModule.from_artifacts(artifacts).to("cuda")
    runner = MjlabPolicyRunner.from_torch(module)

    obs = torch.zeros(1024, artifacts.obs_dim, device="cuda")
    actions = runner.act(obs)   # (1024, act_dim), clamped to [-1, 1]

Usage with ONNX backend::

    from myosuite.integrations.musclemimic.actor_onnx import load_onnx_session
    from myosuite.integrations.musclemimic.mjlab_policy_runner import (
        MjlabPolicyRunner,
    )

    session = load_onnx_session("actor.onnx")
    runner = MjlabPolicyRunner.from_onnx(session)

    obs = torch.zeros(1024, session.obs_dim, device="cpu")
    actions = runner.act(obs)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from myosuite.integrations.musclemimic.actor_onnx import OnnxActorSession
    from myosuite.integrations.musclemimic.actor_torch import (
        MimicActorModule,
        DenseActorModule,
    )

    _TorchActor = MimicActorModule | DenseActorModule
else:
    _TorchActor = Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol-like ABC (no ABC dep to keep things simple)
# ---------------------------------------------------------------------------


class _ActorBackend:
    """Internal interface for actor backends."""

    def infer(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class _TorchBackend(_ActorBackend):
    def __init__(self, module: _TorchActor) -> None:
        self._module = module

    def infer(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._module(obs)


class _OnnxBackend(_ActorBackend):
    def __init__(self, session: OnnxActorSession) -> None:
        self._session = session

    def infer(self, obs: torch.Tensor) -> torch.Tensor:
        # ORT runs on CPU; move back to obs device afterwards
        device = obs.device
        actions_np = self._session.act(obs.detach().cpu().numpy())
        return torch.as_tensor(actions_np, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------


@dataclass
class MjlabPolicyRunner:
    """Batched inference runner compatible with the mjlab training loop.

    Accepts ``(N, obs_dim)`` observations on any device and returns
    ``(N, act_dim)`` actions on the same device, clamped to ``[-1, 1]``.

    Do not construct directly; use :meth:`from_torch` or :meth:`from_onnx`.

    Args:
        obs_dim: Expected observation dimension.
        act_dim: Output action dimension.
        _backend: Internal actor backend (torch or ONNX).
    """

    obs_dim: int
    act_dim: int
    _backend: _ActorBackend

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_torch(cls, module: _TorchActor) -> MjlabPolicyRunner:
        """Create a runner backed by a :class:`~...MimicActorModule`.

        The module should already be on the target device and in eval mode.

        Args:
            module: Eval-mode torch actor module.

        Returns:
            :class:`MjlabPolicyRunner` using the torch backend.
        """
        return cls(
            obs_dim=module.obs_dim,
            act_dim=module.act_dim,
            _backend=_TorchBackend(module),
        )

    @classmethod
    def from_onnx(cls, session: OnnxActorSession) -> MjlabPolicyRunner:
        """Create a runner backed by an :class:`~...OnnxActorSession`.

        Args:
            session: Loaded ONNX session from :func:`~...load_onnx_session`.

        Returns:
            :class:`MjlabPolicyRunner` using the ONNX backend.
        """
        return cls(
            obs_dim=session.obs_dim,
            act_dim=session.act_dim,
            _backend=_OnnxBackend(session),
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Return deterministic actions for a batch of observations.

        Args:
            obs: Raw observations, shape ``(N, obs_dim)``, any device.
                 Will be cast to ``float32`` if needed.

        Returns:
            Actions clamped to ``[-1, 1]``, shape ``(N, act_dim)``,
            on the same device as *obs*.

        Raises:
            ValueError: If ``obs`` last dimension does not match ``obs_dim``.
        """
        if obs.dtype != torch.float32:
            obs = obs.float()
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(
                f"Expected obs last dim {self.obs_dim}, got {obs.shape[-1]}"
            )
        actions = self._backend.infer(obs)
        return torch.clamp(actions, -1.0, 1.0)

    def act_stochastic(
        self,
        obs: torch.Tensor,
        log_std: np.ndarray | torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample actions from the policy's Gaussian distribution.

        Args:
            obs: Raw observations, shape ``(N, obs_dim)``.
            log_std: Log standard deviations, shape ``(act_dim,)``.
            generator: Optional torch RNG for reproducibility.

        Returns:
            Sampled actions clamped to ``[-1, 1]``, shape ``(N, act_dim)``.
        """
        mean = self._backend.infer(obs if obs.dtype == torch.float32 else obs.float())
        log_std_t = torch.as_tensor(
            np.asarray(log_std, dtype=np.float32), device=obs.device
        )
        std = torch.exp(torch.clamp(log_std_t, -20.0, 2.0))
        noise = torch.randn(
            mean.shape,
            dtype=mean.dtype,
            device=mean.device,
            generator=generator,
        )
        return torch.clamp(mean + noise * std, -1.0, 1.0)


__all__ = ["MjlabPolicyRunner"]


# ---------------------------------------------------------------------------
# ONNX-checkpointing runner and training utilities
# (moved from tutorials/mc26 mimic-init training scripts)
# ---------------------------------------------------------------------------

try:
    import tempfile
    from pathlib import Path

    import mujoco
    import wandb
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner
    from mjlab.sim.sim import Simulation, SimulationCfg
    from mjlab.utils.lab_api.math import axis_angle_from_quat, quat_from_matrix
    from mjlab.utils.spaces import Box
    from mjlab.utils.spaces import Dict as DictSpace
    from mjlab.utils.spaces import batch_space
    from rsl_rl.env import VecEnv
    from tensordict import TensorDict

    import torch.nn as nn

    from myosuite.integrations.musclemimic.actor_torch import (
        _load_dense_checkpoint_into_model,
    )
    from myosuite.integrations.musclemimic.fullbody_checkpoint_io import (
        resolve_checkpoint_ref,
    )
    from myosuite.integrations.musclemimic.fullbody_local_policy import (
        FullbodyObsAdapter,
        fullbody_history_settings_from_metadata,
        fullbody_obs_adapter_params_from_metadata,
        load_local_policy_artifacts,
        read_checkpoint_config_metadata,
    )
    from myosuite.integrations.musclemimic.fullbody_model import (
        compile_mimic_fullbody_mjmodel,
        default_mimic_fullbody_config,
    )
    from myosuite.integrations.musclemimic.model_bridge import (
        SharedModelStateBridge,
        make_fullbody_checkpoint_bridged_policy,
    )
    from myosuite.integrations.musclemimic.trajectory_io import load_motion_clip
    from myosuite.utils.onnx_checkpoint import (
        bundle_onnx_with_checkpoint,
        extract_checkpoint_from_onnx,
    )

    _TRAINING_DEPS_AVAILABLE = True
except ImportError:
    _TRAINING_DEPS_AVAILABLE = False


class _ActorExportWrapper(torch.nn.Module):
    """Wrap mjlab's Gaussian actor to export a deterministic mean-action ONNX."""

    def __init__(self, actor: torch.nn.Module) -> None:
        super().__init__()
        self.actor = actor

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs_td = TensorDict({"actor": obs}, batch_size=[obs.shape[0]])
        return self.actor(obs_td)


class OnnxCheckpointingMjlabRunner(MjlabOnPolicyRunner):
    """RSL-RL runner that saves ONNX bundles instead of raw ``.pt`` files.

    Extends :class:`~mjlab.rl.MjlabOnPolicyRunner` so that every
    ``runner.save(path)`` call produces an ONNX-bundled checkpoint that can be
    loaded back with :meth:`load_onnx` or evaluated with the standard myosuite
    ONNX playback tools.
    """

    def __init__(
        self,
        env: Any,
        train_cfg: dict[str, Any],
        log_dir: str | None,
        device: str,
        *,
        task_id: str,
    ) -> None:
        super().__init__(env=env, train_cfg=train_cfg, log_dir=log_dir, device=device)
        self._task_id = task_id
        first_layer = self.alg.actor.state_dict()["mlp.0.weight"]
        self._obs_dim = int(first_layer.shape[1])
        self._act_dim = int(getattr(self.env, "num_actions"))

    def _export_actor_onnx(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        actor_device = next(self.alg.actor.parameters()).device
        wrapper = _ActorExportWrapper(self.alg.actor)
        was_training = self.alg.actor.training
        self.alg.actor.eval()
        try:
            dummy_obs = torch.zeros(
                1, self._obs_dim, dtype=torch.float32, device=actor_device
            )
            torch.onnx.export(
                wrapper,
                dummy_obs,
                str(output_path),
                input_names=["obs"],
                output_names=["action"],
                dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
                opset_version=17,
                do_constant_folding=True,
                dynamo=False,
            )
        finally:
            if was_training:
                self.alg.actor.train()

    def save(self, path: str, infos: Any = None) -> None:
        native_path = Path(path)
        onnx_path = native_path.with_suffix(".onnx")
        with tempfile.TemporaryDirectory(prefix="onnx-ckpt-") as tmp_dir:
            tmp_root = Path(tmp_dir)
            tmp_native = tmp_root / native_path.name
            tmp_onnx = tmp_root / onnx_path.name
            env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
            saved_dict = self.alg.save()
            saved_dict["iter"] = self.current_learning_iteration
            saved_dict["infos"] = {**(infos or {}), "env_state": env_state}
            torch.save(saved_dict, tmp_native)
            self._export_actor_onnx(tmp_onnx)
            obs_groups = getattr(self.alg.actor, "obs_groups", None)
            obs_key = obs_groups[0] if obs_groups and len(obs_groups) == 1 else "actor"
            metadata: dict[str, Any] = {
                "task_id": self._task_id,
                "obs_dim": self._obs_dim,
                "act_dim": self._act_dim,
                "iteration": int(self.current_learning_iteration),
                "obs_key": obs_key,
            }
            fatigue_state = get_env_fatigue_state(self.env)
            if fatigue_state is not None:
                metadata[_FATIGUE_STATE_KEY] = fatigue_state
            bundle_onnx_with_checkpoint(
                onnx_path=tmp_onnx,
                checkpoint_path=tmp_native,
                framework="mjlab-rslrl",
                metadata={
                    "task_id": self._task_id,
                    "obs_dim": self._obs_dim,
                    "act_dim": self._act_dim,
                    "iteration": int(self.current_learning_iteration),
                },
                output_path=onnx_path,
            )
            if self.cfg["upload_model"] and wandb.run is not None:
                self.logger.save_model(str(onnx_path), self.current_learning_iteration)

    def load_onnx(self, path: str | Path) -> dict[str, Any]:
        """Load a previously saved ONNX bundle back into this runner."""
        checkpoint_path, _, temp_dir = extract_checkpoint_from_onnx(path)
        try:
            return self.load(str(checkpoint_path), map_location=self.device)
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()


# ---------------------------------------------------------------------------
# TorchFullbodyObsAdapter — GPU-accelerated fullbody observation builder
# ---------------------------------------------------------------------------


class TorchFullbodyObsAdapter:
    """Torch port of :class:`~...fullbody_local_policy.FullbodyObsAdapter`.

    Moves all index arrays to *device* at construction time and uses batched
    Torch operations for the per-step observation build.  Required for the
    GPU-parallel :class:`CheckpointVecEnv` path; CPU path uses the NumPy
    :class:`~...fullbody_local_policy.FullbodyObsAdapter` directly.
    """

    def __init__(self, adapter: FullbodyObsAdapter, *, device: torch.device) -> None:
        self.device = device
        self._goal = adapter._goal
        self._obs_flags = dict(adapter._obs_flags)
        self._traj_len = int(adapter._traj_len)
        self.goal_dim = int(adapter.goal_dim)

        def _li(v: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(np.asarray(v, dtype=np.int64), device=device)

        def _lf(v: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(np.asarray(v, dtype=np.float32), device=device)

        self._root_qpos_idx_full = _li(adapter._root_qpos_idx_full)
        self._root_qvel_idx_full = _li(adapter._root_qvel_idx_full)
        self._root_qpos_idx_xyz = _li(adapter._root_qpos_idx_xyz)
        self._qpos_ind = _li(adapter._qpos_ind)
        self._qvel_ind = _li(adapter._qvel_ind)
        self._qpos_non_root_ind = _li(adapter._qpos_non_root_ind)
        self._qvel_non_root_ind = _li(adapter._qvel_non_root_ind)
        self._actuator_ids = _li(adapter._actuator_ids)
        self._touch_sensor_ids = _li(adapter._touch_sensor_ids)
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
        self._clip_qvel = (
            _lf(adapter._clip.qvel) if adapter._clip.qvel is not None else None
        )
        sensor_adr = np.asarray(adapter._model.sensor_adr, dtype=np.int64)
        sensor_dim = np.asarray(adapter._model.sensor_dim, dtype=np.int64)
        self._touch_sensor_slices = tuple(
            (int(sensor_adr[int(s)]), int(sensor_adr[int(s)] + sensor_dim[int(s)]))
            for s in np.asarray(adapter._touch_sensor_ids, dtype=np.int64)
        )

    def _calc_site_velocities(
        self,
        *,
        site_ids,
        site_xpos,
        cvel_parent,
        subtree_com_root,
        site_bodyid,
        body_rootid,
    ) -> torch.Tensor:
        parent_body_id = site_bodyid[site_ids]
        root_body_id = body_rootid[parent_body_id]
        body_cvel = cvel_parent[:, parent_body_id, :]
        root_com = subtree_com_root[:, root_body_id, :]
        rpos = site_xpos[:, site_ids, :] - root_com
        lin_vel = body_cvel[..., 3:] - torch.cross(rpos, body_cvel[..., :3], dim=-1)
        return torch.cat([body_cvel[..., :3], lin_vel], dim=-1)

    def _relative_site_quantities(
        self,
        *,
        site_ids,
        site_xpos,
        site_xmat,
        cvel_parent,
        subtree_com_root,
        site_bodyid,
        body_rootid,
    ):
        site_vel = self._calc_site_velocities(
            site_ids=site_ids,
            site_xpos=site_xpos,
            cvel_parent=cvel_parent,
            subtree_com_root=subtree_com_root,
            site_bodyid=site_bodyid,
            body_rootid=body_rootid,
        )
        main_pos = site_xpos[:, int(site_ids[0].item()), :]
        main_mat = site_xmat[:, int(site_ids[0].item()), :].reshape(-1, 3, 3)
        main_vel = site_vel[:, 0, :]
        other_ids = site_ids[1:]
        other_pos = site_xpos[:, other_ids, :]
        other_mat = site_xmat[:, other_ids, :].reshape(site_xmat.shape[0], -1, 3, 3)
        other_vel = site_vel[:, 1:, :]
        site_rpos = other_pos - main_pos.unsqueeze(1)
        rel_rot = torch.einsum("bij,bnjk->bnik", main_mat.transpose(1, 2), other_mat)
        rel_quat = quat_from_matrix(rel_rot.reshape(-1, 3, 3))
        site_rangles = axis_angle_from_quat(rel_quat).reshape(
            rel_rot.shape[0], rel_rot.shape[1], 3
        )
        rel_lin = torch.einsum(
            "bij,bnj->bni", main_mat, main_vel[:, None, 3:] - other_vel[:, :, 3:]
        )
        other_ang = torch.einsum("bnik,bnk->bni", rel_rot, other_vel[:, :, :3])
        site_rvel = torch.cat([other_ang - main_vel[:, None, :3], rel_lin], dim=-1)
        return site_rpos, site_rangles, site_rvel

    def _traj_goal_obs(self, frame_idx: torch.Tensor) -> torch.Tensor:
        goal = self._goal
        batch = int(frame_idx.shape[0])
        offsets = torch.arange(
            goal.n_step_lookahead, device=self.device, dtype=torch.long
        )
        future = torch.clamp(
            frame_idx[:, None] + offsets[None, :] * int(goal.n_step_stride),
            max=self._traj_len - 1,
        )
        flat = future.reshape(-1)
        qpos = self._clip_qpos[flat].reshape(batch, goal.n_step_lookahead, -1)
        if self._clip_qvel is not None:
            qvel = self._clip_qvel[flat].reshape(batch, goal.n_step_lookahead, -1)
        else:
            qvel = torch.zeros(
                batch,
                goal.n_step_lookahead,
                int(self._qvel_ind.shape[0]),
                dtype=torch.float32,
                device=self.device,
            )
        site_rpos, site_rangles, site_rvel = self._relative_site_quantities(
            site_ids=self._traj_site_ids,
            site_xpos=self._traj_site_xpos[flat],
            site_xmat=self._traj_site_xmat[flat],
            cvel_parent=self._traj_cvel[flat],
            subtree_com_root=self._traj_subtree_com[flat],
            site_bodyid=self._traj_site_bodyid,
            body_rootid=self._traj_body_rootid,
        )
        site_rpos = site_rpos.reshape(batch, goal.n_step_lookahead, -1)
        site_rangles = site_rangles.reshape(batch, goal.n_step_lookahead, -1)
        site_rvel = site_rvel.reshape(batch, goal.n_step_lookahead, -1)
        if goal.use_concise_lookahead:
            ref_qpos = self._clip_qpos[frame_idx]
            ref_root_pos = ref_qpos[:, self._root_qpos_idx_xyz]
            ref_root_vel = (
                self._clip_qvel[frame_idx]
                if self._clip_qvel is not None
                else torch.zeros(
                    batch,
                    int(self._qvel_ind.shape[0]),
                    dtype=torch.float32,
                    device=self.device,
                )
            )[:, self._root_qvel_idx_full]
            parts: list[torch.Tensor] = [site_rpos[:, 0, :]]
            for s in range(1, goal.n_step_lookahead):
                parts += [
                    qpos[:, s, :][:, self._root_qpos_idx_xyz] - ref_root_pos,
                    qvel[:, s, :][:, self._root_qvel_idx_full] - ref_root_vel,
                    site_rpos[:, s, :],
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
        goal = self._goal
        obs: list[torch.Tensor] = []
        root_qpos = data.qpos[:, self._root_qpos_idx_full]
        root_qvel = data.qvel[:, self._root_qvel_idx_full]
        if self._obs_flags["enable_joint_pos_observations"]:
            obs.extend([root_qpos[:, 2:], data.qpos[:, self._qpos_non_root_ind]])
        if self._obs_flags["enable_joint_vel_observations"]:
            obs.extend([root_qvel, data.qvel[:, self._qvel_non_root_ind]])
        muscle_blocks: list[torch.Tensor] = []
        flag_map = [
            ("enable_muscle_length_observations", data.actuator_length),
            ("enable_muscle_velocity_observations", data.actuator_velocity),
            ("enable_muscle_force_observations", data.actuator_force),
            ("enable_muscle_excitation_observations", data.ctrl),
            ("enable_muscle_activation_observations", data.act),
        ]
        for flag, arr in flag_map:
            if self._obs_flags[flag]:
                muscle_blocks.append(arr[:, self._actuator_ids])
        if muscle_blocks:
            obs.append(
                torch.stack(muscle_blocks, dim=2).reshape(data.qpos.shape[0], -1)
            )
        if self._touch_sensor_slices:
            obs.extend(
                data.sensordata[:, s:e].sum(dim=1, keepdim=True)
                for s, e in self._touch_sensor_slices
            )
        site_rpos, site_rangles, site_rvel = self._relative_site_quantities(
            site_ids=self._site_ids,
            site_xpos=data.site_xpos,
            site_xmat=data.site_xmat,
            cvel_parent=data.cvel,
            subtree_com_root=data.subtree_com,
            site_bodyid=self._sim_site_bodyid,
            body_rootid=self._sim_body_rootid,
        )
        traj_obs = self._traj_goal_obs(frame_idx)
        goal_parts: list[torch.Tensor] = []
        if goal.enable_mimic_site_rpos_observations:
            goal_parts.append(site_rpos.reshape(site_rpos.shape[0], -1))
        goal_parts += [
            site_rangles.reshape(site_rangles.shape[0], -1),
            site_rvel.reshape(site_rvel.shape[0], -1),
            traj_obs,
        ]
        if goal.enable_motion_phase:
            goal_parts.append(
                frame_idx.to(dtype=torch.float32).unsqueeze(1)
                / float(max(self._traj_len, 1))
            )
        obs.append(torch.cat(goal_parts, dim=1))
        return torch.cat(obs, dim=1).to(dtype=torch.float32)


# ---------------------------------------------------------------------------
# CheckpointVecEnv — RSL-RL VecEnv wrapping mjlab saber with fullbody obs
# ---------------------------------------------------------------------------

_SABER_ENTITY_NAME = "saber_p0_robot"


def _to_numpy_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _to_muscle_activations(action: np.ndarray) -> np.ndarray:
    """Map full-body logits in ``[-1, 1]`` to saber muscle activations in ``[0, 1]``."""
    return np.clip(0.5 * (action + 1.0), 0.0, 1.0).astype(np.float32)


class CheckpointVecEnv(VecEnv):
    """RSL-RL :class:`~rsl_rl.env.VecEnv` exposing the mimic checkpoint interface on native mjlab saber.

    Training keeps this bridge on the *environment* side because the rollout
    storage must retain checkpoint-space observations for later PPO updates —
    the bridge cannot be moved fully into the policy during training as it can
    during evaluation.
    """

    def __init__(
        self,
        env: ManagerBasedRlEnv,
        *,
        checkpoint_root: Path,
        motion_path: Path,
    ) -> None:
        self.env = env
        self.checkpoint_root = Path(checkpoint_root)
        self.motion_path = Path(motion_path)
        self.device = torch.device(self.unwrapped.device)
        self._use_gpu_compat = self.device.type == "cuda"
        self.num_envs = int(self.unwrapped.num_envs)
        self.max_episode_length = int(self.unwrapped.max_episode_length)
        metadata = read_checkpoint_config_metadata(self.checkpoint_root)
        self._history_settings = fullbody_history_settings_from_metadata(metadata)
        self._artifacts = load_local_policy_artifacts(self.checkpoint_root)
        self._fullbody_model, _, _ = compile_mimic_fullbody_mjmodel(
            default_mimic_fullbody_config()
        )
        self._clip = load_motion_clip(
            self.motion_path,
            expected_nq=self._fullbody_model.nq,
            expected_nv=self._fullbody_model.nv,
        )
        self._obs_adapter = FullbodyObsAdapter(
            self._fullbody_model,
            self._clip,
            fullbody_obs_adapter_params_from_metadata(metadata),
        )
        self._traj_len = int(self._obs_adapter._traj_len)
        self._source_model = self.unwrapped._saber_logic.mj_model
        self._bridge = SharedModelStateBridge(self._source_model, self._fullbody_model)
        if self._use_gpu_compat:
            self._target_sim = Simulation(
                num_envs=self.num_envs,
                cfg=SimulationCfg(),
                model=self._fullbody_model,
                device=str(self.device),
            )
            self._target_data = None
            self._torch_obs_adapter = TorchFullbodyObsAdapter(
                self._obs_adapter, device=self.device
            )

            def _t(a):
                return torch.as_tensor(
                    np.asarray(a, dtype=np.float32), device=self.device
                )

            def _i(a):
                return torch.as_tensor(
                    np.asarray(a, dtype=np.int64), device=self.device
                )

            self._target_ref_qpos = _t(self._bridge._target_ref_qpos)
            self._target_ref_qvel = _t(self._bridge._target_ref_qvel)
            self._target_ref_ctrl = _t(self._bridge._target_ref_ctrl)
            self._target_ref_act = (
                _t(self._bridge._target_ref_act)
                if self._bridge._target_ref_act is not None
                else None
            )
            self._source_qpos_idx = _i(self._bridge._source_qpos_idx)
            self._target_qpos_idx = _i(self._bridge._target_qpos_idx)
            self._source_qvel_idx = _i(self._bridge._source_qvel_idx)
            self._target_qvel_idx = _i(self._bridge._target_qvel_idx)
            self._source_act_idx = _i(self._bridge._source_act_idx)
            self._target_act_idx = _i(self._bridge._target_act_idx)
        else:
            self._target_sim = None
            self._target_data = [
                mujoco.MjData(self._fullbody_model) for _ in range(self.num_envs)
            ]
            self._torch_obs_adapter = None
        self._ctrl_dt = float(self.unwrapped._saber_logic.task_cfg.backend.ctrl_dt)
        self.num_actions = int(self._fullbody_model.nu)
        self.single_action_space = Box(shape=(self.num_actions,), low=-1.0, high=1.0)
        obs_box = Box(
            shape=(int(self._artifacts.obs_dim),), low=-float("inf"), high=float("inf")
        )
        self.single_observation_space = DictSpace(
            spaces={"actor": obs_box, "critic": obs_box}
        )
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self.observation_space = batch_space(
            self.single_observation_space, self.num_envs
        )
        self.env.reset()

    @property
    def cfg(self) -> Any:
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        return self.env.render_mode

    @property
    def unwrapped(self) -> ManagerBasedRlEnv:
        return self.env.unwrapped

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.unwrapped.episode_length_buf = value

    def seed(self, seed: int = -1) -> int:
        return self.unwrapped.seed(seed)

    def _frame_idx_from_time(self, sim_time: Any) -> int:
        t = float(np.asarray(_to_numpy_array(sim_time)).reshape(-1)[0])
        return int(round(t / self._ctrl_dt)) % int(self._clip.qpos.shape[0])

    def _build_single_obs(self, env_idx: int) -> np.ndarray:
        if self._target_data is None:
            raise RuntimeError("_build_single_obs is CPU-path only.")
        live = self.unwrapped.scene[_SABER_ENTITY_NAME].data.data
        td = self._target_data[env_idx]
        self._bridge.copy_source_state_into_target(
            source_qpos=live.qpos[env_idx],
            source_qvel=live.qvel[env_idx],
            source_ctrl=live.ctrl[env_idx],
            source_act=live.act[env_idx]
            if getattr(live, "act", None) is not None
            else None,
            target_data=td,
        )
        return np.asarray(
            self._obs_adapter.build(td, self._frame_idx_from_time(live.time[env_idx])),
            dtype=np.float32,
        )

    def _build_obs_batch(self) -> torch.Tensor:
        if self._target_sim is not None:
            live = self.unwrapped.scene[_SABER_ENTITY_NAME].data.data
            td = self._target_sim.data
            td.qpos[:] = self._target_ref_qpos.unsqueeze(0)
            td.qvel[:] = self._target_ref_qvel.unsqueeze(0)
            td.ctrl[:] = self._target_ref_ctrl.unsqueeze(0)
            if self._target_ref_act is not None and hasattr(td, "act"):
                td.act[:] = self._target_ref_act.unsqueeze(0)
            if self._source_qpos_idx.numel() > 0:
                td.qpos[:, self._target_qpos_idx] = live.qpos[:, self._source_qpos_idx]
            if self._source_qvel_idx.numel() > 0:
                td.qvel[:, self._target_qvel_idx] = live.qvel[:, self._source_qvel_idx]
            if self._source_act_idx.numel() > 0:
                td.ctrl[:, self._target_act_idx] = live.ctrl[:, self._source_act_idx]
                if (
                    self._target_ref_act is not None
                    and hasattr(td, "act")
                    and getattr(live, "act", None) is not None
                ):
                    td.act[:, self._target_act_idx] = live.act[:, self._source_act_idx]
            self._target_sim.forward()
            frame_idx = torch.remainder(
                torch.round(live.time / self._ctrl_dt).to(dtype=torch.long),
                self._traj_len,
            )
            return self._torch_obs_adapter.build(td, frame_idx)
        obs_np = np.stack(
            [self._build_single_obs(i) for i in range(self.num_envs)], axis=0
        ).astype(np.float32)
        return torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)

    def get_observations(self) -> TensorDict:
        obs = self._build_obs_batch()
        return TensorDict(
            {"actor": obs, "critic": obs},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def reset(self) -> tuple[TensorDict, dict]:
        self.env.reset()
        return self.get_observations(), {}

    def step(
        self, actions: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, Any]]:
        if self._target_sim is not None:
            projected = torch.full(
                (self.num_envs, int(self._source_model.nu)),
                -1.0,
                dtype=torch.float32,
                device=self.device,
            )
            if self._source_act_idx.numel() > 0:
                projected[:, self._source_act_idx] = actions[:, self._target_act_idx]
            native_actions = torch.clamp(0.5 * (projected + 1.0), 0.0, 1.0)
        else:
            action_np = np.asarray(actions.detach().cpu(), dtype=np.float32)
            projected = np.stack(
                [
                    _to_muscle_activations(
                        self._bridge.project_target_action_to_source(
                            action_np[i], fill_value=-1.0
                        )
                    )
                    for i in range(self.num_envs)
                ],
                axis=0,
            ).astype(np.float32)
            native_actions = torch.as_tensor(
                projected, dtype=torch.float32, device=self.device
            )
        _obs_dict, rew, terminated, truncated, extras = self.env.step(native_actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        if not self.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        return self.get_observations(), rew, dones, extras

    def close(self) -> None:
        if self._target_sim is not None:
            del self._target_sim
        self.env.close()


# ---------------------------------------------------------------------------
# Convenience utilities for the mimic-init training workflow
# ---------------------------------------------------------------------------


def _initialize_runner_from_mimic_checkpoint(
    runner: OnnxCheckpointingMjlabRunner,
    checkpoint_root: Path,
) -> None:
    """Load actor and critic weights from a mimic Orbax checkpoint into *runner*."""
    checkpoint = resolve_checkpoint_ref(str(checkpoint_root))
    artifacts = load_local_policy_artifacts(checkpoint.local_path)
    _load_dense_checkpoint_into_model(
        runner.alg.actor,
        artifacts.params["actor"],
        obs_mean=artifacts.obs_mean,
        obs_var=artifacts.obs_var,
        obs_count=artifacts.obs_count,
    )
    _load_dense_checkpoint_into_model(
        runner.alg.critic,
        artifacts.params["critic"],
        obs_mean=artifacts.obs_mean,
        obs_var=artifacts.obs_var,
        obs_count=artifacts.obs_count,
    )


def _maybe_freeze_actor_std(
    runner: OnnxCheckpointingMjlabRunner, *, learn_actor_std: bool
) -> None:
    """Freeze the Gaussian exploration std unless ``learn_actor_std`` is ``True``."""
    if learn_actor_std:
        return
    distribution = getattr(runner.alg.actor, "distribution", None)
    if distribution is None:
        return
    for attr in ("std_param", "log_std_param"):
        param = getattr(distribution, attr, None)
        if isinstance(param, nn.Parameter):
            param.requires_grad_(False)


def _run_sanity_rollouts(
    policy: Any,
    env: ManagerBasedRlEnv,
    *,
    checkpoint_root: Path,
    motion_path: Path,
    device: str,
    num_episodes: int,
    min_episode_length: int,
) -> list[int]:
    """Run deterministic zero-exploration sanity rollouts on the native saber env.

    Wraps *policy* with :func:`~...model_bridge.make_fullbody_checkpoint_bridged_policy`
    and steps through the environment until *num_episodes* are complete, returning
    the per-episode step counts.  Raises :exc:`RuntimeError` if any episode is
    shorter than *min_episode_length*.
    """
    if num_episodes <= 0:
        return []
    bridged = make_fullbody_checkpoint_bridged_policy(
        source_model=env._saber_logic.mj_model,
        policy=policy,
        checkpoint_root=checkpoint_root,
        motion_path=motion_path,
        ctrl_dt=float(env._saber_logic.task_cfg.backend.ctrl_dt),
        source_action_fill=-1.0,
        source_entity_name=_SABER_ENTITY_NAME,
        policy_device=device,
    )
    completed: list[int] = []
    live_lengths = np.zeros(int(env.num_envs), dtype=np.int32)
    tracked = np.zeros(int(env.num_envs), dtype=bool)
    tracked[: min(int(env.num_envs), num_episodes)] = True
    env.reset()
    reset_fn = getattr(bridged, "reset", None)
    if callable(reset_fn):
        reset_fn()
    with torch.no_grad():
        while len(completed) < num_episodes:
            actions = np.stack(
                [
                    np.asarray(
                        bridged.predict_from_env(env, env_idx=i), dtype=np.float32
                    )
                    for i in range(int(env.num_envs))
                ],
                axis=0,
            )
            native_actions = torch.as_tensor(
                actions, dtype=torch.float32, device=device
            )
            _obs, _rew, terminated, truncated, _extras = env.step(native_actions)
            live_lengths += 1
            done_np = (terminated | truncated).detach().cpu().numpy().astype(bool)
            for idx in np.where(done_np & tracked)[0]:
                completed.append(int(live_lengths[idx]))
                live_lengths[idx] = 0
                if len(completed) >= num_episodes:
                    break
    if any(length < min_episode_length for length in completed):
        raise RuntimeError(
            f"Sanity rollout lengths {completed} below required minimum {min_episode_length}."
        )
    return completed
