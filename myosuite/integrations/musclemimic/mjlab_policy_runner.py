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

    import wandb
    from mjlab.rl import MjlabOnPolicyRunner
    from mjlab.utils.lab_api.math import axis_angle_from_quat, quat_from_matrix
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
        load_local_policy_artifacts,
    )
    from myosuite.utils.onnx_checkpoint import (
        _FATIGUE_STATE_KEY,
        bundle_onnx_with_checkpoint,
        extract_checkpoint_from_onnx,
        get_env_fatigue_state,
        set_env_fatigue_state,
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
        # Use the model's own obs_groups so the dummy TensorDict has the right key.
        # For models with a single group (the common case) this resolves to that
        # group name (e.g. "proprioception"); fall back to "actor" for legacy models.
        obs_groups = getattr(self.actor, "obs_groups", None)
        key = obs_groups[0] if obs_groups and len(obs_groups) == 1 else "actor"
        obs_td = TensorDict({key: obs}, batch_size=[obs.shape[0]])
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
        task_id: str = "",
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
            metadata: dict[str, Any] = {
                "task_id": self._task_id,
                "obs_dim": self._obs_dim,
                "act_dim": self._act_dim,
                "iteration": int(self.current_learning_iteration),
            }
            fatigue_state = get_env_fatigue_state(self.env)
            if fatigue_state is not None:
                metadata[_FATIGUE_STATE_KEY] = fatigue_state
            bundle_onnx_with_checkpoint(
                onnx_path=tmp_onnx,
                checkpoint_path=tmp_native,
                framework="mjlab-rslrl",
                metadata=metadata,
                output_path=onnx_path,
            )
            if self.cfg["upload_model"] and wandb.run is not None:
                self.logger.save_model(str(onnx_path), self.current_learning_iteration)

    def load(self, path: str, **kwargs: Any) -> dict[str, Any]:
        """Load a checkpoint; delegates to :meth:`load_onnx` for ``.onnx`` bundles."""
        if Path(path).suffix == ".onnx":
            return self.load_onnx(path)
        return super().load(path, **kwargs)  # type: ignore[return-value]

    def load_onnx(self, path: str | Path) -> dict[str, Any]:
        """Load a previously saved ONNX bundle back into this runner."""
        checkpoint_path, meta, temp_dir = extract_checkpoint_from_onnx(path)
        try:
            result = self.load(str(checkpoint_path), map_location=self.device)
            fatigue_state = meta.get("metadata", {}).get(_FATIGUE_STATE_KEY)
            if fatigue_state is not None:
                set_env_fatigue_state(self.env, fatigue_state)
            return result
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()


# ---------------------------------------------------------------------------
# TorchFullbodyObsAdapter — GPU-accelerated fullbody observation builder
# ---------------------------------------------------------------------------


class TorchFullbodyObsAdapter:
    """Torch port of :class:`~...fullbody_local_policy.FullbodyObsAdapter`.

    Moves all index arrays to *device* at construction time and uses batched
    Torch operations for the per-step observation build, for GPU-parallel
    training; CPU path uses the NumPy
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
