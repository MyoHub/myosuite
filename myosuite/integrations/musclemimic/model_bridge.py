# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Bridge related MuJoCo models by shared joint and actuator names."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


def _joint_qpos_size(jnt_type: int) -> int:
    return {
        mujoco.mjtJoint.mjJNT_FREE: 7,
        mujoco.mjtJoint.mjJNT_BALL: 4,
        mujoco.mjtJoint.mjJNT_SLIDE: 1,
        mujoco.mjtJoint.mjJNT_HINGE: 1,
    }[mujoco.mjtJoint(jnt_type)]


def _joint_qvel_size(jnt_type: int) -> int:
    return {
        mujoco.mjtJoint.mjJNT_FREE: 6,
        mujoco.mjtJoint.mjJNT_BALL: 3,
        mujoco.mjtJoint.mjJNT_SLIDE: 1,
        mujoco.mjtJoint.mjJNT_HINGE: 1,
    }[mujoco.mjtJoint(jnt_type)]


def _to_numpy_array(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    """Convert numpy / torch-like values to a numpy array."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=dtype)


def _unwrap_env_like(env: Any) -> Any:
    """Return the innermost env-like object by following common wrapper attrs."""

    current = env
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        for attr_name in ("unwrapped", "env"):
            candidate = getattr(current, attr_name, None)
            if candidate is not None and candidate is not current:
                current = candidate
                break
        else:
            return current
    return current


@dataclass(frozen=True)
class SharedModelStateBridge:
    """Bridge state and actions between related MuJoCo models.

    The bridge matches joints and actuators by exact MuJoCo name. Shared source
    state can then be copied into a target model with different extra DoFs, and
    target-model policy actions can be projected back into the source model's
    actuator space.

    Args:
        source_model: Runtime model being controlled, such as a task env model.
        target_model: Model expected by the policy or observation adapter.
        target_keyframe: Reference keyframe used to initialize target-only state.
    """

    source_model: mujoco.MjModel
    target_model: mujoco.MjModel
    target_keyframe: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_source_qpos_idx", self._build_joint_index_array(use_qvel=False)[0]
        )
        object.__setattr__(
            self, "_target_qpos_idx", self._build_joint_index_array(use_qvel=False)[1]
        )
        object.__setattr__(
            self, "_source_qvel_idx", self._build_joint_index_array(use_qvel=True)[0]
        )
        object.__setattr__(
            self, "_target_qvel_idx", self._build_joint_index_array(use_qvel=True)[1]
        )
        source_act_idx, target_act_idx, shared_actuators = (
            self._build_actuator_indices()
        )
        object.__setattr__(self, "_source_act_idx", source_act_idx)
        object.__setattr__(self, "_target_act_idx", target_act_idx)
        object.__setattr__(self, "_shared_actuator_names", shared_actuators)
        object.__setattr__(
            self, "_shared_joint_names", self._build_shared_joint_names()
        )

        ref_data = mujoco.MjData(self.target_model)
        if int(self.target_model.nkey) > self.target_keyframe:
            mujoco.mj_resetDataKeyframe(
                self.target_model, ref_data, self.target_keyframe
            )
        else:
            mujoco.mj_resetData(self.target_model, ref_data)
        mujoco.mj_forward(self.target_model, ref_data)
        object.__setattr__(self, "_target_ref_qpos", np.asarray(ref_data.qpos).copy())
        object.__setattr__(self, "_target_ref_qvel", np.asarray(ref_data.qvel).copy())
        object.__setattr__(self, "_target_ref_ctrl", np.asarray(ref_data.ctrl).copy())
        if getattr(ref_data, "act", None) is not None and int(ref_data.act.size) > 0:
            object.__setattr__(self, "_target_ref_act", np.asarray(ref_data.act).copy())
        else:
            object.__setattr__(self, "_target_ref_act", None)

    @property
    def shared_joint_names(self) -> tuple[str, ...]:
        """Return shared joint names in source-model order."""

        return self._shared_joint_names

    @property
    def shared_actuator_names(self) -> tuple[str, ...]:
        """Return shared actuator names in source-model order."""

        return self._shared_actuator_names

    def copy_source_into_target(
        self,
        source_data: mujoco.MjData,
        target_data: mujoco.MjData,
    ) -> None:
        """Populate *target_data* from *source_data* and run forward dynamics.

        Args:
            source_data: Live MuJoCo data for ``source_model``.
            target_data: MuJoCo data for ``target_model`` to be overwritten.
        """

        self.copy_source_state_into_target(
            source_qpos=source_data.qpos,
            source_qvel=source_data.qvel,
            source_ctrl=source_data.ctrl,
            source_act=getattr(source_data, "act", None),
            target_data=target_data,
        )

    def copy_source_state_into_target(
        self,
        *,
        source_qpos: Any,
        source_qvel: Any,
        source_ctrl: Any,
        source_act: Any | None,
        target_data: mujoco.MjData,
    ) -> None:
        """Populate *target_data* from source state arrays and run forward dynamics."""

        target_data.qpos[:] = self._target_ref_qpos
        target_data.qvel[:] = self._target_ref_qvel
        target_data.ctrl[:] = self._target_ref_ctrl
        if (
            self._target_ref_act is not None
            and getattr(target_data, "act", None) is not None
        ):
            target_data.act[:] = self._target_ref_act

        if self._source_qpos_idx.size:
            qpos = _to_numpy_array(source_qpos, dtype=target_data.qpos.dtype).reshape(
                -1
            )
            target_data.qpos[self._target_qpos_idx] = qpos[self._source_qpos_idx]
        if self._source_qvel_idx.size:
            qvel = _to_numpy_array(source_qvel, dtype=target_data.qvel.dtype).reshape(
                -1
            )
            target_data.qvel[self._target_qvel_idx] = qvel[self._source_qvel_idx]
        if self._source_act_idx.size:
            ctrl = _to_numpy_array(source_ctrl, dtype=target_data.ctrl.dtype).reshape(
                -1
            )
            target_data.ctrl[self._target_act_idx] = ctrl[self._source_act_idx]
            if (
                self._target_ref_act is not None
                and source_act is not None
                and int(np.asarray(_to_numpy_array(source_act)).size) > 0
            ):
                act = _to_numpy_array(source_act, dtype=target_data.act.dtype).reshape(
                    -1
                )
                target_data.act[self._target_act_idx] = act[self._source_act_idx]

        mujoco.mj_forward(self.target_model, target_data)

    def project_target_action_to_source(
        self,
        target_action: np.ndarray,
        *,
        fill_value: float = 0.0,
    ) -> np.ndarray:
        """Project a target-model action vector into source-model actuator space.

        Args:
            target_action: Action vector in ``target_model`` actuator order.
            fill_value: Value assigned to source actuators with no target counterpart.

        Returns:
            Source-space action vector with shape ``(source_model.nu,)``.
        """

        action = np.asarray(target_action, dtype=np.float32).reshape(-1)
        if action.shape[0] != int(self.target_model.nu):
            raise ValueError(
                "target_action has width "
                f"{action.shape[0]}, expected {int(self.target_model.nu)}."
            )
        projected = np.full((int(self.source_model.nu),), fill_value, dtype=np.float32)
        if self._source_act_idx.size:
            projected[self._source_act_idx] = action[self._target_act_idx]
        return projected

    def _build_shared_joint_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for source_joint_id in range(int(self.source_model.njnt)):
            joint_name = mujoco.mj_id2name(
                self.source_model,
                mujoco.mjtObj.mjOBJ_JOINT,
                source_joint_id,
            )
            if not joint_name:
                continue
            if (
                mujoco.mj_name2id(
                    self.target_model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    joint_name,
                )
                >= 0
            ):
                names.append(joint_name)
        return tuple(names)

    def _build_joint_index_array(
        self, *, use_qvel: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        source_idx: list[int] = []
        target_idx: list[int] = []
        for source_joint_id in range(int(self.source_model.njnt)):
            joint_name = mujoco.mj_id2name(
                self.source_model,
                mujoco.mjtObj.mjOBJ_JOINT,
                source_joint_id,
            )
            if not joint_name:
                continue
            target_joint_id = mujoco.mj_name2id(
                self.target_model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            if target_joint_id < 0:
                continue

            source_type = int(self.source_model.jnt_type[source_joint_id])
            target_type = int(self.target_model.jnt_type[target_joint_id])
            if source_type != target_type:
                raise ValueError(
                    f"Joint {joint_name!r} has incompatible types: "
                    f"{mujoco.mjtJoint(source_type).name} vs {mujoco.mjtJoint(target_type).name}."
                )

            if use_qvel:
                width = _joint_qvel_size(source_type)
                source_start = int(self.source_model.jnt_dofadr[source_joint_id])
                target_start = int(self.target_model.jnt_dofadr[target_joint_id])
            else:
                width = _joint_qpos_size(source_type)
                source_start = int(self.source_model.jnt_qposadr[source_joint_id])
                target_start = int(self.target_model.jnt_qposadr[target_joint_id])
            source_idx.extend(range(source_start, source_start + width))
            target_idx.extend(range(target_start, target_start + width))
        return (
            np.asarray(source_idx, dtype=np.int32),
            np.asarray(target_idx, dtype=np.int32),
        )

    def _build_actuator_indices(self) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        target_actuator_ids = {
            name: act_id
            for act_id in range(int(self.target_model.nu))
            if (
                name := mujoco.mj_id2name(
                    self.target_model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    act_id,
                )
            )
        }

        source_idx: list[int] = []
        target_idx: list[int] = []
        shared_names: list[str] = []
        for source_act_id in range(int(self.source_model.nu)):
            name = mujoco.mj_id2name(
                self.source_model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                source_act_id,
            )
            if not name or name not in target_actuator_ids:
                continue
            source_idx.append(source_act_id)
            target_idx.append(target_actuator_ids[name])
            shared_names.append(name)
        return (
            np.asarray(source_idx, dtype=np.int32),
            np.asarray(target_idx, dtype=np.int32),
            tuple(shared_names),
        )


class BridgedPredictPolicy:
    """Run a target-model ``predict`` policy on source-model MuJoCo state.

    Args:
        source_model: Runtime model being controlled.
        target_model: Model expected by the wrapped policy.
        obs_builder: Callable building the target-model observation from
            ``(target_data, frame_idx)``.
        policy: Policy exposing a numpy ``predict(obs)`` method.
        clip_frame_count: Number of reference frames used by ``obs_builder``.
        ctrl_dt: Control timestep used to convert simulation time to frame index.
        source_action_fill: Fill value for unmapped source actuators before any
            final action transform.
        source_action_transform: Optional post-processing applied to projected
            source actions, for example mapping ``[-1, 1]`` logits to ``[0, 1]``.
        target_keyframe: Reference keyframe for target-only state.
        source_entity_name: Optional scene entity key used by
            :meth:`predict_from_env` for mjlab environments.
        source_env_idx: Default batch index used by :meth:`predict_from_env`
            for batched mjlab environments.
        source_env: Optional bound env-like object used by :meth:`__call__`.
        output_device: Optional torch device used by :meth:`__call__` when
            returning a batched tensor action.
    """

    def __init__(
        self,
        source_model: mujoco.MjModel,
        target_model: mujoco.MjModel,
        obs_builder: Any,
        policy: Any,
        *,
        clip_frame_count: int,
        ctrl_dt: float,
        source_action_fill: float = 0.0,
        source_action_transform: Any | None = None,
        target_keyframe: int = 0,
        source_entity_name: str | None = None,
        source_env_idx: int = 0,
        source_env: Any | None = None,
        output_device: str | None = None,
    ) -> None:
        if int(clip_frame_count) <= 0:
            raise ValueError(
                f"clip_frame_count must be positive, got {clip_frame_count}."
            )
        if float(ctrl_dt) <= 0.0:
            raise ValueError(f"ctrl_dt must be positive, got {ctrl_dt}.")
        self._bridge = SharedModelStateBridge(
            source_model,
            target_model,
            target_keyframe=target_keyframe,
        )
        self._target_model = target_model
        self._target_data = mujoco.MjData(target_model)
        if int(target_model.nkey) > target_keyframe:
            mujoco.mj_resetDataKeyframe(
                target_model, self._target_data, target_keyframe
            )
        else:
            mujoco.mj_resetData(target_model, self._target_data)
        mujoco.mj_forward(target_model, self._target_data)
        self._obs_builder = obs_builder
        self._policy = policy
        self._clip_frame_count = int(clip_frame_count)
        self._ctrl_dt = float(ctrl_dt)
        self._source_action_fill = float(source_action_fill)
        self._source_action_transform = source_action_transform
        self._source_entity_name = source_entity_name
        self._source_env_idx = int(source_env_idx)
        self._source_env = source_env
        self._output_device = output_device

    @property
    def bridge(self) -> SharedModelStateBridge:
        """Return the underlying shared-state bridge."""

        return self._bridge

    def frame_idx_from_time(self, sim_time: float) -> int:
        """Return the reference-frame index for a source simulation time."""

        return int(round(float(sim_time) / self._ctrl_dt)) % self._clip_frame_count

    def _predict_from_target_state(self, *, frame_idx: int) -> np.ndarray:
        obs = np.asarray(
            self._obs_builder(self._target_data, frame_idx),
            dtype=np.float32,
        )
        target_action = np.asarray(self._policy.predict(obs), dtype=np.float32)
        source_action = self._bridge.project_target_action_to_source(
            target_action,
            fill_value=self._source_action_fill,
        )
        if self._source_action_transform is None:
            return source_action
        return np.asarray(
            self._source_action_transform(source_action),
            dtype=np.float32,
        )

    def predict_from_source_data(self, source_data: mujoco.MjData) -> np.ndarray:
        """Predict a source-model action from live source MuJoCo state."""

        return self.predict_from_source_state(
            source_qpos=source_data.qpos,
            source_qvel=source_data.qvel,
            source_ctrl=source_data.ctrl,
            source_act=getattr(source_data, "act", None),
            sim_time=source_data.time,
        )

    def predict_from_source_state(
        self,
        *,
        source_qpos: Any,
        source_qvel: Any,
        source_ctrl: Any,
        sim_time: Any,
        source_act: Any | None = None,
    ) -> np.ndarray:
        """Predict a source-model action from raw source state arrays."""

        frame_idx = self.frame_idx_from_time(
            float(_to_numpy_array(sim_time).reshape(-1)[0])
        )
        self._bridge.copy_source_state_into_target(
            source_qpos=source_qpos,
            source_qvel=source_qvel,
            source_ctrl=source_ctrl,
            source_act=source_act,
            target_data=self._target_data,
        )
        return self._predict_from_target_state(frame_idx=frame_idx)

    def predict_from_env(
        self,
        source_env: Any,
        *,
        entity_name: str | None = None,
        env_idx: int | None = None,
    ) -> np.ndarray:
        """Predict a source-model action from a CPU or mjlab env-like object."""

        unwrapped = _unwrap_env_like(source_env)
        if hasattr(unwrapped, "data") and hasattr(unwrapped, "model"):
            return self.predict_from_source_data(unwrapped.data)

        source_entity_name = entity_name or self._source_entity_name
        if source_entity_name is not None and hasattr(unwrapped, "scene"):
            live_data = unwrapped.scene[source_entity_name].data.data
            live_env_idx = self._source_env_idx if env_idx is None else int(env_idx)
            return self.predict_from_source_state(
                source_qpos=live_data.qpos[live_env_idx],
                source_qvel=live_data.qvel[live_env_idx],
                source_ctrl=live_data.ctrl[live_env_idx],
                source_act=(
                    live_data.act[live_env_idx]
                    if getattr(live_data, "act", None) is not None
                    else None
                ),
                sim_time=live_data.time[live_env_idx],
            )
        raise TypeError(
            "predict_from_env expected a CPU env with model/data or an mjlab env "
            "with scene state and a configured source_entity_name."
        )

    def __call__(self, obs: Any) -> Any:
        """Return a batched action for a bound env-like source.

        This is a thin compatibility shim for mjlab-style inference loops that
        expect a callable policy taking an observation argument. The observation
        is ignored because the bridge reads the live state from ``source_env``.
        """

        del obs
        if self._source_env is None:
            raise TypeError(
                "__call__ requires source_env to be configured on "
                "BridgedPredictPolicy."
            )
        action = self.predict_from_env(self._source_env)[None, :]
        if self._output_device is None:
            return action
        import torch

        return torch.as_tensor(
            action,
            device=self._output_device,
            dtype=torch.float32,
        )


class TensorDictPredictPolicyAdapter:
    """Adapt a callable TensorDict policy to a numpy ``predict(obs)`` API.

    This is useful for reusing mjlab/RSL-RL inference callables with
    :class:`BridgedPredictPolicy`, which expects the wrapped policy to expose a
    synchronous ``predict(obs)`` method returning a flat action vector.

    Args:
        policy: Callable accepting a TensorDict with an ``actor`` observation key.
        device: Torch device hosting the wrapped policy.
        obs_key: Observation-group key expected by the wrapped policy.
    """

    def __init__(
        self,
        policy: Any,
        *,
        device: str = "cpu",
        obs_key: str = "actor",
    ) -> None:
        self._policy = policy
        self._device = str(device)
        self._obs_key = str(obs_key)

    def predict(self, obs: Any) -> np.ndarray:
        """Return a flat numpy action for one observation vector."""

        import torch
        from tensordict import TensorDict

        obs_np = _to_numpy_array(obs, dtype=np.float32).reshape(1, -1)
        obs_tensor = torch.as_tensor(
            obs_np,
            device=self._device,
            dtype=torch.float32,
        )
        obs_td = TensorDict(
            {self._obs_key: obs_tensor},
            batch_size=[obs_tensor.shape[0]],
            device=obs_tensor.device,
        )
        action = self._policy(obs_td)
        if isinstance(action, (tuple, list)):
            if len(action) < 1:
                raise ValueError("Wrapped policy returned an empty tuple/list action.")
            action = action[0]
        return _to_numpy_array(action, dtype=np.float32).reshape(-1)

    def reset(self) -> None:
        """Forward resets to the wrapped policy when available."""

        reset_fn = getattr(self._policy, "reset", None)
        if callable(reset_fn):
            reset_fn()


def to_muscle_activations(action: Any) -> np.ndarray:
    """Map checkpoint logits in ``[-1, 1]`` into muscle activations in ``[0, 1]``."""

    action_np = _to_numpy_array(action, dtype=np.float32).reshape(-1)
    return np.clip(0.5 * (action_np + 1.0), 0.0, 1.0).astype(np.float32)


def make_fullbody_checkpoint_bridged_policy(
    source_model: mujoco.MjModel,
    policy: Any,
    *,
    checkpoint_root: str | Path,
    motion_path: str | Path,
    ctrl_dt: float,
    source_action_fill: float = -1.0,
    source_action_transform: Any = to_muscle_activations,
    target_keyframe: int = 0,
    source_entity_name: str | None = None,
    source_env_idx: int = 0,
    source_env: Any | None = None,
    output_device: str | None = None,
    policy_device: str | None = None,
) -> BridgedPredictPolicy:
    """Build a full-body checkpoint bridge around a native env policy.

    The returned policy reads live source-model state, reconstructs the
    full-body checkpoint observation expected by the wrapped policy, and
    projects the checkpoint action back into the source model's actuator space.

    Args:
        source_model: Native runtime model being controlled.
        policy: Wrapped policy. Policies exposing ``predict(obs)`` are used
            directly; callable TensorDict policies are adapted automatically.
        checkpoint_root: Boxing/full-body checkpoint root containing the obs
            adapter metadata.
        motion_path: Reference motion clip used by the checkpoint obs adapter.
        ctrl_dt: Source env control timestep used for frame alignment.
        source_action_fill: Fill value for source actuators without a mapped
            checkpoint counterpart.
        source_action_transform: Optional transform applied after projecting the
            checkpoint action back into the source actuator space.
        target_keyframe: Reference keyframe for the full-body target model.
        source_entity_name: Optional mjlab scene entity name used by
            :meth:`BridgedPredictPolicy.predict_from_env`.
        source_env_idx: Default batched mjlab env index.
        source_env: Optional bound env-like object used by
            :meth:`BridgedPredictPolicy.__call__`.
        output_device: Optional torch output device for batched ``__call__``.
        policy_device: Optional torch device hosting a callable TensorDict
            policy. Defaults to ``output_device`` or ``"cpu"``.

    Returns:
        A :class:`BridgedPredictPolicy` configured for the full-body checkpoint
        observation/action contract.
    """

    from myosuite.integrations.musclemimic.fullbody_checkpoint_io import (
        resolve_checkpoint_ref,
    )
    from myosuite.integrations.musclemimic.fullbody_local_policy import (
        FullbodyObsAdapter,
        fullbody_obs_adapter_params_from_metadata,
        read_checkpoint_config_metadata,
    )
    from myosuite.integrations.musclemimic.fullbody_model import (
        compile_mimic_fullbody_mjmodel,
        default_mimic_fullbody_config,
    )
    from myosuite.integrations.musclemimic.trajectory_io import load_motion_clip

    resolved_checkpoint = resolve_checkpoint_ref(str(checkpoint_root))
    metadata = read_checkpoint_config_metadata(Path(resolved_checkpoint.local_path))
    target_model, _, _ = compile_mimic_fullbody_mjmodel(default_mimic_fullbody_config())
    clip = load_motion_clip(
        Path(motion_path),
        expected_nq=target_model.nq,
        expected_nv=target_model.nv,
    )
    obs_adapter = FullbodyObsAdapter(
        target_model,
        clip,
        fullbody_obs_adapter_params_from_metadata(metadata),
    )
    predict_policy = (
        policy
        if hasattr(policy, "predict")
        else TensorDictPredictPolicyAdapter(
            policy,
            device=policy_device or output_device or "cpu",
        )
    )
    return BridgedPredictPolicy(
        source_model=source_model,
        target_model=target_model,
        obs_builder=obs_adapter.build,
        policy=predict_policy,
        clip_frame_count=int(obs_adapter._traj_len),
        ctrl_dt=float(ctrl_dt),
        source_action_fill=float(source_action_fill),
        source_action_transform=source_action_transform,
        target_keyframe=int(target_keyframe),
        source_entity_name=source_entity_name,
        source_env_idx=int(source_env_idx),
        source_env=source_env,
        output_device=output_device,
    )


__all__ = [
    "BridgedPredictPolicy",
    "SharedModelStateBridge",
    "TensorDictPredictPolicyAdapter",
    "make_fullbody_checkpoint_bridged_policy",
    "to_muscle_activations",
]
