# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""CPU Gymnasium environment for MuscleMimic clip-following with DeepMimic reward.

Wraps a MuscleMimic full-body model and drives it to track a retargeted
MotionClip using the multi-term DeepMimic reward and k-step lookahead
observations matching the MuscleMimic paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.utils import EzPickle

import gymnasium as gym
from myosuite.integrations.musclemimic.fullbody_model import (
    FULLBODY_BODY2SITES_FOR_MIMIC,
    compile_mimic_fullbody_mjmodel,
    default_mimic_fullbody_config,
)
from myosuite.physics.running_stats import RunningMeanStd, normalize, update
from myosuite.terms.mimic_obs import (
    mimic_lookahead_obs,
    mimic_lookahead_obs_size,
    mimic_should_terminate,
)
from myosuite.terms.mimic_reward import (
    MimicRewardScales,
    MimicRewardWeights,
    mimic_composite_reward,
)


# ---------------------------------------------------------------------------
# Site-order mapping between clip and model
# ---------------------------------------------------------------------------

# The clip stores sites in this order (from musclemimic retargeting).
_CLIP_SITE_ORDER: tuple[str, ...] = (
    "upper_body_mimic",
    "head_mimic",
    "right_shoulder_mimic",
    "right_elbow_mimic",
    "right_hand_mimic",
    "left_shoulder_mimic",
    "left_elbow_mimic",
    "left_hand_mimic",
    "pelvis_mimic",
    "right_hip_mimic",
    "right_knee_mimic",
    "right_ankle_mimic",
    "right_toes_mimic",
    "left_hip_mimic",
    "left_knee_mimic",
    "left_ankle_mimic",
    "left_toes_mimic",
)

# Model site order (from FULLBODY_BODY2SITES_FOR_MIMIC.values())
_MODEL_SITE_ORDER: tuple[str, ...] = tuple(FULLBODY_BODY2SITES_FOR_MIMIC.values())

# Map: model_index → clip_index
_CLIP_IDX_FOR_MODEL: np.ndarray = np.array(
    [_CLIP_SITE_ORDER.index(s) for s in _MODEL_SITE_ORDER], dtype=np.int32
)

_CLIP_QPOS_NAME_KEYS: tuple[str, ...] = (
    "qpos_names",
    "qpos_joint_names",
    "joint_names",
)
_CLIP_QVEL_NAME_KEYS: tuple[str, ...] = (
    "qvel_names",
    "qvel_joint_names",
    "joint_names",
)


@dataclass(frozen=True)
class _ClipJointFieldSelection:
    """Mapping from clip columns to model qpos/qvel indices."""

    clip_indices: np.ndarray
    model_indices: np.ndarray
    joint_names: tuple[str, ...]


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


def _decode_joint_names(
    npz: np.lib.npyio.NpzFile,
    candidate_keys: tuple[str, ...],
) -> tuple[str, ...] | None:
    for key in candidate_keys:
        if key not in npz.files:
            continue
        raw = np.asarray(npz[key]).reshape(-1)
        names = tuple(str(name) for name in raw.tolist())
        if not names:
            raise ValueError(f"Clip metadata key {key!r} is empty.")
        return names
    return None


def _build_model_joint_index_map(
    model: mujoco.MjModel,
    *,
    use_qvel: bool,
) -> dict[str, np.ndarray]:
    index_map: dict[str, np.ndarray] = {}
    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if not joint_name:
            continue
        joint_type = int(model.jnt_type[joint_id])
        if use_qvel:
            start = int(model.jnt_dofadr[joint_id])
            size = _joint_qvel_size(joint_type)
        else:
            start = int(model.jnt_qposadr[joint_id])
            size = _joint_qpos_size(joint_type)
        index_map[joint_name] = np.arange(start, start + size, dtype=np.int32)
    return index_map


def _resolve_joint_field_selection(
    *,
    model: mujoco.MjModel,
    clip_width: int,
    expected_width: int,
    clip_joint_names: tuple[str, ...] | None,
    field_name: str,
    use_qvel: bool,
) -> _ClipJointFieldSelection:
    if clip_joint_names is None:
        if clip_width != expected_width:
            raise ValueError(
                f"Clip {field_name} width mismatch: expected {expected_width}, got "
                f"{clip_width}. Provide joint-name metadata via "
                f"{field_name}_names/{field_name}_joint_names or joint_names."
            )
        return _ClipJointFieldSelection(
            clip_indices=np.arange(expected_width, dtype=np.int32),
            model_indices=np.arange(expected_width, dtype=np.int32),
            joint_names=tuple(
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                or f"joint_{joint_id}"
                for joint_id in range(model.njnt)
            ),
        )

    model_index_map = _build_model_joint_index_map(model, use_qvel=use_qvel)
    missing = sorted(set(clip_joint_names) - set(model_index_map))
    if missing:
        raise ValueError(
            f"Clip {field_name} metadata references unknown model joints: {missing}."
        )

    if len(clip_joint_names) == clip_width:
        counters = {name: 0 for name in clip_joint_names}
        model_indices: list[int] = []
        for joint_name in clip_joint_names:
            joint_indices = model_index_map[joint_name]
            offset = counters[joint_name]
            if offset >= len(joint_indices):
                raise ValueError(
                    f"Clip {field_name} metadata over-specifies joint {joint_name!r}."
                )
            model_indices.append(int(joint_indices[offset]))
            counters[joint_name] = offset + 1
        bad_counts = sorted(
            name
            for name, count in counters.items()
            if count != len(model_index_map[name])
        )
        if bad_counts:
            raise ValueError(
                f"Coordinate-level {field_name} metadata must cover complete joints; "
                f"got incomplete coverage for {bad_counts}."
            )
        return _ClipJointFieldSelection(
            clip_indices=np.arange(clip_width, dtype=np.int32),
            model_indices=np.asarray(model_indices, dtype=np.int32),
            joint_names=clip_joint_names,
        )

    if len(set(clip_joint_names)) != len(clip_joint_names):
        raise ValueError(
            f"Joint-level {field_name} metadata must not repeat joint names: "
            f"{clip_joint_names}."
        )
    model_indices = np.concatenate(
        [model_index_map[joint_name] for joint_name in clip_joint_names]
    )
    if int(model_indices.size) != clip_width:
        raise ValueError(
            f"Clip {field_name} width {clip_width} does not match the selected model "
            f"joint coordinates {int(model_indices.size)} for {clip_joint_names}."
        )
    return _ClipJointFieldSelection(
        clip_indices=np.arange(clip_width, dtype=np.int32),
        model_indices=model_indices.astype(np.int32),
        joint_names=clip_joint_names,
    )


def _expand_clip_field(
    default_values: np.ndarray,
    clip_values: np.ndarray,
    selection: _ClipJointFieldSelection,
) -> np.ndarray:
    expanded = np.repeat(default_values[None, :], clip_values.shape[0], axis=0)
    expanded[:, selection.model_indices] = clip_values[:, selection.clip_indices]
    return expanded


def _select_required_indices(
    expanded_values: np.ndarray,
    selection: _ClipJointFieldSelection,
    required_indices: range,
) -> np.ndarray | None:
    required = np.asarray(tuple(required_indices), dtype=np.int32)
    if not np.isin(required, selection.model_indices).all():
        return None
    return expanded_values[:, required]


class MuscleMimicClipEnvV0(gym.Env, EzPickle):
    """CPU Gymnasium env: DeepMimic-style clip tracking on MyoFullBody.

    Args:
        clip_path: Path to a retargeted .npz motion clip.
            Full-width clips use the historical ``qpos`` / ``qvel`` layout.
            Partial clips may instead provide ``qpos_names`` / ``qvel_names``
            (or shared ``joint_names``) naming the model joints covered by the
            reduced arrays; qpos/qvel reward terms then track only that subset.
        lookahead_k: Number of lookahead frames.
        lookahead_stride: Frame stride for lookahead.
        site_err_threshold: Early-termination site-error threshold (metres).
        root_err_threshold: Early-termination root-error threshold (metres).
        use_obs_normalizer: Whether to update and apply RunningMeanStd on obs.
        frame_skip: Number of simulation sub-steps per control step.
        seed: Optional RNG seed.
    """

    def __init__(
        self,
        clip_path: str | Path,
        lookahead_k: int = 5,
        lookahead_stride: int = 20,
        site_err_threshold: float = 1.0,
        root_err_threshold: float = 0.3,
        use_obs_normalizer: bool = True,
        frame_skip: int = 5,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        gym.Env.__init__(self)
        EzPickle.__init__(
            self,
            clip_path,
            lookahead_k,
            lookahead_stride,
            site_err_threshold,
            root_err_threshold,
            use_obs_normalizer,
            frame_skip,
            seed,
            **kwargs,
        )
        self.frame_skip = frame_skip
        self.np_random = np.random.default_rng(seed)
        self._lookahead_k = lookahead_k
        self._lookahead_stride = lookahead_stride
        self._site_err_threshold = site_err_threshold
        self._root_err_threshold = root_err_threshold
        self._use_obs_normalizer = use_obs_normalizer
        self._reward_weights = MimicRewardWeights()
        self._reward_scales = MimicRewardScales()

        # Build model
        cfg = default_mimic_fullbody_config()
        self.model, self._mj_spec, _ = compile_mimic_fullbody_mjmodel(cfg)
        self.data = mujoco.MjData(self.model)
        self._ctrl_dt = float(self.model.opt.timestep * self.frame_skip)
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._default_qpos = self.data.qpos.copy()
        self._default_qvel = self.data.qvel.copy()
        self._default_act = self.data.act.copy()

        # Resolve site IDs in model order
        self._site_ids: np.ndarray = np.array(
            [self.model.site(_name).id for _name in _MODEL_SITE_ORDER], dtype=np.int32
        )

        # Load clip
        clip_path = Path(clip_path)
        npz = np.load(clip_path, allow_pickle=True)
        if (
            "qpos" not in npz.files
            or "qvel" not in npz.files
            or "site_xpos" not in npz.files
        ):
            raise KeyError("Clip NPZ must contain qpos, qvel, and site_xpos arrays.")
        raw_qpos = npz["qpos"].astype(np.float64)
        raw_qvel = npz["qvel"].astype(np.float64)
        if raw_qpos.ndim != 2:
            raise ValueError(f"Clip qpos must be rank-2, got shape {raw_qpos.shape}.")
        if raw_qvel.ndim != 2:
            raise ValueError(f"Clip qvel must be rank-2, got shape {raw_qvel.shape}.")
        qpos_names = _decode_joint_names(npz, _CLIP_QPOS_NAME_KEYS)
        qvel_names = _decode_joint_names(npz, _CLIP_QVEL_NAME_KEYS)
        self._qpos_selection = _resolve_joint_field_selection(
            model=self.model,
            clip_width=int(raw_qpos.shape[1]),
            expected_width=int(self.model.nq),
            clip_joint_names=qpos_names,
            field_name="qpos",
            use_qvel=False,
        )
        self._qvel_selection = _resolve_joint_field_selection(
            model=self.model,
            clip_width=int(raw_qvel.shape[1]),
            expected_width=int(self.model.nv),
            clip_joint_names=qvel_names,
            field_name="qvel",
            use_qvel=True,
        )
        self._clip_qpos: np.ndarray = _expand_clip_field(
            self._default_qpos, raw_qpos, self._qpos_selection
        )
        self._clip_qvel: np.ndarray = _expand_clip_field(
            self._default_qvel, raw_qvel, self._qvel_selection
        )
        self._clip_joint_qpos: np.ndarray = raw_qpos[
            :, self._qpos_selection.clip_indices
        ]
        self._clip_joint_qvel: np.ndarray = raw_qvel[
            :, self._qvel_selection.clip_indices
        ]
        self._clip_root_pos: np.ndarray | None = _select_required_indices(
            self._clip_qpos, self._qpos_selection, range(0, 3)
        )
        self._clip_root_quat: np.ndarray | None = _select_required_indices(
            self._clip_qpos, self._qpos_selection, range(3, 7)
        )
        self._clip_root_vel: np.ndarray | None = _select_required_indices(
            self._clip_qvel, self._qvel_selection, range(0, 3)
        )
        # clip site_xpos shape (T, 17, 3) — reorder to model site order
        raw_sites = npz["site_xpos"].astype(np.float64)  # (T, 17, 3)
        if raw_sites.ndim != 3 or raw_sites.shape[1:] != (len(_CLIP_SITE_ORDER), 3):
            raise ValueError(
                "Clip site_xpos must have shape "
                f"(T, {len(_CLIP_SITE_ORDER)}, 3), got {raw_sites.shape}."
            )
        self._clip_site_xpos: np.ndarray = raw_sites[:, _CLIP_IDX_FOR_MODEL, :]
        self._clip_T: int = self._clip_qpos.shape[0]
        self._frame_offset: int = 0  # random start frame, set at reset
        self._step_count: int = 0

        # Running stats for obs normalisation
        obs_dim = self._obs_dim()
        self._running_stats: RunningMeanStd = RunningMeanStd.zeros(obs_dim)

        self._setup_spaces()

    # ------------------------------------------------------------------

    def _obs_dim(self) -> int:
        nq = self.model.nq if hasattr(self, "model") else 89
        nv = self.model.nv if hasattr(self, "model") else 88
        na = self.model.na if hasattr(self, "model") else 354
        lookahead = mimic_lookahead_obs_size(
            n_sites=len(_MODEL_SITE_ORDER),
            has_root_pos=self._clip_root_pos is not None,
            has_root_vel=self._clip_root_vel is not None,
            k=self._lookahead_k,
        )
        return nq + nv + na + lookahead

    def _setup_spaces(self) -> None:
        ctrl = self.model.actuator_ctrlrange.astype(np.float32)
        self.action_space = spaces.Box(
            low=ctrl[:, 0], high=ctrl[:, 1], dtype=np.float32
        )
        obs_dim = self._obs_dim()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    # ------------------------------------------------------------------

    def _current_frame(self) -> int:
        step = int(round(self.data.time / self._ctrl_dt))
        return (self._frame_offset + step) % self._clip_T

    def _build_obs(self) -> np.ndarray:
        frame = self._current_frame()
        root_pos = self.data.qpos[:3].copy()
        la = mimic_lookahead_obs(
            current_frame=frame,
            clip_site_xpos=self._clip_site_xpos,
            clip_root_pos=self._clip_root_pos,
            clip_root_vel=self._clip_root_vel,
            current_root_pos=root_pos.astype(np.float32),
            k=self._lookahead_k,
            stride=self._lookahead_stride,
        )
        obs = np.concatenate(
            [
                self.data.qpos.astype(np.float32),
                self.data.qvel.astype(np.float32),
                self.data.act.astype(np.float32),
                la,
            ]
        )
        if self._use_obs_normalizer:
            self._running_stats = update(self._running_stats, obs)
            obs = normalize(self._running_stats, obs)
        return obs

    def _compute_reward(self) -> tuple[float, dict[str, Any]]:
        frame = self._current_frame()
        ref_sites = self._clip_site_xpos[frame]  # (17, 3)
        ref_qpos = self._clip_joint_qpos[frame]
        ref_qvel = self._clip_joint_qvel[frame]
        ref_root_pos = (
            None if self._clip_root_pos is None else self._clip_root_pos[frame]
        )
        ref_root_vel = (
            None if self._clip_root_vel is None else self._clip_root_vel[frame]
        )
        ref_root_quat = (
            None if self._clip_root_quat is None else self._clip_root_quat[frame]
        )

        current_sites = self.data.site_xpos[self._site_ids].copy()
        current_qpos = self.data.qpos[self._qpos_selection.model_indices].copy()
        current_qvel = self.data.qvel[self._qvel_selection.model_indices].copy()
        current_root_pos = None if ref_root_pos is None else self.data.qpos[:3].copy()
        current_root_vel = None if ref_root_vel is None else self.data.qvel[:3].copy()
        current_root_quat = (
            None if ref_root_quat is None else self.data.qpos[3:7].copy()
        )

        result = mimic_composite_reward(
            np,
            current_sites,
            ref_sites,
            current_qpos,
            ref_qpos,
            current_qvel,
            ref_qvel,
            current_root_pos,
            ref_root_pos,
            current_root_vel,
            ref_root_vel,
            current_root_quat,
            ref_root_quat,
            weights=self._reward_weights,
            scales=self._reward_scales,
        )
        return float(result["dense"]), result

    def _check_termination(self) -> bool:
        frame = self._current_frame()
        ref_sites = self._clip_site_xpos[frame]
        ref_root_pos = (
            None if self._clip_root_pos is None else self._clip_root_pos[frame]
        )
        current_sites = self.data.site_xpos[self._site_ids].copy()
        current_root_pos = self.data.qpos[:3]
        return mimic_should_terminate(
            current_sites,
            ref_sites,
            current_root_pos,
            ref_root_pos,
            site_err_threshold=self._site_err_threshold,
            root_err_threshold=self._root_err_threshold,
        )

    # ------------------------------------------------------------------
    # Gymnasium interface

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        # Random start frame
        self._frame_offset = int(self.np_random.integers(0, self._clip_T))
        self._step_count = 0

        # Set state from clip
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
        self.data.act[:] = self._default_act
        self.data.qpos[:] = self._clip_qpos[self._frame_offset]
        self.data.qvel[:] = self._clip_qvel[self._frame_offset]
        mujoco.mj_forward(self.model, self.data)

        obs = self._build_obs()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        obs = self._build_obs()
        reward, info = self._compute_reward()
        terminated = self._check_termination()
        truncated = False  # episode length managed externally

        info.update(
            {
                "step": self._step_count,
                "frame": self._current_frame(),
            }
        )
        return obs, reward, terminated, truncated, info
