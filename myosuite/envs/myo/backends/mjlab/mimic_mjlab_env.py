# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
# pylint: disable=import-error,no-member,broad-except
"""mjlab task wiring for Mimic bimanual and full-body.

Registers ``myoMimicBimanual-v0`` and ``myoMimicFullbody-v0`` (and legacy
``myoMuscleMimic*`` aliases) with
``mjlab.tasks.registry.register_mjlab_task`` so
``make_env(..., backend="mjlab")`` can construct a
:class:`~mjlab.envs.ManagerBasedRlEnv` for these ids when ``musclemimic_models``
and mjlab are installed.

Two target-sourcing modes
-------------------------
**Random (default)**
    Episode targets are sampled uniformly from the bounding box defined in the
    model's tracking config.  Matches the CPU / MJX behaviour when no motion
    clip is supplied.

**Trajectory (when a** :class:`~myosuite.core.trajectory_io.MotionClip` **is provided)**
    Targets are taken from the clip's ``site_xpos`` array at the frame
    corresponding to each environment's current simulation time.  Each of the N
    parallel environments starts at a *different random frame* in the clip;
    on episode reset that offset is resampled.  This produces a diverse
    distribution of motion phases across the batch while keeping each episode's
    target sequence coherent with the reference motion.

    Trajectory mode also exposes additional observation terms:

    ``clip_ref_qpos``
        Reference joint positions from the clip at the current frame
        ``(N, nq)``.  Added when ``clip.qpos`` is available.

    ``clip_ref_qvel``
        Reference joint velocities from the clip ``(N, nv)``.  Added when
        ``clip.qvel`` is available.

    ``clip_phase``
        Normalised position in ``[0, 1]`` along the clip ``(N, 1)``.

Usage::

    from pathlib import Path
    from myosuite.core.trajectory_io import load_motion_clip
    from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import register_mimic_mjlab_tasks_with_clip

    clip = load_motion_clip(Path("walk.npz"), expected_nq=69, expected_nv=68)

    # At mjlab startup, after calling register_mjlab_task:
    register_mimic_mjlab_tasks_with_clip(
        register_mjlab_task=mjlab.tasks.registry.register_mjlab_task,
        rl_cfg_fn=lambda: mjlab.runner.DefaultRlCfg(),
        clip=clip,
    )
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from myosuite.core.trajectory_io import MotionClip
from myosuite.terms.mimic_reward import MimicTrackingConfig

if TYPE_CHECKING:
    from myosuite.envs.myo.backends.mjlab.clip_trajectory_source import (
        ClipTrajectorySource,
        MultiClipTrajectorySource,
    )
    from myosuite.integrations.musclemimic.sar_extraction import SynergyModel

# ---------------------------------------------------------------------------
# Module-level cache keyed by (env_instance_id, entity_name, variant).
# Populated lazily on first obs/reward evaluation.
# ---------------------------------------------------------------------------

_mimic_mjlab_cache: dict[tuple[int, str, str], dict[str, Any]] = {}
_MIMIC_REWARD_MODE_MIMIC = "mimic"
_MIMIC_REWARD_MODE_ENV = "env"
_MIMIC_REWARD_MODE_AUGMENTED = "augmented"
_VALID_MIMIC_REWARD_MODES = (
    _MIMIC_REWARD_MODE_MIMIC,
    _MIMIC_REWARD_MODE_ENV,
    _MIMIC_REWARD_MODE_AUGMENTED,
)


def _normalize_motion_clip_bank(
    clip: MotionClip | tuple[MotionClip, ...] | list[MotionClip] | None,
) -> tuple[MotionClip, ...]:
    """Return *clip* as a validated clip tuple."""
    if clip is None:
        return ()
    if isinstance(clip, tuple):
        bank = clip
    elif isinstance(clip, list):
        bank = tuple(clip)
    else:
        bank = (clip,)
    if not bank:
        raise ValueError("At least one motion clip is required.")
    return bank


def _canonicalize_clip_sites(
    clip: MotionClip,
    *,
    required_site_names: tuple[str, ...],
) -> MotionClip:
    """Return *clip* with ``site_xpos`` reordered into *required_site_names*."""
    if clip.site_xpos is None:
        raise ValueError("MotionClip.site_xpos is required for mimic tracking.")
    if clip.site_names is None:
        if int(clip.site_xpos.shape[1]) < len(required_site_names):
            raise ValueError(
                "MotionClip.site_xpos has fewer sites than required for mimic tracking: "
                f"{clip.site_xpos.shape[1]} < {len(required_site_names)}"
            )
        return MotionClip(
            qpos=clip.qpos,
            qvel=clip.qvel,
            site_xpos=clip.site_xpos[:, : len(required_site_names), :],
            site_names=list(required_site_names),
            qpos_joint_names=clip.qpos_joint_names,
            qvel_joint_names=clip.qvel_joint_names,
            qpos_model_indices=clip.qpos_model_indices,
            qvel_model_indices=clip.qvel_model_indices,
            frequency_hz=clip.frequency_hz,
            source_path=clip.source_path,
        )
    try:
        clip_site_ids = np.asarray(
            [clip.site_names.index(name) for name in required_site_names],
            dtype=np.int64,
        )
    except ValueError as exc:
        raise ValueError(
            f"Clip site_names do not contain all required model sites: {exc}"
        ) from exc
    return MotionClip(
        qpos=clip.qpos,
        qvel=clip.qvel,
        site_xpos=clip.site_xpos[:, clip_site_ids, :],
        site_names=list(required_site_names),
        qpos_joint_names=clip.qpos_joint_names,
        qvel_joint_names=clip.qvel_joint_names,
        qpos_model_indices=clip.qpos_model_indices,
        qvel_model_indices=clip.qvel_model_indices,
        frequency_hz=clip.frequency_hz,
        source_path=clip.source_path,
    )


def _normalize_mimic_reward_mode(reward_mode: str) -> str:
    """Validate and normalise the clip-mimic reward composition mode."""
    normalized = reward_mode.strip().lower()
    if normalized not in _VALID_MIMIC_REWARD_MODES:
        raise ValueError(
            "reward_mode must be one of "
            f"{_VALID_MIMIC_REWARD_MODES}, got {reward_mode!r}."
        )
    return normalized


def _reward_mode_uses_mimic_objective(reward_mode: str) -> bool:
    """Return whether *reward_mode* includes the clip-tracking objective."""
    return reward_mode in (_MIMIC_REWARD_MODE_MIMIC, _MIMIC_REWARD_MODE_AUGMENTED)


def _reward_mode_uses_env_objective(reward_mode: str) -> bool:
    """Return whether *reward_mode* includes the native saber task objective."""
    return reward_mode in (_MIMIC_REWARD_MODE_ENV, _MIMIC_REWARD_MODE_AUGMENTED)


def _clip_has_required_indices(
    indices: np.ndarray | None,
    required: range,
) -> bool:
    if indices is None:
        return False
    required_arr = np.asarray(tuple(required), dtype=np.int32)
    return bool(np.isin(required_arr, indices).all())


def _select_clip_columns(values: Any, indices: np.ndarray | None) -> Any:
    if values is None or indices is None:
        return None
    if values.shape[-1] == int(indices.size) and np.array_equal(
        indices, np.arange(int(indices.size), dtype=np.int32)
    ):
        return values
    if hasattr(values, "index_select") and hasattr(values, "device"):
        import torch

        return values.index_select(
            1,
            torch.as_tensor(indices, device=values.device, dtype=torch.long),
        )
    return values[..., indices]


# ---------------------------------------------------------------------------
# Model-level helpers (mujoco-only, no mjlab import)
# ---------------------------------------------------------------------------


def _strip_spec_keyframes(spec: Any) -> None:
    """Remove keyframes from *spec* (mjlab entity adds its own init_state)."""
    for k in list(spec.keys):
        spec.delete(k)


def _init_state_from_model(mj_model: Any) -> Any:
    """Build an ``EntityCfg.InitialStateCfg`` from the model's first keyframe.

    The default keyframe is embedded in the included XML files and only
    surfaces in the *compiled* model.  We extract it here and populate an
    explicit ``InitialStateCfg`` (pos, rot, per-joint dict) so the body
    resets to the correct standing height rather than all-zeros.

    Using the explicit ``joint_pos`` dict path avoids a float64→float32 dtype
    mismatch that occurs when ``joint_pos=None`` and mjlab copies the keyframe
    qpos directly from MuJoCo's float64 buffers into its float32 tensors.
    """
    import mujoco as _mj
    from mjlab.entity import EntityCfg

    if mj_model.nkey == 0:
        return EntityCfg.InitialStateCfg()

    kqpos = mj_model.key_qpos[0]  # float64, shape (nq,)
    kqvel = (
        mj_model.key_qvel[0]
        if getattr(mj_model, "key_qvel", None) is not None
        else None
    )

    pos = (0.0, 0.0, 0.0)
    rot = (1.0, 0.0, 0.0, 0.0)
    lin_vel = (0.0, 0.0, 0.0)
    ang_vel = (0.0, 0.0, 0.0)

    if (
        mj_model.njnt > 0
        and int(mj_model.jnt_type[0]) == int(_mj.mjtJoint.mjJNT_FREE)
        and int(mj_model.jnt_qposadr[0]) == 0
    ):
        pos = tuple(float(x) for x in kqpos[:3])
        rot = tuple(float(x) for x in kqpos[3:7])
        if kqvel is not None and kqvel.shape[0] >= 6:
            lin_vel = tuple(float(x) for x in kqvel[:3])
            ang_vel = tuple(float(x) for x in kqvel[3:6])

    # Per-joint positions for all non-free joints.
    joint_pos: dict[str, float] = {}
    joint_vel: dict[str, float] = {}
    for j in range(mj_model.njnt):
        name = _mj.mj_id2name(mj_model, _mj.mjtObj.mjOBJ_JOINT, j)
        jtype = int(mj_model.jnt_type[j])
        if not name:
            continue
        if jtype in (int(_mj.mjtJoint.mjJNT_SLIDE), int(_mj.mjtJoint.mjJNT_HINGE)):
            qpos_adr = int(mj_model.jnt_qposadr[j])
            joint_pos[name] = float(kqpos[qpos_adr])
            if kqvel is not None:
                dof_adr = int(mj_model.jnt_dofadr[j])
                joint_vel[name] = float(kqvel[dof_adr])

    return EntityCfg.InitialStateCfg(
        pos=pos,
        rot=rot,
        lin_vel=lin_vel,
        ang_vel=ang_vel,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
    )


def _mimic_keyframe_reset_event(
    entity_name: str, mj_model: Any
) -> Callable[[Any, Any], None]:
    """Reset-event that restores the model keyframe, including extra free joints.

    Some mimic variants do not have clip qpos/qvel
    widths that match the articulated model, so RSI cannot be used. They still
    need an explicit reset event because mjlab's default ``reset_scene_to_default``
    only handles the articulation root and joint vectors, not auxiliary free joints
    like detached props embedded in the same entity.
    """
    import mujoco as _mj

    if mj_model.nkey == 0:
        raise ValueError("Keyframe reset requires a model with at least one keyframe.")

    key_qpos = np.asarray(mj_model.key_qpos[0], dtype=np.float32)
    if (
        getattr(mj_model, "key_qvel", None) is not None
        and mj_model.key_qvel.shape[0] > 0
    ):
        key_qvel = np.asarray(mj_model.key_qvel[0], dtype=np.float32)
    else:
        key_qvel = np.zeros(int(mj_model.nv), dtype=np.float32)
    if getattr(mj_model, "key_act", None) is not None and mj_model.key_act.shape[0] > 0:
        key_act = np.asarray(mj_model.key_act[0], dtype=np.float32)
    else:
        key_act = None
    free_qpos_adrs = tuple(
        int(mj_model.jnt_qposadr[j])
        for j in range(mj_model.njnt)
        if int(mj_model.jnt_type[j]) == int(_mj.mjtJoint.mjJNT_FREE)
    )

    def _fn(env: Any, env_ids: Any) -> None:
        import torch

        data = env.scene[entity_name].data.data
        device = data.qpos.device
        n_envs = int(data.qpos.shape[0])

        if env_ids is None:
            env_ids_long = torch.arange(n_envs, device=device, dtype=torch.long)
        else:
            env_ids_long = torch.as_tensor(
                env_ids, device=device, dtype=torch.long
            ).reshape(-1)

        n_reset = int(env_ids_long.shape[0])
        qpos = (
            torch.as_tensor(key_qpos, device=device, dtype=data.qpos.dtype)
            .unsqueeze(0)
            .repeat(n_reset, 1)
        )
        qvel = (
            torch.as_tensor(key_qvel, device=device, dtype=data.qvel.dtype)
            .unsqueeze(0)
            .repeat(n_reset, 1)
        )

        env_origins = getattr(env.scene, "env_origins", None)
        if env_origins is not None and free_qpos_adrs:
            origins = env_origins[env_ids_long].to(device=device, dtype=data.qpos.dtype)
            for qadr in free_qpos_adrs:
                qpos[:, qadr : qadr + 3] += origins

        # TODO: migrate to entity.write_root_state_to_sim + write_joint_state_to_sim.
        # This function writes the full qpos/qvel in one shot, including auxiliary
        # free joints (e.g. detached prop sabers) that are not covered by the
        # entity root+joints decomposition.  Synchronise around the raw Warp writes
        # to prevent CUDA error 700 under certain allocator states (GitHub #40).
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        data.qpos[env_ids_long] = qpos
        data.qvel[env_ids_long] = qvel
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if hasattr(data, "act"):
            if key_act is None:
                data.act[env_ids_long] = 0.0
            else:
                act = (
                    torch.as_tensor(key_act, device=device, dtype=data.act.dtype)
                    .unsqueeze(0)
                    .repeat(n_reset, 1)
                )
                data.act[env_ids_long] = act
        env.sim.forward()

    return _fn


def _muscle_actuator_names(mj_model: Any) -> tuple[str, ...]:
    """Return actuator names whose dynamics type is muscle."""
    import mujoco

    names: list[str] = []
    for i in range(mj_model.nu):
        if int(mj_model.actuator_dyntype[i]) != int(mujoco.mjtDyn.mjDYN_MUSCLE):
            continue
        n = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if n:
            names.append(n)
    return tuple(names)


def _muscle_tendon_names(mj_model: Any) -> tuple[str, ...]:
    """Return tendon names referenced by muscle actuators (for XmlMuscle cfg)."""
    import mujoco

    names: list[str] = []
    for i in range(mj_model.nu):
        if int(mj_model.actuator_dyntype[i]) != int(mujoco.mjtDyn.mjDYN_MUSCLE):
            continue
        tid = int(mj_model.actuator_trnid[i, 0])
        if tid < 0:
            continue
        tn = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_TENDON, tid)
        if tn:
            names.append(tn)
    return tuple(names)


# ---------------------------------------------------------------------------
# Time-coercion helpers
# ---------------------------------------------------------------------------


def _mjlab_sim_time_to_float(tim: Any) -> float:
    """Coerce mjlab ``data.time`` to a Python ``float`` (first env's value).

    Warp-backed fields may be wrapped as ``mjlab.sim.sim_data.TorchArray``;
    ``isinstance(..., torch.Tensor)`` is false and ``float(tim)`` raises
    ``TypeError``.
    """
    import torch

    if isinstance(tim, (int, float, np.integer, np.floating)):
        return float(tim)
    if isinstance(tim, torch.Tensor):
        t = tim
    else:
        try:
            t = tim.reshape(-1)
        except (AttributeError, TypeError):
            t = torch.as_tensor(tim).reshape(-1)
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t).reshape(-1)
    return float(t.reshape(-1)[0].detach().cpu())


def _mjlab_sim_time_to_tensor(tim: Any) -> Any:
    """Coerce mjlab ``data.time`` to a ``(N,)`` float32 :class:`torch.Tensor`.

    Args:
        tim: mjlab time field (``torch.Tensor``, TorchArray, scalar, or
             numpy value).

    Returns:
        1-D float32 tensor of per-env simulation times.
    """
    import torch

    if isinstance(tim, torch.Tensor):
        return tim.reshape(-1).float()
    if isinstance(tim, (int, float, np.integer, np.floating)):
        return torch.tensor([float(tim)], dtype=torch.float32)
    try:
        t = tim.reshape(-1)
    except (AttributeError, TypeError):
        t = torch.as_tensor(tim)
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t)
    return t.reshape(-1).float()


# ---------------------------------------------------------------------------
# Target synchronisation  (random box  OR  trajectory clip)
# ---------------------------------------------------------------------------


def _sync_mimic_mjlab_targets(
    env: Any,
    entity_name: str,
    cache: dict[str, Any],
) -> None:
    """Update ``cache["target_torch"]`` for the current step.

    Dispatches to trajectory-clip mode when ``cache["clip_source"]`` is set,
    otherwise falls back to the original random-box sampling that resamples
    once per episode (on time regression).

    Args:
        env: mjlab environment instance.
        entity_name: Scene entity name (e.g. ``"mimic_bimanual_robot"``).
        cache: Per-env cache dict produced by :func:`_resolve_mimic_mjlab_ids`.
    """
    import torch

    data = env.scene[entity_name].data.data
    clip_source: ClipTrajectorySource | None = cache.get("clip_source")

    if clip_source is not None:
        # --- Trajectory mode: targets come from the MotionClip ---
        t = _mjlab_sim_time_to_tensor(data.time)  # (N,) float32
        clip_source.update(t)
        cache["target_torch"] = clip_source.site_targets(t)  # (N, n_tracked, 3)
    else:
        # --- Random mode: resample once per episode (original behaviour) ---
        t_now = _mjlab_sim_time_to_float(data.time)
        last = cache.get("last_sim_time")
        need = cache.get("target_torch") is None
        if last is not None and t_now < last - 1e-6:
            need = True
        if need:
            n_env = int(data.qpos.shape[0])
            n_sites = int(cache["site_ids"].shape[0])
            device = data.qpos.device
            lo = torch.as_tensor(cache["lo"], device=device, dtype=torch.float32)
            hi = torch.as_tensor(cache["hi"], device=device, dtype=torch.float32)
            u = torch.rand((n_env, n_sites, 3), device=device, dtype=torch.float32)
            cache["target_torch"] = lo + (hi - lo) * u
        cache["last_sim_time"] = t_now


# ---------------------------------------------------------------------------
# Cache population
# ---------------------------------------------------------------------------


def _resolve_mimic_mjlab_ids(
    env: Any,
    entity_name: str,
    variant: str,
    clip: MotionClip | tuple[MotionClip, ...] | list[MotionClip] | None = None,
    ctrl_dt: float | None = None,
) -> dict[str, Any]:
    """Lazily build and return the per-env mimic cache.

    On the first call for a given ``(env, entity_name, variant)`` triple the
    cache is populated with model site ids, box bounds, tracking config, and
    (optionally) a :class:`~myosuite.envs.myo.backends.mjlab.clip_trajectory_source.ClipTrajectorySource`.
    Subsequent calls just synchronise the targets and return the cached dict.

    Args:
        env: mjlab environment instance.
        entity_name: Scene entity name.
        variant: ``"bimanual"`` or ``"fullbody"``.
        clip: Optional MotionClip to use as a target source.  When supplied
              *ctrl_dt* must also be provided.  Ignored if the cache entry
              already exists (the clip passed on the first call wins).
        ctrl_dt: Control timestep in seconds.  Required when *clip* is given.

    Returns:
        The per-env cache dict.
    """
    from ml_collections import config_dict

    key = (id(env), entity_name, variant)
    if key in _mimic_mjlab_cache:
        cache = _mimic_mjlab_cache[key]
        _sync_mimic_mjlab_targets(env, entity_name, cache)
        return cache

    if variant == "bimanual":
        from myosuite.integrations.musclemimic.bimanual_model import (
            BODY2SITES_FOR_MIMIC,
            build_mimic_bimanual_spec,
            default_mimic_config,
        )

        cfg = default_mimic_config()
        spec, _ = build_mimic_bimanual_spec(config_dict.create(**dict(cfg)))
        mj_model = spec.compile()
        site_names = tuple(BODY2SITES_FOR_MIMIC.values())
        tracking = MimicTrackingConfig(
            reward_scale=float(cfg.tracking_reward_scale),
            success_threshold=float(cfg.tracking_success_threshold),
        )
        lo = np.asarray(cfg.target_site_range.low, dtype=np.float64)
        hi = np.asarray(cfg.target_site_range.high, dtype=np.float64)
    elif variant == "saber":
        from myosuite.integrations.musclemimic.bimanual_model import (
            BODY2SITES_FOR_MIMIC,
        )
        from myosuite.integrations.musclemimic.myotorso_bimanual_model import (
            default_myotorso_bimanual_mimic_config,
        )
        from myosuite.envs.myo.tasks.challenge.saber_scene_assets import (
            build_saber_scene_mjmodel_and_spec,
        )

        cfg = default_myotorso_bimanual_mimic_config()
        mj_model, _ = build_saber_scene_mjmodel_and_spec()
        site_names = tuple(BODY2SITES_FOR_MIMIC.values())
        tracking = MimicTrackingConfig(
            reward_scale=2.0,  # saber clip tasks benefit from a denser tracking scale
            success_threshold=float(cfg.tracking_success_threshold),
        )
        lo = np.asarray(cfg.target_site_range.low, dtype=np.float64)
        hi = np.asarray(cfg.target_site_range.high, dtype=np.float64)
    else:
        from myosuite.integrations.musclemimic.fullbody_model import (
            FULLBODY_BODY2SITES_FOR_MIMIC,
            build_mimic_fullbody_spec,
            default_mimic_fullbody_config,
        )

        cfg = default_mimic_fullbody_config()
        spec, _ = build_mimic_fullbody_spec(config_dict.create(**dict(cfg)))
        mj_model = spec.compile()
        site_names = tuple(FULLBODY_BODY2SITES_FOR_MIMIC.values())
        tracking = MimicTrackingConfig(
            reward_scale=float(cfg.tracking_reward_scale),
            success_threshold=float(cfg.tracking_success_threshold),
        )
        lo = np.asarray(cfg.target_site_range.low, dtype=np.float64)
        hi = np.asarray(cfg.target_site_range.high, dtype=np.float64)

    site_ids = np.asarray(
        [mj_model.site(name).id for name in site_names],
        dtype=np.int32,
    )

    clip_source: ClipTrajectorySource | MultiClipTrajectorySource | None = None
    resolved_clip = None
    clip_bank = _normalize_motion_clip_bank(clip)
    if clip_bank:
        if ctrl_dt is None:
            raise ValueError("ctrl_dt must be provided when clip is given")
        from myosuite.core.trajectory_io import expand_motion_clip_to_model
        from myosuite.envs.myo.backends.mjlab.clip_trajectory_source import (
            ClipTrajectorySource,
            MultiClipTrajectorySource,
        )

        resolved_clips = tuple(
            _canonicalize_clip_sites(
                expand_motion_clip_to_model(one_clip, mj_model),
                required_site_names=site_names,
            )
            for one_clip in clip_bank
        )
        resolved_clip = resolved_clips[0]
        tracked_site_ids = np.arange(len(site_names), dtype=np.int64)
        if len(resolved_clips) == 1:
            clip_source = ClipTrajectorySource(
                clip=resolved_clip,
                tracked_site_ids=tracked_site_ids,
                ctrl_dt=ctrl_dt,
            )
        else:
            clip_source = MultiClipTrajectorySource(
                clips=resolved_clips,
                tracked_site_ids=tracked_site_ids,
                ctrl_dt=ctrl_dt,
            )

    _mimic_mjlab_cache[key] = dict(
        site_ids=site_ids,
        lo=lo.astype(np.float64),
        hi=hi.astype(np.float64),
        tracking=tracking,
        clip=resolved_clip,
        clip_source=clip_source,
        last_sim_time=None,
        target_torch=None,
    )
    cache = _mimic_mjlab_cache[key]
    _sync_mimic_mjlab_targets(env, entity_name, cache)
    return cache


# ---------------------------------------------------------------------------
# Observation closure factories
# ---------------------------------------------------------------------------


def _mimic_obs_qpos(entity_name: str) -> Callable[[Any], Any]:
    """Joint positions ``(N, nq)``."""

    def _fn(env: Any) -> Any:
        return env.scene[entity_name].data.data.qpos.clone()

    return _fn


def _mimic_obs_qvel(entity_name: str) -> Callable[[Any], Any]:
    """Joint velocities scaled by ctrl_dt, ``(N, nv)``."""

    def _fn(env: Any) -> Any:
        data = env.scene[entity_name].data.data
        ctrl_dt = env.physics_dt * env.cfg.decimation
        return data.qvel * ctrl_dt

    return _fn


def _mimic_obs_act(entity_name: str) -> Callable[[Any], Any]:
    """Muscle activation state ``(N, na)``."""

    def _fn(env: Any) -> Any:
        return env.scene[entity_name].data.data.act.clone()

    return _fn


def _mimic_obs_site_pos(
    entity_name: str,
    variant: str,
    clip: MotionClip | None = None,
    ctrl_dt: float | None = None,
) -> Callable[[Any], Any]:
    """Current tracked site positions, flattened to ``(N, n_tracked * 3)``."""

    def _fn(env: Any) -> Any:
        ids = _resolve_mimic_mjlab_ids(env, entity_name, variant, clip, ctrl_dt)[
            "site_ids"
        ]
        data = env.scene[entity_name].data.data
        pos = data.site_xpos[:, ids, :]
        return pos.reshape(pos.shape[0], -1)

    return _fn


def _mimic_obs_target(
    entity_name: str,
    variant: str,
    clip: MotionClip | None = None,
    ctrl_dt: float | None = None,
) -> Callable[[Any], Any]:
    """Target site positions, flattened to ``(N, n_tracked * 3)``."""

    def _fn(env: Any) -> Any:
        cache = _resolve_mimic_mjlab_ids(env, entity_name, variant, clip, ctrl_dt)
        tgt = cache["target_torch"]
        assert tgt is not None
        n_env = int(env.scene[entity_name].data.data.qpos.shape[0])
        return tgt.reshape(n_env, -1)

    return _fn


def _mimic_obs_err(
    entity_name: str,
    variant: str,
    clip: MotionClip | None = None,
    ctrl_dt: float | None = None,
) -> Callable[[Any], Any]:
    """Target-minus-current site error, flattened to ``(N, n_tracked * 3)``."""

    def _fn(env: Any) -> Any:
        cache = _resolve_mimic_mjlab_ids(env, entity_name, variant, clip, ctrl_dt)
        ids = cache["site_ids"]
        tgt = cache["target_torch"]
        assert tgt is not None
        data = env.scene[entity_name].data.data
        pos = data.site_xpos[:, ids, :]
        err = tgt - pos
        return err.reshape(err.shape[0], -1)

    return _fn


# ---------------------------------------------------------------------------
# Trajectory-mode-only observation factories
# ---------------------------------------------------------------------------


def _mimic_obs_clip_ref_qpos(
    entity_name: str,
    variant: str,
    clip: MotionClip,
    ctrl_dt: float,
) -> Callable[[Any], Any]:
    """Reference joint positions from the clip at the current frame ``(N, nq)``.

    Only valid in trajectory mode (``clip`` must have ``qpos`` populated).

    Args:
        entity_name: Scene entity name.
        variant: ``"bimanual"`` or ``"fullbody"``.
        clip: Source motion clip.
        ctrl_dt: Control timestep.

    Returns:
        Closure returning ``(N, nq)`` reference qpos tensor.
    """

    def _fn(env: Any) -> Any:
        cache = _resolve_mimic_mjlab_ids(env, entity_name, variant, clip, ctrl_dt)
        clip_source = cache.get("clip_source")
        if clip_source is None:
            raise RuntimeError(
                "clip_ref_qpos obs requires trajectory mode (clip_source is None)"
            )
        t = _mjlab_sim_time_to_tensor(env.scene[entity_name].data.data.time)
        ref = clip_source.ref_qpos(t)
        if ref is None:
            raise RuntimeError("clip.qpos is not available in this MotionClip")
        return ref

    return _fn


def _mimic_obs_clip_ref_qvel(
    entity_name: str,
    variant: str,
    clip: MotionClip,
    ctrl_dt: float,
) -> Callable[[Any], Any]:
    """Reference joint velocities from the clip at the current frame ``(N, nv)``."""

    def _fn(env: Any) -> Any:
        cache = _resolve_mimic_mjlab_ids(env, entity_name, variant, clip, ctrl_dt)
        clip_source = cache.get("clip_source")
        if clip_source is None:
            raise RuntimeError(
                "clip_ref_qvel obs requires trajectory mode (clip_source is None)"
            )
        t = _mjlab_sim_time_to_tensor(env.scene[entity_name].data.data.time)
        ref = clip_source.ref_qvel(t)
        if ref is None:
            raise RuntimeError("clip.qvel is not available in this MotionClip")
        return ref

    return _fn


def _mimic_obs_clip_phase(
    entity_name: str,
    variant: str,
    clip: MotionClip,
    ctrl_dt: float,
) -> Callable[[Any], Any]:
    """Normalised phase in ``[0, 1]`` along the clip ``(N, 1)``."""

    def _fn(env: Any) -> Any:
        cache = _resolve_mimic_mjlab_ids(env, entity_name, variant, clip, ctrl_dt)
        clip_source = cache.get("clip_source")
        if clip_source is None:
            raise RuntimeError(
                "clip_phase obs requires trajectory mode (clip_source is None)"
            )
        t = _mjlab_sim_time_to_tensor(env.scene[entity_name].data.data.time)
        return clip_source.phase(t)

    return _fn


# ---------------------------------------------------------------------------
# Reward closure factory
# ---------------------------------------------------------------------------


def _mimic_tracking_reward(
    entity_name: str,
    variant: str,
    clip: MotionClip | None = None,
    ctrl_dt: float | None = None,
) -> Callable[[Any], Any]:
    """Dense reward ``exp(-scale * mean(||target - pos||))`` matching MJX base.

    Works in both random and trajectory modes.
    """

    def _fn(env: Any) -> Any:
        import torch

        cache = _resolve_mimic_mjlab_ids(env, entity_name, variant, clip, ctrl_dt)
        tracking: MimicTrackingConfig = cache["tracking"]
        ids = cache["site_ids"]
        tgt = cache["target_torch"]
        assert tgt is not None
        data = env.scene[entity_name].data.data
        pos = data.site_xpos[:, ids, :]
        err = tgt - pos
        dist = torch.sqrt(torch.sum(err * err, dim=-1)).mean(dim=-1)  # (N,)
        return torch.exp(-tracking.reward_scale * dist)

    return _fn


# ---------------------------------------------------------------------------
# DeepMimic reward closure factory
# ---------------------------------------------------------------------------


def _mimic_deepmimic_reward(
    entity_name: str,
    variant: str,
    clip: MotionClip,
    ctrl_dt: float,
) -> Callable[[Any], Any]:
    """Weighted DeepMimic composite reward for trajectory mode.

    Combines site tracking (w=0.6) + joint pos/vel (0.1 each) + root
    kinematics (pos/vel/orient 0.1/0.1/0.01).  Only valid when a clip is
    provided (trajectory mode).

    Args:
        entity_name: Scene entity name.
        variant: ``"bimanual"`` or ``"fullbody"``.
        clip: Source motion clip (must have ``qpos``, ``qvel`` and
              ``site_xpos`` populated).
        ctrl_dt: Control timestep in seconds.

    Returns:
        Closure returning per-env reward tensor ``(N,)``.
    """
    from myosuite.terms.mimic_reward import (
        DEFAULT_SCALES,
        DEFAULT_WEIGHTS,
        mimic_composite_reward,
    )

    def _fn(env: Any) -> Any:
        import torch

        cache = _resolve_mimic_mjlab_ids(env, entity_name, variant, clip, ctrl_dt)
        resolved_clip: MotionClip | None = cache.get("clip")
        ids = cache["site_ids"]
        tgt_sites = cache["target_torch"]  # (N, n_sites, 3)
        assert tgt_sites is not None
        data = env.scene[entity_name].data.data
        cur_sites = data.site_xpos[:, ids, :]  # (N, n_sites, 3)

        clip_source: ClipTrajectorySource | None = cache.get("clip_source")
        t = _mjlab_sim_time_to_tensor(data.time)

        if (
            clip_source is not None
            and resolved_clip is not None
            and resolved_clip.qpos is not None
            and resolved_clip.qvel is not None
        ):
            ref_qpos = clip_source.ref_qpos(t)  # (N, nq)  or None
            ref_qvel = clip_source.ref_qvel(t)  # (N, nv) or None
        else:
            ref_qpos = None
            ref_qvel = None

        if ref_qpos is None or ref_qvel is None:
            # Fallback: single-term site tracking
            err = tgt_sites - cur_sites
            dist = torch.sqrt((err * err).sum(dim=-1)).mean(dim=-1)
            return torch.exp(-DEFAULT_SCALES.site * dist)

        ref_qpos_full = ref_qpos
        ref_qvel_full = ref_qvel
        qpos_indices = resolved_clip.qpos_model_indices
        qvel_indices = resolved_clip.qvel_model_indices
        cur_qpos = _select_clip_columns(data.qpos, qpos_indices)
        cur_qvel = _select_clip_columns(data.qvel, qvel_indices)
        ref_qpos = _select_clip_columns(ref_qpos_full, qpos_indices)
        ref_qvel = _select_clip_columns(ref_qvel_full, qvel_indices)

        has_root_pos = _clip_has_required_indices(qpos_indices, range(0, 3))
        has_root_quat = _clip_has_required_indices(qpos_indices, range(3, 7))
        has_root_vel = _clip_has_required_indices(qvel_indices, range(0, 3))
        cur_root_pos = data.qpos[:, :3] if has_root_pos else None
        cur_root_vel = data.qvel[:, :3] if has_root_vel else None
        cur_root_quat = data.qpos[:, 3:7] if has_root_quat else None
        ref_root_pos = ref_qpos_full[:, :3] if has_root_pos else None
        ref_root_vel = ref_qvel_full[:, :3] if has_root_vel else None
        ref_root_quat = ref_qpos_full[:, 3:7] if has_root_quat else None

        result = mimic_composite_reward(
            torch,
            cur_sites,
            tgt_sites,
            cur_qpos,
            ref_qpos,
            cur_qvel,
            ref_qvel,
            cur_root_pos,
            ref_root_pos,
            cur_root_vel,
            ref_root_vel,
            cur_root_quat,
            ref_root_quat,
            weights=DEFAULT_WEIGHTS,
            scales=DEFAULT_SCALES,
        )
        return result["dense"]  # (N,)

    return _fn


# ---------------------------------------------------------------------------
# Lookahead observation closure factory
# ---------------------------------------------------------------------------


def _mimic_obs_lookahead(
    entity_name: str,
    variant: str,
    clip: MotionClip,
    ctrl_dt: float,
    k: int = 5,
    stride: int = 20,
) -> Callable[[Any], Any]:
    """k-step lookahead observation over future clip targets.

    Returns flattened relative tracked-site positions plus future phase. For

    Args:
        entity_name: Scene entity name.
        variant: ``"bimanual"`` or ``"fullbody"``.
        clip: Source motion clip.
        ctrl_dt: Control timestep.
        k: Number of lookahead steps.
        stride: Frame stride between steps.
    """

    def _fn(env: Any) -> Any:
        import torch

        cache = _resolve_mimic_mjlab_ids(env, entity_name, variant, clip, ctrl_dt)
        resolved_clip: MotionClip | None = cache.get("clip")
        clip_source = cache.get("clip_source")
        if clip_source is None:
            raise RuntimeError("lookahead obs requires trajectory mode")
        if resolved_clip is None:
            raise RuntimeError("lookahead obs requires a resolved MotionClip")

        data = env.scene[entity_name].data.data
        t = _mjlab_sim_time_to_tensor(data.time)  # (N,)
        n_envs = int(t.shape[0])
        device = t.device
        clip_lengths = clip_source.clip_lengths(t)

        cur_frames = clip_source.frame_indices(t)
        cur_root_pos = data.qpos[:, :3]  # (N, 3)

        has_root_pos = _clip_has_required_indices(
            resolved_clip.qpos_model_indices, range(0, 3)
        )
        has_root_vel = _clip_has_required_indices(
            resolved_clip.qvel_model_indices, range(0, 3)
        )
        per_step_dim = (
            int(clip_source.n_tracked) * 3
            + (3 if has_root_pos else 0)
            + (3 if has_root_vel else 0)
            + 1  # phase
        )
        out = torch.zeros(n_envs, k * per_step_dim, device=device, dtype=torch.float32)
        has_qpos = has_root_pos and clip_source.ref_qpos(t) is not None
        has_qvel = has_root_vel and clip_source.ref_qvel(t) is not None

        offset = 0
        for step_i in range(1, k + 1):
            future_frames = (cur_frames + step_i * stride) % clip_lengths
            future_sites = clip_source.site_targets_at_frames(future_frames)
            rel_sites = (future_sites - cur_root_pos.unsqueeze(1)).reshape(n_envs, -1)
            n_site_dim = rel_sites.shape[1]
            out[:, offset : offset + n_site_dim] = rel_sites
            offset += n_site_dim
            if has_qpos:
                future_qpos = clip_source.ref_qpos_at_frames(future_frames)
                assert future_qpos is not None
                future_rpos = future_qpos[:, :3]
                delta = (future_rpos - cur_root_pos).float()
                out[:, offset : offset + 3] = delta
                offset += 3
            if has_qvel:
                future_qvel = clip_source.ref_qvel_at_frames(future_frames)
                assert future_qvel is not None
                future_rvel = future_qvel[:, :3].float()
                out[:, offset : offset + 3] = future_rvel
                offset += 3
            phase = future_frames.float() / torch.clamp(
                clip_lengths.float() - 1.0,
                min=1.0,
            )
            out[:, offset] = phase
            offset += 1

        return out

    return _fn


# ---------------------------------------------------------------------------
# RSI (Reference State Initialization) event closure factory
# ---------------------------------------------------------------------------


def _mimic_rsi_event(
    entity_name: str,
    variant: str,
    clip: MotionClip | tuple[MotionClip, ...],
    ctrl_dt: float,
    mj_model: Any | None = None,
) -> Callable[[Any, Any], None]:
    """Reset-event that sets qpos/qvel from the motion clip (RSI).

    On each episode reset, each environment is placed at a random frame of the
    reference clip rather than at the standing keyframe.  This is the
    *Reference State Initialization* technique from DeepMimic — without it
    the policy never sees the middle of a motion and reward signals stay weak.

    Args:
        entity_name: Scene entity name.
        variant: ``"bimanual"`` or ``"fullbody"``.
        clip: Source motion clip or clip bank with ``qpos`` and ``qvel`` populated.
        ctrl_dt: Control timestep (used to initialise :class:`ClipTrajectorySource`).
        mj_model: Optional MuJoCo model for handling fixed-base entities with
            auxiliary free joints.

    Returns:
        Closure ``(env, env_ids) -> None`` suitable for
        ``EventTermCfg(mode="reset")``.
    """

    def _fn(env: Any, env_ids: Any) -> None:
        import torch
        import mujoco

        data = env.scene[entity_name].data.data
        n_envs = int(data.qpos.shape[0])
        device = data.qpos.device

        # Populate (or retrieve) the shared cache — also initialises clip_source.
        cache = _resolve_mimic_mjlab_ids(env, entity_name, variant, clip, ctrl_dt)
        clip_source = cache.get("clip_source")
        if clip_source is None:
            return
        has_qpos = getattr(clip_source, "_qpos_tensor", None) is not None or (
            getattr(clip_source, "_qpos_tensors", None) is not None
        )
        if not has_qpos:
            return

        # Ensure internal tensors are on the correct device.
        clip_source._ensure_device(device, n_envs)

        # Resample start offsets for the envs that just reset.
        env_ids_long = torch.as_tensor(env_ids, device=device, dtype=torch.long)
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
        # Prevent _detect_and_resample_resets from overwriting these offsets on
        # the very next update() call (t=0 would look like a regression from
        # whatever _last_t was before the reset).
        if clip_source._last_t is not None:
            clip_source._last_t[env_ids_long] = 0.0

        # --- Write root state (pos + quat + lin_vel + ang_vel) ---
        ref_qpos = clip_source.ref_qpos_at_frames(new_offsets)
        if ref_qpos is None:
            return
        # Add per-env world origins so the bodies appear in the right place.
        env_origins = env.scene.env_origins[env_ids_long]  # (n_reset, 3)
        root_pos = ref_qpos[:, :3].float() + env_origins
        root_quat = ref_qpos[:, 3:7].float()  # (w, x, y, z)

        ref_qvel = clip_source.ref_qvel_at_frames(new_offsets)
        if ref_qvel is not None:
            root_lin_vel = ref_qvel[:, :3].float()
            root_ang_vel = ref_qvel[:, 3:6].float()
        else:
            root_lin_vel = torch.zeros(n_reset, 3, device=device)
            root_ang_vel = torch.zeros(n_reset, 3, device=device)

        root_state = torch.cat(
            [root_pos, root_quat, root_lin_vel, root_ang_vel], dim=-1
        )  # (n_reset, 13)
        entity = env.scene[entity_name]

        # --- Write joint state (non-root DOFs) ---
        joint_pos = ref_qpos[:, 7:].float()  # (n_reset, nq-7)
        if ref_qvel is not None:
            joint_vel = ref_qvel[:, 6:].float()  # (n_reset, nv-6)
        else:
            joint_vel = torch.zeros(n_reset, ref_qpos.shape[1] - 7, device=device)
        try:
            entity.write_root_state_to_sim(root_state, env_ids=env_ids_long)
            entity.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_long)
        except ValueError as exc:
            if "fixed-base entity" not in str(exc) or mj_model is None:
                raise
            qpos = ref_qpos.clone()
            env_origins = env.scene.env_origins[env_ids_long].to(
                device=device, dtype=qpos.dtype
            )
            free_qpos_adrs = tuple(
                int(mj_model.jnt_qposadr[j])
                for j in range(int(mj_model.njnt))
                if int(mj_model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)
            )
            for qadr in free_qpos_adrs:
                qpos[:, qadr : qadr + 3] += env_origins
            data.qpos[env_ids_long] = qpos
            if ref_qvel is not None:
                data.qvel[env_ids_long] = ref_qvel.float()
            else:
                data.qvel[env_ids_long] = 0.0
            env.sim.forward()

    return _fn


# ---------------------------------------------------------------------------
# Early termination closure
# ---------------------------------------------------------------------------


def _mimic_early_termination(
    entity_name: str,
    variant: str,
    clip: MotionClip | None = None,
    ctrl_dt: float | None = None,
    site_err_threshold: float = 1.0,
    root_err_threshold: float = 0.3,
) -> Callable[[Any], Any]:
    """Terminate episodes where tracking error exceeds recovery threshold.

    Returns boolean tensor ``(N,)`` — ``True`` for envs that should terminate.
    """

    def _fn(env: Any) -> Any:
        import torch

        cache = _resolve_mimic_mjlab_ids(env, entity_name, variant, clip, ctrl_dt)
        ids = cache["site_ids"]
        tgt = cache["target_torch"]
        if tgt is None:
            return torch.zeros(
                env.scene[entity_name].data.data.qpos.shape[0],
                dtype=torch.bool,
                device=env.scene[entity_name].data.data.qpos.device,
            )
        data = env.scene[entity_name].data.data
        cur_sites = data.site_xpos[:, ids, :]  # (N, n_sites, 3)
        err = cur_sites - tgt
        mean_dist = torch.sqrt((err * err).sum(dim=-1)).mean(dim=-1)  # (N,)
        site_term = mean_dist > site_err_threshold

        root_err = torch.sqrt(((data.qpos[:, :3] - tgt.mean(dim=1)) ** 2).sum(dim=-1))
        root_term = root_err > root_err_threshold
        return site_term | root_term

    return _fn


# ---------------------------------------------------------------------------
# mjlab env-config builder
# ---------------------------------------------------------------------------


def _make_mimic_env_cfg(
    *,
    _task_id: str,
    entity_name: str,
    variant: str,
    spec_fn: Callable[[], Any],
    muscle_actuators: tuple[str, ...],
    tendon_targets: tuple[str, ...],
    sim_dt: float,
    ctrl_dt: float,
    max_episode_steps: int,
    clip: MotionClip | None = None,
    num_envs: int = 1,
    use_deepmimic_reward: bool = True,
    use_lookahead: bool = True,
    use_early_termination: bool = True,
    action_mode: str = "sigmoid",
    mj_model: Any = None,
    enable_clip_state_terms: bool = True,
    reward_mode: str = _MIMIC_REWARD_MODE_MIMIC,
    mimic_reward_weight: float = 5.0,
) -> Any:
    """Build :class:`~mjlab.envs.ManagerBasedRlEnvCfg` for one Mimic task.

    When *clip* is supplied the environment operates in trajectory mode:
    targets are taken from ``clip.site_xpos`` and additional observation terms
    (``clip_ref_qpos``, ``clip_ref_qvel``, ``clip_phase``) are added.

    Args:
        _task_id: Task ID string (informational only).
        entity_name: Scene entity name for the articulated body.
        variant: ``"bimanual"`` or ``"fullbody"``.
        spec_fn: Callable that returns an :class:`mujoco.MjSpec`.
        muscle_actuators: Tuple of muscle actuator names.
        tendon_targets: Tuple of tendon names for the XmlMuscle actuator.
        sim_dt: Physics timestep in seconds.
        ctrl_dt: Control timestep in seconds (``sim_dt × decimation``).
        max_episode_steps: Episode length in control steps.
        clip: Optional MotionClip for trajectory-mode targets.
        num_envs: Number of parallel environments.  Use ``1`` for play/
            visualization (default) and a larger value (e.g. ``1024``) for
            GPU-parallel training.
        enable_clip_state_terms: Whether to expose clip qpos/qvel references and
            use RSI from clip state. Disable this when clip state widths do not
            match the model but ``site_xpos`` remains usable.
        reward_mode: Reward composition mode (``"mimic"``, ``"augmented"``).
        action_mode: Muscle action interpretation. ``"sigmoid"`` keeps the
            training path's canonical action normalisation; ``"direct"``
            preserves checkpoint playback by clipping incoming values to
            ``[-1, 1]`` before writing them as muscle controls.

    Returns:
        A configured :class:`~mjlab.envs.ManagerBasedRlEnvCfg` instance.
    """
    from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
        MyoMuscleActivationActionCfg,
    )
    from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
        _XmlWrappedActuatorCfg,
    )
    from mjlab.actuator.actuator import TransmissionType
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
    from mjlab.envs import ManagerBasedRlEnvCfg
    from mjlab.envs.mdp import terminations as mdp_terminations
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.observation_manager import (
        ObservationGroupCfg,
        ObservationTermCfg,
    )
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.scene import SceneCfg
    from mjlab.sim import MujocoCfg, SimulationCfg

    reward_mode = _normalize_mimic_reward_mode(reward_mode)

    articulation = EntityArticulationInfoCfg(
        actuators=(
            _XmlWrappedActuatorCfg(
                target_names_expr=tendon_targets,
                transmission_type=TransmissionType.TENDON,
            ),
        )
    )
    # Use the compiled model's keyframe for explicit init_state so the body
    # resets to the standing pose rather than the all-zeros default.
    init_state = (
        _init_state_from_model(mj_model)
        if mj_model is not None
        else EntityCfg.InitialStateCfg()
    )
    entity_cfg = EntityCfg(
        spec_fn=spec_fn, articulation=articulation, init_state=init_state
    )
    scene_cfg = SceneCfg(num_envs=num_envs, entities={entity_name: entity_cfg})

    # --- Core observation terms (both modes) ---
    obs_terms: dict[str, Any] = {
        "qpos": ObservationTermCfg(func=_mimic_obs_qpos(entity_name)),
        "qvel": ObservationTermCfg(func=_mimic_obs_qvel(entity_name)),
        "act": ObservationTermCfg(func=_mimic_obs_act(entity_name)),
        "mimic_site_pos": ObservationTermCfg(
            func=_mimic_obs_site_pos(entity_name, variant, clip, ctrl_dt)
        ),
        "mimic_site_target": ObservationTermCfg(
            func=_mimic_obs_target(entity_name, variant, clip, ctrl_dt)
        ),
        "mimic_site_err": ObservationTermCfg(
            func=_mimic_obs_err(entity_name, variant, clip, ctrl_dt)
        ),
    }
    # --- Trajectory-mode extras ---
    if clip is not None:
        if enable_clip_state_terms and clip.qpos is not None:
            obs_terms["clip_ref_qpos"] = ObservationTermCfg(
                func=_mimic_obs_clip_ref_qpos(entity_name, variant, clip, ctrl_dt)
            )
        if enable_clip_state_terms and clip.qvel is not None:
            obs_terms["clip_ref_qvel"] = ObservationTermCfg(
                func=_mimic_obs_clip_ref_qvel(entity_name, variant, clip, ctrl_dt)
            )
        obs_terms["clip_phase"] = ObservationTermCfg(
            func=_mimic_obs_clip_phase(entity_name, variant, clip, ctrl_dt)
        )
        if use_lookahead:
            obs_terms["mimic_lookahead"] = ObservationTermCfg(
                func=_mimic_obs_lookahead(entity_name, variant, clip, ctrl_dt)
            )

    observations = {
        "policy": ObservationGroupCfg(terms=obs_terms),
        "actor": ObservationGroupCfg(terms=obs_terms),
        "critic": ObservationGroupCfg(terms=obs_terms),
    }
    actions = {
        "muscles": MyoMuscleActivationActionCfg(
            entity_name=entity_name,
            actuator_names=muscle_actuators,
            tendon_names=tendon_targets,
            action_mode=action_mode,
        ),
    }
    terminations = {
        "time_out": TerminationTermCfg(func=mdp_terminations.time_out, time_out=True),
    }
    if (
        _reward_mode_uses_mimic_objective(reward_mode)
        and use_early_termination
        and clip is not None
    ):
        terminations["mimic_deviation"] = TerminationTermCfg(
            func=_mimic_early_termination(entity_name, variant, clip, ctrl_dt),
            time_out=False,
        )

    # Reset handling:
    # - use RSI when clip qpos/qvel align with the model
    # - otherwise restore the compiled model keyframe so auxiliary free joints
    #   (e.g. saber props) do not reset to all zeros
    events: dict[str, Any] = {}
    if clip is not None and enable_clip_state_terms and clip.qpos is not None:
        events["rsi"] = EventTermCfg(
            func=_mimic_rsi_event(
                entity_name, variant, clip, ctrl_dt, mj_model=mj_model
            ),
            mode="reset",
        )
    elif clip is not None and mj_model is not None and getattr(mj_model, "nkey", 0) > 0:
        events["keyframe_reset"] = EventTermCfg(
            func=_mimic_keyframe_reset_event(entity_name, mj_model),
            mode="reset",
        )

    rewards: dict[str, Any] = {}
    if _reward_mode_uses_mimic_objective(reward_mode):
        if (
            use_deepmimic_reward
            and clip is not None
            and enable_clip_state_terms
            and clip.qpos is not None
            and clip.qvel is not None
        ):
            reward_fn = _mimic_deepmimic_reward(entity_name, variant, clip, ctrl_dt)
        else:
            reward_fn = _mimic_tracking_reward(entity_name, variant, clip, ctrl_dt)

        # mjlab RewardManager multiplies every term by ctrl_dt (scale_by_dt=True).
        # Compensate so the logged reward matches the raw composite reward (0–1).
        reward_weight = float(mimic_reward_weight) / ctrl_dt if ctrl_dt > 0 else 1.0
        rewards["tracking"] = RewardTermCfg(func=reward_fn, weight=reward_weight)
    decimation = max(1, int(round(ctrl_dt / sim_dt)))
    episode_length_s = float(max_episode_steps) * ctrl_dt

    env_cfg_kwargs: dict[str, Any] = dict(
        scene=scene_cfg,
        decimation=decimation,
        episode_length_s=episode_length_s,
        observations=observations,
        actions=actions,
        terminations=terminations,
        rewards=rewards,
        sim=SimulationCfg(
            mujoco=MujocoCfg(timestep=sim_dt, ccd_iterations=500),
            njmax=512,
            nconmax=256,
        ),
    )
    if events:
        env_cfg_kwargs["events"] = events
    return ManagerBasedRlEnvCfg(**env_cfg_kwargs)


# ---------------------------------------------------------------------------
# Task registration entry points
# ---------------------------------------------------------------------------


def register_mimic_mjlab_tasks(
    register_mjlab_task: Callable[..., None],
    rl_cfg_fn: Callable[[], Any],
) -> None:
    """Register Mimic tasks in random-target mode (original behaviour).

    No motion clip is required.  Targets are sampled uniformly from the
    bounding box on each episode reset.

    Args:
        register_mjlab_task: mjlab task registry function.
        rl_cfg_fn: Callable returning a default RL runner config.
    """
    _register_mimic_tasks(
        register_mjlab_task=register_mjlab_task,
        rl_cfg_fn=rl_cfg_fn,
        clip=None,
    )


def register_mimic_mjlab_tasks_with_clip(
    register_mjlab_task: Callable[..., None],
    rl_cfg_fn: Callable[[], Any],
    clip: MotionClip,
    use_deepmimic_reward: bool = True,
    use_lookahead: bool = True,
    use_early_termination: bool = True,
    action_mode: str = "sigmoid",
    reward_mode: str = _MIMIC_REWARD_MODE_MIMIC,
    mimic_reward_weight: float = 5.0,
    env_reward_weight: float = 1.0,
) -> None:
    """Register Mimic tasks in trajectory mode.

    Targets are taken from *clip*'s ``site_xpos`` at the frame corresponding
    to each environment's current simulation time.  Each of the N parallel
    environments starts at a random frame; on reset that offset is resampled.

    Additional observation terms are added automatically:

    * ``clip_ref_qpos`` — reference qpos at the current frame (if available)
    * ``clip_ref_qvel`` — reference qvel at the current frame (if available)
    * ``clip_phase``    — normalised phase in ``[0, 1]``

    Args:
        register_mjlab_task: mjlab task registry function.
        rl_cfg_fn: Callable returning a default RL runner config.
        clip: Loaded :class:`~myosuite.core.trajectory_io.MotionClip` with
              ``site_xpos`` populated.
        action_mode: Muscle action interpretation. Leave as ``"sigmoid"`` for
            training; use ``"direct"`` for fullbody checkpoint inference.
        reward_mode: Reward composition for the saber mimic task (``"mimic"``,
            ``"env"``, or ``"augmented"``). Forwarded to
            :func:`register_saber_p0_mjlab_task_with_mimic_mix`.

    Raises:
        ValueError: If ``clip.site_xpos`` is ``None``.
    """
    reward_mode = _normalize_mimic_reward_mode(reward_mode)
    if clip.site_xpos is None:
        raise ValueError(
            "register_mimic_mjlab_tasks_with_clip requires clip.site_xpos; "
            "reload the clip with a file that includes site positions."
        )
    _register_mimic_tasks(
        register_mjlab_task=register_mjlab_task,
        rl_cfg_fn=rl_cfg_fn,
        clip=clip,
        use_deepmimic_reward=use_deepmimic_reward,
        use_lookahead=use_lookahead,
        use_early_termination=use_early_termination,
        action_mode=action_mode,
        mimic_reward_weight=mimic_reward_weight,
    )
    from myosuite.envs.myo.backends.mjlab.register_mjlab_saber import (
        SaberMimicMixCfg,
        register_saber_p0_mjlab_task_with_mimic_mix,
    )

    register_saber_p0_mjlab_task_with_mimic_mix(
        register_mjlab_task=register_mjlab_task,
        rl_cfg_fn=rl_cfg_fn,
        mimic_mix=SaberMimicMixCfg(
            clips=(clip,),
            reward_mode=reward_mode,
            mimic_reward_weight=mimic_reward_weight,
            env_reward_weight=env_reward_weight,
            use_deepmimic_reward=use_deepmimic_reward,
            use_lookahead=use_lookahead,
            use_early_termination=use_early_termination,
            use_reference_state_initialization=True,
        ),
    )
    from mjlab.tasks.registry import list_tasks

    if "myoMimicFullbody-v0" not in list_tasks():
        raise RuntimeError(
            "Registration of myoMimicFullbody-v0 failed silently. "
            "Ensure musclemimic_models is installed "
            "('pip install myosuite[musclemimic]') and the clip is valid."
        )
    from myosuite.envs.myo.tasks.challenge.saber_task_spec import (
        SABER_P0_MIMIC_ENV_ID,
    )

    if SABER_P0_MIMIC_ENV_ID not in list_tasks():
        raise RuntimeError(
            f"Registration of {SABER_P0_MIMIC_ENV_ID} failed silently. "
            "Ensure musclemimic_models is installed and the clip contains "
            "site_xpos compatible with the saber tracking setup."
        )


def _register_mimic_tasks(
    register_mjlab_task: Callable[..., None],
    rl_cfg_fn: Callable[[], Any],
    clip: MotionClip | None,
    use_deepmimic_reward: bool = True,
    use_lookahead: bool = True,
    use_early_termination: bool = True,
    action_mode: str = "sigmoid",
    mimic_reward_weight: float = 1.0,
) -> None:
    """Internal implementation shared by both registration entry points."""
    import importlib.util

    if importlib.util.find_spec("musclemimic_models") is None:
        raise ImportError(
            "musclemimic_models is not installed. "
            "Install it with: pip install 'myosuite[musclemimic]'"
        )

    from ml_collections import config_dict

    from myosuite.integrations.musclemimic.bimanual_model import (
        build_mimic_bimanual_spec,
        default_mimic_config,
    )
    from myosuite.integrations.musclemimic.fullbody_model import (
        build_mimic_fullbody_spec,
        default_mimic_fullbody_config,
    )

    # --- Bimanual ---
    try:
        from myosuite.envs.myo.backends.mjlab.configs.musclemimic_bimanual_cfg import (
            MuscleMimicBimanualCfg,
        )

        b_cfg = default_mimic_config()
        b_dict = config_dict.create(**dict(b_cfg))
        b_spec_probe, _ = build_mimic_bimanual_spec(b_dict)
        b_mj = b_spec_probe.compile()
        b_muscles = _muscle_actuator_names(b_mj)
        b_tendons = _muscle_tendon_names(b_mj)
        if b_muscles and b_tendons:

            def _bimanual_spec_fn() -> Any:
                spec, _ = build_mimic_bimanual_spec(b_dict)
                return spec

            b_ctrl_dt = float(b_cfg.ctrl_dt)
            _b_common = dict(
                entity_name="mimic_bimanual_robot",
                variant="bimanual",
                spec_fn=_bimanual_spec_fn,
                muscle_actuators=b_muscles,
                tendon_targets=b_tendons,
                sim_dt=float(b_cfg.sim_dt),
                ctrl_dt=b_ctrl_dt,
                max_episode_steps=int(b_cfg.max_episode_steps),
                clip=clip,
                use_deepmimic_reward=use_deepmimic_reward,
                use_lookahead=use_lookahead,
                use_early_termination=use_early_termination,
                action_mode=action_mode,
                mj_model=b_mj,
                mimic_reward_weight=mimic_reward_weight,
            )
            b_train_env = _make_mimic_env_cfg(
                _task_id="myoMimicBimanual-v0",
                num_envs=MuscleMimicBimanualCfg.num_envs,
                **_b_common,
            )
            b_play_env = _make_mimic_env_cfg(
                _task_id="myoMimicBimanual-v0",
                num_envs=1,
                **_b_common,
            )
            for task_id in ("myoMimicBimanual-v0", "myoMuscleMimicBimanual-v0"):
                _register_or_replace_mjlab_task(
                    register_mjlab_task,
                    task_id=task_id,
                    env_cfg=b_train_env,
                    play_env_cfg=b_play_env,
                    rl_cfg=rl_cfg_fn(),
                    runner_cls=None,
                )
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning(
            "Bimanual mimic registration failed: %s", exc
        )

    # --- Full body --- (errors propagate so the caller can report them)
    from myosuite.envs.myo.backends.mjlab.configs.musclemimic_fullbody_cfg import (
        MuscleMimicFullbodyCfg,
    )

    f_cfg = default_mimic_fullbody_config()
    f_dict = config_dict.create(**dict(f_cfg))
    f_spec_probe, _ = build_mimic_fullbody_spec(f_dict)
    f_mj = f_spec_probe.compile()
    f_muscles = _muscle_actuator_names(f_mj)
    f_tendons = _muscle_tendon_names(f_mj)
    if not f_muscles or not f_tendons:
        raise RuntimeError(
            "No muscle actuators or tendons found in the fullbody model. "
            "Check that musclemimic_models is correctly installed."
        )

    def _fullbody_spec_fn() -> Any:
        spec, _ = build_mimic_fullbody_spec(f_dict)
        return spec

    f_ctrl_dt = float(f_cfg.ctrl_dt)
    _f_common = dict(
        entity_name="mimic_fullbody_robot",
        variant="fullbody",
        spec_fn=_fullbody_spec_fn,
        muscle_actuators=f_muscles,
        tendon_targets=f_tendons,
        sim_dt=float(f_cfg.sim_dt),
        ctrl_dt=f_ctrl_dt,
        max_episode_steps=int(f_cfg.max_episode_steps),
        clip=clip,
        use_deepmimic_reward=use_deepmimic_reward,
        use_lookahead=use_lookahead,
        use_early_termination=use_early_termination,
        action_mode=action_mode,
        mj_model=f_mj,
        mimic_reward_weight=mimic_reward_weight,
    )
    f_train_env = _make_mimic_env_cfg(
        _task_id="myoMimicFullbody-v0",
        num_envs=MuscleMimicFullbodyCfg.num_envs,
        **_f_common,
    )
    f_play_env = _make_mimic_env_cfg(
        _task_id="myoMimicFullbody-v0",
        num_envs=1,
        **_f_common,
    )
    for task_id in ("myoMimicFullbody-v0", "myoMuscleMimicFullbody-v0"):
        _register_or_replace_mjlab_task(
            register_mjlab_task,
            task_id=task_id,
            env_cfg=f_train_env,
            play_env_cfg=f_play_env,
            rl_cfg=rl_cfg_fn(),
            runner_cls=None,
        )


def _register_or_replace_mjlab_task(
    register_mjlab_task: Callable[..., None],
    *,
    task_id: str,
    env_cfg: Any,
    play_env_cfg: Any,
    rl_cfg: Any,
    runner_cls: Any,
) -> None:
    """Register a task, replacing an existing MJLab registry entry if needed.

    MJLab's public registration function is append-only.  Notebooks and
    scripts can first bootstrap random-mode tasks and later register clip-mode
    tasks under the same ids; replacing the private registry entry keeps that
    workflow idempotent while preserving the normal error path for non-MJLab
    registries or unrelated failures.
    """
    already_registered_exc: ValueError | None = None
    try:
        register_mjlab_task(
            task_id=task_id,
            env_cfg=env_cfg,
            play_env_cfg=play_env_cfg,
            rl_cfg=rl_cfg,
            runner_cls=runner_cls,
        )
        return
    except ValueError as exc:
        if "already registered" not in str(exc):
            raise
        already_registered_exc = exc

    assert already_registered_exc is not None
    import mjlab.tasks.registry as registry

    registry_dict = getattr(registry, "_REGISTRY", None)
    task_cfg_cls = getattr(registry, "_TaskCfg", None)
    if (
        register_mjlab_task is not getattr(registry, "register_mjlab_task", None)
        or registry_dict is None
        or task_cfg_cls is None
        or task_id not in registry_dict
    ):
        raise already_registered_exc

    registry_dict[task_id] = task_cfg_cls(env_cfg, play_env_cfg, rl_cfg, runner_cls)


# ---------------------------------------------------------------------------
# SAR action term
# ---------------------------------------------------------------------------


class SARMuscleActivationActionCfg:
    """Action term config that maps SAR synergy actions to muscle activations.

    The policy outputs ``n_synergies`` values in ``[-1, 1]``.  These are
    passed through a :class:`~benchmarks.sar_backends.sar_torch_transform.SARTorchTransform`
    (MinMaxScaler⁻¹ → FastICA⁻¹ → PCA⁻¹ → clamp) to produce ``n_muscles``
    activations in ``[0, 1]``, which are then written to MuJoCo's ``ctrl``
    array.

    Args:
        entity_name: mjlab scene entity name.
        actuator_names: Tuple of muscle actuator names (length = n_muscles).
        sar_transform: Fitted :class:`~...SARTorchTransform`.
    """

    def __init__(
        self,
        *,
        entity_name: str,
        actuator_names: tuple[str, ...],
        sar_transform: Any,
    ) -> None:
        self.entity_name = entity_name
        self.actuator_names = actuator_names
        self.sar_transform = sar_transform

    def build(self, env: Any) -> SARMuscleActivationAction:
        return SARMuscleActivationAction(self, env)


class SARMuscleActivationAction:
    """Action term: SAR synergy actions → muscle activations → MuJoCo ctrl.

    Applies the SAR inverse transform so the policy learns in a low-dimensional
    synergy space while MuJoCo receives full-dimensional muscle activations.

    Args:
        cfg: :class:`SARMuscleActivationActionCfg` instance.
        env: mjlab ``ManagerBasedRlEnv``.
    """

    def __init__(self, cfg: SARMuscleActivationActionCfg, env: Any) -> None:
        import torch

        self.cfg = cfg
        self._env = env
        self.num_envs = env.num_envs
        self.device = env.device

        self._sar: Any = cfg.sar_transform.to(self.device)

        if self._sar.n_muscles != len(cfg.actuator_names):
            raise ValueError(
                f"SARTorchTransform has n_muscles={self._sar.n_muscles} "
                f"but {len(cfg.actuator_names)} actuator names were provided"
            )
        entity = env.scene[cfg.entity_name]
        target_ids, _ = entity.find_actuators(cfg.actuator_names)
        if len(target_ids) != len(cfg.actuator_names):
            raise ValueError(
                f"SARMuscleActivationAction expected {len(cfg.actuator_names)} "
                f"actuators, resolved {len(target_ids)}"
            )
        self._target_ids = torch.tensor(
            target_ids, device=self.device, dtype=torch.long
        )

        self._raw_actions = torch.zeros(
            (self.num_envs, self._sar.n_syn), device=self.device
        )
        self._processed_actions = torch.zeros(
            (self.num_envs, self._sar.n_muscles), device=self.device
        )

    @property
    def action_dim(self) -> int:
        """Policy action dimension (number of synergies)."""
        return self._sar.n_syn

    @property
    def raw_action(self) -> Any:
        return self._raw_actions

    def process_actions(self, actions: Any) -> None:
        """Apply SAR inverse transform to synergy actions.

        Args:
            actions: ``(N, n_syn)`` tensor from the policy, values in ``[-1, 1]``.
        """
        self._raw_actions[:] = actions.to(self.device)
        with_no_grad = __import__("torch").no_grad()
        with with_no_grad:
            self._processed_actions[:] = self._sar(self._raw_actions)

    def apply_actions(self) -> None:
        """Write muscle activations into MuJoCo ctrl slots."""
        entity = self._env.scene[self.cfg.entity_name]
        indexing = entity.data.indexing
        global_ctrl_ids = indexing.ctrl_ids[self._target_ids]
        entity.write_ctrl_to_sim(self._processed_actions, ctrl_ids=global_ctrl_ids)


# ---------------------------------------------------------------------------
# SAR task registration
# ---------------------------------------------------------------------------


def register_mimic_mjlab_tasks_with_sar(
    register_mjlab_task: Callable[..., None],
    rl_cfg_fn: Callable[[], Any],
    sar_dir: Path | str,
    clip: MotionClip | None = None,
) -> None:
    """Register Mimic tasks with SAR (synergy-based) action space.

    The policy learns to output ``n_synergies`` actions.  The SAR inverse
    transform maps them to full-dimensional muscle activations before they
    are applied in simulation.

    Task IDs registered:
    * ``myoMimicFullbody-SAR-v0``  /  ``myoMuscleMimicFullbody-SAR-v0``
    * ``myoMimicBimanual-SAR-v0``  /  ``myoMuscleMimicBimanual-SAR-v0``

    Args:
        register_mjlab_task: mjlab task registry function.
        rl_cfg_fn: Callable returning a default RL runner config.
        sar_dir: Directory written by
            :func:`~myosuite.integrations.musclemimic.sar_extraction.save_synergy_model`.
        clip: Optional :class:`~myosuite.core.trajectory_io.MotionClip` for
              trajectory-mode targets (same as
              :func:`register_mimic_mjlab_tasks_with_clip`).

    Raises:
        FileNotFoundError: If *sar_dir* does not contain the expected SAR files.
        ImportError: If ``torch`` or ``scikit-learn`` or ``joblib`` are missing.
    """
    from myosuite.integrations.musclemimic.sar_extraction import load_synergy_model

    sar_model: SynergyModel = load_synergy_model(sar_dir)
    _register_mimic_sar_tasks(
        register_mjlab_task=register_mjlab_task,
        rl_cfg_fn=rl_cfg_fn,
        sar_model=sar_model,
        clip=clip,
    )


def _register_mimic_sar_tasks(
    register_mjlab_task: Callable[..., None],
    rl_cfg_fn: Callable[[], Any],
    sar_model: SynergyModel,
    clip: MotionClip | None,
) -> None:
    """Internal implementation for SAR task registration."""
    import importlib.util

    if importlib.util.find_spec("musclemimic_models") is None:
        return

    from benchmarks.sar_backends.sar_torch_transform import SARTorchTransform
    from ml_collections import config_dict

    from myosuite.integrations.musclemimic.bimanual_model import (
        build_mimic_bimanual_spec,
        default_mimic_config,
    )
    from myosuite.integrations.musclemimic.fullbody_model import (
        build_mimic_fullbody_spec,
        default_mimic_fullbody_config,
    )

    def _sar(n_muscles: int) -> SARTorchTransform:
        if sar_model.n_muscles != n_muscles:
            raise ValueError(
                f"SAR model has {sar_model.n_muscles} muscles but model has {n_muscles}"
            )
        return SARTorchTransform(sar_model.ica, sar_model.pca, sar_model.scaler)

    # --- Bimanual ---
    try:
        b_cfg = default_mimic_config()
        b_dict = config_dict.create(**dict(b_cfg))
        b_spec_probe, _ = build_mimic_bimanual_spec(b_dict)
        b_mj = b_spec_probe.compile()
        b_muscles = _muscle_actuator_names(b_mj)
        b_tendons = _muscle_tendon_names(b_mj)
        if b_muscles and b_tendons:
            b_transform = _sar(len(b_muscles))

            def _bimanual_spec_fn() -> Any:
                spec, _ = build_mimic_bimanual_spec(b_dict)
                _strip_spec_keyframes(spec)
                return spec

            b_ctrl_dt = float(b_cfg.ctrl_dt)
            b_env = _make_mimic_sar_env_cfg(
                _task_id="myoMimicBimanual-SAR-v0",
                entity_name="mimic_bimanual_robot",
                variant="bimanual",
                spec_fn=_bimanual_spec_fn,
                muscle_actuators=b_muscles,
                tendon_targets=b_tendons,
                sar_transform=b_transform,
                sim_dt=float(b_cfg.sim_dt),
                ctrl_dt=b_ctrl_dt,
                max_episode_steps=int(b_cfg.max_episode_steps),
                clip=clip,
            )
            for task_id in ("myoMimicBimanual-SAR-v0", "myoMuscleMimicBimanual-SAR-v0"):
                register_mjlab_task(
                    task_id=task_id,
                    env_cfg=b_env,
                    play_env_cfg=b_env,
                    rl_cfg=rl_cfg_fn(),
                    runner_cls=None,
                )
    except Exception:
        pass

    # --- Full body ---
    try:
        f_cfg = default_mimic_fullbody_config()
        f_dict = config_dict.create(**dict(f_cfg))
        f_spec_probe, _ = build_mimic_fullbody_spec(f_dict)
        f_mj = f_spec_probe.compile()
        f_muscles = _muscle_actuator_names(f_mj)
        f_tendons = _muscle_tendon_names(f_mj)
        if f_muscles and f_tendons:
            f_transform = _sar(len(f_muscles))

            def _fullbody_spec_fn() -> Any:
                spec, _ = build_mimic_fullbody_spec(f_dict)
                _strip_spec_keyframes(spec)
                return spec

            f_ctrl_dt = float(f_cfg.ctrl_dt)
            f_env = _make_mimic_sar_env_cfg(
                _task_id="myoMimicFullbody-SAR-v0",
                entity_name="mimic_fullbody_robot",
                variant="fullbody",
                spec_fn=_fullbody_spec_fn,
                muscle_actuators=f_muscles,
                tendon_targets=f_tendons,
                sar_transform=f_transform,
                sim_dt=float(f_cfg.sim_dt),
                ctrl_dt=f_ctrl_dt,
                max_episode_steps=int(f_cfg.max_episode_steps),
                clip=clip,
            )
            for task_id in ("myoMimicFullbody-SAR-v0", "myoMuscleMimicFullbody-SAR-v0"):
                register_mjlab_task(
                    task_id=task_id,
                    env_cfg=f_env,
                    play_env_cfg=f_env,
                    rl_cfg=rl_cfg_fn(),
                    runner_cls=None,
                )
    except Exception:
        pass


def _make_mimic_sar_env_cfg(
    *,
    _task_id: str,
    entity_name: str,
    variant: str,
    spec_fn: Callable[[], Any],
    muscle_actuators: tuple[str, ...],
    tendon_targets: tuple[str, ...],
    sar_transform: Any,
    sim_dt: float,
    ctrl_dt: float,
    max_episode_steps: int,
    clip: MotionClip | None = None,
) -> Any:
    """Build a :class:`~mjlab.envs.ManagerBasedRlEnvCfg` with SAR action space.

    Identical to :func:`_make_mimic_env_cfg` except the ``"muscles"`` action
    term uses :class:`SARMuscleActivationActionCfg` so the policy operates in
    synergy space.

    Args:
        _task_id: Task ID (informational).
        entity_name: Scene entity name.
        variant: ``"bimanual"`` or ``"fullbody"``.
        spec_fn: Callable returning an :class:`mujoco.MjSpec`.
        muscle_actuators: Full-dimensional muscle actuator names.
        tendon_targets: Tendon names for XmlMuscle actuator config.
        sar_transform: Fitted :class:`~benchmarks.sar_backends.sar_torch_transform.SARTorchTransform`.
        sim_dt: Physics timestep.
        ctrl_dt: Control timestep.
        max_episode_steps: Episode length in control steps.
        clip: Optional MotionClip for trajectory-mode targets.

    Returns:
        Configured :class:`~mjlab.envs.ManagerBasedRlEnvCfg`.
    """
    from mjlab.actuator import XmlActuatorCfg as _XmlActuatorCfg
    from mjlab.actuator.actuator import TransmissionType
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
    from mjlab.envs import ManagerBasedRlEnvCfg
    from mjlab.envs.mdp import terminations as mdp_terminations
    from mjlab.managers.observation_manager import (
        ObservationGroupCfg,
        ObservationTermCfg,
    )
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.scene import SceneCfg
    from mjlab.sim import MujocoCfg, SimulationCfg

    articulation = EntityArticulationInfoCfg(
        actuators=(
            _XmlActuatorCfg(
                target_names_expr=tendon_targets,
                transmission_type=TransmissionType.TENDON,
            ),
        )
    )
    entity_cfg = EntityCfg(spec_fn=spec_fn, articulation=articulation)
    scene_cfg = SceneCfg(num_envs=1, entities={entity_name: entity_cfg})

    obs_terms: dict[str, Any] = {
        "qpos": ObservationTermCfg(func=_mimic_obs_qpos(entity_name)),
        "qvel": ObservationTermCfg(func=_mimic_obs_qvel(entity_name)),
        "act": ObservationTermCfg(func=_mimic_obs_act(entity_name)),
        "mimic_site_pos": ObservationTermCfg(
            func=_mimic_obs_site_pos(entity_name, variant, clip, ctrl_dt)
        ),
        "mimic_site_target": ObservationTermCfg(
            func=_mimic_obs_target(entity_name, variant, clip, ctrl_dt)
        ),
        "mimic_site_err": ObservationTermCfg(
            func=_mimic_obs_err(entity_name, variant, clip, ctrl_dt)
        ),
    }

    if clip is not None:
        if clip.qpos is not None:
            obs_terms["clip_ref_qpos"] = ObservationTermCfg(
                func=_mimic_obs_clip_ref_qpos(entity_name, variant, clip, ctrl_dt)
            )
        if clip.qvel is not None:
            obs_terms["clip_ref_qvel"] = ObservationTermCfg(
                func=_mimic_obs_clip_ref_qvel(entity_name, variant, clip, ctrl_dt)
            )
        obs_terms["clip_phase"] = ObservationTermCfg(
            func=_mimic_obs_clip_phase(entity_name, variant, clip, ctrl_dt)
        )

    observations = {"policy": ObservationGroupCfg(terms=obs_terms)}
    actions = {
        "muscles": SARMuscleActivationActionCfg(
            entity_name=entity_name,
            actuator_names=muscle_actuators,
            sar_transform=sar_transform,
        ),
    }
    terminations = {
        "time_out": TerminationTermCfg(func=mdp_terminations.time_out, time_out=True),
    }
    rewards = {
        "tracking": RewardTermCfg(
            func=_mimic_tracking_reward(entity_name, variant, clip, ctrl_dt),
            weight=1.0,
        ),
    }

    decimation = max(1, int(round(ctrl_dt / sim_dt)))
    episode_length_s = float(max_episode_steps) * ctrl_dt

    return ManagerBasedRlEnvCfg(
        scene=scene_cfg,
        decimation=decimation,
        episode_length_s=episode_length_s,
        observations=observations,
        actions=actions,
        terminations=terminations,
        rewards=rewards,
        sim=SimulationCfg(
            mujoco=MujocoCfg(timestep=sim_dt, ccd_iterations=500),
            njmax=512,
            nconmax=256,
        ),
    )


# ---------------------------------------------------------------------------
# Directional walk task with SAR action space
# ---------------------------------------------------------------------------
# A simpler alternative to trajectory-following Mimic tasks.  The policy
# receives a forward-velocity reward and an alive bonus; no motion-clip is
# required.  Because actions live in synergy space (n_syn << n_muscles)
# training converges faster and the resulting gait looks more natural.
#
# Task id: ``myoFullBodyWalkSAR-v0``
#
# Observations: qpos (without root XY), qvel (scaled), act, root_vel (xyz)
# Actions:      SAR synergies → SARMuscleActivationAction → MuJoCo ctrl
# Reward:       5 × exp(−(v_fwd − v_target)²)
#             + alive_bonus × (height > min_height)
#             − act_reg_weight × mean(act²)
# ---------------------------------------------------------------------------

_DIR_FWD_IDX: int = 1  # qvel index for the forward (Y) direction
_DIR_MIN_HEIGHT: float = 0.5  # root z-pos fall threshold for fullbody model


def _dir_obs_qpos_wo_root_xy(entity_name: str) -> Callable[[Any], Any]:
    """qpos without root XY translation (indices 0–1 removed). Shape (N, nq-2)."""

    def _fn(env: Any) -> Any:
        return env.scene[entity_name].data.data.qpos[:, 2:].clone()

    return _fn


def _dir_obs_root_vel(entity_name: str) -> Callable[[Any], Any]:
    """Root translational velocity (x, y, z). Shape (N, 3)."""

    def _fn(env: Any) -> Any:
        return env.scene[entity_name].data.data.qvel[:, :3].clone()

    return _fn


def _dir_vel_reward(
    entity_name: str,
    *,
    target_vel: float,
    fwd_idx: int = _DIR_FWD_IDX,
) -> Callable[[Any], Any]:
    """exp(−(target_vel − qvel[fwd_idx])²). Shape (N,)."""

    def _fn(env: Any) -> Any:
        import torch  # noqa: PLC0415

        fwd_vel = env.scene[entity_name].data.data.qvel[:, fwd_idx]
        return torch.exp(-torch.square(target_vel - fwd_vel))

    return _fn


def _dir_alive_reward(
    entity_name: str,
    *,
    min_height: float = _DIR_MIN_HEIGHT,
) -> Callable[[Any], Any]:
    """1.0 while root z-position ≥ min_height, else 0.0. Shape (N,)."""

    def _fn(env: Any) -> Any:
        import torch  # noqa: PLC0415

        height = env.scene[entity_name].data.data.qpos[:, 2]
        return (height >= min_height).to(dtype=torch.float32)

    return _fn


def _dir_act_reg(entity_name: str) -> Callable[[Any], Any]:
    """Mean squared activation regularisation. Shape (N,)."""

    def _fn(env: Any) -> Any:
        import torch  # noqa: PLC0415

        act = env.scene[entity_name].data.data.act
        return torch.mean(torch.square(act), dim=1)

    return _fn


def _make_directional_sar_env_cfg(
    *,
    _task_id: str,
    entity_name: str,
    spec_fn: Callable[[], Any],
    muscle_actuators: tuple[str, ...],
    tendon_targets: tuple[str, ...],
    sar_transform: Any,
    sim_dt: float,
    ctrl_dt: float,
    max_episode_steps: int,
    target_vel: float = 1.2,
    alive_bonus: float = 0.2,
    act_reg_weight: float = 0.001,
    num_envs: int = 1,
    mj_model: Any = None,
) -> Any:
    """Build :class:`~mjlab.envs.ManagerBasedRlEnvCfg` for the directional SAR walk task.

    The environment rewards walking forward at *target_vel* m/s using only the
    compressed SAR action space.  No motion-clip is required.

    Args:
        _task_id: Informational task name string.
        entity_name: Scene entity name for the articulated body.
        spec_fn: Callable returning a :class:`mujoco.MjSpec`.
        muscle_actuators: Muscle actuator names (length must match SAR n_muscles).
        tendon_targets: Tendon names for the :class:`XmlMuscleActuatorCfg`.
        sar_transform: :class:`SARTorchTransform` inverse-mapping synergies→activations.
        sim_dt: Physics timestep in seconds.
        ctrl_dt: Control timestep in seconds.
        max_episode_steps: Episode truncation length in control steps.
        target_vel: Forward walking speed target in m/s.
        alive_bonus: Weight for the alive (upright) reward term.
        act_reg_weight: Weight for the L2 activation penalty (negated internally).
        num_envs: Number of parallel environments.
        mj_model: Optional compiled :class:`mujoco.MjModel` (e.g. from ``spec.compile()``)
            with keyframes, used to seed :class:`~mjlab.entity.EntityCfg.InitialStateCfg`.
            If ``None``, mjlab's default initial state is used.

    Returns:
        A fully configured :class:`~mjlab.envs.ManagerBasedRlEnvCfg`.
    """
    from mjlab.actuator.actuator import TransmissionType
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
    from mjlab.envs import ManagerBasedRlEnvCfg
    from mjlab.envs.mdp import terminations as mdp_terminations
    from mjlab.managers.observation_manager import (
        ObservationGroupCfg,
        ObservationTermCfg,
    )
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.scene import SceneCfg
    from mjlab.sim import MujocoCfg, SimulationCfg
    from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
        _XmlWrappedActuatorCfg,
    )

    decimation = max(1, round(ctrl_dt / sim_dt))

    articulation = EntityArticulationInfoCfg(
        actuators=(
            _XmlWrappedActuatorCfg(
                target_names_expr=tendon_targets,
                transmission_type=TransmissionType.TENDON,
            ),
        )
    )
    # Use the compiled model's keyframe for explicit init_state so the body
    # resets to the standing pose rather than the all-zeros default.
    init_state = (
        _init_state_from_model(mj_model)
        if mj_model is not None
        else EntityCfg.InitialStateCfg()
    )
    entity_cfg = EntityCfg(
        spec_fn=spec_fn, articulation=articulation, init_state=init_state
    )
    scene_cfg = SceneCfg(num_envs=num_envs, entities={entity_name: entity_cfg})

    observations = {
        "policy": ObservationGroupCfg(
            terms={
                "qpos": ObservationTermCfg(func=_dir_obs_qpos_wo_root_xy(entity_name)),
                "qvel": ObservationTermCfg(func=_mimic_obs_qvel(entity_name)),
                "act": ObservationTermCfg(func=_mimic_obs_act(entity_name)),
                "root_vel": ObservationTermCfg(func=_dir_obs_root_vel(entity_name)),
            },
        ),
    }

    actions = {
        "muscles": SARMuscleActivationActionCfg(
            entity_name=entity_name,
            actuator_names=muscle_actuators,
            sar_transform=sar_transform,
        ),
    }

    rewards = {
        "forward_vel": RewardTermCfg(
            func=_dir_vel_reward(entity_name, target_vel=target_vel),
            weight=5.0,
        ),
        "alive": RewardTermCfg(
            func=_dir_alive_reward(entity_name),
            weight=alive_bonus,
        ),
        "act_reg": RewardTermCfg(
            func=_dir_act_reg(entity_name),
            weight=-act_reg_weight,
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(
            func=mdp_terminations.time_out,
            time_out=True,
        ),
    }

    return ManagerBasedRlEnvCfg(
        scene=scene_cfg,
        decimation=decimation,
        episode_length_s=float(max_episode_steps) * ctrl_dt,
        observations=observations,
        actions=actions,
        terminations=terminations,
        rewards=rewards,
        sim=SimulationCfg(mujoco=MujocoCfg(timestep=sim_dt, ccd_iterations=500)),
    )


def register_directional_walk_sar(
    register_mjlab_task: Callable[..., None],
    rl_cfg_fn: Callable[[], Any],
    sar_dir: str | Path,
    *,
    target_vel: float = 1.2,
    alive_bonus: float = 0.2,
    act_reg_weight: float = 0.001,
) -> None:
    """Register ``myoFullBodyWalkSAR-v0``: directional forward walk in synergy space.

    Unlike the Mimic tasks this task does **not** require a motion clip.
    The reward is a simple forward-velocity term plus an alive bonus, making it
    well-suited for quick finetuning experiments:

    - Action dim is ``n_syn`` (e.g. 28) instead of 354 → faster convergence.
    - Synergies encode natural walking patterns → better initial inductive bias.

    Args:
        register_mjlab_task: ``mjlab.tasks.registry.register_mjlab_task`` (or mock).
        rl_cfg_fn: Callable returning an :class:`~mjlab.rl.RslRlOnPolicyRunnerCfg`.
        sar_dir: Directory produced by :func:`save_synergy_model`.
        target_vel: Target forward walking speed in m/s (default 1.2).
        alive_bonus: Weight for the alive reward term (default 0.2).
        act_reg_weight: Weight for the activation L2 penalty (default 0.001).

    Raises:
        FileNotFoundError: If *sar_dir* does not exist.

    Note:
        Silently exits if ``musclemimic_models`` is not installed.
    """
    import importlib.util

    sar_dir = Path(sar_dir)
    if not sar_dir.exists():
        raise FileNotFoundError(f"SAR directory not found: {sar_dir}")

    if importlib.util.find_spec("musclemimic_models") is None:
        return

    from benchmarks.sar_backends.sar_torch_transform import SARTorchTransform
    from ml_collections import config_dict

    from myosuite.integrations.musclemimic.fullbody_model import (
        build_mimic_fullbody_spec,
        default_mimic_fullbody_config,
    )
    from myosuite.integrations.musclemimic.sar_extraction import load_synergy_model

    sar_model = load_synergy_model(sar_dir)

    try:
        f_cfg = default_mimic_fullbody_config()
        f_dict = config_dict.create(**dict(f_cfg))
        f_spec_probe, _ = build_mimic_fullbody_spec(f_dict)
        f_mj = f_spec_probe.compile()
        f_muscles = _muscle_actuator_names(f_mj)
        f_tendons = _muscle_tendon_names(f_mj)
        if not (f_muscles and f_tendons):
            return
        if sar_model.n_muscles != len(f_muscles):
            raise ValueError(
                f"SAR model has {sar_model.n_muscles} muscles "
                f"but the fullbody model has {len(f_muscles)}.  "
                "Re-extract synergies using the same fullbody model."
            )
        f_transform = SARTorchTransform(sar_model.ica, sar_model.pca, sar_model.scaler)

        def _spec_fn() -> Any:
            spec, _ = build_mimic_fullbody_spec(f_dict)
            _strip_spec_keyframes(spec)
            return spec

        env_cfg = _make_directional_sar_env_cfg(
            _task_id="myoFullBodyWalkSAR-v0",
            entity_name="fullbody_walk_robot",
            spec_fn=_spec_fn,
            muscle_actuators=f_muscles,
            tendon_targets=f_tendons,
            sar_transform=f_transform,
            sim_dt=float(f_cfg.sim_dt),
            ctrl_dt=float(f_cfg.ctrl_dt),
            max_episode_steps=int(f_cfg.max_episode_steps),
            target_vel=target_vel,
            alive_bonus=alive_bonus,
            act_reg_weight=act_reg_weight,
            mj_model=f_mj,
        )
        register_mjlab_task(
            task_id="myoFullBodyWalkSAR-v0",
            env_cfg=env_cfg,
            play_env_cfg=env_cfg,
            rl_cfg=rl_cfg_fn(),
            runner_cls=None,
        )
    except Exception:
        pass


def default_mimic_clip_on_policy_runner_cfg(**kwargs) -> Any:
    """Return mjlab PPO runner defaults tuned for clip-mode MuscleMimic.

    Hyperparameters and network width match the spirit of
    ``tutorials/mimic/train_mimic.py`` (256×4 MLP, lower LR, lower entropy,
    fixed LR schedule, observation normalization, advantage norm per
    minibatch) while remaining compatible with ``MjlabOnPolicyRunner``.

    Pass as ``rl_cfg_fn`` to :func:`register_mimic_mjlab_tasks_with_clip`
    instead of ``lambda: RslRlOnPolicyRunnerCfg()`` for better sample
    efficiency on the high-dimensional mimic observation.

    Returns:
        A configured :class:`~mjlab.rl.RslRlOnPolicyRunnerCfg` instance.
    """
    from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

    actor = RslRlModelCfg(
        hidden_dims=(256, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    )
    critic = RslRlModelCfg(
        hidden_dims=(256, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg=None,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="adaptive",
        entropy_coef=0.01,
        clip_param=0.2,
        value_loss_coef=1.0,
        normalize_advantage_per_mini_batch=False,
        use_clipped_value_loss=True,
    )
    return RslRlOnPolicyRunnerCfg(
        actor=actor,
        critic=critic,
        algorithm=algorithm,
        # upload_model=False,
        **kwargs,
    )


__all__ = [
    "SARMuscleActivationAction",
    "SARMuscleActivationActionCfg",
    "ClipTrajectorySource",
    "default_mimic_clip_on_policy_runner_cfg",
    "register_mimic_mjlab_tasks",
    "register_mimic_mjlab_tasks_with_clip",
    "register_mimic_mjlab_tasks_with_sar",
    "register_directional_walk_sar",
]
