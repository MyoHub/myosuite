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
    Targets are taken from the clip's ``site_xpos`` array, one frame per
    control step since each environment's last reset (mjlab's integer
    ``episode_length_buf``).  Each of the N parallel environments starts at a
    *different random frame* in the clip; on episode reset that offset is
    resampled.  This produces a diverse
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

import logging
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from myosuite.core.trajectory_io import MotionClip
from myosuite.envs.myo.backends.mjlab.mjlab_env_base import MjlabEntityAccessor
from myosuite.physics.quat_math import quat2mat
from myosuite.terms.mimic_reward import MimicTrackingConfig, mimic_site_tracking_reward

if TYPE_CHECKING:
    from myosuite.envs.myo.backends.mjlab.clip_trajectory_source import (
        ClipTrajectorySource,
        MultiClipTrajectorySource,
    )
    from myosuite.integrations.musclemimic.sar_extraction import SynergyModel

# ---------------------------------------------------------------------------
# Module-level cache keyed by (env_config_id, entity_name, variant).
# Populated lazily on first obs/reward evaluation.
# ---------------------------------------------------------------------------

_mimic_mjlab_cache: dict[tuple[int, str, str], dict[str, Any]] = {}


def _mimic_cache_key(env: Any, entity_name: str, variant: str) -> tuple[int, str, str]:
    """Cache key that survives env wrappers.

    ``id(env)`` differs per wrapper, so every view of one env would get its own
    entry — and hence its own :class:`ClipTrajectorySource`, which draws random
    start frames on first use. ``env.cfg`` is shared by all views; keying on
    it also lets the entry be dropped when the config is collected.
    """
    owner = getattr(env, "cfg", None)
    if owner is None:
        return (id(env), entity_name, variant)
    key = (id(owner), entity_name, variant)
    if key not in _mimic_mjlab_cache:
        try:
            weakref.finalize(owner, _mimic_mjlab_cache.pop, key, None)
        except TypeError:
            # Not weak-referenceable (e.g. a SimpleNamespace test stub).
            # the entry then lives as long as the process, as it did before.
            pass
    return key


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
            weights=clip.weights,
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
        weights=clip.weights,
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


def _reward_mode_uses_env_objective(reward_mode: str) -> bool:
    """Return whether *reward_mode* includes the native task objective."""
    return reward_mode in (_MIMIC_REWARD_MODE_ENV, _MIMIC_REWARD_MODE_AUGMENTED)


def _require_supported_reward_mode(reward_mode: str) -> str:
    """Validate *reward_mode* and reject modes the Mimic tasks cannot honour.

    ``"env"`` and ``"augmented"`` add a native task reward; the Mimic tasks
    define none, only the clip-tracking objective.

    Raises:
        ValueError: If *reward_mode* is not a known mode.
        NotImplementedError: If *reward_mode* needs a native task reward.
    """
    normalized = _normalize_mimic_reward_mode(reward_mode)
    if _reward_mode_uses_env_objective(normalized):
        raise NotImplementedError(
            f"reward_mode={normalized!r} needs a native task reward, which the "
            "mjlab Mimic tasks do not define; only 'mimic' is supported."
        )
    return normalized


def _clip_has_required_indices(
    indices: np.ndarray | None,
    required: range,
) -> bool:
    if indices is None:
        return False
    required_arr = np.asarray(tuple(required), dtype=np.int32)
    return bool(np.isin(required_arr, indices).all())


def _select_clip_columns(
    values: Any, indices: np.ndarray | None, index: Any | None = None
) -> Any:
    """Columns *indices* of *values*; *index* is the same indices as a device tensor."""
    if values is None or indices is None:
        return None
    if values.shape[-1] == int(indices.size) and np.array_equal(
        indices, np.arange(int(indices.size), dtype=np.int32)
    ):
        return values
    if hasattr(values, "index_select") and hasattr(values, "device"):
        import torch

        if index is None:
            index = torch.as_tensor(indices, device=values.device, dtype=torch.long)
        return values.index_select(1, index)
    return values[..., indices]


def _clip_columns(
    cache: dict[str, Any], name: str, values: Any, indices: np.ndarray | None
) -> Any:
    """:func:`_select_clip_columns` with the index uploaded once per cache."""
    import torch

    if values is None or indices is None:
        return None
    index = _device_constant(cache, f"{name}_cols", indices, values.device, torch.long)
    return _select_clip_columns(values, indices, index)


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
            # qvel[3:6] of a free joint is the body-frame angular velocity.
            ang_vel = tuple(
                float(x) for x in quat2mat(np.asarray(rot)) @ np.asarray(kqvel[3:6])
            )

    # Per-joint positions for all non-free joints. Keys are anchored regexes,
    # because mjlab matches them as such (``knee_angle_r`` alone also hits
    # ``knee_angle_rotation2_r``).
    joint_pos: dict[str, float] = {}
    joint_vel: dict[str, float] = {}
    for j in range(mj_model.njnt):
        name = _mj.mj_id2name(mj_model, _mj.mjtObj.mjOBJ_JOINT, j)
        jtype = int(mj_model.jnt_type[j])
        if not name:
            continue
        if jtype in (int(_mj.mjtJoint.mjJNT_SLIDE), int(_mj.mjtJoint.mjJNT_HINGE)):
            qpos_adr = int(mj_model.jnt_qposadr[j])
            joint_pos[f"^{name}$"] = float(kqpos[qpos_adr])
            if kqvel is not None:
                dof_adr = int(mj_model.jnt_dofadr[j])
                joint_vel[f"^{name}$"] = float(kqvel[dof_adr])

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
        # free joints (e.g. detached props) that are not covered by the
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


def _mimic_episode_steps(env: Any) -> Any:
    """Per-env control steps since the last reset, ``(N,)`` int64.

    Clip frames advance one per control step, so they are indexed by mjlab's
    integer ``episode_length_buf``, like the CPU twin's step counter.  The
    float32 ``data.time`` drifts: ``floor(time / ctrl_dt)`` lags the counter
    on most steps of a 1000-step episode.
    """
    return env.episode_length_buf


# ---------------------------------------------------------------------------
# Target synchronisation  (random box  OR  trajectory clip)
# ---------------------------------------------------------------------------


def _tensor_state(tensor: Any) -> tuple[Any, int] | None:
    """*tensor* with its in-place version counter, or ``None`` if it keeps none.

    Equal states (the same object at the same version) hold equal values: every
    in-place write bumps the version, as ``torch.utils.checkpoint`` relies on.
    Inference tensors (and ``None``) keep no version, so they never compare equal.
    """
    try:
        return tensor, tensor._version
    except (AttributeError, RuntimeError):
        return None


def _target_inputs(env: Any, cache: dict[str, Any]) -> tuple[Any, ...]:
    """States of the tensors the step's targets derive from.

    The per-env step counter and, in trajectory mode, the clip source's start
    offsets (and clip indices of a clip bank), which reset events rewrite.
    """
    source = cache.get("clip_source")
    tensors = [_mimic_episode_steps(env)]
    if source is not None:
        tensors.append(getattr(source, "_start_offsets", None))
        if hasattr(source, "_clip_indices"):
            tensors.append(source._clip_indices)
    return tuple(_tensor_state(t) for t in tensors)


def _same_state(a: tuple[Any, int] | None, b: tuple[Any, int] | None) -> bool:
    return a is not None and b is not None and a[0] is b[0] and a[1] == b[1]


_CURRENT, _STEPPED, _CHANGED = "current", "stepped", "changed"


def _target_inputs_change(env: Any, cache: dict[str, Any]) -> str:
    """How the target inputs changed since the last sync of *cache*.

    ``"current"``: unchanged, the synced targets hold.  ``"stepped"``: only
    mjlab's per-step ``episode_length_buf += 1`` (one in-place write to the same
    buffer while ``common_step_counter`` advanced by one), which cannot start
    an episode.  ``"changed"``: anything else (resets, a replaced buffer, new
    clip offsets), which needs the reset check.
    """
    synced = cache.get("synced_inputs")
    current = _target_inputs(env, cache)
    if synced is None or len(current) != len(synced):
        return _CHANGED
    if all(_same_state(a, b) for a, b in zip(current, synced)):
        return _CURRENT
    step_now, step_then = current[0], synced[0]
    counter = getattr(env, "common_step_counter", None)
    synced_counter = cache.get("synced_step_counter")
    if (
        all(_same_state(a, b) for a, b in zip(current[1:], synced[1:]))
        and step_now is not None
        and step_then is not None
        and step_now[0] is step_then[0]
        and step_now[1] == step_then[1] + 1
        and isinstance(counter, int)
        and isinstance(synced_counter, int)
        and counter == synced_counter + 1
    ):
        return _STEPPED
    return _CHANGED


def _clip_value(cache: dict[str, Any], name: str, compute: Callable[[], Any]) -> Any:
    """Clip-only value *name* of the synced step, computed once per target sync.

    The observation groups, the reward and the terminations of a step share it.
    """
    values = cache.setdefault("clip_values", {})
    if name not in values:
        values[name] = compute()
    return values[name]


def _device_constant(
    cache: dict[str, Any], name: str, value: Any, device: Any, dtype: Any
) -> Any:
    """Host array *value* as a tensor on *device*, uploaded once per cache.

    Indexing a device tensor with a NumPy array, or ``torch.as_tensor`` of one,
    copies it to the device (a host sync) on every call.
    """
    import torch

    consts = cache.setdefault("device_constants", {})
    tensor = consts.get(name)
    if tensor is None or tensor.device != torch.device(device):
        tensor = torch.as_tensor(np.asarray(value), dtype=dtype, device=device)
        consts[name] = tensor
    return tensor


def _site_index(cache: dict[str, Any], device: Any) -> Any:
    """``cache["site_ids"]`` (tracked model sites) as a device index."""
    import torch

    return _device_constant(cache, "site_ids", cache["site_ids"], device, torch.long)


def _sync_mimic_mjlab_targets(
    env: Any,
    entity_name: str,
    cache: dict[str, Any],
    *,
    check_resets: bool = True,
) -> None:
    """Update ``cache["target_torch"]`` for the current step.

    Dispatches to trajectory-clip mode when ``cache["clip_source"]`` is set,
    otherwise falls back to the original random-box sampling that resamples
    once per episode (on time regression).

    Args:
        env: mjlab environment instance.
        entity_name: Scene entity name (e.g. ``"mimic_bimanual_robot"``).
        cache: Per-env cache dict produced by :func:`_resolve_mimic_mjlab_ids`.
        check_resets: ``False`` when the step counter only advanced since the
            last sync, so no episode can have restarted: skips the reset
            check (a host sync on a GPU).
    """
    import torch

    data = env.scene[entity_name].data.data
    clip_source: ClipTrajectorySource | None = cache.get("clip_source")

    if clip_source is not None:
        # --- Trajectory mode: targets come from the MotionClip ---
        step = _mimic_episode_steps(env)  # (N,) int64
        clip_source.update(step, check_resets=check_resets)
        cache["target_torch"] = clip_source.site_targets(step)  # (N, n_tracked, 3)
    else:
        # --- Random mode: each env resamples when its own episode restarts ---
        step = _mimic_episode_steps(env)
        last = cache.get("last_step")
        target = cache.get("target_torch")
        n_env = int(data.qpos.shape[0])
        if target is None or last is None:
            reset = torch.ones(n_env, dtype=torch.bool, device=step.device)
        else:
            reset = step < last
        resample = (
            target is None or last is None or (check_resets and bool(reset.any()))
        )
        if resample:
            n_sites = int(cache["site_ids"].shape[0])
            device = data.qpos.device
            lo = _device_constant(cache, "lo", cache["lo"], device, torch.float32)
            hi = _device_constant(cache, "hi", cache["hi"], device, torch.float32)
            u = torch.rand((n_env, n_sites, 3), device=device, dtype=torch.float32)
            fresh = lo + (hi - lo) * u
            cache["target_torch"] = (
                fresh
                if target is None
                else torch.where(reset.to(device)[:, None, None], fresh, target)
            )
        cache["last_step"] = step.clone()
    cache["clip_values"] = {}  # derived from the previous inputs
    cache["synced_inputs"] = _target_inputs(env, cache)
    cache["synced_step_counter"] = getattr(env, "common_step_counter", None)


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
    Subsequent calls synchronise the targets when their inputs (the step
    counter, the clip start offsets) changed since the last sync, so the
    terms of one step phase share one sync instead of each repeating it; the
    reset check (a host sync on a GPU) is skipped after a plain env step.

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

    key = _mimic_cache_key(env, entity_name, variant)
    if key in _mimic_mjlab_cache:
        cache = _mimic_mjlab_cache[key]
        change = _target_inputs_change(env, cache)
        if change != _CURRENT:
            _sync_mimic_mjlab_targets(
                env, entity_name, cache, check_resets=change == _CHANGED
            )
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
        last_step=None,
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
        return MjlabEntityAccessor(env, entity_name).joint_pos().clone()

    return _fn


def _mimic_obs_qvel(entity_name: str) -> Callable[[Any], Any]:
    """Joint velocities scaled by ctrl_dt, ``(N, nv)``."""

    def _fn(env: Any) -> Any:
        ctrl_dt = env.physics_dt * env.cfg.decimation
        return MjlabEntityAccessor(env, entity_name).joint_vel() * ctrl_dt

    return _fn


def _mimic_obs_act(entity_name: str) -> Callable[[Any], Any]:
    """Muscle activation state ``(N, na)``."""

    def _fn(env: Any) -> Any:
        return env.scene[entity_name].data.data.act.clone()

    return _fn


def _tracked_site_pos(env: Any, entity_name: str, cache: dict[str, Any]) -> Any:
    """World positions of the tracked sites, ``(N, n_tracked, 3)``."""
    site_xpos = env.scene[entity_name].data.data.site_xpos
    return site_xpos[:, _site_index(cache, site_xpos.device), :]


def _clip_ref(env: Any, cache: dict[str, Any], name: str) -> Any:
    """``ref_qpos`` / ``ref_qvel`` / ``phase`` / ``clip_end`` of the clip source
    at the current step (computed once per target sync)."""
    source = cache["clip_source"]
    return _clip_value(
        cache, name, lambda: getattr(source, name)(_mimic_episode_steps(env))
    )


def _mimic_obs_site_pos(
    entity_name: str,
    variant: str,
    clip: MotionClip | None = None,
    ctrl_dt: float | None = None,
) -> Callable[[Any], Any]:
    """Current tracked site positions, flattened to ``(N, n_tracked * 3)``."""

    def _fn(env: Any) -> Any:
        cache = _resolve_mimic_mjlab_ids(env, entity_name, variant, clip, ctrl_dt)
        pos = _tracked_site_pos(env, entity_name, cache)
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
        tgt = cache["target_torch"]
        assert tgt is not None
        err = tgt - _tracked_site_pos(env, entity_name, cache)
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
        if cache.get("clip_source") is None:
            raise RuntimeError(
                "clip_ref_qpos obs requires trajectory mode (clip_source is None)"
            )
        ref = _clip_ref(env, cache, "ref_qpos")
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
        if cache.get("clip_source") is None:
            raise RuntimeError(
                "clip_ref_qvel obs requires trajectory mode (clip_source is None)"
            )
        ref = _clip_ref(env, cache, "ref_qvel")
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
        if cache.get("clip_source") is None:
            raise RuntimeError(
                "clip_phase obs requires trajectory mode (clip_source is None)"
            )
        return _clip_ref(env, cache, "phase")

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

    Works in both random and trajectory modes; the same term as the CPU twin.
    """

    def _fn(env: Any) -> Any:
        import torch

        cache = _resolve_mimic_mjlab_ids(env, entity_name, variant, clip, ctrl_dt)
        tracking: MimicTrackingConfig = cache["tracking"]
        tgt = cache["target_torch"]
        assert tgt is not None
        pos = _tracked_site_pos(env, entity_name, cache)
        return mimic_site_tracking_reward(
            torch, pos, tgt, scale=tracking.reward_scale
        )  # (N,)

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
        tgt_sites = cache["target_torch"]  # (N, n_sites, 3)
        assert tgt_sites is not None
        data = env.scene[entity_name].data.data
        cur_sites = _tracked_site_pos(env, entity_name, cache)  # (N, n_sites, 3)

        if (
            cache.get("clip_source") is not None
            and resolved_clip is not None
            and resolved_clip.qpos is not None
            and resolved_clip.qvel is not None
        ):
            ref_qpos = _clip_ref(env, cache, "ref_qpos")  # (N, nq)  or None
            ref_qvel = _clip_ref(env, cache, "ref_qvel")  # (N, nv) or None
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
        cur_qpos = _clip_columns(cache, "qpos", data.qpos, qpos_indices)
        cur_qvel = _clip_columns(cache, "qvel", data.qvel, qvel_indices)
        ref_qpos = _clip_columns(cache, "qpos", ref_qpos_full, qpos_indices)
        ref_qvel = _clip_columns(cache, "qvel", ref_qvel_full, qvel_indices)

        # A fixed-base entity (bimanual) has no root: qpos[:7] are hinge angles.
        free_root = not env.scene[entity_name].is_fixed_base
        has_root_pos = free_root and _clip_has_required_indices(qpos_indices, range(3))
        has_root_quat = free_root and _clip_has_required_indices(
            qpos_indices, range(3, 7)
        )
        has_root_vel = free_root and _clip_has_required_indices(qvel_indices, range(3))
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

    Returns flattened relative tracked-site positions plus future phase, and the
    root position delta / velocity when the entity has a free root.

    Args:
        entity_name: Scene entity name.
        variant: ``"bimanual"`` or ``"fullbody"``.
        clip: Source motion clip.
        ctrl_dt: Control timestep.
        k: Number of lookahead steps.
        stride: Frame stride between steps.
    """

    def _future(env: Any, clip_source: Any, root_pos: bool, root_vel: bool) -> Any:
        """Clip part of the lookahead: future sites, root pos/vel and phase."""
        import torch

        step = _mimic_episode_steps(env)  # (N,)
        clip_lengths = clip_source.clip_lengths(step)
        cur_frames = clip_source.frame_indices(step)
        frames = [(cur_frames + i * stride) % clip_lengths for i in range(1, k + 1)]
        sites = torch.stack([clip_source.site_targets_at_frames(f) for f in frames], 1)
        rpos = rvel = None
        if root_pos and clip_source.ref_qpos(step) is not None:
            rpos = torch.stack(
                [clip_source.ref_qpos_at_frames(f)[:, :3] for f in frames], 1
            )
        if root_vel and clip_source.ref_qvel(step) is not None:
            rvel = torch.stack(
                [clip_source.ref_qvel_at_frames(f)[:, :3].float() for f in frames], 1
            )
        last = torch.clamp(clip_lengths.float() - 1.0, min=1.0)
        phase = torch.stack([f.float() / last for f in frames], 1)
        return sites, rpos, rvel, phase

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
        n_envs = int(_mimic_episode_steps(env).shape[0])
        # Site targets are relative to the root; a fixed base (bimanual) has none, so
        # they stay in the world frame and carry no root terms.
        free_root = not env.scene[entity_name].is_fixed_base
        cur_root_pos = (
            data.qpos[:, :3]
            if free_root
            else torch.zeros(n_envs, 3, device=data.qpos.device)
        )

        has_root_pos = free_root and _clip_has_required_indices(
            resolved_clip.qpos_model_indices, range(3)
        )
        has_root_vel = free_root and _clip_has_required_indices(
            resolved_clip.qvel_model_indices, range(3)
        )
        per_step_dim = (
            int(clip_source.n_tracked) * 3
            + (3 if has_root_pos else 0)
            + (3 if has_root_vel else 0)
            + 1  # phase
        )
        # The clip part depends on the step only: shared by the observation groups.
        sites, rpos, rvel, phase = _clip_value(
            cache,
            f"lookahead/{k}/{stride}/{has_root_pos}/{has_root_vel}",
            lambda: _future(env, clip_source, has_root_pos, has_root_vel),
        )
        # Per lookahead step: sites rel. root, [root delta], [root vel], phase.
        pieces = [(sites - cur_root_pos[:, None, None, :]).reshape(n_envs, k, -1)]
        if rpos is not None:
            pieces.append((rpos - cur_root_pos[:, None, :]).float())
        if rvel is not None:
            pieces.append(rvel)
        pieces.append(phase[..., None])
        out = torch.cat(pieces, dim=-1).reshape(n_envs, -1)
        # A root term without clip data keeps its width as zero padding at the end.
        pad = k * per_step_dim - out.shape[1]
        if pad:
            out = torch.cat([out, out.new_zeros(n_envs, pad)], dim=1)
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
        from mjlab.utils.lab_api.math import quat_apply

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
        # env_ids=None means "all envs" in mjlab's event contract.
        if env_ids is None:
            env_ids_long = torch.arange(n_envs, device=device, dtype=torch.long)
        else:
            env_ids_long = torch.as_tensor(
                env_ids, device=device, dtype=torch.long
            ).reshape(-1)
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
        # mjlab zeroes episode_length_buf after the reset events; mark these
        # envs as already at step 0 so the next update() does not see the drop
        # as another reset and overwrite the offsets just drawn.
        if clip_source._last_step is not None:
            clip_source._last_step[env_ids_long] = 0

        # --- Write root state (pos + quat + lin_vel + ang_vel) ---
        ref_qpos = clip_source.ref_qpos_at_frames(new_offsets, env_ids_long)
        if ref_qpos is None:
            return
        # Add per-env world origins so the bodies appear in the right place.
        env_origins = env.scene.env_origins[env_ids_long]  # (n_reset, 3)
        root_pos = ref_qpos[:, :3].float() + env_origins
        root_quat = ref_qpos[:, 3:7].float()  # (w, x, y, z)

        ref_qvel = clip_source.ref_qvel_at_frames(new_offsets, env_ids_long)
        if ref_qvel is not None:
            # Free-joint qvel holds the world-frame linear but the body-frame
            # angular velocity; write_root_state_to_sim expects both in world.
            root_lin_vel = ref_qvel[:, :3].float()
            root_ang_vel = quat_apply(root_quat, ref_qvel[:, 3:6].float())
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
    use_clip_root: bool = True,
) -> Callable[[Any], Any]:
    """Terminate episodes where tracking error exceeds recovery threshold.

    Same rule as the CPU ``mimic_should_terminate``: mean site error against
    the clip targets, plus the root position error against the clip's
    reference root.  The root check runs only when the entity has a free root
    joint, *use_clip_root* is set and the clip covers the root qpos.

    Returns boolean tensor ``(N,)`` — ``True`` for envs that should terminate.
    """
    from myosuite.terms.mimic_obs import mimic_termination_mask

    def _fn(env: Any) -> Any:
        import torch

        cache = _resolve_mimic_mjlab_ids(env, entity_name, variant, clip, ctrl_dt)
        entity = env.scene[entity_name]
        cur_sites = _tracked_site_pos(env, entity_name, cache)
        tgt = cache["target_torch"]
        if tgt is None:
            return torch.zeros(
                cur_sites.shape[0], dtype=torch.bool, device=cur_sites.device
            )

        # Root error vs the clip's reference root, only for a free root joint
        # (bimanual is fixed-base: its qpos[:3] are hinge angles).
        cur_root = ref_root = None
        resolved_clip: MotionClip | None = cache.get("clip")
        clip_source = cache.get("clip_source")
        if (
            use_clip_root
            and not entity.is_fixed_base
            and clip_source is not None
            and resolved_clip is not None
            and _clip_has_required_indices(
                resolved_clip.qpos_model_indices, range(0, 3)
            )
        ):
            ref_qpos = _clip_ref(env, cache, "ref_qpos")
            if ref_qpos is not None:
                ref_root = ref_qpos[:, :3]
                cur_root = entity.data.root_link_pos_w
        deviated = mimic_termination_mask(
            torch,
            cur_sites,
            tgt,
            cur_root,
            ref_root,
            site_err_threshold=site_err_threshold,
            root_err_threshold=root_err_threshold,
        )
        # Past the clip end the reference is undefined: that is a truncation.
        if clip_source is not None:
            deviated = deviated & ~_clip_ref(env, cache, "clip_end")
        return deviated

    return _fn


def _mimic_clip_end(
    entity_name: str,
    variant: str,
    clip: MotionClip | None = None,
    ctrl_dt: float | None = None,
) -> Callable[[Any], Any]:
    """Time-out term: ``True`` for envs that have played past the end of their clip."""

    def _fn(env: Any) -> Any:
        cache = _resolve_mimic_mjlab_ids(env, entity_name, variant, clip, ctrl_dt)
        return _clip_ref(env, cache, "clip_end")

    return _fn


# ---------------------------------------------------------------------------
# mjlab env-config builder
# ---------------------------------------------------------------------------


def _policy_actor_critic_groups(obs_terms: dict[str, Any]) -> dict[str, Any]:
    """One observation group per name rsl_rl may ask for, all with *obs_terms*.

    mjlab's default runner cfg maps ``actor``/``critic`` to groups of the same
    name and rsl_rl raises when they are missing; ``policy`` serves play.
    """
    from mjlab.managers.observation_manager import ObservationGroupCfg

    return {
        name: ObservationGroupCfg(terms=obs_terms)
        for name in ("policy", "actor", "critic")
    }


def mimic_viewer_cfg(
    entity_name: str, body_name: str = "pelvis", **overrides: Any
) -> Any:
    """Viewer camera that follows *body_name* of a full-body Mimic entity.

    A three-quarter side view of the whole body (distance 3.2 m, elevation -6
    degrees, azimuth 50 degrees). It follows the pelvis, so the walker stays in
    frame; mjlab's default camera is fixed in the world. The image size is left
    at mjlab's default; set ``width`` and ``height`` to render larger frames.

    Args:
        entity_name: Scene entity name (e.g. ``"mimic_fullbody_robot"``).
        body_name: Body of the entity to track.
        **overrides: Any other :class:`mjlab.viewer.ViewerConfig` field, e.g.
            ``width=1280, height=720`` or ``azimuth=120.0``.

    Returns:
        A :class:`mjlab.viewer.ViewerConfig`.
    """
    from mjlab.viewer import ViewerConfig

    kwargs: dict[str, Any] = dict(
        origin_type=ViewerConfig.OriginType.ASSET_BODY,
        entity_name=entity_name,
        body_name=body_name,
        distance=3.2,
        elevation=-6.0,
        azimuth=50.0,
        lookat=(0.0, 0.0, -0.05),
    )
    kwargs.update(overrides)
    return ViewerConfig(**kwargs)


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
        reward_mode: Reward composition mode.  Only ``"mimic"`` (the
            clip-tracking objective) is implemented.
        action_mode: Muscle action interpretation. ``"sigmoid"`` keeps the
            training path's canonical action normalisation; ``"direct"``
            preserves checkpoint playback by clipping incoming values to
            ``[-1, 1]`` before writing them as muscle controls.

    Returns:
        A configured :class:`~mjlab.envs.ManagerBasedRlEnvCfg` instance.

    Raises:
        NotImplementedError: If *reward_mode* needs a native task reward
            (``"env"``, ``"augmented"``).
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
    from mjlab.managers.observation_manager import ObservationTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.scene import SceneCfg
    from mjlab.sim import SimulationCfg
    from myosuite.envs.myo.backends.mjlab.tasks.cpu_reference import (
        musclemimic_mujoco_cfg,
    )
    from myosuite.envs.myo.backends.mjlab.tasks.mdp.terminations import (
        SYNC_TERM,
        sync_forward,
    )

    _require_supported_reward_mode(reward_mode)

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
        clip_bank = _normalize_motion_clip_bank(clip)
        has_clip_qpos = all(c.qpos is not None for c in clip_bank)
        has_clip_qvel = all(c.qvel is not None for c in clip_bank)
        if enable_clip_state_terms and has_clip_qpos:
            obs_terms["clip_ref_qpos"] = ObservationTermCfg(
                func=_mimic_obs_clip_ref_qpos(entity_name, variant, clip, ctrl_dt)
            )
        if enable_clip_state_terms and has_clip_qvel:
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

    observations = _policy_actor_critic_groups(obs_terms)
    actions = {
        "muscles": MyoMuscleActivationActionCfg(
            entity_name=entity_name,
            actuator_names=muscle_actuators,
            tendon_names=tendon_targets,
            action_mode=action_mode,
        ),
    }
    terminations = {
        # Site positions are derived quantities: refresh them before the reward and
        # the deviation check score them (the CPU twin runs mj_forward after stepping).
        SYNC_TERM: TerminationTermCfg(func=sync_forward),
        "time_out": TerminationTermCfg(func=mdp_terminations.time_out, time_out=True),
    }
    if clip is not None:
        terminations["clip_end"] = TerminationTermCfg(
            func=_mimic_clip_end(entity_name, variant, clip, ctrl_dt), time_out=True
        )
    if use_early_termination and clip is not None:
        terminations["mimic_deviation"] = TerminationTermCfg(
            func=_mimic_early_termination(
                entity_name,
                variant,
                clip,
                ctrl_dt,
                use_clip_root=enable_clip_state_terms,
            ),
            time_out=False,
        )

    # Reset handling:
    # - use RSI when clip qpos/qvel align with the model
    # - otherwise restore the compiled model keyframe so auxiliary free joints
    #   (e.g. detached props) do not reset to all zeros
    events: dict[str, Any] = {}
    if clip is not None and enable_clip_state_terms and has_clip_qpos:
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

    if (
        use_deepmimic_reward
        and clip is not None
        and enable_clip_state_terms
        and has_clip_qpos
        and has_clip_qvel
    ):
        reward_fn = _mimic_deepmimic_reward(entity_name, variant, clip, ctrl_dt)
    else:
        reward_fn = _mimic_tracking_reward(entity_name, variant, clip, ctrl_dt)

    # mjlab RewardManager multiplies every term by ctrl_dt (scale_by_dt=True).
    # Compensate so the logged reward matches the raw composite reward (0–1).
    reward_weight = float(mimic_reward_weight) / ctrl_dt if ctrl_dt > 0 else 1.0
    rewards = {"tracking": RewardTermCfg(func=reward_fn, weight=reward_weight)}
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
            mujoco=musclemimic_mujoco_cfg(variant, timestep=sim_dt),
            njmax=512,
            nconmax=256,
        ),
    )
    if events:
        env_cfg_kwargs["events"] = events
    if variant == "fullbody":  # the bimanual body has no pelvis: keep mjlab's camera
        env_cfg_kwargs["viewer"] = mimic_viewer_cfg(entity_name)
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
    clip: MotionClip | tuple[MotionClip, ...] | list[MotionClip],
    use_deepmimic_reward: bool = True,
    use_lookahead: bool = True,
    use_early_termination: bool = True,
    action_mode: str = "sigmoid",
    reward_mode: str = _MIMIC_REWARD_MODE_MIMIC,
    mimic_reward_weight: float = 5.0,
    env_reward_weight: float = 1.0,
) -> None:
    """Register Mimic tasks in trajectory mode.

    Targets are taken from *clip*'s ``site_xpos``, advancing one frame per
    control step of each environment's episode.  Each of the N parallel
    environments starts at a random frame; on reset that offset is resampled.

    Additional observation terms are added automatically:

    * ``clip_ref_qpos`` — reference qpos at the current frame (if available)
    * ``clip_ref_qvel`` — reference qvel at the current frame (if available)
    * ``clip_phase``    — normalised phase in ``[0, 1]``

    Args:
        register_mjlab_task: mjlab task registry function.
        rl_cfg_fn: Callable returning a default RL runner config.
        clip: Loaded :class:`~myosuite.core.trajectory_io.MotionClip` with
              ``site_xpos`` populated, or a tuple/list of such clips: every env
              then draws its clip and start frame on each reset.
        action_mode: Muscle action interpretation. Leave as ``"sigmoid"`` for
            training; use ``"direct"`` for fullbody checkpoint inference.
        reward_mode: Reward composition for the mimic task.  Only ``"mimic"``
            (clip tracking) is implemented: the Mimic tasks have no native
            task reward for ``"env"`` or ``"augmented"`` to use.
        mimic_reward_weight: Weight of the clip-tracking reward.
        env_reward_weight: Weight of the native task reward; must stay
            ``1.0`` since the Mimic tasks define none.

    Raises:
        ValueError: If ``clip.site_xpos`` is ``None`` or *reward_mode* is
            unknown.
        NotImplementedError: If *reward_mode* or *env_reward_weight* needs a
            native task reward.
    """
    reward_mode = _require_supported_reward_mode(reward_mode)
    if env_reward_weight != 1.0:
        raise NotImplementedError(
            f"env_reward_weight={env_reward_weight} weights a native task reward, "
            "which the mjlab Mimic tasks do not define."
        )
    if any(c.site_xpos is None for c in _normalize_motion_clip_bank(clip)):
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
        reward_mode=reward_mode,
        mimic_reward_weight=mimic_reward_weight,
    )
    from mjlab.tasks.registry import list_tasks

    if "myoMimicFullbody-v0" not in list_tasks():
        raise RuntimeError(
            "Registration of myoMimicFullbody-v0 failed silently. "
            "Ensure musclemimic_models is installed "
            "('pip install myosuite[musclemimic]') and the clip is valid."
        )


def _register_mimic_tasks(
    register_mjlab_task: Callable[..., None],
    rl_cfg_fn: Callable[[], Any],
    clip: MotionClip | None,
    use_deepmimic_reward: bool = True,
    use_lookahead: bool = True,
    use_early_termination: bool = True,
    action_mode: str = "sigmoid",
    reward_mode: str = _MIMIC_REWARD_MODE_MIMIC,
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
                reward_mode=reward_mode,
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
        reward_mode=reward_mode,
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
    passed through a :class:`~myosuite.integrations.musclemimic.sar_torch_transform.SARTorchTransform`
    (MinMaxScaler⁻¹ → FastICA⁻¹ → PCA⁻¹ → clamp) to produce ``n_muscles``
    activations in ``[0, 1]``, which are then written as the effort targets
    of the muscle tendons (the ``XmlActuatorCfg`` wrapping the muscles copies
    them into MuJoCo's ``ctrl`` before every physics step).

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
        import mujoco
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
        # mjlab rewrites the ctrl of XmlActuatorCfg-wrapped actuators from their
        # targets before every physics step, so a direct ctrl write is lost:
        # write each muscle's activation to the tendon it pulls instead.
        model = env.sim.mj_model
        entity_tendons = entity.indexing.tendon_ids.tolist()
        tendon_ids = []
        for act_id in entity.indexing.ctrl_ids[target_ids].tolist():
            if int(model.actuator_trntype[act_id]) != mujoco.mjtTrn.mjTRN_TENDON:
                raise ValueError(
                    f"SARMuscleActivationAction: actuator "
                    f"{model.actuator(act_id).name!r} has no tendon transmission"
                )
            tendon = int(model.actuator_trnid[act_id, 0])
            tendon_ids.append(entity_tendons.index(tendon))
        self._entity = entity
        self._tendon_ids = torch.tensor(
            tendon_ids, device=self.device, dtype=torch.long
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
        """Write muscle activations as the effort targets of the muscle tendons."""
        self._entity.set_tendon_effort_target(
            self._processed_actions, tendon_ids=self._tendon_ids
        )

    def reset(self, env_ids: Any = None) -> None:
        """Zero the stored actions of the reset environments (all if ``None``)."""
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0


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
    are applied in simulation.  Everything else (initial state, observations,
    RSI, early termination, rewards) is the ``myoMimic*-v0`` task that
    :func:`register_mimic_mjlab_tasks` (no clip) or
    :func:`register_mimic_mjlab_tasks_with_clip` registers.

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

    from myosuite.integrations.musclemimic.sar_torch_transform import SARTorchTransform
    from ml_collections import config_dict

    from myosuite.envs.myo.backends.mjlab.configs.musclemimic_bimanual_cfg import (
        MuscleMimicBimanualCfg,
    )
    from myosuite.envs.myo.backends.mjlab.configs.musclemimic_fullbody_cfg import (
        MuscleMimicFullbodyCfg,
    )
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

    # Reward weight defaults of the muscle-space entry points:
    # register_mimic_mjlab_tasks_with_clip (clip) / register_mimic_mjlab_tasks.
    mimic_reward_weight = 5.0 if clip is not None else 1.0

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

            _b_common = dict(
                _task_id="myoMimicBimanual-SAR-v0",
                entity_name="mimic_bimanual_robot",
                variant="bimanual",
                spec_fn=_bimanual_spec_fn,
                muscle_actuators=b_muscles,
                tendon_targets=b_tendons,
                sar_transform=b_transform,
                sim_dt=float(b_cfg.sim_dt),
                ctrl_dt=float(b_cfg.ctrl_dt),
                max_episode_steps=int(b_cfg.max_episode_steps),
                clip=clip,
                mj_model=b_mj,
                mimic_reward_weight=mimic_reward_weight,
            )
            b_env = _make_mimic_sar_env_cfg(
                num_envs=MuscleMimicBimanualCfg.num_envs, **_b_common
            )
            b_play_env = _make_mimic_sar_env_cfg(num_envs=1, **_b_common)
            for task_id in ("myoMimicBimanual-SAR-v0", "myoMuscleMimicBimanual-SAR-v0"):
                register_mjlab_task(
                    task_id=task_id,
                    env_cfg=b_env,
                    play_env_cfg=b_play_env,
                    rl_cfg=rl_cfg_fn(),
                    runner_cls=None,
                )
    except Exception as exc:
        # Bimanual SAR mjlab registration is optional (needs a matching SAR
        # model + mjlab). Log at debug so the reason is discoverable instead of
        # the task silently vanishing; don't fail package import.
        logging.getLogger(__name__).debug(
            "Skipped bimanual SAR mjlab registration: %s", exc, exc_info=exc
        )

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

            # The compiled model keeps the keyframe the spec_fn strips, so the
            # entity's initial state is the standing pose, not pelvis z = 0.
            _f_common = dict(
                _task_id="myoMimicFullbody-SAR-v0",
                entity_name="mimic_fullbody_robot",
                variant="fullbody",
                spec_fn=_fullbody_spec_fn,
                muscle_actuators=f_muscles,
                tendon_targets=f_tendons,
                sar_transform=f_transform,
                sim_dt=float(f_cfg.sim_dt),
                ctrl_dt=float(f_cfg.ctrl_dt),
                max_episode_steps=int(f_cfg.max_episode_steps),
                clip=clip,
                mj_model=f_mj,
                mimic_reward_weight=mimic_reward_weight,
            )
            f_env = _make_mimic_sar_env_cfg(
                num_envs=MuscleMimicFullbodyCfg.num_envs, **_f_common
            )
            f_play_env = _make_mimic_sar_env_cfg(num_envs=1, **_f_common)
            for task_id in ("myoMimicFullbody-SAR-v0", "myoMuscleMimicFullbody-SAR-v0"):
                register_mjlab_task(
                    task_id=task_id,
                    env_cfg=f_env,
                    play_env_cfg=f_play_env,
                    rl_cfg=rl_cfg_fn(),
                    runner_cls=None,
                )
    except Exception as exc:
        # Full-body SAR mjlab registration is optional (see bimanual note above).
        logging.getLogger(__name__).debug(
            "Skipped full-body SAR mjlab registration: %s", exc, exc_info=exc
        )


def _make_mimic_sar_env_cfg(
    *,
    entity_name: str,
    muscle_actuators: tuple[str, ...],
    sar_transform: Any,
    **mimic_cfg_kwargs: Any,
) -> Any:
    """Build the :func:`_make_mimic_env_cfg` task with a SAR action space.

    Scene, initial state, observations, RSI, early termination and rewards all
    come from :func:`_make_mimic_env_cfg`; only the ``"muscles"`` action term
    is replaced by :class:`SARMuscleActivationActionCfg`, so the policy acts
    in synergy space on the same task as ``myoMimic*-v0``.

    Args:
        entity_name: Scene entity name.
        muscle_actuators: Full-dimensional muscle actuator names.
        sar_transform: Fitted :class:`~myosuite.integrations.musclemimic.sar_torch_transform.SARTorchTransform`.
        **mimic_cfg_kwargs: Remaining keyword arguments of
            :func:`_make_mimic_env_cfg` (model, timing, clip, ...).

    Returns:
        Configured :class:`~mjlab.envs.ManagerBasedRlEnvCfg`.
    """
    cfg = _make_mimic_env_cfg(
        entity_name=entity_name,
        muscle_actuators=muscle_actuators,
        **mimic_cfg_kwargs,
    )
    cfg.actions["muscles"] = SARMuscleActivationActionCfg(
        entity_name=entity_name,
        actuator_names=muscle_actuators,
        sar_transform=sar_transform,
    )
    return cfg


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
    from mjlab.managers.observation_manager import ObservationTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.scene import SceneCfg
    from mjlab.sim import SimulationCfg
    from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
        _XmlWrappedActuatorCfg,
    )
    from myosuite.envs.myo.backends.mjlab.tasks.cpu_reference import (
        musclemimic_mujoco_cfg,
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

    observations = _policy_actor_critic_groups(
        {
            "qpos": ObservationTermCfg(func=_dir_obs_qpos_wo_root_xy(entity_name)),
            "qvel": ObservationTermCfg(func=_mimic_obs_qvel(entity_name)),
            "act": ObservationTermCfg(func=_mimic_obs_act(entity_name)),
            "root_vel": ObservationTermCfg(func=_dir_obs_root_vel(entity_name)),
        }
    )

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
        sim=SimulationCfg(mujoco=musclemimic_mujoco_cfg("fullbody", timestep=sim_dt)),
        viewer=mimic_viewer_cfg(entity_name),
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

    from myosuite.integrations.musclemimic.sar_torch_transform import SARTorchTransform
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
    except Exception as exc:
        # myoFullBodyWalkSAR mjlab registration is optional (needs SAR model + mjlab).
        logging.getLogger(__name__).debug(
            "Skipped myoFullBodyWalkSAR-v0 mjlab registration: %s", exc, exc_info=exc
        )


def default_mimic_clip_on_policy_runner_cfg(**kwargs) -> Any:
    """Return mjlab PPO runner defaults tuned for clip-mode MuscleMimic.

    Hyperparameters and network width match the spirit of
    ``tutorials/files/5.2/train_mimic.py`` (256×4 MLP, lower LR, lower entropy,
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
