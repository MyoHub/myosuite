from typing import Callable, Any

from mjlab_myosuite.trajectory_io import MotionClip


def _mimic_rsi_event(
    entity_name: str,
    variant: str,
    clip: MotionClip,
    ctrl_dt: float,
) -> Callable[[Any, Any], None]:
  """Reset-event that sets qpos/qvel from the motion clip (RSI).

  On each episode reset, each environment is placed at a random frame of the
  reference clip rather than at the standing keyframe.  This is the
  *Reference State Initialization* technique from DeepMimic — without it
  the policy never sees the middle of a motion and reward signals stay weak.

  Args:
      entity_name: Scene entity name.
      variant: ``"bimanual"`` or ``"fullbody"``.
      clip: Source motion clip with ``qpos`` and ``qvel`` populated.
      ctrl_dt: Control timestep (used to initialise :class:`ClipTrajectorySource`).

  Returns:
      Closure ``(env, env_ids) -> None`` suitable for
      ``EventTermCfg(mode="reset")``.
  """

  def _fn(env: Any, env_ids: Any) -> None:
    import torch

    data = env.scene[entity_name].data.data
    n_envs = int(data.qpos.shape[0])
    device = data.qpos.device

    # Populate (or retrieve) the shared cache — also initialises clip_source.
    cache = _resolve_mimic_mjlab_ids(env, entity_name, variant, clip, ctrl_dt)
    clip_source: ClipTrajectorySource | None = cache.get("clip_source")
    if clip_source is None or clip_source._qpos_tensor is None:
      return

    # Ensure internal tensors are on the correct device.
    clip_source._ensure_device(device, n_envs)

    # Resample start offsets for the envs that just reset.
    env_ids_long = torch.as_tensor(env_ids, device=device, dtype=torch.long)
    n_reset = int(env_ids_long.shape[0])
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
    ref_qpos = clip_source._qpos_tensor[new_offsets]  # (n_reset, nq)
    # Add per-env world origins so the bodies appear in the right place.
    env_origins = env.scene.env_origins[env_ids_long]  # (n_reset, 3)
    root_pos = ref_qpos[:, :3].float() + env_origins
    root_quat = ref_qpos[:, 3:7].float()  # (w, x, y, z)

    if clip_source._qvel_tensor is not None:
      ref_qvel = clip_source._qvel_tensor[new_offsets]  # (n_reset, nv)
      root_lin_vel = ref_qvel[:, :3].float()
      root_ang_vel = ref_qvel[:, 3:6].float()
    else:
      root_lin_vel = torch.zeros(n_reset, 3, device=device)
      root_ang_vel = torch.zeros(n_reset, 3, device=device)

    root_state = torch.cat(
      [root_pos, root_quat, root_lin_vel, root_ang_vel], dim=-1
    )  # (n_reset, 13)
    entity = env.scene[entity_name]
    entity.write_root_state_to_sim(root_state, env_ids=env_ids_long)

    # --- Write joint state (non-root DOFs) ---
    joint_pos = ref_qpos[:, 7:].float()  # (n_reset, nq-7)
    if clip_source._qvel_tensor is not None:
      joint_vel = ref_qvel[:, 6:].float()  # (n_reset, nv-6)
    else:
      joint_vel = torch.zeros(n_reset, ref_qpos.shape[1] - 7, device=device)
    entity.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_long)

  return _fn