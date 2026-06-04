# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""Native trajectory playback for MyoFullBody.

Does not import upstream ``musclemimic`` runtime modules.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path

# pylint: disable=no-member
import mujoco
import numpy as np

from myosuite.core.playback_contract import (
    PlaybackArtifacts,
    PlaybackRequest,
    PlaybackResult,
    PlaybackRunner,
)
from myosuite.integrations.musclemimic.fullbody_checkpoint_io import (
    describe_checkpoint_dir,
    resolve_checkpoint_ref,
)
from myosuite.integrations.musclemimic.fullbody_model import (
    FULLBODY_BODY2SITES_FOR_MIMIC,
    compile_mimic_fullbody_mjmodel,
    default_mimic_fullbody_config,
)
from myosuite.integrations.musclemimic.fullbody_local_policy import (
    FullbodyObsAdapter,
    LocalPolicyRunner,
    OnnxPolicyRunner,
    fullbody_history_settings_from_metadata,
    fullbody_obs_adapter_params_from_metadata,
    has_local_policy_artifacts,
    load_local_policy_artifacts,
    read_checkpoint_config_metadata,
)
from myosuite.integrations.musclemimic.trajectory_io import (
    MotionClip,
    load_motion_clip,
    resolve_motion_path,
)
from myosuite.viz.site_marker_viz import make_site_marker_viz_fn
from myosuite.integrations.musclemimic.playback_runners import (
    run_passive_viewer_loop,
)
from myosuite.utils.video_io import write_video

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NativePlaybackArgs:
    """Arguments for native fullbody trajectory playback."""

    checkpoint_path: str
    motion_path: str
    n_steps: int
    eval_seed: int
    use_mujoco: bool
    mujoco_viewer: bool
    stochastic: bool
    record: bool
    record_path: str
    record_fps: int | None
    show_targets: bool = True
    onnx: str | None = None


def parse_native_playback_argv(argv: list[str]) -> NativePlaybackArgs:
    """Parse CLI args needed by native playback."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--path", required=True, type=str)
    p.add_argument("--motion_path", required=True, type=str)
    p.add_argument("--n_steps", type=int, default=1000)
    p.add_argument("--eval_seed", type=int, default=0)
    p.add_argument("--use_mujoco", action="store_true")
    p.add_argument("--mujoco_viewer", action="store_true")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--record", action="store_true")
    p.add_argument(
        "--record_path",
        type=str,
        default="fullbody_native_playback.mp4",
    )
    p.add_argument("--record_fps", type=int, default=None)
    p.add_argument(
        "--no_show_targets",
        action="store_true",
        help="Disable orange reference-site sphere markers in the viewer.",
    )
    p.add_argument(
        "--onnx",
        type=str,
        default=None,
        help=(
            "Path to an exported ONNX policy file (e.g. mm-10m-2.onnx). "
            "When provided, ONNX inference replaces the Orbax/numpy local policy. "
            "Obs normalization must be baked into the ONNX graph "
            "(as produced by: export_onnx.py export --framework orbax …)."
        ),
    )
    ns = p.parse_args(argv)
    return NativePlaybackArgs(
        checkpoint_path=ns.path,
        motion_path=ns.motion_path,
        n_steps=int(ns.n_steps),
        eval_seed=int(ns.eval_seed),
        use_mujoco=bool(ns.use_mujoco),
        mujoco_viewer=bool(ns.mujoco_viewer),
        stochastic=bool(ns.stochastic),
        record=bool(ns.record),
        record_path=str(ns.record_path),
        record_fps=(int(ns.record_fps) if ns.record_fps is not None else None),
        show_targets=not bool(ns.no_show_targets),
        onnx=ns.onnx if ns.onnx else None,
    )


def _build_model() -> tuple[mujoco.MjModel, str]:
    cfg = default_mimic_fullbody_config()
    model, _spec, xml_path = compile_mimic_fullbody_mjmodel(cfg)
    floor_z_offset = float(os.environ.get("MYOSUITE_MIMIC_FLOOR_Z_OFFSET", "0.0"))
    if abs(floor_z_offset) > 0.0:
        for geom_id in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if name.lower() == "floor":
                model.geom_pos[geom_id, 2] = (
                    float(model.geom_pos[geom_id, 2]) + floor_z_offset
                )
                logger.warning(
                    "Applied floor z offset %.6f to geom '%s' (new z=%.6f).",
                    floor_z_offset,
                    name,
                    float(model.geom_pos[geom_id, 2]),
                )
                break
    return model, xml_path


class FullbodyNativePlaybackRunner(PlaybackRunner):
    """Native fullbody replay runner for MuJoCo passive viewer."""

    def __init__(
        self,
        model: mujoco.MjModel,
        clip: MotionClip,
        show_targets: bool = True,
    ) -> None:
        self._model = model
        self._clip = clip
        self._show_targets = show_targets

    def run(
        self,
        request: PlaybackRequest,
        artifacts: PlaybackArtifacts,
    ) -> PlaybackResult:
        """Run trajectory replay and return tracking metrics."""
        data = mujoco.MjData(self._model)
        n_frames = int(self._clip.qpos.shape[0])
        n_steps = min(max(1, request.n_steps), n_frames)
        if request.options.get("record", False):
            site_errors = _run_offscreen_record_loop(
                model=self._model,
                data=data,
                clip=self._clip,
                n_steps=n_steps,
                output_path=Path(str(request.options["record_path"])),
                fps=request.options.get("record_fps"),
            )
        else:
            viz_fn = (
                _make_target_viz_fn(self._clip, self._model)
                if self._show_targets and self._clip.site_xpos is not None
                else None
            )
            site_errors = run_passive_viewer_loop(
                model=self._model,
                data=data,
                n_steps=n_steps,
                step_fn=lambda i, m, d: _step_with_clip(i, m, d, self._clip),
                metric_fn=lambda i, m, d: _compute_site_tracking_error(
                    m, d, self._clip, i
                ),
                viz_fn=viz_fn,
            )
        mean_err = float(np.mean(site_errors)) if site_errors else None
        warnings: list[str] = []
        if request.options.get("stochastic", False):
            warnings.append("--stochastic is currently ignored for native replay mode.")
        if request.options.get("record", False) and request.options.get(
            "mujoco_viewer", False
        ):
            warnings.append(
                "--mujoco_viewer is ignored when --record is enabled in native "
                "playback (offscreen render is used for MP4 export)."
            )
        warnings.append(
            "Checkpoint inference is not wired yet; replay uses motion states."
        )
        return PlaybackResult(
            frames_executed=n_steps,
            mean_tracking_error=mean_err,
            warnings=tuple(warnings),
        )


class FullbodyLocalPolicyPlaybackRunner(PlaybackRunner):
    """Native fullbody policy runner using local checkpoint artifacts only."""

    def __init__(
        self,
        model: mujoco.MjModel,
        clip: MotionClip,
        policy_runner: LocalPolicyRunner,
        show_targets: bool = True,
    ) -> None:
        self._model = model
        self._clip = clip
        self._policy_runner = policy_runner
        self._show_targets = show_targets

    def run(
        self,
        request: PlaybackRequest,
        artifacts: PlaybackArtifacts,
    ) -> PlaybackResult:
        """Run policy inference in MuJoCo and return tracking metrics."""
        del artifacts
        data = mujoco.MjData(self._model)
        n_frames = int(self._clip.qpos.shape[0])
        n_steps = min(max(1, request.n_steps), n_frames)
        # Initialize from clip frame 0 for reproducible initial pose.
        data.qpos[:] = self._clip.qpos[0]
        if self._clip.qvel is not None and self._clip.qvel.shape[0] > 0:
            data.qvel[:] = self._clip.qvel[0]
        mujoco.mj_forward(self._model, data)
        if request.options.get("record", False):
            site_errors = _run_offscreen_policy_record_loop(
                model=self._model,
                data=data,
                clip=self._clip,
                policy_runner=self._policy_runner,
                n_steps=n_steps,
                output_path=Path(str(request.options["record_path"])),
                fps=request.options.get("record_fps"),
            )
            mean_err = float(np.mean(site_errors)) if site_errors else None
            return PlaybackResult(
                frames_executed=n_steps,
                mean_tracking_error=mean_err,
                warnings=(),
            )
        viz_fn = (
            _make_target_viz_fn(self._clip, self._model)
            if self._show_targets and self._clip.site_xpos is not None
            else None
        )
        site_errors = run_passive_viewer_loop(
            model=self._model,
            data=data,
            n_steps=n_steps,
            step_fn=lambda i, m, d: self._policy_runner.step(
                m, d, self._policy_runner.action_for(d, self._clip, i)
            ),
            metric_fn=lambda i, m, d: _compute_site_tracking_error(
                m, d, self._clip, min(i, n_frames - 1)
            ),
            viz_fn=viz_fn,
        )
        mean_err = float(np.mean(site_errors)) if site_errors else None
        return PlaybackResult(
            frames_executed=n_steps,
            mean_tracking_error=mean_err,
            warnings=(),
        )


def _make_target_viz_fn(
    clip: MotionClip,
    model: mujoco.MjModel,
) -> object:
    """Build a VizCallback that renders orange target spheres from *clip*.

    Args:
        clip: Motion clip with ``site_xpos`` array ``(T, n_sites, 3)``.
        model: Compiled MuJoCo model.

    Returns:
        A ``VizCallback``-compatible callable for ``run_passive_viewer_loop``.
    """
    site_names = tuple(FULLBODY_BODY2SITES_FOR_MIMIC.values())
    return make_site_marker_viz_fn(
        clip=clip,
        site_names=site_names,
        model=model,
    )


def _compute_site_tracking_error(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    clip: MotionClip,
    frame_idx: int,
) -> float | None:
    """Mean Euclidean site error for available mimic sites at frame_idx."""
    if clip.site_xpos is None or clip.site_xpos.ndim != 3:
        return None
    if frame_idx >= clip.site_xpos.shape[0]:
        return None
    n_available = clip.site_xpos.shape[1]
    if n_available <= 0:
        return None
    errs: list[float] = []
    for i, _name in enumerate(FULLBODY_BODY2SITES_FOR_MIMIC.values()):
        if i >= n_available:
            break
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, _name)
        if site_id < 0:
            continue
        current = np.asarray(data.site_xpos[site_id], dtype=np.float64)
        target = np.asarray(clip.site_xpos[frame_idx, i], dtype=np.float64)
        errs.append(float(np.linalg.norm(current - target)))
    if not errs:
        return None
    return float(np.mean(errs))


def _step_with_clip(
    frame_idx: int,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    clip: MotionClip,
) -> None:
    """Advance one frame by writing qpos/qvel from the clip."""
    data.qpos[:] = clip.qpos[frame_idx]
    if clip.qvel is not None and frame_idx < clip.qvel.shape[0]:
        data.qvel[:] = clip.qvel[frame_idx]
    mujoco.mj_forward(model, data)


def _run_offscreen_record_loop(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    clip: MotionClip,
    n_steps: int,
    output_path: Path,
    fps: int | None,
) -> list[float]:
    """Replay clip offscreen and save an MP4."""
    width, height = _resolve_offscreen_render_size(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    site_errors: list[float] = []
    frames: list[np.ndarray] = []
    try:
        for frame_idx in range(n_steps):
            _step_with_clip(frame_idx, model, data, clip)
            err = _compute_site_tracking_error(model, data, clip, frame_idx)
            if err is not None:
                site_errors.append(float(err))
            renderer.update_scene(data)
            frames.append(np.asarray(renderer.render(), dtype=np.uint8))
    finally:
        renderer.close()

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_fps = (
        int(fps)
        if fps is not None
        else max(1, int(round(1.0 / float(model.opt.timestep))))
    )
    write_video(
        str(output_path),
        np.asarray(frames, dtype=np.uint8),
        outputdict={"-r": str(target_fps)},
    )
    return site_errors


def _run_offscreen_policy_record_loop(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    clip: MotionClip,
    policy_runner: LocalPolicyRunner,
    n_steps: int,
    output_path: Path,
    fps: int | None,
) -> list[float]:
    """Run policy rollout offscreen and save an MP4."""
    width, height = _resolve_offscreen_render_size(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    site_errors: list[float] = []
    frames: list[np.ndarray] = []
    try:
        for frame_idx in range(n_steps):
            action = policy_runner.action_for(
                data=data,
                clip=clip,
                frame_idx=frame_idx,
            )
            policy_runner.step(model=model, data=data, action=action)
            err = _compute_site_tracking_error(model, data, clip, frame_idx)
            if err is not None:
                site_errors.append(float(err))
            renderer.update_scene(data)
            frames.append(np.asarray(renderer.render(), dtype=np.uint8))
    finally:
        renderer.close()

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_fps = (
        int(fps)
        if fps is not None
        else max(1, int(round(1.0 / float(model.opt.timestep))))
    )
    write_video(
        str(output_path),
        np.asarray(frames, dtype=np.uint8),
        outputdict={"-r": str(target_fps)},
    )
    return site_errors


def _resolve_offscreen_render_size(
    model: mujoco.MjModel,
) -> tuple[int, int]:
    """Choose a safe render size bounded by model offscreen framebuffer."""
    default_width = 1280
    default_height = 720
    vis = getattr(model, "vis", None)
    global_vis = getattr(vis, "global_", None) if vis is not None else None
    offwidth = getattr(global_vis, "offwidth", 0) if global_vis else 0
    offheight = getattr(global_vis, "offheight", 0) if global_vis else 0

    max_width = int(offwidth) if int(offwidth) > 0 else default_width
    max_height = int(offheight) if int(offheight) > 0 else default_height
    width = max(1, min(default_width, max_width))
    height = max(1, min(default_height, max_height))
    return width, height


def run_native_playback(parsed: NativePlaybackArgs) -> int:
    """Run native motion replay in MuJoCo viewer."""
    if not parsed.use_mujoco:
        raise ValueError("Native fullbody playback requires --use_mujoco.")
    if not parsed.mujoco_viewer and not parsed.record:
        raise ValueError(
            "Native fullbody playback requires --mujoco_viewer or --record."
        )
    ckpt = resolve_checkpoint_ref(parsed.checkpoint_path)
    ckpt_diag = describe_checkpoint_dir(ckpt.local_path)
    model, xml_path = _build_model()
    motion_file = resolve_motion_path(
        parsed.motion_path,
        env_name="MyoFullBody",
    )
    clip = load_motion_clip(
        motion_file,
        expected_nq=model.nq,
        expected_nv=model.nv,
    )
    n_frames = int(clip.qpos.shape[0])
    n_steps = min(max(1, parsed.n_steps), n_frames)
    request = PlaybackRequest(
        checkpoint_ref=parsed.checkpoint_path,
        motion_ref=parsed.motion_path,
        n_steps=n_steps,
        seed=parsed.eval_seed,
        backend="gym",
        render_mode=("record" if parsed.record else "mujoco_viewer"),
        options={
            "stochastic": parsed.stochastic,
            "record": parsed.record,
            "record_path": parsed.record_path,
            "record_fps": parsed.record_fps,
            "mujoco_viewer": parsed.mujoco_viewer,
            "show_targets": parsed.show_targets,
        },
    )
    artifacts = PlaybackArtifacts(
        checkpoint_path=ckpt.local_path,
        motion_path=motion_file,
        metadata={"checkpoint_dir": ckpt_diag, "xml_path": xml_path},
    )
    logger.info(
        "Native playback: checkpoint=%s resolved=%s (diag=%s), motion=%s "
        "frames=%d n_steps=%d xml=%s",
        parsed.checkpoint_path,
        ckpt.local_path,
        ckpt_diag,
        motion_file,
        n_frames,
        n_steps,
        xml_path,
    )
    runner: PlaybackRunner = FullbodyNativePlaybackRunner(
        model, clip, show_targets=parsed.show_targets
    )
    onnx_path = parsed.onnx
    if onnx_path is not None:
        # ONNX policy takes priority over Orbax local policy.
        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            logger.error("--onnx file not found: %s", onnx_path)
            return 2
        try:
            cfg_meta = read_checkpoint_config_metadata(ckpt.local_path)
            goal_params = fullbody_obs_adapter_params_from_metadata(cfg_meta)
            history_settings = fullbody_history_settings_from_metadata(cfg_meta)
            obs_adapter = FullbodyObsAdapter(
                model=model, clip=clip, goal_params=goal_params
            )
            # Read obs_dim from the ONNX graph itself.
            import onnx as _onnx

            _m = _onnx.load(str(onnx_path), load_external_data=False)
            _obs_dim = _m.graph.input[0].type.tensor_type.shape.dim[1].dim_value
            onnx_runner = OnnxPolicyRunner(
                onnx_path=onnx_path,
                obs_dim=int(_obs_dim),
                action_dim=model.nu,
                obs_adapter=obs_adapter,
                **history_settings,
            )
            runner = FullbodyLocalPolicyPlaybackRunner(
                model, clip, onnx_runner, show_targets=parsed.show_targets
            )
            logger.info(
                "Using ONNX policy runner: %s (obs_adapter obs; action_dim=%d)",
                onnx_path.name,
                model.nu,
            )
        except Exception as err:
            logger.error("Failed to initialize ONNX policy runner (%s). Aborting.", err)
            return 2
    disable_local_policy = (
        os.environ.get("MYOSUITE_MUSCLEMIMIC_DISABLE_LOCAL_POLICY", "0") == "1"
    )
    if (
        onnx_path is None
        and has_local_policy_artifacts(ckpt.local_path)
        and not disable_local_policy
    ):
        try:
            local_policy = load_local_policy_artifacts(ckpt.local_path)
            cfg_meta = read_checkpoint_config_metadata(ckpt.local_path)
            goal_params: dict[str, object] = fullbody_obs_adapter_params_from_metadata(
                cfg_meta
            )
            history_settings = fullbody_history_settings_from_metadata(cfg_meta)
            obs_adapter = FullbodyObsAdapter(
                model=model,
                clip=clip,
                goal_params=goal_params,
            )
            policy_runner = LocalPolicyRunner(
                artifacts=local_policy,
                stochastic=parsed.stochastic,
                seed=parsed.eval_seed,
                obs_adapter=obs_adapter,
                **history_settings,
            )
            runner = FullbodyLocalPolicyPlaybackRunner(
                model, clip, policy_runner, show_targets=parsed.show_targets
            )
            logger.warning(
                "Using local policy runner (obs_dim=%d, action_dim=%d).",
                local_policy.obs_dim,
                local_policy.action_dim,
            )
        except (ImportError, KeyError, FileNotFoundError, ValueError) as err:
            err_msg = str(err)
            if "orbax.checkpoint is required" in err_msg:
                logger.warning(
                    "Local policy artifacts found but Orbax is unavailable. "
                    "Install `orbax-checkpoint` in this environment to enable "
                    "policy actions; falling back to trajectory replay."
                )
            else:
                logger.warning(
                    "Local policy artifacts found but failed to initialize (%s). "
                    "Falling back to trajectory replay.",
                    err,
                )
    result = runner.run(request, artifacts)
    for warn in result.warnings:
        logger.warning("%s", warn)
    if result.mean_tracking_error is not None:
        logger.info(
            "Native replay mean site-tracking error over %d frames: %.6f",
            result.frames_executed,
            result.mean_tracking_error,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry-point for native fullbody trajectory playback."""
    if argv is None:
        import sys

        argv = sys.argv[1:]
    parsed = parse_native_playback_argv(argv)
    return run_native_playback(parsed)


__all__ = [
    "FullbodyLocalPolicyPlaybackRunner",
    "FullbodyNativePlaybackRunner",
    "NativePlaybackArgs",
    "main",
    "parse_native_playback_argv",
    "run_native_playback",
]
