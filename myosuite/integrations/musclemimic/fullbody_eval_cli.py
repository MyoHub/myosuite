# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.

"""MyoSuite full-body evaluation entrypoint.

**Policy evaluation with viewer** (HF/local checkpoint path + motion + MuJoCo
viewer): pass ``--path`` and ``--motion_path`` with
``--use_mujoco --mujoco_viewer``. This delegates to upstream
``python -m fullbody.eval`` so policy actions are used.

**Native local playback**: pass ``--path`` and ``--motion_path`` without
MuJoCo viewer policy delegation. This path avoids upstream imports and can run
local policy inference when checkpoint ``train_state`` artifacts are present;
otherwise it replays trajectory states for deterministic diagnostics.

Native playback replays trajectory states from the motion clip for
deterministic tracking diagnostics; checkpoint directories are resolved and
validated for CLI contract, but policy inference is delegated only in the
viewer policy mode above.

**MyoSuite-only smoke test**: omit ``--path`` and use ``--n-steps`` /
``--seed`` for a short MJX rollout on ``MjxMimicFullbody-v0``.

On **macOS** with ``--mujoco_viewer``, MuJoCo passive viewer requires
``mjpython``. Do **not** run MyoSuite as the **outer** process with
``mjpython`` (including ``mjpython -m myosuite…``) or ``uv run mjpython``.
Use ``python -m myosuite…`` or ``uv run python -m myosuite…`` so the uv-based
venv can start.

Environment:
    MYOSUITE_PREVIEW_REEXEC_MJPYTHON: Internal guard for one-time preview
        relaunch under ``mjpython`` on macOS.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from myosuite.core.subprocess_orchestration import run_command
from myosuite.utils.video_io import write_video

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CliModeConfig:
    """Parsed mode selection flags for dispatch."""

    has_path: bool
    use_mujoco: bool
    mujoco_viewer: bool
    backend: str  # "mjx" or "mjlab"


def _strip_dispatch_only_flags_for_native_path(argv: list[str]) -> list[str]:
    """Drop ``--backend`` flags before native playback argparse.

    Top-level dispatch understands ``--backend`` for the no-``--path`` smoke
    tests, but :mod:`fullbody_native_playback` only accepts MuJoCo playback
    flags. Passing ``--backend mjlab`` together with ``--path`` is a common
    copy-paste mistake; strip known values so native playback still runs.

    Args:
        argv: Raw CLI argv after ``sys.argv[1:]`` style stripping.

    Returns:
        A new argv list without ``--backend`` / ``--backend=…`` when the value
        is ``mjx`` or ``mjlab``.
    """
    out: list[str] = []
    i = 0
    n = len(argv)
    while i < n:
        arg = argv[i]
        if arg == "--backend" and i + 1 < n and argv[i + 1] in ("mjx", "mjlab"):
            i += 2
            continue
        if arg.startswith("--backend="):
            _, _, val = arg.partition("=")
            if val in ("mjx", "mjlab"):
                i += 1
                continue
        out.append(arg)
        i += 1
    return out


def _parse_mode_config(argv: list[str]) -> CliModeConfig:
    """Parse only mode-selection flags from argv."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--path", type=str, default="")
    p.add_argument("--use_mujoco", action="store_true")
    p.add_argument("--mujoco_viewer", action="store_true")
    p.add_argument("--backend", type=str, default="mjx", choices=["mjx", "mjlab"])
    ns, _unknown = p.parse_known_args(argv)
    return CliModeConfig(
        has_path=bool(ns.path),
        use_mujoco=bool(ns.use_mujoco),
        mujoco_viewer=bool(ns.mujoco_viewer),
        backend=ns.backend,
    )


def _reject_deprecated_flags(argv: list[str]) -> int | None:
    """Return an error code when legacy CLI flags are provided.

    Args:
        argv: Raw CLI argv list.

    Returns:
        ``2`` when deprecated flags are present, else ``None``.
    """
    if "--playback" not in argv:
        return None
    print(
        "Flag '--playback' is obsolete and no longer supported.\n"
        "Use '--path ... --motion_path ... --use_mujoco --mujoco_viewer' for "
        "interactive playback, or '--record' for MP4 export.",
        file=sys.stderr,
    )
    return 2


def _exit_no_jax_for_smoke() -> NoReturn:
    """Log why MJX smoke cannot run and exit with code 1."""
    if sys.platform == "darwin":
        logger.error(
            "JAX is not available. On macOS, MyoSuite does not ship the `mjx` "
            "optional stack via PyPI markers; use `--path` with an installed "
            "MuscleMimic tree + MuJoCo, or run this smoke test on Linux. See "
            "myosuite/integrations/musclemimic/README.md."
        )
    else:
        logger.error(
            "JAX is required for MJX smoke: pip install 'myosuite[mjx]' on a "
            "supported platform (Linux, non-Windows)."
        )
    raise SystemExit(1)


def _preview_requested(cfg: CliModeConfig) -> bool:
    """True if the caller asked for MuJoCo interactive preview/viewer."""
    return cfg.use_mujoco and cfg.mujoco_viewer


def _maybe_reexec_preview_under_mjpython(argv: list[str]) -> int | None:
    """On macOS, relaunch preview with ``mjpython`` when needed.

    MuJoCo ``viewer.launch_passive`` requires ``mjpython`` on macOS. When the
    outer process is regular ``python`` and this command is preview-only
    (no ``--path``), attempt a one-time self-reexec with ``mjpython``.
    """
    if sys.platform != "darwin":
        return None
    if Path(sys.executable).name == "mjpython":
        return None
    if os.environ.get("MYOSUITE_PREVIEW_REEXEC_MJPYTHON") == "1":
        return None
    mjpython = shutil.which("mjpython")
    if not mjpython:
        return None
    logger.info("Relaunching preview under mjpython: %s", mjpython)
    env = os.environ.copy()
    env["MYOSUITE_PREVIEW_REEXEC_MJPYTHON"] = "1"
    cmd = [
        mjpython,
        "-m",
        "myosuite.integrations.musclemimic.fullbody_eval_cli",
        *argv,
    ]
    return run_command(cmd=cmd, env=env)


def _native_path_main(argv: list[str]) -> int:
    """Run MyoSuite-native trajectory playback for --path invocations."""
    from myosuite.integrations.musclemimic import (
        fullbody_native_playback as _native_playback,
    )

    try:
        return _native_playback.main(argv)
    except SystemExit as err:
        code = int(getattr(err, "code", 2) or 2)
        print(
            "Native --path playback argument validation failed.",
            file=sys.stderr,
        )
        return code if code > 0 else 2
    except Exception as err:
        print(
            "Native --path playback failed. This mode requires "
            "--path, --motion_path, --use_mujoco, and either --mujoco_viewer "
            f"or --record, plus valid artifacts.\nOriginal error: {err}",
            file=sys.stderr,
        )
        return 2


def _policy_path_main(argv: list[str]) -> int:
    """Run upstream fullbody policy inference via MuscleMimic eval entrypoint."""
    from myosuite.integrations.musclemimic.runtime import (
        build_fullbody_eval_command,
        resolve_musclemimic_exec_cwd,
    )

    exec_cwd = resolve_musclemimic_exec_cwd()
    cmd, env = build_fullbody_eval_command(exec_cwd, argv)
    return run_command(cmd=cmd, cwd=exec_cwd, env=env)


def _path_mode_main(argv: list[str]) -> int:
    """Run MyoSuite-native replay path without upstream dependency."""
    return _native_path_main(_strip_dispatch_only_flags_for_native_path(argv))


def _should_use_policy_eval(cfg: CliModeConfig) -> bool:
    """True when --path invocation should run upstream policy evaluation."""
    return cfg.has_path and cfg.use_mujoco and cfg.mujoco_viewer


def _use_upstream_policy_backend() -> bool:
    """Whether upstream ``fullbody.eval`` delegation is explicitly enabled."""
    return os.environ.get("MYOSUITE_MUSCLEMIMIC_USE_UPSTREAM", "0") == "1"


def _minimal_mjx_main(argv: list[str]) -> int:
    """Short MJX rollout on ``MjxMimicFullbody-v0`` (no MM eval)."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a short MJX full-body rollout (MyoSuite core only). "
            "For HF/motion/MuJoCo playback, pass --path … (needs musclemimic)."
        )
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=16,
        help="Number of control steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for reset and actions.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only print errors (default logs a short summary).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO)

    try:
        import jax
        import jax.numpy as jp
    except ImportError:
        _exit_no_jax_for_smoke()

    try:
        from myosuite.envs.myo.backends.mjx import make
    except Exception as err:
        logger.error("MJX stack not available: %s", err)
        msg = str(err)
        if "mjDSBL_PASSIVE" in msg or "mjtDisableBit" in msg:
            logger.error(
                "MuJoCo and mujoco.mjx versions appear mismatched (try: pip "
                "install -U 'mujoco>=3.3' or use Linux with myosuite[mjx]). "
                "See myosuite/integrations/musclemimic/README.md "
                "(Troubleshooting)"
            )
        raise SystemExit(1) from err

    try:
        env = make("MjxMimicFullbody-v0")
    except Exception as err:
        logger.error(
            "Failed to build MjxMimicFullbody-v0 "
            "(install myosuite[musclemimic] for MJCF): %s",
            err,
        )
        raise SystemExit(1) from err

    rng = jax.random.PRNGKey(int(args.seed))
    state = env.reset(rng)
    total_reward = jp.array(0.0, dtype=jp.float32)
    for _ in range(int(args.n_steps)):
        rng, k_act = jax.random.split(rng)
        action = jax.random.uniform(
            k_act,
            (env.action_size,),
            minval=-1.0,
            maxval=1.0,
        )
        state = env.step(state, action)
        total_reward = total_reward + state.reward

    mean_r = float(total_reward / float(args.n_steps))
    obs_dim = int(state.obs["state"].shape[0])
    if not args.quiet:
        logger.info(
            "MjxMimicFullbody-v0: steps=%d obs_dim=%d mean_reward=%.6f",
            args.n_steps,
            obs_dim,
            mean_r,
        )
    return 0


def _mjlab_onnx_main(
    onnx_path: str,
    motion_path: str | None,
    path_ref: str | None,
    record: bool = False,
    record_path: str = "fullbody_onnx_mjlab.mp4",
    n_steps: int = 1000,
) -> int:
    """Run ONNX policy with CPU MuJoCo simulation and optional viewer/record.

    Uses pure CPU physics (no warp/GPU) to avoid platform-specific crashes.
    The ONNX policy runs on CPU via onnxruntime; obs are built by
    ``FullbodyObsAdapter`` from the CPU ``MjData`` each step.

    On macOS, ``mujoco.viewer.launch_passive`` requires ``mjpython``.  When
    the viewer cannot be opened (e.g. missing ``mjpython``), pass ``--record``
    to save an MP4 instead.

    Args:
        onnx_path: Path to the exported ``.onnx`` policy file.
        motion_path: MuscleMimic motion key (e.g. ``KIT/314/walking_medium09_poses``).
        path_ref: Checkpoint reference (``hf://…`` or local) for goal_params.
        record: Save MP4 instead of opening viewer.
        record_path: Output MP4 path (used when ``record=True``).
        n_steps: Maximum number of steps to run.

    Returns:
        Exit code (0 on success).
    """
    from pathlib import Path as _Path

    import mujoco
    import numpy as np

    from myosuite.integrations.musclemimic.fullbody_checkpoint_io import (
        resolve_checkpoint_ref,
    )
    from myosuite.integrations.musclemimic.fullbody_local_policy import (
        FullbodyObsAdapter,
        OnnxPolicyRunner,
        fullbody_history_settings_from_metadata,
        fullbody_obs_adapter_params_from_metadata,
        read_checkpoint_config_metadata,
    )
    from myosuite.integrations.musclemimic.fullbody_model import (
        compile_mimic_fullbody_mjmodel,
        default_mimic_fullbody_config,
    )
    from myosuite.integrations.musclemimic.trajectory_io import (
        load_motion_clip,
        resolve_motion_path,
    )

    onnx_path_obj = _Path(onnx_path)
    if not onnx_path_obj.exists():
        logger.error("--onnx file not found: %s", onnx_path_obj)
        return 2
    if not motion_path:
        logger.error(
            "--motion_path is required with --onnx. "
            "Example: --motion_path KIT/314/walking_medium09_poses"
        )
        return 2

    # --- Build CPU model ---
    model, _, _ = compile_mimic_fullbody_mjmodel(default_mimic_fullbody_config())

    # --- Load motion clip ---
    try:
        motion_file = resolve_motion_path(motion_path)
        clip = load_motion_clip(motion_file, expected_nq=model.nq, expected_nv=model.nv)
    except FileNotFoundError as err:
        logger.error("Could not resolve --motion_path %r: %s", motion_path, err)
        return 2

    # --- Load goal_params from checkpoint metadata ---
    goal_params: dict = {}
    history_settings: dict = {}
    if path_ref:
        try:
            ckpt_ref = resolve_checkpoint_ref(path_ref)
            cfg_meta = read_checkpoint_config_metadata(ckpt_ref.local_path)
            goal_params = fullbody_obs_adapter_params_from_metadata(cfg_meta)
            history_settings = fullbody_history_settings_from_metadata(cfg_meta)
        except Exception as err:
            logger.warning(
                "Could not load goal_params from --path (%s); using defaults.", err
            )

    # --- Build obs adapter + ONNX runner ---
    try:
        obs_adapter = FullbodyObsAdapter(
            model=model, clip=clip, goal_params=goal_params
        )
    except ValueError as err:
        logger.error("FullbodyObsAdapter init failed: %s", err)
        return 2

    try:
        import onnx as _onnx

        _m = _onnx.load(str(onnx_path_obj), load_external_data=False)
        obs_dim = int(_m.graph.input[0].type.tensor_type.shape.dim[1].dim_value)
    except Exception as err:
        logger.error("Could not read obs_dim from ONNX graph: %s", err)
        return 2

    runner = OnnxPolicyRunner(
        onnx_path=onnx_path_obj,
        obs_dim=obs_dim,
        action_dim=model.nu,
        obs_adapter=obs_adapter,
        **history_settings,
    )
    logger.info(
        "ONNX policy loaded: %s  obs_dim=%d  action_dim=%d",
        onnx_path_obj.name,
        obs_dim,
        model.nu,
    )

    # --- Run ---
    data = mujoco.MjData(model)
    data.qpos[:] = clip.qpos[0]
    if clip.qvel is not None and clip.qvel.shape[0] > 0:
        data.qvel[:] = clip.qvel[0]
    mujoco.mj_forward(model, data)

    n_frames = int(clip.qpos.shape[0])
    steps = min(max(1, n_steps), n_frames)

    if record:
        # Offscreen MP4 record — works on all platforms.
        offwidth = int(
            getattr(getattr(model, "vis", None) and model.vis.global_, "offwidth", 0)
            or 0
        )
        offheight = int(
            getattr(getattr(model, "vis", None) and model.vis.global_, "offheight", 0)
            or 0
        )
        width = min(1280, offwidth) if offwidth > 0 else 640
        height = min(720, offheight) if offheight > 0 else 480
        renderer = mujoco.Renderer(model, height=height, width=width)
        frames: list[np.ndarray] = []
        try:
            for i in range(steps):
                action = runner.action_for(data, clip, i)
                runner.step(model, data, action)
                renderer.update_scene(data)
                frames.append(np.asarray(renderer.render(), dtype=np.uint8))
        finally:
            renderer.close()
        out = _Path(record_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        fps = max(1, int(round(1.0 / float(model.opt.timestep))))
        write_video(str(out), np.asarray(frames), outputdict={"-r": str(fps)})
        logger.info("Saved %d-frame policy rollout to %s", len(frames), out)
        return 0

    # Interactive viewer.
    import mujoco.viewer as _mjviewer  # noqa: F401 — ensure submodule loaded

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            i = 0
            while viewer.is_running() and i < steps:
                action = runner.action_for(data, clip, i)
                runner.step(model, data, action)
                viewer.sync()
                i += 1
        return 0
    except RuntimeError as err:
        if "mjpython" in str(err):
            logger.error(
                "mujoco.viewer requires mjpython on macOS but could not be loaded. "
                "Use --record to save an MP4 instead:\n"
                "  uv run myosuite-musclemimic-fullbody-eval "
                "--backend mjlab --path %s --motion_path %s --onnx %s --record",
                path_ref or "<path>",
                motion_path,
                onnx_path,
            )
            return 1
        raise


def _mjlab_play_main(argv: list[str]) -> int:
    """Delegate to mjlab's ``play`` CLI for interactive GPU→CPU visualization.

    mjlab's ``NativeMujocoViewer`` syncs GPU simulation state to a CPU
    ``MjData`` each frame (``qpos``, ``qvel``, ``ctrl`` → ``.cpu().numpy()``
    → ``mj_forward``), then hands it to MuJoCo's passive viewer via
    ``viewer.sync()``.  This gives a true GPU-physics render without copying
    the full simulation to CPU.

    Accepts a subset of mjlab play flags forwarded to the task:

        --agent zero|random   Built-in dummy agent (default: zero)
        --checkpoint PATH     Load a local checkpoint file
        --wandb-run-path ORG/PROJ/RUN  Fetch checkpoint from W&B

    Any additional flags are passed through to ``mjlab play`` unchanged.

    Args:
        argv: CLI arguments list (may include --backend mjlab).

    Returns:
        Exit code from the mjlab play process.
    """
    try:
        import mjlab  # noqa: F401
    except ImportError:
        logger.error(
            "mjlab is not installed. Install it with: pip install 'myosuite[mjlab]'. "
            "See myosuite/integrations/musclemimic/README.md (mjlab section)."
        )
        raise SystemExit(1)

    # Strip CPU-path flags and extract mjlab-relevant ones.
    # These flags are understood by the CPU eval path but not by mjlab play.
    _CPU_ONLY_FLAGS = {
        "--use_mujoco",
        "--mujoco_viewer",
        "--record",
        "--stochastic",
        "--no_show_targets",
    }
    _CPU_ONLY_VALUE_FLAGS = {
        "--record_path",
        "--eval_seed",
        "--n_steps",
    }

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--backend", type=str, default="mjlab")
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("--motion_path", type=str, default=None)
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--onnx", type=str, default=None)
    # Consume CPU-only value flags so they don't leak to mjlab play.
    for _f in _CPU_ONLY_VALUE_FLAGS:
        parser.add_argument(_f, type=str, default=None)
    # Consume CPU-only boolean flags.
    for _f in _CPU_ONLY_FLAGS:
        parser.add_argument(_f, action="store_true", default=False)
    _known, extra_argv = parser.parse_known_args(argv)

    logging.basicConfig(level=logging.WARNING if _known.quiet else logging.INFO)

    # If --motion_path given, register Mimic tasks in trajectory mode so
    # mjlab play's NativeMujocoViewer tracks the clip's reference targets.
    try:
        from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
            register_mjlab_tasks,
        )

        if _known.motion_path:
            from myosuite.core.trajectory_io import (
                load_motion_clip,
                resolve_motion_path,
            )
            from myosuite.envs.myo.backends.mjlab.mimic_mjlab_env import (
                register_mimic_mjlab_tasks_with_clip,
            )
            from mjlab.tasks.registry import register_mjlab_task as _rmt
            from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import (
                _elbow_ppo_runner_cfg,
            )

            try:
                motion_file = resolve_motion_path(_known.motion_path)
                clip = load_motion_clip(motion_file, expected_nq=89, expected_nv=88)
                register_mimic_mjlab_tasks_with_clip(
                    _rmt, _elbow_ppo_runner_cfg, clip=clip
                )
                logger.info(
                    "Registered trajectory mode with clip: %s", _known.motion_path
                )
            except Exception as err:
                logger.warning(
                    "Could not load --motion_path %r (%s); "
                    "falling back to random-target mode.",
                    _known.motion_path,
                    err,
                )
                register_mjlab_tasks()
        else:
            register_mjlab_tasks()
    except Exception as err:
        logger.warning("Task registration failed (may already be registered): %s", err)

    # Resolve --path to a mjlab-compatible --checkpoint if possible.
    # --path may reference an Orbax/JAX checkpoint (hf://... or local dir).
    # mjlab play needs a PyTorch .pt file; if none found, fall back to
    # --agent zero and warn the user.
    path_ref = _known.path  # type: ignore[attr-defined]
    if (
        path_ref
        and "--checkpoint" not in extra_argv
        and "--wandb-run-path" not in extra_argv
    ):
        try:
            from myosuite.integrations.musclemimic.fullbody_checkpoint_io import (
                resolve_checkpoint_ref,
            )

            ckpt_ref = resolve_checkpoint_ref(path_ref)
            pt_files = sorted(ckpt_ref.local_path.rglob("*.pt"))
            if pt_files:
                extra_argv = ["--checkpoint", str(pt_files[0])] + extra_argv
                logger.info("Resolved --path to mjlab checkpoint: %s", pt_files[0])
            else:
                logger.warning(
                    "--path %r resolved to %s which contains no .pt checkpoint "
                    "files (found Orbax/JAX format). "
                    "mjlab play requires a PyTorch .pt checkpoint. "
                    "Falling back to --agent zero. "
                    "To use the trained policy, run without --backend mjlab:\n"
                    "  uv run myosuite-musclemimic-fullbody-eval "
                    "--path %s --motion_path ... --use_mujoco --mujoco_viewer",
                    path_ref,
                    ckpt_ref.local_path,
                    path_ref,
                )
                if "--agent" not in extra_argv:
                    extra_argv = ["--agent", "zero"] + extra_argv
        except Exception as err:
            logger.warning("Could not resolve --path %r: %s", path_ref, err)
            if "--agent" not in extra_argv:
                extra_argv = ["--agent", "zero"] + extra_argv

    # If --onnx is provided, run the CPU policy loop directly (no warp/GPU).
    # mjlab's NativeMujocoViewer requires ManagerBasedRlEnv.reset() which can
    # crash with warp JIT issues on some platforms; the CPU path is portable.
    if _known.onnx:
        record_flag = getattr(_known, "record", False)
        record_path_val = (
            getattr(_known, "record_path", "fullbody_onnx_mjlab.mp4")
            or "fullbody_onnx_mjlab.mp4"
        )
        n_steps_val = int(getattr(_known, "n_steps", None) or 1000)
        return _mjlab_onnx_main(
            onnx_path=_known.onnx,
            motion_path=_known.motion_path,
            path_ref=_known.path,
            record=bool(record_flag),
            record_path=str(record_path_val),
            n_steps=n_steps_val,
        )

    # mjlab play reads sys.argv directly, so we patch it before calling.
    task_argv = ["play", "myoMimicFullbody-v0"] + extra_argv
    logger.info("Delegating to mjlab play: %s", " ".join(task_argv[1:]))

    try:
        from mjlab.scripts.play import main as _mjlab_play

        _orig_argv = sys.argv[:]
        sys.argv = task_argv
        try:
            result = _mjlab_play()
        finally:
            sys.argv = _orig_argv
        return int(result or 0)
    except ImportError:
        pass  # fall through to subprocess

    # Subprocess fallback: spawn a Python process that registers tasks first.
    cmd = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "from myosuite.envs.myo.backends.mjlab.register_mjlab_tasks import register_mjlab_tasks; "
            "register_mjlab_tasks(); "
            "sys.argv = " + repr(task_argv) + "; "
            "from mjlab.scripts.play import main; "
            "raise SystemExit(main())"
        ),
    ]
    return run_command(cmd=cmd)


def main(argv: list[str] | None = None) -> int:
    """Run native playback with ``--path``; else preview or backend smoke test."""
    if argv is None:
        argv = sys.argv[1:]
    deprecated_code = _reject_deprecated_flags(argv)
    if deprecated_code is not None:
        return deprecated_code
    cfg = _parse_mode_config(argv)

    # ``--backend`` is dispatch-only for no-``--path`` mode. When ``--path`` is
    # present we run native path playback and strip backend flags there.
    if cfg.backend == "mjlab" and not cfg.has_path:
        return _mjlab_play_main(argv)

    if _preview_requested(cfg):
        reexec_code = _maybe_reexec_preview_under_mjpython(argv)
        if reexec_code is not None:
            return reexec_code
    if _preview_requested(cfg) and not cfg.has_path:
        from myosuite.integrations.musclemimic import (
            fullbody_mujoco_preview as _fb_preview,
        )

        return _fb_preview.main(argv)
    if cfg.has_path:
        if _should_use_policy_eval(cfg):
            if _use_upstream_policy_backend():
                try:
                    return _policy_path_main(argv)
                except ImportError as err:
                    logger.warning(
                        "Upstream policy backend unavailable (%s). "
                        "Falling back to MyoSuite-native playback.",
                        err,
                    )
            else:
                logger.info(
                    "Using MyoSuite-native local policy runner. "
                    "Set MYOSUITE_MUSCLEMIMIC_USE_UPSTREAM=1 to force upstream "
                    "fullbody.eval delegation."
                )
            return _path_mode_main(argv)
        return _path_mode_main(argv)
    return _minimal_mjx_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
